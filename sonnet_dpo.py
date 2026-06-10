'''
[Variation 2: DPO Only]
소네트 생성 모델 - DPO Only 버전

실행:
  `python sonnet_dpo.py --epochs 1 --batch_size 2`
'''

import argparse
import os
import random
import torch
import re
import copy

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import SonnetsDataset
from models.gpt2 import GPT2Model
from optimizer import AdamW

TQDM_DISABLE = False


def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class DPODataset(Dataset):
  """
  DPO를 위해 선호 소네트(Winner)와 비선호 소네트(Loser)의 쌍을 반환하는 데이터셋.
  Loser는 Winner 소네트의 단어 순서를 문장 내에서 무작위로 섞어 시적 리듬과 구문을 인위적으로 파괴하여 생성합니다.
  """
  def __init__(self, original_dataset, tokenizer, max_length=192):
    self.original_dataset = original_dataset
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.original_dataset.sonnets)

  def _corrupt_text(self, text):
    lines = text.split('\n')
    corrupted_lines = []
    for line in lines:
      if not line.strip():
        corrupted_lines.append(line)
        continue
      words = line.split()
      if len(words) > 2:
        random.shuffle(words)
      corrupted_lines.append(" ".join(words))
    return "\n".join(corrupted_lines)

  def __getitem__(self, idx):
    sonnet_id, winner_text = self.original_dataset[idx]
    loser_text = self._corrupt_text(winner_text)
    
    winner_enc = self.tokenizer(winner_text, max_length=self.max_length, truncation=True, padding='max_length')
    loser_enc = self.tokenizer(loser_text, max_length=self.max_length, truncation=True, padding='max_length')
    
    return {
      'winner_ids': torch.tensor(winner_enc['input_ids'], dtype=torch.long),
      'winner_mask': torch.tensor(winner_enc['attention_mask'], dtype=torch.long),
      'loser_ids': torch.tensor(loser_enc['input_ids'], dtype=torch.long),
      'loser_mask': torch.tensor(loser_enc['attention_mask'], dtype=torch.long)
    }


class SonnetGPT(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    output = self.gpt(input_ids, attention_mask)
    hidden_states = output['last_hidden_state']
    logits = self.gpt.hidden_state_to_token(hidden_states)
    return logits

  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=1.0, top_p=0.9, max_length=128):
    device = self.get_device()
    tokenizer = self.tokenizer
    newline_id = tokenizer.encode('\n')[0]
    
    prompt_text = tokenizer.decode(encoding[0].tolist())
    lines = [l.strip() for l in prompt_text.strip().split('\n') if l.strip()]
    lines = lines[:3]
    while len(lines) < 3:
      lines.append("And write of thee in verse as time goes on,")
      
    full_prompt = "\n".join(lines) + "\n"
    context_ids = tokenizer.encode(full_prompt, return_tensors='pt').to(device)
    attn_mask = torch.ones(context_ids.shape, dtype=torch.int64).to(device)
    
    import pronouncing
    import re
    from evaluation import count_syllables_and_stress
    
    rhyme_map = {2: 0, 3: 1, 6: 4, 7: 5, 10: 8, 11: 9, 13: 12}
    
    for i in range(3, 14):
      rhyme_token_ids = []
      if i in rhyme_map:
        base_line_idx = rhyme_map[i]
        if base_line_idx < len(lines):
          base_line = lines[base_line_idx]
          words = re.findall(r"\b\w+(?:'\w+)?\b", base_line)
          if words:
            last_word = words[-1].lower().strip(".,;:!?-\"()'")
            rhymes = pronouncing.rhymes(last_word)
            rhymes.append(last_word)
            for rw in rhymes:
              for prefix in ["", " "]:
                ids = tokenizer.encode(prefix + rw)
                if len(ids) == 1:
                  rhyme_token_ids.append(ids[0])
            rhyme_token_ids = list(set(rhyme_token_ids))
            
      current_line_tokens = []
      rhyme_selected = False
      
      for step in range(30):
        logits = self.forward(context_ids, attn_mask)[:, -1, :].clone() / temperature
        
        current_text = tokenizer.decode(current_line_tokens).strip()
        syllable_count, _ = count_syllables_and_stress(current_text) if current_text else (0, [])
        
        # 5음절 미만이거나 토큰이 3개 미만인 경우 조기 개행 생성 방지
        if syllable_count < 5 or len(current_line_tokens) < 3:
          logits[0, newline_id] -= 100.0
          
        if 8 <= syllable_count <= 11 and rhyme_token_ids and not rhyme_selected:
          for tid in rhyme_token_ids:
            if tid < logits.shape[-1]:
              logits[0, tid] += 50.0
          logits[0, newline_id] -= 100.0
          
        if rhyme_selected or syllable_count >= 11:
          logits[0, newline_id] += 80.0
          
        probs = torch.nn.functional.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumulative_probs <= top_p
        top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
        top_p_mask[..., 0] = True
        filtered_probs = sorted_probs * top_p_mask
        
        prob_sum = filtered_probs.sum(dim=-1, keepdim=True)
        if prob_sum.item() == 0.0:
          filtered_probs[..., 0] = 1.0
          prob_sum = filtered_probs.sum(dim=-1, keepdim=True)
        filtered_probs /= prob_sum
        
        sampled_index = torch.multinomial(filtered_probs, 1)
        sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)
        token_id = sampled_token.item()
        
        if token_id in rhyme_token_ids:
          rhyme_selected = True
          
        if token_id == newline_id:
          break
          
        current_line_tokens.append(token_id)
        context_ids = torch.cat([context_ids, sampled_token], dim=1)
        attn_mask = torch.cat([attn_mask, torch.ones((1, 1), dtype=torch.int64).to(device)], dim=1)
        
      new_line = tokenizer.decode(current_line_tokens).strip()
      lines.append(new_line)
      
      newline_token = torch.tensor([[newline_id]]).to(device)
      context_ids = torch.cat([context_ids, newline_token], dim=1)
      attn_mask = torch.cat([attn_mask, torch.ones((1, 1), dtype=torch.int64).to(device)], dim=1)
      
    generated_tokens = context_ids[0][encoding.shape[1]:]
    generated_output = tokenizer.decode(generated_tokens.cpu().numpy().tolist())
    return context_ids, generated_output


def get_log_probs(logits, labels):
  """
  생성 토큰의 로그 확률을 구합니다. (Label과 대응하는 Logits 간의 log_softmax 매핑)
  """
  # logits: (batch_size, seq_len, vocab_size)
  # labels: (batch_size, seq_len)
  shift_logits = logits[:, :-1, :].contiguous()
  shift_labels = labels[:, 1:].contiguous()
  
  log_probs = F.log_softmax(shift_logits, dim=-1)
  per_token_logps = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(-1)).squeeze(-1)
  
  # padding 토큰(eos_token_id)은 마스킹하여 로그 확률 계산에서 배제
  mask = (shift_labels != 50256).float()
  return (per_token_logps * mask).sum(dim=-1)


def dpo_loss(policy_win_logps, policy_lose_logps, ref_win_logps, ref_lose_logps, beta=0.1):
  """
  DPO Loss = -logsigmoid(beta * (policy_win_logps - ref_win_logps) - beta * (policy_lose_logps - ref_lose_logps))
  """
  policy_ratio = policy_win_logps - policy_lose_logps
  ref_ratio = ref_win_logps - ref_lose_logps
  logits = policy_ratio - ref_ratio
  loss = -F.logsigmoid(beta * logits).mean()
  return loss


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
  }
  torch.save(save_info, filepath)
  print(f"saved the model to {filepath}")


def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  base_sonnet_dataset = SonnetsDataset(args.sonnet_path)
  
  # DPO 전용 Winner/Loser 데이터셋 구축
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  tokenizer.pad_token = tokenizer.eos_token
  dpo_dataset = DPODataset(base_sonnet_dataset, tokenizer)
  
  val_size = int(0.1 * len(dpo_dataset))
  if val_size == 0 and len(dpo_dataset) > 1:
    val_size = 1
  train_size = len(dpo_dataset) - val_size
  train_dataset, val_dataset = torch.utils.data.random_split(
    dpo_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(args.seed)
  )

  train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
  val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)

  args = add_arguments(args)
  
  # 1. Policy 모델 (학습 진행)
  policy_model = SonnetGPT(args).to(device)
  
  # 2. Reference 모델 (동결/Freeze)
  ref_model = SonnetGPT(args).to(device)
  ref_model.eval()
  for param in ref_model.parameters():
    param.requires_grad = False

  optimizer = AdamW(policy_model.parameters(), lr=args.lr)
  best_val_loss = float('inf')
  patience = 3
  patience_counter = 0

  for epoch in range(args.epochs):
    policy_model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(train_loader, desc=f'DPO train-{epoch}', disable=TQDM_DISABLE):
      w_ids = batch['winner_ids'].to(device)
      w_mask = batch['winner_mask'].to(device)
      l_ids = batch['loser_ids'].to(device)
      l_mask = batch['loser_mask'].to(device)

      # Reference 모델의 로그 확률 계산
      with torch.no_grad():
        ref_win_logits = ref_model(w_ids, w_mask)
        ref_lose_logits = ref_model(l_ids, l_mask)
        ref_win_logps = get_log_probs(ref_win_logits, w_ids)
        ref_lose_logps = get_log_probs(ref_lose_logits, l_ids)

      # Policy 모델의 로그 확률 계산
      optimizer.zero_grad()
      pol_win_logits = policy_model(w_ids, w_mask)
      pol_lose_logits = policy_model(l_ids, l_mask)
      pol_win_logps = get_log_probs(pol_win_logits, w_ids)
      pol_lose_logps = get_log_probs(pol_lose_logits, l_ids)

      # DPO Loss 연산
      loss = dpo_loss(pol_win_logps, pol_lose_logps, ref_win_logps, ref_lose_logps, beta=args.dpo_beta)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    # 검증 루프
    policy_model.eval()
    val_loss = 0
    val_batches = 0
    with torch.no_grad():
      for batch in val_loader:
        w_ids = batch['winner_ids'].to(device)
        w_mask = batch['winner_mask'].to(device)
        l_ids = batch['loser_ids'].to(device)
        l_mask = batch['loser_mask'].to(device)

        ref_win_logits = ref_model(w_ids, w_mask)
        ref_lose_logits = ref_model(l_ids, l_mask)
        ref_win_logps = get_log_probs(ref_win_logits, w_ids)
        ref_lose_logps = get_log_probs(ref_lose_logits, l_ids)

        pol_win_logits = policy_model(w_ids, w_mask)
        pol_lose_logits = policy_model(l_ids, l_mask)
        pol_win_logps = get_log_probs(pol_win_logits, w_ids)
        pol_lose_logps = get_log_probs(pol_lose_logits, l_ids)

        loss = dpo_loss(pol_win_logps, pol_lose_logps, ref_win_logps, ref_lose_logps, beta=args.dpo_beta)
        val_loss += loss.item()
        val_batches += 1
    val_loss = val_loss / val_batches

    print(f"Epoch {epoch}: DPO train loss :: {train_loss:.3f} | val loss :: {val_loss:.3f}")
    
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      patience_counter = 0
      save_model(policy_model, optimizer, args, f'best_{args.filepath}')
    else:
      patience_counter += 1
      print(f"  [Early Stopping] No improvement for {patience_counter}/{patience} epochs.")

    if patience_counter >= patience:
      print(f"Early stopping triggered at epoch {epoch}. Training finished.")
      break


@torch.no_grad()
def compute_lexical_diversity(text):
  import re
  words = re.findall(r"\b\w+(?:'\w+)?\b", text.lower())
  if not words:
    return 0.0
  return len(set(words)) / len(words)


@torch.no_grad()
def compute_sonnet_or_not_bot(text, metrics):
  lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
  if len(lines) != 14:
    return 0.0
  if metrics['mean_syllable_error'] > 2.5:
    return 0.0
  if metrics['rhyme_accuracy'] < 0.10:
    return 0.0
  return 1.0


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  best_path = f'best_{args.filepath}'
  
  if os.path.exists(best_path):
    saved = torch.load(best_path, weights_only=False)
    print(f"Loading best model checkpoint from {best_path} for generation...")
  else:
    print(f"Warning: {best_path} not found. Trying fallback checkpoints...")
    saved_files = [f for f in os.listdir('.') if f.endswith(args.filepath) and f[0].isdigit()]
    if saved_files:
      saved_files.sort(key=lambda x: int(x.split('_')[0]), reverse=True)
      saved = torch.load(saved_files[0], weights_only=False)
    else:
      raise FileNotFoundError("No trained checkpoint found to generate sonnets.")

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))
    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])

  # 생성된 소네트들의 평가 지표 연산
  from evaluation import evaluate_poetic_metrics, test_sonnet, evaluate_theme_alignment
  try:
    gold_path = 'data/TRUE_sonnets_held_out_dev.txt'
    chrf_score = test_sonnet(test_path=args.sonnet_out, gold_path=gold_path)
  except Exception as e:
    chrf_score = 0.0

  gold_dataset = SonnetsDataset(gold_path)
  gold_sonnets = {i: text for i, text in gold_dataset}

  all_poetic_metrics = []
  sonnet_or_not_scores = []
  lexical_diversities = []
  theme_alignments = []
  
  for idx, (sonnet_id, full_sonnet) in enumerate(generated_sonnets):
    m = evaluate_poetic_metrics(full_sonnet)
    all_poetic_metrics.append(m)
    
    sonnet_or_not_val = compute_sonnet_or_not_bot(full_sonnet, m)
    sonnet_or_not_scores.append(sonnet_or_not_val)
    
    lex_div = compute_lexical_diversity(full_sonnet)
    lexical_diversities.append(lex_div)

    gold_text = gold_sonnets.get(idx, "")
    theme_align = evaluate_theme_alignment(full_sonnet, gold_text)
    theme_alignments.append(theme_align)

  avg_syllable_err = sum(m['mean_syllable_error'] for m in all_poetic_metrics) / len(all_poetic_metrics)
  avg_syllable_acc = sum(m['syllable_accuracy'] for m in all_poetic_metrics) / len(all_poetic_metrics)
  avg_meter_acc = sum(m['meter_accuracy'] for m in all_poetic_metrics) / len(all_poetic_metrics)
  avg_rhyme_acc = sum(m['rhyme_accuracy'] for m in all_poetic_metrics) / len(all_poetic_metrics)
  
  avg_sonnet_or_not = sum(sonnet_or_not_scores) / len(sonnet_or_not_scores)
  avg_lexical_diversity = sum(lexical_diversities) / len(lexical_diversities)
  avg_theme_alignment = sum(theme_alignments) / len(theme_alignments)
  
  form_accuracy = (avg_syllable_acc + avg_meter_acc + avg_rhyme_acc) / 3.0
  overall_quality = chrf_score / 100.0
  poe_metric = (form_accuracy * 0.3) + (avg_lexical_diversity * 0.2) + (overall_quality * 0.3) + (avg_theme_alignment * 0.2)

  print("\n=============================================")
  print("Final Evaluation on Fixed Metric Suite (DPO ONLY)")
  print("=============================================")
  print(f"1. chrF Score: {chrf_score:.3f}")
  print(f"2. Sonnet or Not, Bot? (Pass Rate): {avg_sonnet_or_not*100:.1f}%")
  print(f"3. POEMetric Score: {poe_metric:.3f}")
  print(f"   - Form Accuracy: {form_accuracy:.3f}")
  print(f"   - Lexical Diversity: {avg_lexical_diversity:.3f}")
  print(f"   - Overall Quality: {overall_quality:.3f}")
  print(f"   - Theme Alignment: {avg_theme_alignment:.3f}")
  print("---------------------------------------------")
  print(f"   (Detailed Poetic Specs)")
  print(f"   - Avg Syllable Deviation: {avg_syllable_err:.3f}")
  print(f"   - Avg Syllable Accuracy: {avg_syllable_acc*100:.1f}%")
  print(f"   - Avg Meter Accuracy: {avg_meter_acc*100:.1f}%")
  print(f"   - Avg Rhyme Accuracy: {avg_rhyme_acc*100:.1f}%")
  print("=============================================\n")


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--temperature", type=float, default=1.2)
  parser.add_argument("--top_p", type=float, default=0.9)
  parser.add_argument("--batch_size", type=int, default=4)
  parser.add_argument("--lr", type=float, default=5e-6) # DPO는 일반 파인튜닝보다 약간 작은 lr 사용
  parser.add_argument("--dpo_beta", type=float, default=0.1)
  parser.add_argument("--model_size", type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
  return parser.parse_args()


def add_arguments(args):
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet-dpo.pt'
  seed_everything(args.seed)
  train(args)
  generate_submission_sonnets(args)
