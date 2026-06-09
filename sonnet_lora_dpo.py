'''
[Variation 3: LoRA + DPO]
소네트 생성 모델 - LoRA + DPO 버전

실행:
  `python sonnet_lora_dpo.py --epochs 1 --batch_size 2`
'''

import argparse
import os
import random
import torch
import re
import math
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


# Native LoRA Wrapper for GPT-2 Linear Layer
class LoRALinear(nn.Module):
  def __init__(self, original_linear, r=8, alpha=16):
    super().__init__()
    self.linear = original_linear
    self.r = r
    self.alpha = alpha
    self.scaling = alpha / r
    
    in_features = original_linear.in_features
    out_features = original_linear.out_features
    
    self.lora_A = nn.Parameter(torch.zeros((in_features, r)))
    self.lora_B = nn.Parameter(torch.zeros((r, out_features)))
    
    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    nn.init.zeros_(self.lora_B)

  def forward(self, x):
    orig_out = self.linear(x)
    lora_out = (x @ self.lora_A) @ self.lora_B * self.scaling
    return orig_out + lora_out


class SonnetGPT(nn.Module):
  def __init__(self, args, use_lora=True):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.use_lora = use_lora

    if use_lora:
      self.apply_lora(r=args.lora_r, alpha=args.lora_alpha)
    else:
      for param in self.gpt.parameters():
        param.requires_grad = True

  def apply_lora(self, r=8, alpha=16):
    # GPT-2 모델의 모든 GPT2Layer 내의 query, value 레이어에 LoRA 주입
    for layer in self.gpt.gpt_layers:
      layer.self_attention.query = LoRALinear(layer.self_attention.query, r=r, alpha=alpha)
      layer.self_attention.value = LoRALinear(layer.self_attention.value, r=r, alpha=alpha)
        
    # LoRA 파라미터만 학습 가능하도록 설정하고 나머지는 동결(Freeze)
    for name, param in self.named_parameters():
      if 'lora_' in name:
        param.requires_grad = True
      else:
        param.requires_grad = False

  def forward(self, input_ids, attention_mask):
    output = self.gpt(input_ids, attention_mask)
    hidden_states = output['last_hidden_state']
    logits = self.gpt.hidden_state_to_token(hidden_states)
    return logits

  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=1.2, top_p=0.9, max_length=128):
    token_ids = encoding.to(self.get_device())
    attn_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())
    
    newline_id = self.tokenizer.encode('\n')[0]
    prompt_text = self.tokenizer.decode(token_ids[0].tolist())
    prompt_newlines = prompt_text.count('\n')
    
    remaining_lines = max(13 - prompt_newlines, 1)
    generated_newlines = 0

    for _ in range(max_length):
      logits_sequence = self.forward(token_ids, attn_mask)
      logits_last_token = logits_sequence[:, -1, :].clone() / temperature

      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)
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

      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attn_mask = torch.cat(
        [attn_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

      if sampled_token.item() == newline_id:
          generated_newlines += 1
          if generated_newlines >= remaining_lines:
              break

    generated_tokens = token_ids[0][encoding.shape[1]:]
    generated_output = self.tokenizer.decode(generated_tokens.cpu().numpy().tolist())
    return token_ids, generated_output


def get_log_probs(logits, labels):
  shift_logits = logits[:, :-1, :].contiguous()
  shift_labels = labels[:, 1:].contiguous()
  log_probs = F.log_softmax(shift_logits, dim=-1)
  per_token_logps = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(-1)).squeeze(-1)
  mask = (shift_labels != 50256).float()
  return (per_token_logps * mask).sum(dim=-1)


def dpo_loss(policy_win_logps, policy_lose_logps, ref_win_logps, ref_lose_logps, beta=0.1):
  policy_ratio = policy_win_logps - policy_lose_logps
  ref_ratio = ref_win_logps - ref_lose_logps
  logits = policy_ratio - ref_ratio
  loss = -F.logsigmoid(beta * logits).mean()
  return loss


def save_model(model, optimizer, args, filepath):
  # 추론 시 가볍게 로딩할 수 있도록 trainable 파라미터(LoRA)만 선별 저장도 가능하나,
  # 호환성을 위해 state_dict 전체를 저장합니다.
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
  
  # 1. Policy 모델 (LoRA 주입 및 학습 활성화)
  policy_model = SonnetGPT(args, use_lora=True).to(device)
  
  # 2. Reference 모델 (베이스라인 모델 동결)
  ref_model = SonnetGPT(args, use_lora=False).to(device)
  ref_model.eval()
  for param in ref_model.parameters():
    param.requires_grad = False

  # 오직 requires_grad=True인 LoRA 어댑터 가중치만 옵티마이저에 전달
  trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
  optimizer = AdamW(trainable_params, lr=args.lr)
  
  best_val_loss = float('inf')
  patience = 3
  patience_counter = 0

  for epoch in range(args.epochs):
    policy_model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(train_loader, desc=f'LoRA+DPO train-{epoch}', disable=TQDM_DISABLE):
      w_ids = batch['winner_ids'].to(device)
      w_mask = batch['winner_mask'].to(device)
      l_ids = batch['loser_ids'].to(device)
      l_mask = batch['loser_mask'].to(device)

      with torch.no_grad():
        ref_win_logits = ref_model(w_ids, w_mask)
        ref_lose_logits = ref_model(l_ids, l_mask)
        ref_win_logps = get_log_probs(ref_win_logits, w_ids)
        ref_lose_logps = get_log_probs(ref_lose_logits, l_ids)

      optimizer.zero_grad()
      pol_win_logits = policy_model(w_ids, w_mask)
      pol_lose_logits = policy_model(l_ids, l_mask)
      pol_win_logps = get_log_probs(pol_win_logits, w_ids)
      pol_lose_logps = get_log_probs(pol_lose_logits, l_ids)

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

    print(f"Epoch {epoch}: LoRA+DPO train loss :: {train_loss:.3f} | val loss :: {val_loss:.3f}")
    
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

  model = SonnetGPT(saved['args'], use_lora=True)
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
  from evaluation import evaluate_poetic_metrics, test_sonnet
  try:
    gold_subset = 'data/TRUE_sonnets_held_out_dev_subset.txt'
    chrf_score = test_sonnet(test_path=args.sonnet_out, gold_path=gold_subset)
  except Exception as e:
    chrf_score = 0.0

  all_poetic_metrics = []
  sonnet_or_not_scores = []
  lexical_diversities = []
  
  for sonnet_id, full_sonnet in generated_sonnets:
    m = evaluate_poetic_metrics(full_sonnet)
    all_poetic_metrics.append(m)
    
    sonnet_or_not_val = compute_sonnet_or_not_bot(full_sonnet, m)
    sonnet_or_not_scores.append(sonnet_or_not_val)
    
    lex_div = compute_lexical_diversity(full_sonnet)
    lexical_diversities.append(lex_div)

  avg_syllable_err = sum(m['mean_syllable_error'] for m in all_poetic_metrics) / len(all_poetic_metrics)
  avg_syllable_acc = sum(m['syllable_accuracy'] for m in all_poetic_metrics) / len(all_poetic_metrics)
  avg_meter_acc = sum(m['meter_accuracy'] for m in all_poetic_metrics) / len(all_poetic_metrics)
  avg_rhyme_acc = sum(m['rhyme_accuracy'] for m in all_poetic_metrics) / len(all_poetic_metrics)
  
  avg_sonnet_or_not = sum(sonnet_or_not_scores) / len(sonnet_or_not_scores)
  avg_lexical_diversity = sum(lexical_diversities) / len(lexical_diversities)
  
  form_accuracy = (avg_syllable_acc + avg_meter_acc + avg_rhyme_acc) / 3.0
  overall_quality = chrf_score / 100.0
  poe_metric = (form_accuracy * 0.4) + (avg_lexical_diversity * 0.3) + (overall_quality * 0.3)

  print("\n=============================================")
  print("Final Evaluation on Fixed Metric Suite (LoRA + DPO)")
  print("=============================================")
  print(f"1. chrF Score: {chrf_score:.3f}")
  print(f"2. Sonnet or Not, Bot? (Pass Rate): {avg_sonnet_or_not*100:.1f}%")
  print(f"3. POEMetric Score: {poe_metric:.3f}")
  print(f"   - Form Accuracy: {form_accuracy:.3f}")
  print(f"   - Lexical Diversity: {avg_lexical_diversity:.3f}")
  print(f"   - Overall Quality: {overall_quality:.3f}")
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
  parser.add_argument("--lr", type=float, default=1e-4) # LoRA의 경우 조금 더 큰 lr 권장
  parser.add_argument("--lora_r", type=int, default=8)
  parser.add_argument("--lora_alpha", type=int, default=16)
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
  args.filepath = f'{args.epochs}-{args.lr}-sonnet-lora-dpo.pt'
  seed_everything(args.seed)
  train(args)
  generate_submission_sonnets(args)
