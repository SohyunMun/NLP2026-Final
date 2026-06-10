'''
[Variation 1: Baseline]
소네트 생성 모델 - Baseline 버전

실행:
  `python sonnet_baseline.py --epochs 1 --batch_size 4`
'''

import argparse
import os
import random
import torch
import re

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
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
  def generate(self, encoding, temperature=1.2, top_p=0.9, max_length=128):
    token_ids = encoding.to(self.get_device())
    attn_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())
    
    newline_id = self.tokenizer.encode('\n')[0]
    prompt_text = self.tokenizer.decode(token_ids[0].tolist())
    prompt_newlines = prompt_text.count('\n')
    
    remaining_lines = max(13 - prompt_newlines, 1)
    generated_newlines = 0
    current_line_tokens = []

    for _ in range(max_length):
      logits_sequence = self.forward(token_ids, attn_mask)
      logits_last_token = logits_sequence[:, -1, :].clone() / temperature

      # Baseline 디코딩: 어떠한 하드 제약 조건도 없음
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
          current_line_tokens = []
          if generated_newlines >= remaining_lines:
              break
      else:
          current_line_tokens.append(sampled_token.item())

    generated_tokens = token_ids[0][encoding.shape[1]:]
    generated_output = self.tokenizer.decode(generated_tokens.cpu().numpy().tolist())
    return token_ids, generated_output


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }
  torch.save(save_info, filepath)
  print(f"saved the model to {filepath}")


def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  
  val_size = int(0.1 * len(sonnet_dataset))
  if val_size == 0 and len(sonnet_dataset) > 1:
    val_size = 1
  train_size = len(sonnet_dataset) - val_size
  train_dataset, val_dataset = torch.utils.data.random_split(
    sonnet_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(args.seed)
  )

  sonnet_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)
  val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size,
                               collate_fn=sonnet_dataset.collate_fn)

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  optimizer = AdamW(model.parameters(), lr=args.lr)
  best_val_loss = float('inf')
  patience = 3
  patience_counter = 0

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
      labels = b_ids[:, 1:].contiguous().flatten()

      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    model.eval()
    val_loss = 0
    val_batches = 0
    with torch.no_grad():
      for batch in val_dataloader:
        b_ids, b_mask = batch['token_ids'], batch['attention_mask']
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        logits = model(b_ids, b_mask)
        logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
        labels = b_ids[:, 1:].contiguous().flatten()
        loss = F.cross_entropy(logits, labels, reduction='mean')
        val_loss += loss.item()
        val_batches += 1
    val_loss = val_loss / val_batches

    print(f"Epoch {epoch}: train loss :: {train_loss:.3f} | val loss :: {val_loss:.3f}")
    
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      patience_counter = 0
      save_model(model, optimizer, args, f'best_{args.filepath}')
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
  print("Final Evaluation on Fixed Metric Suite (BASELINE)")
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
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--lr", type=float, default=1e-5)
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
  args.filepath = f'{args.epochs}-{args.lr}-sonnet-baseline.pt'
  seed_everything(args.seed)
  train(args)
  generate_submission_sonnets(args)
