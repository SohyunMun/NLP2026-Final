'''
[The Ultimate Poetic Pipeline]
소네트 생성 모델 성능 극대화를 위한 3단계 점진적 학습

구조:
  - Stage 1: DAPT (Domain-Adaptive Pre-training) -> 전체 가중치 사전 적응
  - Stage 2: SFT with PEFT -> Prefix Tuning + LoRA 결합 구조 미세조정
  - Stage 3: Fine-grained DPO -> 선호도 쌍을 통한 정형률 최종 얼라인먼트
  - Evaluation: chrF, Sonnet or Not, POEMetric 종합 평가

실행 예시:
  * 전체 파이프라인 순차 실행:
    `python sonnet_DAPT_LORA_PEPT_DPO.py --stage all --epochs_dapt 1 --epochs_sft 1 --epochs_dpo 1 --batch_size 4`
  * 특정 단계만 단독 실행:
    `python sonnet_DAPT_LORA_PEPT_DPO.py --stage dpo --epochs_dpo 1 --batch_size 4`
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


# =====================================================================
# PEFT Components (LoRA + Prefix Tuning)
# =====================================================================

class LoRALinear(nn.Module):
  """Native PyTorch LoRA wrapper for Linear layers."""
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


class PrefixTuningEmbedding(nn.Module):
  """Prefix Tuning implementation that prepends virtual tokens at the embedding level."""
  def __init__(self, prefix_len, d_model):
    super().__init__()
    self.prefix_len = prefix_len
    self.d_model = d_model
    self.prefix_param = nn.Parameter(torch.randn(prefix_len, d_model) * 0.02)

  def forward(self, token_embeddings):
    batch_size = token_embeddings.size(0)
    prefix = self.prefix_param.unsqueeze(0).expand(batch_size, -1, -1)
    return torch.cat([prefix, token_embeddings], dim=1)


# =====================================================================
# DPO Dataset
# =====================================================================

class DPODataset(Dataset):
  """DPO Preference Dataset yielding Winner and Loser pairs."""
  def __init__(self, original_dataset, tokenizer, max_length=192):
    self.original_dataset = original_dataset
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.original_dataset)

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


# =====================================================================
# Ultimate Sonnet GPT Model Wrapper
# =====================================================================

class SonnetGPT(nn.Module):
  def __init__(self, args, use_peft=True):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.use_peft = use_peft
    self.prefix_len = args.prefix_len

    if use_peft:
      self.prefix_tuner = PrefixTuningEmbedding(args.prefix_len, args.d)
      self.apply_lora(r=args.lora_r, alpha=args.lora_alpha)
    else:
      for param in self.gpt.parameters():
        param.requires_grad = True

  def apply_lora(self, r=8, alpha=16):
    for layer in self.gpt.gpt_layers:
      layer.self_attention.query = LoRALinear(layer.self_attention.query, r=r, alpha=alpha)
      layer.self_attention.value = LoRALinear(layer.self_attention.value, r=r, alpha=alpha)
        
    # Freeze standard parameters, active LoRA and Prefix params
    for name, param in self.named_parameters():
      if 'lora_' in name or 'prefix_param' in name:
        param.requires_grad = True
      else:
        param.requires_grad = False

  def forward(self, input_ids, attention_mask):
    if self.use_peft:
      wte = self.gpt.word_embedding(input_ids)
      pos_ids = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)
      wpe = self.gpt.pos_embedding(pos_ids)
      embeddings = wte + wpe
      
      extended_embeddings = self.prefix_tuner(embeddings)
      
      batch_size = input_ids.size(0)
      prefix_mask = torch.ones(batch_size, self.prefix_len, dtype=torch.int64, device=input_ids.device)
      extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)
      
      hidden_states = extended_embeddings
      from utils import get_extended_attention_mask
      extended_attn_mask = get_extended_attention_mask(extended_mask, hidden_states.dtype)
      
      for block in self.gpt.gpt_layers:
        hidden_states = block(hidden_states, extended_attn_mask)
        
      hidden_states = self.gpt.final_layer_norm(hidden_states)
      hidden_states_sliced = hidden_states[:, self.prefix_len:]
      logits = self.gpt.hidden_state_to_token(hidden_states_sliced)
    else:
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


# =====================================================================
# Log Probabilities Helper and DPO Loss
# =====================================================================

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


# =====================================================================
# Stages Implementation
# =====================================================================

def run_dapt(args, device):
  """Stage 1: Domain-Adaptive Pre-training (Full-Model SFT on general poetry)."""
  print("\n>>> Launching Stage 1: Domain-Adaptive Pre-training (DAPT)...")
  dapt_dataset = SonnetsDataset(args.dapt_path)
  
  val_size = int(0.1 * len(dapt_dataset))
  if val_size == 0 and len(dapt_dataset) > 1:
    val_size = 1
  train_size = len(dapt_dataset) - val_size
  train_dataset, val_dataset = torch.utils.data.random_split(
    dapt_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(args.seed)
  )

  train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=dapt_dataset.collate_fn)
  val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dapt_dataset.collate_fn)

  # Full model is trainable
  model = SonnetGPT(args, use_peft=False).to(device)
  optimizer = AdamW(model.parameters(), lr=args.lr_dapt)

  best_val_loss = float('inf')
  patience_counter = 0

  for epoch in range(args.epochs_dapt):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(train_loader, desc=f'DAPT Epoch {epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask = batch['token_ids'].to(device), batch['attention_mask'].to(device)
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
      labels = b_ids[:, 1:].contiguous().flatten()
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
      num_batches += 1
    
    train_loss /= num_batches

    model.eval()
    val_loss = 0
    val_batches = 0
    with torch.no_grad():
      for batch in val_loader:
        b_ids, b_mask = batch['token_ids'].to(device), batch['attention_mask'].to(device)
        logits = model(b_ids, b_mask)
        logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
        labels = b_ids[:, 1:].contiguous().flatten()
        loss = F.cross_entropy(logits, labels, reduction='mean')
        val_loss += loss.item()
        val_batches += 1
    val_loss /= val_batches

    print(f"Epoch {epoch}: DAPT Train Loss :: {train_loss:.3f} | Val Loss :: {val_loss:.3f}")
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      patience_counter = 0
      # Save intermediate DAPT checkpoint
      torch.save({'model': model.state_dict(), 'args': args}, args.dapt_checkpoint)
    else:
      patience_counter += 1
    
    if patience_counter >= 3:
      print("Early stopping Stage 1.")
      break


def run_sft(args, device):
  """Stage 2: Supervised Fine-Tuning with PEFT (Prefix + LoRA)."""
  print("\n>>> Launching Stage 2: Supervised Fine-Tuning with PEFT...")
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  
  val_size = int(0.1 * len(sonnet_dataset))
  if val_size == 0 and len(sonnet_dataset) > 1:
    val_size = 1
  train_size = len(sonnet_dataset) - val_size
  train_dataset, val_dataset = torch.utils.data.random_split(
    sonnet_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(args.seed)
  )

  train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=sonnet_dataset.collate_fn)
  val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=sonnet_dataset.collate_fn)

  # Initialize model with PEFT wrappers active
  model = SonnetGPT(args, use_peft=True).to(device)

  # Load DAPT weights if available
  if os.path.exists(args.dapt_checkpoint):
    print(f"Loading Stage 1 (DAPT) weights from {args.dapt_checkpoint}...")
    saved_dapt = torch.load(args.dapt_checkpoint, weights_only=False)
    # Filter state dict keys (since DAPT didn't have Prefix/LoRA wrappers)
    model_state = model.state_dict()
    for name, param in saved_dapt['model'].items():
      # map core weight keys back into our gpt prefix
      if name in model_state and model_state[name].shape == param.shape:
        model_state[name].copy_(param)
    model.load_state_dict(model_state)
  else:
    print("DAPT checkpoint not found. Starting SFT from raw pretrained GPT-2...")

  # SFT stage only trains LoRA and Prefix variables
  trainable_params = [p for p in model.parameters() if p.requires_grad]
  optimizer = AdamW(trainable_params, lr=args.lr_sft)

  best_val_loss = float('inf')
  patience_counter = 0

  for epoch in range(args.epochs_sft):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(train_loader, desc=f'SFT Epoch {epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask = batch['token_ids'].to(device), batch['attention_mask'].to(device)
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
      labels = b_ids[:, 1:].contiguous().flatten()
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
      num_batches += 1
    
    train_loss /= num_batches

    model.eval()
    val_loss = 0
    val_batches = 0
    with torch.no_grad():
      for batch in val_loader:
        b_ids, b_mask = batch['token_ids'].to(device), batch['attention_mask'].to(device)
        logits = model(b_ids, b_mask)
        logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
        labels = b_ids[:, 1:].contiguous().flatten()
        loss = F.cross_entropy(logits, labels, reduction='mean')
        val_loss += loss.item()
        val_batches += 1
    val_loss /= val_batches

    print(f"Epoch {epoch}: SFT Train Loss :: {train_loss:.3f} | Val Loss :: {val_loss:.3f}")
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      patience_counter = 0
      torch.save({'model': model.state_dict(), 'args': args}, args.sft_checkpoint)
    else:
      patience_counter += 1
    
    if patience_counter >= 3:
      print("Early stopping Stage 2.")
      break


def run_dpo(args, device):
  """Stage 3: Direct Preference Optimization (DPO) for metrics alignment."""
  print("\n>>> Launching Stage 3: Direct Preference Optimization (DPO)...")
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

  # Initialize Policy Model
  policy_model = SonnetGPT(args, use_peft=True).to(device)
  if os.path.exists(args.sft_checkpoint):
    print(f"Loading Stage 2 (SFT) checkpoint from {args.sft_checkpoint}...")
    saved_sft = torch.load(args.sft_checkpoint, weights_only=False)
    policy_model.load_state_dict(saved_sft['model'])
  else:
    print("SFT checkpoint not found. DPO must be loaded from an SFT baseline!")
    
  # Initialize reference model and lock weights
  ref_model = SonnetGPT(args, use_peft=True).to(device)
  ref_model.load_state_dict(policy_model.state_dict())
  ref_model.eval()
  for param in ref_model.parameters():
    param.requires_grad = False

  trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
  optimizer = AdamW(trainable_params, lr=args.lr_dpo)

  best_val_loss = float('inf')
  patience_counter = 0

  for epoch in range(args.epochs_dpo):
    policy_model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(train_loader, desc=f'DPO Epoch {epoch}', disable=TQDM_DISABLE):
      w_ids, w_mask = batch['winner_ids'].to(device), batch['winner_mask'].to(device)
      l_ids, l_mask = batch['loser_ids'].to(device), batch['loser_mask'].to(device)

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
    
    train_loss /= num_batches

    policy_model.eval()
    val_loss = 0
    val_batches = 0
    with torch.no_grad():
      for batch in val_loader:
        w_ids, w_mask = batch['winner_ids'].to(device), batch['winner_mask'].to(device)
        l_ids, l_mask = batch['loser_ids'].to(device), batch['loser_mask'].to(device)

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
    val_loss /= val_batches

    print(f"Epoch {epoch}: DPO Train Loss :: {train_loss:.3f} | Val Loss :: {val_loss:.3f}")
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      patience_counter = 0
      torch.save({'model': policy_model.state_dict(), 'args': args}, args.ultimate_checkpoint)
    else:
      patience_counter += 1
    
    if patience_counter >= 3:
      print("Early stopping Stage 3.")
      break


# =====================================================================
# Metric Evaluation Suite
# =====================================================================

@torch.no_grad()
def compute_lexical_diversity(text):
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


def run_evaluation(args, device):
  """Evaluation: Generate sonnets on held-out subset and calculate POEMetric Suite."""
  print("\n>>> Launching Evaluation on Fixed Metric Suite...")
  
  if os.path.exists(args.ultimate_checkpoint):
    checkpoint_to_load = args.ultimate_checkpoint
  elif os.path.exists(args.sft_checkpoint):
    print("Warning: Ultimate checkpoint not found. Falling back to SFT checkpoint...")
    checkpoint_to_load = args.sft_checkpoint
  else:
    raise FileNotFoundError("No trained checkpoint found to perform evaluation.")

  saved = torch.load(checkpoint_to_load, weights_only=False)
  model = SonnetGPT(saved['args'], use_peft=True).to(device)
  model.load_state_dict(saved['model'])
  model.eval()

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)
  generated_sonnets = []

  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    generated_sonnets.append((sonnet_id, decoded_output))
    print(f"--- Prompt {sonnet_id} ---")
    print(decoded_output)
    print("---------------------\n")

  # Write predictions
  os.makedirs(os.path.dirname(args.sonnet_out), exist_ok=True)
  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets (ULTIMATE PIPELINE)-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(f"{sonnet[1]}\n\n")

  # Metrics calculation
  from evaluation import evaluate_poetic_metrics, test_sonnet
  try:
    gold_subset = 'data/TRUE_sonnets_held_out_dev_subset.txt'
    chrf_score = test_sonnet(test_path=args.sonnet_out, gold_path=gold_subset)
  except Exception:
    chrf_score = 0.0

  all_poetic_metrics = []
  sonnet_or_not_scores = []
  lexical_diversities = []
  
  for sonnet_id, full_sonnet in generated_sonnets:
    m = evaluate_poetic_metrics(full_sonnet)
    all_poetic_metrics.append(m)
    sonnet_or_not_scores.append(compute_sonnet_or_not_bot(full_sonnet, m))
    lexical_diversities.append(compute_lexical_diversity(full_sonnet))

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
  print("Final Evaluation on Fixed Metric Suite (ULTIMATE)")
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


# =====================================================================
# Main Command CLI
# =====================================================================

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--stage", type=str, choices=['all', 'dapt', 'sft', 'dpo', 'eval'], default='all',
                      help="stage to run: dapt, sft, dpo, eval or all sequentially")
  parser.add_argument("--dapt_path", type=str, default="data/sonnets.txt",
                      help="Dataset path for Stage 1 DAPT")
  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt",
                      help="Dataset path for Stage 2 & 3 SFT & DPO")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt",
                      help="Dataset path for evaluation prompts")
  parser.add_argument("--sonnet_out", type=str, default="predictions/ultimate_generated_sonnets.txt")
  parser.add_argument("--seed", type=int, default=11711)
  
  # Epochs and Hyperparameters per stage
  parser.add_argument("--batch_size", type=int, default=4)
  parser.add_argument("--epochs_dapt", type=int, default=5, help="Epochs for DAPT")
  parser.add_argument("--epochs_sft", type=int, default=5, help="Epochs for PEFT SFT")
  parser.add_argument("--epochs_dpo", type=int, default=5, help="Epochs for DPO")
  
  parser.add_argument("--lr_dapt", type=float, default=1e-5, help="Learning rate for Stage 1")
  parser.add_argument("--lr_sft", type=float, default=1e-4, help="Learning rate for Stage 2")
  parser.add_argument("--lr_dpo", type=float, default=5e-5, help="Learning rate for Stage 3")
  
  # PEFT Config
  parser.add_argument("--lora_r", type=int, default=8)
  parser.add_argument("--lora_alpha", type=int, default=16)
  parser.add_argument("--prefix_len", type=int, default=8)
  
  # DPO Config
  parser.add_argument("--dpo_beta", type=float, default=0.1)
  
  # Model Setup
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--temperature", type=float, default=1.2)
  parser.add_argument("--top_p", type=float, default=0.9)
  parser.add_argument("--model_size", type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
  
  # Checkpoint paths
  parser.add_argument("--dapt_checkpoint", type=str, default="best_ultimate_dapt.pt")
  parser.add_argument("--sft_checkpoint", type=str, default="best_ultimate_sft.pt")
  parser.add_argument("--ultimate_checkpoint", type=str, default="best_ultimate_dpo.pt")
  
  return parser.parse_args()


def add_model_dims(args):
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
  args = add_model_dims(args)
  seed_everything(args.seed)
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  if args.stage == 'all':
    run_dapt(args, device)
    run_sft(args, device)
    run_dpo(args, device)
    run_evaluation(args, device)
  elif args.stage == 'dapt':
    run_dapt(args, device)
  elif args.stage == 'sft':
    run_sft(args, device)
  elif args.stage == 'dpo':
    run_dpo(args, device)
  elif args.stage == 'eval':
    run_evaluation(args, device)
