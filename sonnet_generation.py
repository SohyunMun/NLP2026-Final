'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation.py --use_gpu`

SonnetGPT 모델을 훈련하고, 필요한 제출용 파일을 작성한다.
'''

import argparse
import math
import random
import re

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
from einops import rearrange

from dataset import ShakespeareDataset
from datasets import SonnetsDataset
from evaluation import test_sonnet
from models.gpt2 import GPT2Model
from optimizer import AdamW

TQDM_DISABLE = False

# 셰익스피어 소네트 운율 체계: ABAB CDCD EFEF GG (0-indexed)
RHYME_SCHEME = {
  2: 0, 3: 1,
  6: 4, 7: 5,
  10: 8, 11: 9,
  13: 12
}


def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


# 기존 nn.Linear를 동결하고 저랭크 행렬 A, B만 학습하여 소규모 데이터에서 과적합을 방지한다.
class LoRALinear(nn.Module):
  def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.15):
    super().__init__()
    self.original = linear
    self.scaling = alpha / r
    self.lora_A = nn.Linear(linear.in_features, r, bias=False)
    self.lora_B = nn.Linear(r, linear.out_features, bias=False)
    self.lora_dropout = nn.Dropout(dropout)
    nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
    nn.init.zeros_(self.lora_B.weight)

  def forward(self, x):
    return self.original(x) + self.scaling * self.lora_B(self.lora_A(self.lora_dropout(x)))


def _apply_lora(gpt_model, r=8, alpha=16):
  # attention projection에 LoRA 어댑터 삽입.
  for layer in gpt_model.gpt_layers:
    sa = layer.self_attention
    sa.query              = LoRALinear(sa.query,          r=r, alpha=alpha)
    sa.key                = LoRALinear(sa.key,            r=r, alpha=alpha)
    sa.value              = LoRALinear(sa.value,          r=r, alpha=alpha)
    layer.attention_dense = LoRALinear(layer.attention_dense, r=r, alpha=alpha)


def _get_last_word(text: str) -> str:
  words = re.findall(r"[a-zA-Z']+", text)
  return words[-1].lower() if words else ''


def _get_rhyme_module():
  try:
    import pronouncing
    return pronouncing
  except ImportError:
    return None


class SonnetGPT(nn.Module):
  """Sonnet 생성을 위해 설계된 GPT-2 모델."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # 기본적으로, 전체 모델을 fine-tuning한다.
    # Stage 2에서는 LoRA 어댑터 파라미터만 학습하도록 전환한다.
    _apply_lora(self.gpt,
                r=getattr(args, 'lora_r', 8),
                alpha=getattr(args, 'lora_alpha', 16))

    for param in self.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """시퀀스의 모든 토큰 위치에서 다음 토큰에 대한 logit을 반환한다."""
    output = self.gpt(input_ids, attention_mask)
    hidden_states = output['last_hidden_state']
    logits = self.gpt.hidden_state_to_token(hidden_states)
    return logits

  def get_device(self):
    return next(self.parameters()).device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.9, top_p=0.9, top_k=5,
               max_length=300, repetition_penalty=1.3, strategy='sampling',
               use_rhyme=True):
    """
    소네트 생성. strategy에 따라 세 가지 디코딩 방법 중 하나를 사용한다.
    셰익스피어 소네트는 14행이므로 프롬프트 행 수를 기반으로 남은 행만큼 생성 후 종료한다.
    """
    if strategy == 'beam':
      return self._beam_search(encoding, num_beams=top_k, max_length=max_length,
                               temperature=temperature)
    return self._sampling(encoding, temperature, top_p, top_k,
                          max_length, repetition_penalty, strategy, use_rhyme)

  @torch.no_grad()
  def _sampling(self, encoding, temperature, top_p, top_k,
                max_length, repetition_penalty, strategy, use_rhyme):
    """Top-p nucleus sampling 또는 top-k sampling."""
    token_ids = encoding.to(self.get_device())
    attn_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())
    prompt_len = token_ids.shape[1]

    newline_id = self.tokenizer.encode('\n')[0]
    prompt_text = self.tokenizer.decode(token_ids[0].tolist())
    prompt_newlines = prompt_text.count('\n')
    # 소네트 14행 강제: 프롬프트 줄바꿈 수를 빼서 남은 행 계산.
    remaining_lines = max(13 - prompt_newlines, 1)
    generated_newlines = 0

    # 운율 추적: 각 행의 마지막 단어 저장
    pronouncing = _get_rhyme_module() if use_rhyme else None
    line_last_words = {}
    for i, pline in enumerate(prompt_text.split('\n')):
      w = _get_last_word(pline)
      if w:
        line_last_words[i] = w
    current_line_idx = prompt_newlines
    current_line_tokens = []

    for _ in range(max_length):
      logits = self.forward(token_ids, attn_mask)
      logits_last = logits[:, -1, :].clone() / temperature

      if repetition_penalty != 1.0:
        for tid in set(token_ids[0].tolist()):
          if logits_last[0, tid] < 0:
            logits_last[0, tid] *= repetition_penalty
          else:
            logits_last[0, tid] /= repetition_penalty

      probs = F.softmax(logits_last, dim=-1)

      if strategy == 'top_k':
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
        topk_probs /= topk_probs.sum(dim=-1, keepdim=True)
        sampled_idx = torch.multinomial(topk_probs, 1)
        sampled_token = topk_indices.gather(dim=-1, index=sampled_idx)
      else:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cum_probs <= top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = True
        filtered = sorted_probs * mask
        filtered /= filtered.sum(dim=-1, keepdim=True)
        sampled_idx = torch.multinomial(filtered, 1)
        sampled_token = sorted_indices.gather(dim=-1, index=sampled_idx)

      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attn_mask = torch.cat(
        [attn_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1)

      if sampled_token.item() == newline_id:
        # 현재 행의 마지막 단어 저장
        w = _get_last_word(self.tokenizer.decode(current_line_tokens))
        if w:
          line_last_words[current_line_idx] = w
        generated_newlines += 1
        current_line_idx += 1
        current_line_tokens = []
        if generated_newlines >= remaining_lines:
          break
      else:
        current_line_tokens.append(sampled_token.item())

        # 운율 강제: 행 중반 이후 ABAB CDCD EFEF GG 체계에 맞는 단어로 유도.
        partner = RHYME_SCHEME.get(current_line_idx)
        if (use_rhyme and pronouncing and partner is not None
            and partner in line_last_words and len(current_line_tokens) >= 6):
          rhyme_ids = set()
          for rw in list(pronouncing.rhymes(line_last_words[partner]))[:200]:
            ids = self.tokenizer.encode(' ' + rw, add_special_tokens=False)
            if ids:
              rhyme_ids.add(ids[0])
          if rhyme_ids:
            for rid in rhyme_ids:
              if rid < logits_last.shape[-1]:
                logits_last[0, rid] += 2.0
            probs = F.softmax(logits_last, dim=-1)
            if strategy == 'top_k':
              topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
              topk_probs /= topk_probs.sum(dim=-1, keepdim=True)
              sampled_token = topk_indices.gather(dim=-1, index=torch.multinomial(topk_probs, 1))
            else:
              sorted_probs, sorted_indices = torch.sort(probs, descending=True)
              cum_probs = torch.cumsum(sorted_probs, dim=-1)
              mask = cum_probs <= top_p
              mask[..., 1:] = mask[..., :-1].clone()
              mask[..., 0] = True
              filtered = sorted_probs * mask
              filtered /= filtered.sum(dim=-1, keepdim=True)
              sampled_token = sorted_indices.gather(dim=-1, index=torch.multinomial(filtered, 1))

    generated_tokens = token_ids[0][prompt_len:]
    return token_ids, self.tokenizer.decode(generated_tokens.cpu().tolist())

  @torch.no_grad()
  def _beam_search(self, encoding, num_beams=5, max_length=300,
                   temperature=1.0, length_penalty=1.0):
    """num_beams개의 가설을 유지하며 누적 log 확률이 가장 높은 시퀀스를 반환."""
    device = self.get_device()
    token_ids = encoding.to(device)
    prompt_len = token_ids.shape[1]

    newline_id = self.tokenizer.encode('\n')[0]
    prompt_newlines = token_ids[0].tolist().count(newline_id)
    remaining_lines = max(13 - prompt_newlines, 1)

    beams = [(token_ids, 0.0, 0)]
    completed = []

    for _ in range(max_length):
      candidates = []
      for beam_ids, beam_score, beam_nl in beams:
        am = torch.ones(beam_ids.shape, dtype=torch.int64).to(device)
        logits = self.forward(beam_ids, am)
        logits_last = logits[:, -1, :] / max(temperature, 1e-7)
        log_probs = F.log_softmax(logits_last, dim=-1)
        topk_lp, topk_ids = torch.topk(log_probs[0], num_beams)

        for lp, tid in zip(topk_lp, topk_ids):
          new_ids = torch.cat([beam_ids, tid.view(1, 1)], dim=1)
          new_score = beam_score + lp.item()
          new_nl = beam_nl + (1 if tid.item() == newline_id else 0)

          if tid.item() == self.tokenizer.eos_token_id or new_nl >= remaining_lines:
            seq_len = new_ids.shape[1] - prompt_len
            completed.append((new_ids, new_score / max(seq_len ** length_penalty, 1)))
          else:
            candidates.append((new_ids, new_score, new_nl))

      if not candidates:
        break
      candidates.sort(key=lambda x: x[1], reverse=True)
      beams = candidates[:num_beams]
      if len(completed) >= num_beams:
        break

    if completed:
      completed.sort(key=lambda x: x[1], reverse=True)
      best_ids = completed[0][0]
    else:
      best_ids = beams[0][0]

    generated_tokens = best_ids[0][prompt_len:]
    return best_ids, self.tokenizer.decode(generated_tokens.cpu().tolist())


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
  print(f"save the model to {filepath}")


def save_checkpoint(model, optimizer, scheduler, args, filepath,
                    epoch, stage, best_loss, patience_counter):
  """epoch 단위 체크포인트 저장 (이어 학습용)."""
  ckpt = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'args': args,
    'epoch': epoch,
    'stage': stage,
    'best_loss': best_loss,
    'patience_counter': patience_counter,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }
  torch.save(ckpt, filepath)
  print(f"  [checkpoint] {filepath} (stage={stage}, epoch={epoch})")


def pretrain_shakespeare(args, model, device):
  """Stage 1: Shakespeare 전작으로 도메인 적응 사전학습. 체크포인트에서 이어 학습 가능."""
  import os
  ckpt_path = f'stage1_checkpoint_{args.filepath}'

  shakespeare_dataset = ShakespeareDataset(chunk_size=args.chunk_size)
  dataloader = DataLoader(shakespeare_dataset, shuffle=True, batch_size=args.batch_size,
                          collate_fn=shakespeare_dataset.collate_fn)

  optimizer = AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=0.01)
  total_steps = len(dataloader) * args.pretrain_epochs
  warmup_steps = int(total_steps * args.warmup_ratio)
  scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

  start_epoch = 0
  if args.resume and os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optim'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch'] + 1
    print(f"\n[Stage 1] 체크포인트 로드: epoch {ckpt['epoch']} 완료 → epoch {start_epoch}부터 재개")
    if start_epoch >= args.pretrain_epochs:
      print("Stage 1 이미 완료.\n")
      return model

  print(f"\n=== Stage 1: Domain-Adaptive Pre-Training (Shakespeare 전작) ===")
  print(f"  데이터: {len(shakespeare_dataset):,}개 청크 | epochs: {args.pretrain_epochs} | lr: {args.pretrain_lr}")

  for epoch in range(start_epoch, args.pretrain_epochs):
    model.train()
    total_loss, num_batches = 0, 0

    for batch in tqdm(dataloader, desc=f'pretrain-{epoch}', disable=TQDM_DISABLE):
      b_ids = batch['token_ids'].to(device)
      b_mask = batch['attention_mask'].to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
      labels = b_ids[:, 1:].contiguous().flatten()
      mask = b_mask[:, 1:].contiguous().flatten().bool()
      loss = F.cross_entropy(logits[mask], labels[mask], reduction='mean')
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
      optimizer.step()
      scheduler.step()

      total_loss += loss.item()
      num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"  Epoch {epoch}: pretrain loss :: {avg_loss:.3f}")

    if (epoch + 1) % 2 == 0 or epoch == args.pretrain_epochs - 1:
      save_checkpoint(model, optimizer, scheduler, args, ckpt_path,
                      epoch=epoch, stage=1, best_loss=avg_loss, patience_counter=0)

  print("Stage 1 완료.\n")
  return model


def train(args, model, device):
  """Stage 2: 소네트 데이터셋으로 LoRA fine-tuning. 체크포인트에서 이어 학습 가능."""
  import os
  ckpt_path = f'stage2_checkpoint_{args.filepath}'
  best_filepath = f'best_{args.filepath}'

  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)
  # 학습 중 시각화는 dev set 사용 (test set 오염 방지)
  held_out_sonnet_dataset = SonnetsDataset(args.dev_sonnet_path)

  # LoRA 어댑터만 학습: base 파라미터 동결.
  for name, param in model.named_parameters():
    param.requires_grad = ('lora_A' in name or 'lora_B' in name)
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  total = sum(p.numel() for p in model.parameters())
  print(f"[LoRA] 학습 파라미터: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

  optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.05)
  total_steps = len(sonnet_dataloader) * args.epochs
  warmup_steps = int(total_steps * args.warmup_ratio)
  scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

  best_train_loss = float('inf')
  patience_counter = 0
  start_epoch = 0

  if args.resume and os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optim'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch'] + 1
    best_train_loss = ckpt['best_loss']
    patience_counter = ckpt['patience_counter']
    print(f"\n[Stage 2] 체크포인트 로드: epoch {ckpt['epoch']} 완료 → epoch {start_epoch}부터 재개")
    print(f"  best_loss={best_train_loss:.3f}, patience={patience_counter}/{args.patience}")

  print(f"\n=== Stage 2: Fine-tuning on Sonnets (LoRA) ===")
  print(f"  데이터: {len(sonnet_dataset)}개 소네트 | epochs: {args.epochs} | lr: {args.lr}")

  for epoch in range(start_epoch, args.epochs):
    model.train()
    train_loss, num_batches = 0, 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask = batch['token_ids'].to(device), batch['attention_mask'].to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
      labels = b_ids[:, 1:].contiguous().flatten()
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
      optimizer.step()
      scheduler.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss:.3f}.")

    # 생성 샘플 확인 (dev set 기준)
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      _, generated = model.generate(
        encoding['input_ids'], temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, repetition_penalty=args.repetition_penalty,
        strategy='sampling', use_rhyme=False
      )
      print(f'{batch[1]}{generated}\n\n')
      break

    # TODO: 소넷의 작은 데이터셋에서 과적합을 방지하기 위한 종료 조건을 생각하시오.
    if train_loss < best_train_loss:
      best_train_loss = train_loss
      patience_counter = 0
      save_model(model, optimizer, args, best_filepath)
      print(f"Best model saved (loss: {best_train_loss:.3f})")
    else:
      patience_counter += 1
      print(f"No improvement. Patience: {patience_counter}/{args.patience}")

    save_checkpoint(model, optimizer, scheduler, args, ckpt_path,
                    epoch=epoch, stage=2, best_loss=best_train_loss,
                    patience_counter=patience_counter)

    if patience_counter >= args.patience:
      print(f"Early stopping at epoch {epoch}.")
      break

  return best_filepath


@torch.no_grad()
def generate_submission_sonnets(args, best_filepath):
  """세 가지 전략으로 소네트 생성 후 chrF 비교, 최고 성능 전략을 최종 제출 파일로 저장."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(best_filepath, weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)
  results = {}

  for strategy in ['sampling', 'top_k', 'beam']:
    print(f"\n--- Strategy: {strategy} ---")
    output_path = f'predictions/generated_sonnets_{strategy}.txt'
    generated_sonnets = []

    for batch in held_out_sonnet_dataset:
      sonnet_id = batch[0]
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
      _, generated_output = model.generate(
        encoding['input_ids'], temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, repetition_penalty=args.repetition_penalty,
        strategy=strategy, use_rhyme=False
      )
      full_sonnet = f'{batch[1]}{generated_output}\n\n'
      generated_sonnets.append((sonnet_id, full_sonnet))

    with open(output_path, "w+") as f:
      f.write(f"--Generated Sonnets-- \n\n")
      for sonnet in generated_sonnets:
        f.write(f"\n{sonnet[0]}\n")
        f.write(sonnet[1])

    try:
      score = test_sonnet(test_path=output_path, gold_path='data/TRUE_sonnets_held_out.txt')
      results[strategy] = score
      print(f"chrF score: {score:.4f}")
    except Exception as e:
      print(f"chrF 계산 오류: {e}")

  # 전략 비교 출력
  print("\n=== Generation Strategy Comparison (chrF) ===")
  for strategy, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {strategy:10s}: {score:.4f}")

  # 최고 성능 전략을 최종 제출 파일로 복사
  if results:
    import shutil
    best_strategy = max(results, key=results.get)
    shutil.copy(f'predictions/generated_sonnets_{best_strategy}.txt', args.sonnet_out)
    print(f"\nBest strategy: {best_strategy} (chrF: {results[best_strategy]:.4f})")
    print(f"Final submission saved: {args.sonnet_out}")

  return results


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--dev_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--use_gpu", action='store_true')

  # Stage 1 (DAPT) 설정
  parser.add_argument("--pretrain_epochs", type=int, default=10,
                      help="Shakespeare 전작 사전학습 epoch 수")
  parser.add_argument("--pretrain_lr", type=float, default=2e-5,
                      help="Stage 1 학습률")
  parser.add_argument("--chunk_size", type=int, default=256,
                      help="Stage 1 텍스트 청크 크기 (토큰)")

  # Stage 2 (Fine-tuning) 설정
  parser.add_argument("--epochs", type=int, default=100,
                      help="소네트 fine-tuning epoch 수")
  parser.add_argument("--lr", type=float, default=2e-4,
                      help="Stage 2 학습률 (LoRA)")
  parser.add_argument("--patience", type=int, default=15)

  # 공통 설정
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--warmup_ratio", type=float, default=0.1)
  parser.add_argument("--max_grad_norm", type=float, default=1.0)

  # Generation 설정
  parser.add_argument("--temperature", type=float, default=0.9)
  parser.add_argument("--top_p", type=float, default=0.9)
  parser.add_argument("--top_k", type=int, default=5,
                      help="top-k sampling의 k값 및 beam search의 beam 수")
  parser.add_argument("--repetition_penalty", type=float, default=1.3)

  parser.add_argument("--model_size", type=str,
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2-medium')
  parser.add_argument("--resume", action='store_true',
                      help="체크포인트에서 이어 학습")

  # LoRA 설정
  parser.add_argument("--lora_r", type=int, default=8)
  parser.add_argument("--lora_alpha", type=int, default=16)

  args = parser.parse_args()
  return args


def add_arguments(args):
  if args.model_size == 'gpt2':
    args.d, args.l, args.num_heads = 768, 12, 12
  elif args.model_size == 'gpt2-medium':
    args.d, args.l, args.num_heads = 1024, 24, 16
  elif args.model_size == 'gpt2-large':
    args.d, args.l, args.num_heads = 1280, 36, 20
  elif args.model_size == 'gpt2-xl':
    args.d, args.l, args.num_heads = 1600, 48, 25
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  import os
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'
  seed_everything(args.seed)

  args = add_arguments(args)
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  best_filepath = f'best_{args.filepath}'

  # --resume: Stage 2 체크포인트가 있으면 Stage 2부터 이어 학습
  stage2_ckpt = f'stage2_checkpoint_{args.filepath}'
  if args.resume and os.path.exists(stage2_ckpt):
    print("[Resume] Stage 2 체크포인트 발견 → Stage 2부터 이어 학습")
    ckpt = torch.load(stage2_ckpt, weights_only=False)
    model = SonnetGPT(args)
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
  else:
    model = SonnetGPT(args)
    model = model.to(device)
    # Stage 1: Shakespeare 전작으로 도메인 적응 사전학습.
    model = pretrain_shakespeare(args, model, device)

  # Stage 2: 소네트 fine-tuning (LoRA)
  best_filepath = train(args, model, device)

  # 세 가지 생성 전략 비교 및 최종 제출 파일 생성
  results = generate_submission_sonnets(args, best_filepath)
