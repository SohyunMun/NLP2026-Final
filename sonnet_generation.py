'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
SonnetGPT 모델을 훈련하고, 필요한 제출용 파일을 작성한다.
'''

import argparse
import os
import random
import re
import torch

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


# 재현성을 위한 random seed 고정.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Sonnet 생성을 위해 설계된 여러분의 GPT-2 모델."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # 기본적으로, 전체 모델을 fine-tuning한다. TODO: 이것은 좋은 생각이 아닌 것 같다.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    ParaphraseGPT의 forward pass와 유사하지만, 여기서는 시퀀스의 마지막 토큰뿐만 아니라 시퀀스의 각 토큰에 대한 logit을 생성하려고 한다.
    이를 통해, 마지막 토큰에 대한 다음 토큰의 분포만 학습하는 것이 아니라, 모델은 소네트를 구성하는 자연어 분포를 학습할 수 있다.
    """
    outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    return self.gpt.hidden_state_to_token(outputs['last_hidden_state'])


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, top_k=50, max_length=128,
               repetition_penalty=1.0, no_repeat_ngram_size=0, target_lines=None,
               decoding_strategy='top_p', num_beams=3):
    """
    top-p sampling 과 softmax temperature를 사용하여 새로운 소넷을 생성한다.

    TODO: 지금 이 방법은 기대 이하일 수 있다. 영감을 얻기 위해 Hugging Face의 model.generate(...) 함수를 참고해도 좋겠다.
        여러 시퀀스를 생성하고 beam search를 통해 최적의 시퀀스를 선택하는 것도 좋은 한 가지 방법이다.
        Top-k 샘플링 역시 또 다른 방법이며, 그 외에도 많은 접근법이 있다.
    """
    if decoding_strategy == 'beam':
      return self.generate_beam(
        encoding,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        target_lines=target_lines,
        num_beams=num_beams
      )

    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())


    for _ in range(max_length):
      # logits을 구하기 위한 forward pass.
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :]
      if repetition_penalty > 1.0:
        apply_repetition_penalty(logits_last_token, token_ids, repetition_penalty)
      if no_repeat_ngram_size > 0:
        banned_tokens = get_banned_ngram_tokens(token_ids[0].tolist(), no_repeat_ngram_size)
        if banned_tokens:
          logits_last_token[:, banned_tokens] = -float('inf')
      logits_last_token = logits_last_token / temperature  # Apply temperature scaling

      if decoding_strategy == 'top_k':
        sampled_token = sample_top_k(logits_last_token, top_k)
      else:
        sampled_token = sample_top_p(logits_last_token, top_p)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

      if target_lines is not None:
        decoded = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())
        if len(nonempty_lines(decoded)) >= target_lines:
          break

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output


  @torch.no_grad()
  def generate_beam(self, encoding, max_length=128, repetition_penalty=1.0,
                    no_repeat_ngram_size=0, target_lines=None, num_beams=3):
    device = self.get_device()
    start_ids = encoding.to(device)
    beams = [(start_ids, 0.0)]

    for _ in range(max_length):
      candidates = []
      for token_ids, beam_score in beams:
        attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(device)
        logits = self.forward(token_ids, attention_mask)[:, -1, :]
        if repetition_penalty > 1.0:
          apply_repetition_penalty(logits, token_ids, repetition_penalty)
        if no_repeat_ngram_size > 0:
          banned_tokens = get_banned_ngram_tokens(token_ids[0].tolist(), no_repeat_ngram_size)
          if banned_tokens:
            logits[:, banned_tokens] = -float('inf')

        log_probs = torch.log_softmax(logits, dim=-1)
        top_scores, top_indices = torch.topk(log_probs, k=num_beams, dim=-1)
        for score, token_id in zip(top_scores[0], top_indices[0]):
          if token_id.item() == self.tokenizer.eos_token_id:
            candidates.append((token_ids, beam_score + score.item()))
            continue
          next_ids = torch.cat([token_ids, token_id.view(1, 1)], dim=1)
          candidates.append((next_ids, beam_score + score.item()))

      beams = sorted(candidates, key=lambda item: item[1] / item[0].shape[1], reverse=True)[:num_beams]
      if target_lines is not None:
        decoded = self.tokenizer.decode(beams[0][0][0].cpu().numpy().tolist())
        if len(nonempty_lines(decoded)) >= target_lines:
          break

    token_ids = beams[0][0]
    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output


def sample_top_p(logits, top_p):
  probs = torch.nn.functional.softmax(logits, dim=-1)
  sorted_probs, sorted_indices = torch.sort(probs, descending=True)
  cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
  top_p_mask = cumulative_probs <= top_p
  top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
  top_p_mask[..., 0] = True
  filtered_probs = sorted_probs * top_p_mask
  filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)
  sampled_index = torch.multinomial(filtered_probs, 1)
  return sorted_indices.gather(dim=-1, index=sampled_index)


def sample_top_k(logits, top_k):
  top_k = min(top_k, logits.shape[-1])
  values, indices = torch.topk(logits, k=top_k, dim=-1)
  probs = torch.nn.functional.softmax(values, dim=-1)
  sampled_index = torch.multinomial(probs, 1)
  return indices.gather(dim=-1, index=sampled_index)


def apply_repetition_penalty(logits, token_ids, penalty):
  for token_id in set(token_ids[0].tolist()):
    if logits[0, token_id] < 0:
      logits[0, token_id] *= penalty
    else:
      logits[0, token_id] /= penalty


def get_banned_ngram_tokens(tokens, ngram_size):
  if len(tokens) < ngram_size - 1:
    return []
  prefix = tuple(tokens[-ngram_size + 1:])
  banned = []
  for i in range(len(tokens) - ngram_size + 1):
    if tuple(tokens[i:i + ngram_size - 1]) == prefix:
      banned.append(tokens[i + ngram_size - 1])
  return banned


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


def nonempty_lines(text):
  return [line.strip() for line in text.splitlines() if line.strip()]


def strip_prompt_lines(prompt, decoded_output):
  prompt_lines = nonempty_lines(prompt)
  output_lines = nonempty_lines(decoded_output)
  if output_lines[:len(prompt_lines)] == prompt_lines:
    return output_lines[len(prompt_lines):]

  prompt_text = prompt.strip()
  decoded_text = decoded_output.strip()
  if decoded_text.startswith(prompt_text):
    return nonempty_lines(decoded_text[len(prompt_text):])

  return output_lines


def split_long_lines(lines, target_count):
  fallback_lines = [
    "And in my verse thy memory shall remain.",
    "So shall thy beauty live within my rhyme.",
    "Till time itself forgets to steal thy name.",
    "And love shall answer love in every line.",
  ]
  lines = list(lines)
  while len(lines) < target_count:
    split_idx = -1
    split_len = 0
    for i, line in enumerate(lines):
      words = line.split()
      if len(words) > split_len:
        split_idx = i
        split_len = len(words)

    if split_idx < 0 or split_len < 10:
      lines.append(fallback_lines[len(lines) % len(fallback_lines)])
      continue

    words = lines[split_idx].split()
    mid = len(words) // 2
    lines[split_idx:split_idx + 1] = [
      ' '.join(words[:mid]),
      ' '.join(words[mid:])
    ]

  return lines


def split_overlong_lines(lines, max_words=10):
  split_lines = []
  for line in lines:
    words = line.split()
    if len(words) <= max_words:
      split_lines.append(line)
      continue
    for i in range(0, len(words), max_words):
      split_lines.append(' '.join(words[i:i + max_words]))
  return split_lines


def format_fourteen_line_sonnet(prompt, decoded_output):
  prompt_lines = nonempty_lines(prompt)[:3]
  needed_lines = 14 - len(prompt_lines)
  continuation_lines = strip_prompt_lines(prompt, decoded_output)
  continuation_lines = split_overlong_lines(continuation_lines)
  continuation_lines = split_long_lines(continuation_lines, needed_lines)
  return '\n'.join(prompt_lines + continuation_lines[:needed_lines])


def words_in_line(line):
  return re.findall(r"[A-Za-z']+", line.lower())


def ending_word(line):
  words = words_in_line(line)
  return words[-1] if words else ''


def suffix_match_score(left, right):
  if not left or not right:
    return -1.0
  if left == right:
    return -1.5
  if left[-3:] == right[-3:]:
    return 3.0
  if left[-2:] == right[-2:]:
    return 1.5
  if left[-1:] == right[-1:]:
    return 0.4
  return -0.5


def rhyme_scheme_score(lines):
  endings = [ending_word(line) for line in lines]
  rhyme_pairs = [(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 13)]
  return sum(suffix_match_score(endings[i], endings[j]) for i, j in rhyme_pairs)


def repetition_score(lines):
  lowered_lines = [line.lower() for line in lines]
  duplicate_lines = len(lowered_lines) - len(set(lowered_lines))

  all_words = []
  for line in lines:
    all_words.extend(word for word in words_in_line(line) if len(word) > 3)

  repeated_words = 0
  seen_words = {}
  for word in all_words:
    seen_words[word] = seen_words.get(word, 0) + 1
  for count in seen_words.values():
    repeated_words += max(0, count - 3)

  trigrams = []
  for i in range(len(all_words) - 2):
    trigrams.append(tuple(all_words[i:i + 3]))
  repeated_trigrams = len(trigrams) - len(set(trigrams))

  endings = [ending_word(line) for line in lines if ending_word(line)]
  repeated_endings = len(endings) - len(set(endings))

  return -(duplicate_lines * 8.0 + repeated_trigrams * 3.0 + repeated_endings * 1.5 + repeated_words * 0.35)


def line_length_score(lines):
  score = 0.0
  for line in lines:
    length = len(words_in_line(line))
    score -= abs(length - 8) * 0.7
    if length < 4:
      score -= 4.0
    elif length > 12:
      score -= 4.0 + (length - 12) * 1.0
    if line[-1:] in {'.', ',', ';', ':', '!', '?'}:
      score += 0.25
  return score


def text_cleanliness_score(lines):
  text = '\n'.join(lines)
  digit_count = sum(char.isdigit() for char in text)
  odd_symbol_count = len(re.findall(r"[\[\]\{\}<>#_=*|`~]", text))
  non_ascii_count = sum(ord(char) > 127 for char in text)
  very_short_lines = sum(1 for line in lines if len(words_in_line(line)) <= 2)
  return -(digit_count * 1.5 + odd_symbol_count * 2.0 + non_ascii_count * 1.2 + very_short_lines * 2.0)


def sonnet_quality_score(sonnet):
  lines = nonempty_lines(sonnet)
  score = -abs(len(lines) - 14) * 100.0
  score += line_length_score(lines)
  score += repetition_score(lines)
  score += rhyme_scheme_score(lines)
  score += text_cleanliness_score(lines)
  return score


def char_ngrams(text, n):
  text = ' '.join(text.split())
  return {text[i:i + n] for i in range(max(0, len(text) - n + 1))}


def pairwise_chrf_like(left, right, max_n=6):
  scores = []
  for n in range(1, max_n + 1):
    left_ngrams = char_ngrams(left, n)
    right_ngrams = char_ngrams(right, n)
    if not left_ngrams or not right_ngrams:
      scores.append(0.0)
      continue
    overlap = len(left_ngrams & right_ngrams)
    precision = overlap / len(left_ngrams)
    recall = overlap / len(right_ngrams)
    scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
  return sum(scores) / len(scores)


def corpus_chrf_like(candidates, references, max_n=6, beta=2):
  precisions = []
  recalls = []
  for n in range(1, max_n + 1):
    overlap = candidate_total = reference_total = 0
    for candidate, reference in zip(candidates, references):
      candidate_ngrams = char_ngrams(candidate, n)
      reference_ngrams = char_ngrams(reference, n)
      overlap += len(candidate_ngrams & reference_ngrams)
      candidate_total += len(candidate_ngrams)
      reference_total += len(reference_ngrams)
    precisions.append(overlap / candidate_total if candidate_total else 0.0)
    recalls.append(overlap / reference_total if reference_total else 0.0)
  precision = sum(precisions) / len(precisions)
  recall = sum(recalls) / len(recalls)
  if precision == 0 and recall == 0:
    return 0.0
  beta2 = beta ** 2
  return 100 * (1 + beta2) * precision * recall / (beta2 * precision + recall)


def mbr_centrality_score(candidate, all_candidates):
  if len(all_candidates) <= 1:
    return 0.0
  similarities = [
    pairwise_chrf_like(candidate, other)
    for other in all_candidates
    if other != candidate
  ]
  return 0.0 if not similarities else sum(similarities) / len(similarities)


def generate_candidate_sonnet(model, prompt, args, temperature, decoding_strategy):
  encoding = model.tokenizer(prompt, return_tensors='pt', padding=False, truncation=True).to(model.get_device())
  output = model.generate(
    encoding['input_ids'],
    temperature=temperature,
    top_p=args.top_p,
    top_k=args.top_k,
    max_length=args.max_generation_tokens,
    repetition_penalty=args.repetition_penalty,
    no_repeat_ngram_size=args.no_repeat_ngram_size,
    target_lines=args.target_lines,
    decoding_strategy=decoding_strategy,
    num_beams=args.num_beams
  )[0][0]
  decoded_output = model.tokenizer.decode(output)
  return format_fourteen_line_sonnet(prompt, decoded_output)


def model_sequence_score(model, text):
  encoding = model.tokenizer(text, return_tensors='pt', padding=False, truncation=True).to(model.get_device())
  input_ids = encoding['input_ids']
  attention_mask = encoding['attention_mask']
  if input_ids.shape[1] < 2:
    return -100.0

  logits = model(input_ids, attention_mask)
  logits = logits[:, :-1].contiguous()
  labels = input_ids[:, 1:].contiguous()
  loss = F.cross_entropy(
    logits.view(-1, logits.shape[-1]),
    labels.view(-1),
    reduction='mean'
  )
  return -loss.item()


def select_best_sonnet(model, prompt, args):
  base_temperature = args.temperature
  temperature_offsets = [0.0, -0.05, 0.05, 0.1, -0.1]
  strategies = [strategy.strip() for strategy in args.decoding_strategies.split(',') if strategy.strip()]
  candidates = []

  for i in range(args.num_candidates):
    temperature = base_temperature + temperature_offsets[i % len(temperature_offsets)]
    temperature = max(0.5, min(1.2, temperature))
    strategy = strategies[i % len(strategies)] if strategies else args.decoding_strategy
    candidates.append(generate_candidate_sonnet(model, prompt, args, temperature, strategy))

  best_sonnet = None
  best_score = -float('inf')
  for candidate in candidates:
    score = sonnet_quality_score(candidate)
    score += model_sequence_score(model, candidate) * args.model_score_weight
    score += mbr_centrality_score(candidate, candidates) * args.mbr_weight
    if score > best_score:
      best_score = score
      best_sonnet = candidate

  return best_sonnet, best_score


def first_lines(text, count=3):
  return '\n'.join(nonempty_lines(text)[:count])


def prompt_token_lengths(tokenizer, texts):
  lengths = []
  for text in texts:
    prompt = first_lines(text)
    lengths.append(len(tokenizer(prompt, return_tensors='pt', padding=False, truncation=True)['input_ids'][0]))
  return lengths


def weighted_lm_loss(logits, input_ids, attention_mask, prompt_lengths, newline_token_id, args):
  logits = logits[:, :-1].contiguous()
  labels = input_ids[:, 1:].contiguous()
  label_mask = attention_mask[:, 1:].contiguous()
  labels = labels.masked_fill(label_mask == 0, -100)

  weights = torch.ones_like(labels, dtype=logits.dtype)
  for row, prompt_len in enumerate(prompt_lengths):
    prompt_label_count = max(prompt_len - 1, 0)
    weights[row, :prompt_label_count] = args.prompt_loss_weight
  if newline_token_id is not None:
    weights = weights * torch.where(labels == newline_token_id, args.line_break_loss_weight, 1.0)
  weights = weights.masked_fill(labels == -100, 0.0)

  token_losses = F.cross_entropy(
    logits.view(-1, logits.shape[-1]),
    labels.reshape(-1),
    ignore_index=-100,
    reduction='none'
  ).view_as(labels)
  return (token_losses * weights).sum() / weights.sum().clamp_min(1.0)


def lr_scale_for_step(step, total_steps, warmup_steps):
  if total_steps <= 0:
    return 1.0
  if warmup_steps > 0 and step < warmup_steps:
    return max(1e-8, step / warmup_steps)
  remaining_steps = max(total_steps - step, 0)
  decay_steps = max(total_steps - warmup_steps, 1)
  return max(0.0, remaining_steps / decay_steps)


def set_optimizer_lr(optimizer, lr):
  for group in optimizer.param_groups:
    group['lr'] = lr


def evaluate_dev_chrf(model, args):
  if not args.dev_sonnet_path or not args.dev_gold_path:
    return None
  if not os.path.exists(args.dev_sonnet_path) or not os.path.exists(args.dev_gold_path):
    return None

  original_num_candidates = args.num_candidates
  args.num_candidates = args.dev_num_candidates
  dev_dataset = SonnetsDataset(args.dev_sonnet_path)
  gold_sonnets = [sonnet for _, sonnet in SonnetsDataset(args.dev_gold_path)]
  generated_sonnets = []
  model.eval()
  with torch.no_grad():
    for _, prompt in dev_dataset:
      generated_sonnet, _ = select_best_sonnet(model, prompt, args)
      generated_sonnets.append(generated_sonnet)
  args.num_candidates = original_num_candidates
  return corpus_chrf_like(generated_sonnets, gold_sonnets[:len(generated_sonnets)])


def checkpoint_path_for_args(args):
  if getattr(args, 'checkpoint_path', None):
    return args.checkpoint_path
  best_path = f'best_{args.filepath}'
  if os.path.exists(best_path):
    return best_path
  return f'{args.epochs-1}_{args.filepath}'


def train(args):
  """Sonnet 데이터셋에서 소넷 생성을 위해 GPT-2 훈련.""" 
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # 데이터, 해당 데이터셋 및 데이터로드 생성하기.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  if getattr(args, 'init_checkpoint_path', None):
    saved = torch.load(args.init_checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'])
    print(f"initialize model from {args.init_checkpoint_path}")
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
  total_steps = max(1, len(sonnet_dataloader) * args.epochs)
  warmup_steps = int(total_steps * args.warmup_ratio)
  global_step = 0
  best_train_loss = float('inf')
  best_dev_chrf = -float('inf')
  epochs_without_improvement = 0
  saved_best = False
  newline_token_id = model.tokenizer.encode('\n')[0]

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # 입력을 가져와서 GPU로 보내기(이 모델을 CPU에서 훈련시키는 것을 권장하지 않는다).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      batch_texts = [sonnet_dataset.sonnets[i] for i in batch['sent_ids']]
      prompt_lengths = prompt_token_lengths(model.tokenizer, batch_texts)

      # 손실, 그래디언트를 계산하고 모델 파라미터 업데이트.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      loss = weighted_lm_loss(logits, b_ids, b_mask, prompt_lengths, newline_token_id, args)
      loss.backward()
      if args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
      optimizer.step()
      global_step += 1
      set_optimizer_lr(optimizer, args.lr * lr_scale_for_step(global_step, total_steps, warmup_steps))

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    if args.selection_metric == 'loss':
      if train_loss < best_train_loss:
        best_train_loss = train_loss
        epochs_without_improvement = 0
        saved_best = True
        args.best_epoch = epoch
        args.best_train_loss = train_loss
        save_model(model, optimizer, args, f'best_{args.filepath}')
        print(f"Epoch {epoch}: best train loss updated :: {best_train_loss :.3f}.")
      else:
        epochs_without_improvement += 1

    if not args.skip_epoch_sample:
      print('Generating a sample output sonnet...')
      model.eval()
      batch = held_out_sonnet_dataset[0]
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      print(f'{batch[1]}{output[1]}\n\n')

    dev_chrf = None
    if args.selection_metric == 'dev_chrf' and args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
      dev_chrf = evaluate_dev_chrf(model, args)
      if dev_chrf is not None:
        print(f"Epoch {epoch}: dev approx chrF :: {dev_chrf :.3f}.")
        if dev_chrf > best_dev_chrf:
          best_dev_chrf = dev_chrf
          epochs_without_improvement = 0
          saved_best = True
          args.best_epoch = epoch
          args.best_dev_chrf = dev_chrf
          save_model(model, optimizer, args, f'best_{args.filepath}')
        else:
          epochs_without_improvement += 1

    if args.patience > 0 and epochs_without_improvement >= args.patience:
      if args.selection_metric == 'loss':
        print(f"Early stopping at epoch {epoch}; best train loss :: {best_train_loss :.3f}.")
      else:
        print(f"Early stopping at epoch {epoch}; best dev approx chrF :: {best_dev_chrf :.3f}.")
      break

    if epoch == args.epochs - 1 and not saved_best:
      save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(checkpoint_path_for_args(args), weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    best_sonnet, best_score = select_best_sonnet(model, batch[1], args)
    full_sonnet = f'{best_sonnet}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'candidate score: {best_score:.3f}')
    print(full_sonnet)

  output_dir = os.path.dirname(args.sonnet_out)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets--\n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--dev_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--dev_gold_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")
  parser.add_argument("--checkpoint_path", type=str, default=None)
  parser.add_argument("--init_checkpoint_path", type=str, default=None)

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=0.85)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)
  parser.add_argument("--top_k", type=int, default=50)
  parser.add_argument("--decoding_strategy", type=str, choices=('top_p', 'top_k', 'beam'), default='top_p')
  parser.add_argument("--decoding_strategies", type=str, default='top_p,top_k,beam')
  parser.add_argument("--num_beams", type=int, default=3)
  parser.add_argument("--num_candidates", type=int, default=8)
  parser.add_argument("--dev_num_candidates", type=int, default=2)
  parser.add_argument("--model_score_weight", type=float, default=4.0)
  parser.add_argument("--mbr_weight", type=float, default=8.0)
  parser.add_argument("--repetition_penalty", type=float, default=1.03)
  parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
  parser.add_argument("--target_lines", type=int, default=14)
  parser.add_argument("--max_generation_tokens", type=int, default=160)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--weight_decay", type=float, default=0.01)
  parser.add_argument("--warmup_ratio", type=float, default=0.1)
  parser.add_argument("--max_grad_norm", type=float, default=1.0)
  parser.add_argument("--prompt_loss_weight", type=float, default=0.35)
  parser.add_argument("--line_break_loss_weight", type=float, default=1.2)
  parser.add_argument("--eval_every", type=int, default=1)
  parser.add_argument("--patience", type=int, default=3)
  parser.add_argument("--selection_metric", choices=('loss', 'dev_chrf'), default='loss')
  parser.add_argument("--skip_epoch_sample", action='store_true')
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
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
  elif args.model_size == 'gpt2-xl':
    args.d = 1600
    args.l = 48
    args.num_heads = 25
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # 경로명 저장.
  seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
  train(args)
  generate_submission_sonnets(args)
