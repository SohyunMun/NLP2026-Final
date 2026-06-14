#!/usr/bin/env python3
"""Run form-aware SFT-initialized LoRA-DPO for sonnet generation.

This experiment keeps the strong MSK SFT initialization, then teaches DPO to
prefer sonnets that preserve content while rejecting bad form, weak rhyme,
line-length distortions, and repetition-heavy continuations. Generation also
uses a form/rhyme/repetition-aware reranker.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ROOT = PROJECT_ROOT.parent
for search_path in (SCRIPT_DIR, ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

try:
    import pronouncing
except Exception:
    pronouncing = None

from datasets import SonnetsDataset
from optimizer import AdamW
from run_msk_sft_lora_dpo import (
    apply_lora_to_policy,
    chrf_score,
    continuation,
    continuation_logps,
    corrupt_by_mismatch,
    corrupt_by_repetition,
    dpo_loss,
    first_lines,
    load_policy_from_dpo,
    load_sft_model,
    save_checkpoint,
)
from sonnet_generation import (
    generate_candidate_sonnet,
    mbr_centrality_score,
    model_sequence_score,
    nonempty_lines,
    seed_everything,
)


SONNET_RHYME_PAIRS = [(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 13)]
ANTI_RHYME_ENDINGS = [
    "orange",
    "silver",
    "month",
    "purple",
    "wolf",
    "chaos",
    "spirit",
    "music",
    "truth",
    "world",
    "depth",
]


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def replace_last_word(line: str, new_word: str) -> str:
    match = list(re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", line))
    if not match:
        return new_word
    last = match[-1]
    return f"{line[:last.start()]}{new_word}{line[last.end():]}"


def corrupt_by_bad_rhyme(prompt: str, winner: str) -> str:
    """Keep 14 lines but make line endings intentionally unhelpful for rhyme."""
    prompt_lines = nonempty_lines(prompt)
    lines = (prompt_lines + continuation(winner))[:14]
    while len(lines) < 14:
        lines.append("And time removes the color from the day.")

    fixed = []
    for idx, line in enumerate(lines):
        if idx < len(prompt_lines):
            fixed.append(line)
        else:
            fixed.append(replace_last_word(line, ANTI_RHYME_ENDINGS[idx % len(ANTI_RHYME_ENDINGS)]))
    return "\n".join(fixed[:14])


def corrupt_by_bad_line_length(prompt: str, winner: str) -> str:
    """Keep 14 lines but alternate too-short and too-long continuation lines."""
    prompt_lines = nonempty_lines(prompt)
    source = continuation(winner) or nonempty_lines(winner)
    needed = 14 - len(prompt_lines)
    bad = []
    for idx in range(needed):
        base = source[idx % len(source)] if source else "Love fades."
        nxt = source[(idx + 1) % len(source)] if source else "Time follows."
        if idx % 3 == 0:
            bad.append("Love.")
        elif idx % 3 == 1:
            bad.append(f"{base} {nxt}")
        else:
            bad.append(base)
    return "\n".join((prompt_lines + bad)[:14])


def corrupt_by_repeated_endings(prompt: str, winner: str) -> str:
    """Make endings repetitive, which should be rejected despite superficial rhyme."""
    prompt_lines = nonempty_lines(prompt)
    lines = (prompt_lines + continuation(winner))[:14]
    while len(lines) < 14:
        lines.append("And love shall answer love in every line.")
    fixed = []
    for idx, line in enumerate(lines):
        if idx < len(prompt_lines):
            fixed.append(line)
        else:
            fixed.append(replace_last_word(line, "love"))
    return "\n".join(fixed[:14])


def corrupt_by_form_short(prompt: str, winner: str) -> str:
    """Use a too-short continuation so DPO sees non-sonnet structure as rejected."""
    prompt_lines = nonempty_lines(prompt)
    source = continuation(winner) or nonempty_lines(winner)
    short_continuation = source[: max(1, min(5, len(source)))]
    return "\n".join(prompt_lines + short_continuation)


class FormAwarePreferenceDataset(Dataset):
    def __init__(
        self,
        sonnet_path: Path,
        tokenizer,
        max_pairs: int | None = None,
        seed: int = 11711,
        include_short_form_rejects: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_pairs = max_pairs
        self.include_short_form_rejects = include_short_form_rejects
        self.rng = random.Random(seed)
        self.sonnets = [sonnet for _, sonnet in SonnetsDataset(str(sonnet_path))]
        self.pairs = self._build_pairs()
        if max_pairs is not None:
            self.rng.shuffle(self.pairs)
            self.pairs = self.pairs[:max_pairs]

    def _build_pairs(self) -> list[tuple[str, str, str, str]]:
        pairs = []
        n = len(self.sonnets)
        for idx, winner in enumerate(self.sonnets):
            prompt = first_lines(winner)
            donor = self.sonnets[(idx * 37 + 11) % n]
            if donor == winner:
                donor = self.sonnets[(idx + 1) % n]
            rejected_variants = [
                ("mismatch", corrupt_by_mismatch(prompt, donor)),
                ("repetition", corrupt_by_repetition(prompt, winner)),
                ("bad_rhyme", corrupt_by_bad_rhyme(prompt, winner)),
                ("bad_line_length", corrupt_by_bad_line_length(prompt, winner)),
                ("repeated_endings", corrupt_by_repeated_endings(prompt, winner)),
            ]
            if self.include_short_form_rejects:
                rejected_variants.append(("short_form", corrupt_by_form_short(prompt, winner)))
            for reject_type, rejected in rejected_variants:
                pairs.append((prompt, winner, rejected, reject_type))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

    def collate_fn(self, examples):
        prompts, winners, losers, reject_types = zip(*examples)
        winner_enc = self.tokenizer(
            list(winners),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        loser_enc = self.tokenizer(
            list(losers),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        prompt_lengths = [
            len(self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=512)["input_ids"][0])
            for prompt in prompts
        ]
        return {
            "winner_ids": winner_enc["input_ids"],
            "winner_mask": winner_enc["attention_mask"],
            "loser_ids": loser_enc["input_ids"],
            "loser_mask": loser_enc["attention_mask"],
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
            "reject_types": list(reject_types),
        }


def fallback_rhyme_foot(word: str) -> str:
    word = re.sub(r"[^a-z']", "", word.lower()).strip("'")
    if not word:
        return ""
    match = re.search(r"[aeiouy][a-z']*$", word)
    return match.group(0)[-4:] if match else word[-3:]


def estimate_syllables(text: str) -> int:
    total = 0
    for token in words(text):
        count = None
        if pronouncing is not None:
            phones = pronouncing.phones_for_word(token)
            if phones:
                count = pronouncing.syllable_count(phones[0])
        if count is None:
            groups = re.findall(r"[aeiouy]+", token.lower())
            count = max(1, len(groups))
            if token.endswith("e") and count > 1:
                count -= 1
        total += count
    return total


def last_word(line: str) -> str:
    tokens = words(line)
    return tokens[-1] if tokens else ""


def rhyme_words_for(base_word: str, limit: int = 80) -> list[str]:
    base_word = re.sub(r"[^a-z']", "", base_word.lower()).strip("'")
    if not base_word:
        return []
    candidates = [base_word]
    if pronouncing is not None:
        candidates.extend(pronouncing.rhymes(base_word))
    seen = set()
    clean = []
    for word in candidates:
        word = re.sub(r"[^a-z']", "", word.lower()).strip("'")
        if not word or word in seen:
            continue
        seen.add(word)
        clean.append(word)
        if len(clean) >= limit:
            break
    return clean


def rhyme_token_ids(tokenizer, lines: list[str], line_idx: int, limit: int = 80) -> set[int]:
    rhyme_map = {2: 0, 3: 1, 6: 4, 7: 5, 10: 8, 11: 9, 13: 12}
    if line_idx not in rhyme_map:
        return set()
    base_idx = rhyme_map[line_idx]
    if base_idx >= len(lines):
        return set()
    ids = set()
    for rhyme_word in rhyme_words_for(last_word(lines[base_idx]), limit=limit):
        for prefix in ("", " "):
            encoded = tokenizer.encode(prefix + rhyme_word)
            if len(encoded) == 1:
                ids.add(encoded[0])
    return ids


def apply_local_repetition_penalty(logits, token_ids, penalty: float) -> None:
    if penalty <= 1.0:
        return
    for token_id in set(token_ids[0].tolist()):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty


def sample_top_p_token(logits, top_p: float):
    probs = torch.softmax(logits, dim=-1)
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
    filtered_probs = filtered_probs / prob_sum
    sampled_index = torch.multinomial(filtered_probs, 1)
    return sorted_indices.gather(dim=-1, index=sampled_index)


@torch.no_grad()
def generate_line_rhyme_candidate(model, prompt: str, args, temperature: float) -> str:
    device = model.get_device()
    tokenizer = model.tokenizer
    newline_id = tokenizer.encode("\n")[0]
    eos_id = tokenizer.eos_token_id

    lines = nonempty_lines(prompt)[:3]
    while len(lines) < 3:
        lines.append("And write of thee in verse as time goes on,")

    context_ids = tokenizer.encode("\n".join(lines) + "\n", return_tensors="pt").to(device)
    attention_mask = torch.ones(context_ids.shape, dtype=torch.int64).to(device)

    for line_idx in range(len(lines), 14):
        current_line_tokens = []
        current_rhyme_ids = rhyme_token_ids(tokenizer, lines, line_idx, limit=args.rhyme_word_limit)
        rhyme_selected = False

        for _ in range(args.rhyme_line_max_tokens):
            logits = model(context_ids, attention_mask)[:, -1, :].clone()
            apply_local_repetition_penalty(logits, context_ids, args.repetition_penalty)
            logits = logits / max(temperature, 1e-6)

            current_text = tokenizer.decode(current_line_tokens).strip()
            syllables = estimate_syllables(current_text)

            if syllables < args.rhyme_min_syllables or len(current_line_tokens) < args.rhyme_min_tokens:
                logits[0, newline_id] -= args.newline_suppress_bias

            if current_rhyme_ids and not rhyme_selected and syllables >= args.rhyme_bias_start_syllables:
                valid_ids = [tid for tid in current_rhyme_ids if tid < logits.shape[-1]]
                if valid_ids:
                    logits[0, valid_ids] += args.rhyme_token_bias
                    logits[0, newline_id] -= args.newline_suppress_bias

            if rhyme_selected or syllables >= args.rhyme_target_syllables:
                logits[0, newline_id] += args.newline_bonus

            sampled_token = sample_top_p_token(logits, args.top_p)
            token_id = sampled_token.item()
            if token_id == eos_id:
                break
            if token_id == newline_id:
                break

            current_line_tokens.append(token_id)
            if token_id in current_rhyme_ids:
                rhyme_selected = True
            context_ids = torch.cat([context_ids, sampled_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=torch.int64, device=device)],
                dim=1,
            )

        new_line = tokenizer.decode(current_line_tokens).strip()
        if not new_line:
            new_line = "And in my verse thy memory shall remain."
        lines.append(new_line)

        newline_token = torch.tensor([[newline_id]], dtype=torch.long, device=device)
        context_ids = torch.cat([context_ids, newline_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=torch.int64, device=device)],
            dim=1,
        )

    return "\n".join(lines[:14])


def rhymes_similar(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) == 1
    if abs(len(left) - len(right)) == 1:
        longer, shorter = (left, right) if len(left) > len(right) else (right, left)
        return any(longer[:idx] + longer[idx + 1:] == shorter for idx in range(len(longer)))
    return False


def line_count_score(lines: list[str]) -> float:
    if not lines:
        return 0.0
    return max(0.0, 1.0 - abs(len(lines) - 14) / 14.0)


def line_length_score(lines: list[str]) -> float:
    if not lines:
        return 0.0
    scores = []
    for line in lines[:14]:
        length = len(words(line))
        scores.append(max(0.0, 1.0 - abs(length - 10) / 10.0))
    return float(np.mean(scores)) if scores else 0.0


def rhyme_pair_score(lines: list[str]) -> float:
    if len(lines) < 14:
        return 0.0
    feet = []
    for line in lines[:14]:
        tokens = words(line)
        feet.append(fallback_rhyme_foot(tokens[-1]) if tokens else "")
    matches = 0
    for left, right in SONNET_RHYME_PAIRS:
        if rhymes_similar(feet[left], feet[right]):
            matches += 1
    return matches / len(SONNET_RHYME_PAIRS)


def lexical_mattr(text: str, window: int = 50) -> float:
    tokens = words(text)
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    scores = []
    for idx in range(0, len(tokens) - window + 1):
        span = tokens[idx:idx + window]
        scores.append(len(set(span)) / window)
    return float(np.mean(scores))


def repetition_rate(text: str) -> float:
    tokens = [token for token in words(text) if len(token) > 2]
    if not tokens:
        return 0.0
    unigram_counts = Counter(tokens)
    repeated_unigrams = sum(max(0, count - 3) for count in unigram_counts.values())
    trigrams = [tuple(tokens[idx:idx + 3]) for idx in range(0, len(tokens) - 2)]
    repeated_trigrams = len(trigrams) - len(set(trigrams)) if trigrams else 0
    lines = [line.lower() for line in nonempty_lines(text)]
    repeated_lines = len(lines) - len(set(lines))
    return min(1.0, (repeated_unigrams + 3 * repeated_trigrams + 5 * repeated_lines) / max(1, len(tokens)))


def form_features(sonnet: str) -> dict[str, float]:
    lines = nonempty_lines(sonnet)
    count = line_count_score(lines)
    length = line_length_score(lines)
    rhyme = rhyme_pair_score(lines)
    lex = lexical_mattr(sonnet)
    repeat = repetition_rate(sonnet)
    partial = 0.45 * count + 0.20 * length + 0.25 * rhyme + 0.10 * max(0.0, 1.0 - repeat)
    return {
        "line_count_score": count,
        "line_length_score": length,
        "rhyme_pair_score": rhyme,
        "lexical_mattr": lex,
        "repetition_rate": repeat,
        "form_partial": partial,
    }


def generation_args(base_args, args):
    gen_args = copy.deepcopy(base_args)
    gen_args.temperature = args.temperature
    gen_args.top_p = args.top_p
    gen_args.top_k = args.top_k
    gen_args.decoding_strategy = "top_p"
    gen_args.decoding_strategies = args.decoding_strategies
    gen_args.num_beams = args.num_beams
    gen_args.num_candidates = args.num_candidates
    gen_args.dev_num_candidates = 1
    gen_args.model_score_weight = args.model_score_weight
    gen_args.mbr_weight = args.mbr_weight
    gen_args.repetition_penalty = args.repetition_penalty
    gen_args.no_repeat_ngram_size = args.no_repeat_ngram_size
    gen_args.target_lines = 14
    gen_args.max_generation_tokens = args.max_generation_tokens
    return gen_args


def form_aware_score(model, candidate: str, candidates: list[str], args) -> tuple[float, dict[str, float]]:
    features = form_features(candidate)
    model_score = model_sequence_score(model, candidate)
    mbr_score = mbr_centrality_score(candidate, candidates)
    score = (
        args.model_score_weight * model_score
        + args.mbr_weight * mbr_score
        + args.line_count_weight * features["line_count_score"]
        + args.line_length_weight * features["line_length_score"]
        + args.rhyme_weight * features["rhyme_pair_score"]
        + args.lexical_weight * features["lexical_mattr"]
        - args.repeat_weight * features["repetition_rate"]
    )
    features = dict(features)
    features["model_score"] = model_score
    features["mbr_score"] = mbr_score
    features["total_score"] = score
    return score, features


def select_best_form_aware_sonnet(model, prompt: str, base_args, args) -> tuple[str, float, dict[str, float]]:
    gen_args = generation_args(base_args, args)
    base_temperature = args.temperature
    temperature_offsets = [0.0, -0.05, 0.05, 0.10, -0.10, 0.15, -0.15, 0.20]
    strategies = [strategy.strip() for strategy in args.decoding_strategies.split(",") if strategy.strip()]
    candidates = []
    for idx in range(args.num_candidates):
        temperature = max(0.5, min(1.2, base_temperature + temperature_offsets[idx % len(temperature_offsets)]))
        if args.use_line_rhyme_generation and idx < args.line_rhyme_candidates:
            candidates.append(generate_line_rhyme_candidate(model, prompt, args, temperature))
        else:
            strategy = strategies[idx % len(strategies)] if strategies else args.decoding_strategy
            candidates.append(generate_candidate_sonnet(model, prompt, gen_args, temperature, strategy))

    best_sonnet = candidates[0]
    best_score = -float("inf")
    best_features = {}
    for candidate in candidates:
        score, features = form_aware_score(model, candidate, candidates, args)
        if score > best_score:
            best_sonnet = candidate
            best_score = score
            best_features = features
    return best_sonnet, best_score, best_features


@torch.no_grad()
def generate_file(model, base_args, args, prompt_path: Path, output_path: Path, seed: int | None = None) -> None:
    if seed is not None:
        seed_everything(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = SonnetsDataset(str(prompt_path))
    model.eval()
    with output_path.open("w", encoding="utf-8") as f:
        f.write("--Generated Sonnets--\n\n")
        for sonnet_id, prompt in dataset:
            best_sonnet, best_score, features = select_best_form_aware_sonnet(model, prompt, base_args, args)
            f.write(f"\n{sonnet_id}\n{best_sonnet}\n\n")
            print(
                f"[generate] {output_path.name} id={sonnet_id} score={best_score:.3f} "
                f"form={features.get('form_partial', 0.0):.3f} rhyme={features.get('rhyme_pair_score', 0.0):.3f} "
                f"rep={features.get('repetition_rate', 0.0):.3f}"
            )


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda") if args.use_gpu and torch.cuda.is_available() else torch.device("cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_model, base_args = load_sft_model(Path(args.sft_checkpoint), device)
    policy_model, _ = load_sft_model(Path(args.sft_checkpoint), device)
    apply_lora_to_policy(policy_model, args.lora_r, args.lora_alpha, args.lora_dropout)

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    tokenizer = policy_model.tokenizer
    pref_dataset = FormAwarePreferenceDataset(
        Path(args.train_path),
        tokenizer,
        max_pairs=args.max_pairs,
        seed=args.seed,
        include_short_form_rejects=args.include_short_form_rejects,
    )
    val_size = max(1, int(len(pref_dataset) * args.val_ratio))
    train_size = len(pref_dataset) - val_size
    train_dataset, val_dataset = random_split(
        pref_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pref_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pref_dataset.collate_fn)

    trainable = [param for param in policy_model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_dev_chrf = -float("inf")
    best_loss_path = out_dir / "best_loss_lora_dpo_form_rhyme.pt"
    best_chrf_path = out_dir / "best_chrf_lora_dpo_form_rhyme.pt"
    history = []

    reject_counts = Counter(pair[3] for pair in pref_dataset.pairs)
    print(f"[setup] total_pairs={len(pref_dataset)} train_pairs={train_size} val_pairs={val_size}")
    print(f"[setup] reject_counts={dict(reject_counts)}")
    print(f"[setup] trainable_params={sum(p.numel() for p in trainable):,}")

    for epoch in range(args.epochs):
        policy_model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"form-dpo-train-{epoch}"):
            winner_ids = batch["winner_ids"].to(device)
            winner_mask = batch["winner_mask"].to(device)
            loser_ids = batch["loser_ids"].to(device)
            loser_mask = batch["loser_mask"].to(device)
            prompt_lengths = batch["prompt_lengths"].to(device)

            with torch.no_grad():
                ref_win = continuation_logps(ref_model, winner_ids, winner_mask, prompt_lengths)
                ref_lose = continuation_logps(ref_model, loser_ids, loser_mask, prompt_lengths)

            optimizer.zero_grad()
            policy_win = continuation_logps(policy_model, winner_ids, winner_mask, prompt_lengths)
            policy_lose = continuation_logps(policy_model, loser_ids, loser_mask, prompt_lengths)
            loss = dpo_loss(policy_win, policy_lose, ref_win, ref_lose, args.beta)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            train_losses.append(loss.item())

        policy_model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"form-dpo-val-{epoch}"):
                winner_ids = batch["winner_ids"].to(device)
                winner_mask = batch["winner_mask"].to(device)
                loser_ids = batch["loser_ids"].to(device)
                loser_mask = batch["loser_mask"].to(device)
                prompt_lengths = batch["prompt_lengths"].to(device)
                ref_win = continuation_logps(ref_model, winner_ids, winner_mask, prompt_lengths)
                ref_lose = continuation_logps(ref_model, loser_ids, loser_mask, prompt_lengths)
                policy_win = continuation_logps(policy_model, winner_ids, winner_mask, prompt_lengths)
                policy_lose = continuation_logps(policy_model, loser_ids, loser_mask, prompt_lengths)
                val_losses.append(dpo_loss(policy_win, policy_lose, ref_win, ref_lose, args.beta).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        epoch_dev_path = out_dir / "predictions" / f"dev_epoch{epoch}.txt"
        generate_file(policy_model, base_args, args, Path(args.dev_prompt_path), epoch_dev_path, seed=args.seed + epoch)
        dev_chrf = chrf_score(epoch_dev_path, Path(args.dev_gold_path))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_loss_path, policy_model, optimizer, args, base_args, epoch, val_loss, dev_chrf)
        if dev_chrf > best_dev_chrf:
            best_dev_chrf = dev_chrf
            save_checkpoint(best_chrf_path, policy_model, optimizer, args, base_args, epoch, val_loss, dev_chrf)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "dev_chrf": dev_chrf,
            "best_val_loss": best_val_loss,
            "best_dev_chrf": best_dev_chrf,
        }
        history.append(row)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"dev_chrF={dev_chrf:.4f}"
        )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    best_chrf_policy, best_chrf_base_args = load_policy_from_dpo(best_chrf_path, Path(args.sft_checkpoint), device, args)
    best_chrf_dev = out_dir / "predictions" / "dev_best_chrf.txt"
    best_chrf_test = out_dir / "predictions" / "test_best_chrf.txt"
    best_chrf_epoch = max(history, key=lambda item: item["dev_chrf"])["epoch"]
    generate_file(
        best_chrf_policy,
        best_chrf_base_args,
        args,
        Path(args.dev_prompt_path),
        best_chrf_dev,
        seed=args.seed + best_chrf_epoch,
    )
    generate_file(
        best_chrf_policy,
        best_chrf_base_args,
        args,
        Path(args.test_prompt_path),
        best_chrf_test,
        seed=args.seed + 10_000 + best_chrf_epoch,
    )
    best_chrf_score = chrf_score(best_chrf_dev, Path(args.dev_gold_path))

    best_loss_policy, best_loss_base_args = load_policy_from_dpo(best_loss_path, Path(args.sft_checkpoint), device, args)
    best_loss_dev = out_dir / "predictions" / "dev_best_loss.txt"
    best_loss_test = out_dir / "predictions" / "test_best_loss.txt"
    best_loss_epoch = min(history, key=lambda item: item["val_loss"])["epoch"]
    generate_file(
        best_loss_policy,
        best_loss_base_args,
        args,
        Path(args.dev_prompt_path),
        best_loss_dev,
        seed=args.seed + best_loss_epoch,
    )
    generate_file(
        best_loss_policy,
        best_loss_base_args,
        args,
        Path(args.test_prompt_path),
        best_loss_test,
        seed=args.seed + 10_000 + best_loss_epoch,
    )
    best_loss_chrf = chrf_score(best_loss_dev, Path(args.dev_gold_path))

    summary = {
        "method": "MSK SFT checkpoint + form/rhyme/repetition-aware LoRA-DPO",
        "use_line_rhyme_generation": args.use_line_rhyme_generation,
        "line_rhyme_candidates": args.line_rhyme_candidates,
        "sft_checkpoint": str(args.sft_checkpoint),
        "train_path": str(args.train_path),
        "best_chrf_checkpoint": str(best_chrf_path),
        "best_loss_checkpoint": str(best_loss_path),
        "best_dev_chrf": best_chrf_score,
        "best_loss_dev_chrf": best_loss_chrf,
        "best_chrf_epoch": best_chrf_epoch,
        "best_loss_epoch": best_loss_epoch,
        "history": history,
        "dev_prediction": str(best_loss_dev),
        "test_prediction": str(best_loss_test),
        "best_chrf_dev_prediction": str(best_chrf_dev),
        "best_chrf_test_prediction": str(best_chrf_test),
        "best_loss_dev_prediction": str(best_loss_dev),
        "best_loss_test_prediction": str(best_loss_test),
        "reject_counts": dict(reject_counts),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# MSK Form-aware LoRA-DPO Summary",
        "",
        f"- SFT checkpoint: `{args.sft_checkpoint}`",
        f"- train data: `{args.train_path}`",
        f"- line-rhyme generation: `{args.use_line_rhyme_generation}`",
        f"- best chrF dev chrF: `{best_chrf_score:.4f}`",
        f"- best loss dev chrF: `{best_loss_chrf:.4f}`",
        f"- best chrF epoch: `{summary['best_chrf_epoch']}`",
        f"- best DPO val-loss epoch: `{summary['best_loss_epoch']}`",
        f"- reject counts: `{dict(reject_counts)}`",
        f"- best-loss dev prediction: `{best_loss_dev}`",
        f"- best-loss test prediction: `{best_loss_test}`",
        f"- best-chrF dev prediction: `{best_chrf_dev}`",
        f"- best-chrF test prediction: `{best_chrf_test}`",
        "",
        "| epoch | train loss | val loss | dev chrF |",
        "|---:|---:|---:|---:|",
    ]
    for row in history:
        lines.append(f"| {row['epoch']} | {row['train_loss']:.4f} | {row['val_loss']:.4f} | {row['dev_chrf']:.4f} |")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_checkpoint", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--dev_prompt_path", required=True)
    parser.add_argument("--dev_gold_path", required=True)
    parser.add_argument("--test_prompt_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--include_short_form_rejects", action="store_true")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--num_candidates", type=int, default=6)
    parser.add_argument("--decoding_strategy", default="top_p")
    parser.add_argument("--decoding_strategies", default="top_p,top_k,beam")
    parser.add_argument("--model_score_weight", type=float, default=1.0)
    parser.add_argument("--mbr_weight", type=float, default=4.0)
    parser.add_argument("--line_count_weight", type=float, default=3.0)
    parser.add_argument("--line_length_weight", type=float, default=2.0)
    parser.add_argument("--rhyme_weight", type=float, default=7.0)
    parser.add_argument("--lexical_weight", type=float, default=2.0)
    parser.add_argument("--repeat_weight", type=float, default=5.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.06)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--max_generation_tokens", type=int, default=130)
    parser.add_argument("--use_line_rhyme_generation", action="store_true")
    parser.add_argument("--line_rhyme_candidates", type=int, default=2)
    parser.add_argument("--rhyme_token_bias", type=float, default=45.0)
    parser.add_argument("--newline_bonus", type=float, default=70.0)
    parser.add_argument("--newline_suppress_bias", type=float, default=80.0)
    parser.add_argument("--rhyme_min_syllables", type=int, default=5)
    parser.add_argument("--rhyme_bias_start_syllables", type=int, default=8)
    parser.add_argument("--rhyme_target_syllables", type=int, default=11)
    parser.add_argument("--rhyme_min_tokens", type=int, default=3)
    parser.add_argument("--rhyme_line_max_tokens", type=int, default=30)
    parser.add_argument("--rhyme_word_limit", type=int, default=80)
    return parser.parse_args()


if __name__ == "__main__":
    train(get_args())
