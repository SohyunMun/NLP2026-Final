#!/usr/bin/env python3
"""Run SFT-initialized LoRA-DPO for sonnet generation.

This script fixes the earlier DPO setup by:
1. Loading a strong SFT checkpoint into both policy and reference models.
2. Freezing the reference model.
3. Adding LoRA adapters only to the policy model and training only those adapters.
4. Saving both the lowest DPO validation-loss checkpoint and the best dev chrF checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sacrebleu.metrics import CHRF
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ROOT = PROJECT_ROOT.parent
for search_path in (SCRIPT_DIR, ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from datasets import SonnetsDataset
from optimizer import AdamW
from sonnet_generation import (
    SonnetGPT,
    add_arguments,
    nonempty_lines,
    select_best_sonnet,
    seed_everything,
)


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.original = linear
        self.scaling = alpha / r
        self.lora_A = nn.Linear(linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, linear.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


def apply_lora_to_policy(model: SonnetGPT, r: int, alpha: int, dropout: float) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for layer in model.gpt.gpt_layers:
        sa = layer.self_attention
        sa.query = LoRALinear(sa.query, r=r, alpha=alpha, dropout=dropout).to(sa.query.weight.device)
        sa.key = LoRALinear(sa.key, r=r, alpha=alpha, dropout=dropout).to(sa.key.weight.device)
        sa.value = LoRALinear(sa.value, r=r, alpha=alpha, dropout=dropout).to(sa.value.weight.device)
        layer.attention_dense = LoRALinear(
            layer.attention_dense, r=r, alpha=alpha, dropout=dropout
        ).to(layer.attention_dense.weight.device)

    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name


def load_sft_model(checkpoint_path: Path, device: torch.device) -> tuple[SonnetGPT, argparse.Namespace]:
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_args = copy.deepcopy(saved["args"])
    base_args = add_arguments(base_args)
    model = SonnetGPT(base_args)
    model.load_state_dict(saved["model"])
    model = model.to(device)
    return model, base_args


def first_lines(text: str, count: int = 3) -> str:
    return "\n".join(nonempty_lines(text)[:count])


def continuation(text: str, start: int = 3) -> list[str]:
    return nonempty_lines(text)[start:]


def corrupt_by_repetition(prompt: str, winner: str) -> str:
    lines = nonempty_lines(winner)
    prompt_lines = nonempty_lines(prompt)
    repeated = []
    source = continuation(winner) or lines
    while len(prompt_lines) + len(repeated) < 14:
        repeated.append(source[len(repeated) % len(source)])
    return "\n".join(prompt_lines + repeated[: 14 - len(prompt_lines)])


def corrupt_by_mismatch(prompt: str, donor: str) -> str:
    prompt_lines = nonempty_lines(prompt)
    donor_lines = continuation(donor)
    if not donor_lines:
        donor_lines = nonempty_lines(donor)
    return "\n".join((prompt_lines + donor_lines)[:14])


class PreferenceDataset(Dataset):
    def __init__(self, sonnet_path: Path, tokenizer, max_pairs: int | None = None, seed: int = 11711):
        self.tokenizer = tokenizer
        self.max_pairs = max_pairs
        self.rng = random.Random(seed)
        self.sonnets = [sonnet for _, sonnet in SonnetsDataset(str(sonnet_path))]
        self.pairs = self._build_pairs()
        if max_pairs is not None:
            self.rng.shuffle(self.pairs)
            self.pairs = self.pairs[:max_pairs]

    def _build_pairs(self) -> list[tuple[str, str, str]]:
        pairs = []
        n = len(self.sonnets)
        for idx, winner in enumerate(self.sonnets):
            prompt = first_lines(winner)
            donor = self.sonnets[(idx * 37 + 11) % n]
            if donor == winner:
                donor = self.sonnets[(idx + 1) % n]
            pairs.append((prompt, winner, corrupt_by_mismatch(prompt, donor)))
            pairs.append((prompt, winner, corrupt_by_repetition(prompt, winner)))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

    def collate_fn(self, examples):
        prompts, winners, losers = zip(*examples)
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
        }


def continuation_logps(model, input_ids, attention_mask, prompt_lengths):
    logits = model(input_ids, attention_mask)[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()
    label_mask = attention_mask[:, 1:].bool()

    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    response_mask = label_mask.clone()
    positions = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0)
    prompt_cutoffs = (prompt_lengths.to(labels.device) - 1).clamp_min(0).unsqueeze(1)
    response_mask &= positions >= prompt_cutoffs

    token_logps = token_logps.masked_fill(~response_mask, 0.0)
    return token_logps.sum(dim=1)


def dpo_loss(policy_win, policy_lose, ref_win, ref_lose, beta: float):
    policy_logratio = policy_win - policy_lose
    ref_logratio = ref_win - ref_lose
    logits = policy_logratio - ref_logratio
    return -F.logsigmoid(beta * logits).mean()


def split_numbered_blocks(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\d+\s*\n", text)
    return [block.strip() for block in blocks[1:] if block.strip()]


def chrf_score(prediction_path: Path, gold_path: Path) -> float:
    preds = split_numbered_blocks(prediction_path.read_text(encoding="utf-8", errors="ignore"))
    refs = split_numbered_blocks(gold_path.read_text(encoding="utf-8", errors="ignore"))
    max_len = min(len(preds), len(refs))
    if max_len == 0:
        return 0.0
    return float(CHRF().corpus_score(preds[:max_len], [refs[:max_len]]).score)


def generation_args(base_args, args):
    gen_args = copy.deepcopy(base_args)
    gen_args.temperature = args.temperature
    gen_args.top_p = args.top_p
    gen_args.top_k = args.top_k
    gen_args.decoding_strategy = "top_p"
    gen_args.decoding_strategies = "top_p,top_k,beam"
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


@torch.no_grad()
def generate_file(model, base_args, args, prompt_path: Path, output_path: Path, seed: int | None = None) -> None:
    if seed is not None:
        seed_everything(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gen_args = generation_args(base_args, args)
    dataset = SonnetsDataset(str(prompt_path))
    model.eval()
    with output_path.open("w", encoding="utf-8") as f:
        f.write("--Generated Sonnets--\n\n")
        for sonnet_id, prompt in dataset:
            best_sonnet, best_score = select_best_sonnet(model, prompt, gen_args)
            f.write(f"\n{sonnet_id}\n{best_sonnet}\n\n")
            print(f"[generate] {output_path.name} id={sonnet_id} score={best_score:.3f}")


def save_checkpoint(path: Path, policy_model, optimizer, args, base_args, epoch, val_loss, dev_chrf):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": policy_model.state_dict(),
            "optim": optimizer.state_dict(),
            "args": vars(args),
            "base_args": base_args,
            "epoch": epoch,
            "val_loss": val_loss,
            "dev_chrf": dev_chrf,
        },
        path,
    )


def load_policy_from_dpo(path: Path, base_checkpoint: Path, device: torch.device, args) -> tuple[SonnetGPT, argparse.Namespace]:
    policy, base_args = load_sft_model(base_checkpoint, device)
    apply_lora_to_policy(policy, args.lora_r, args.lora_alpha, args.lora_dropout)
    saved = torch.load(path, map_location=device, weights_only=False)
    policy.load_state_dict(saved["model"])
    policy = policy.to(device)
    return policy, base_args


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
    pref_dataset = PreferenceDataset(Path(args.train_path), tokenizer, max_pairs=args.max_pairs, seed=args.seed)
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
    best_loss_path = out_dir / "best_loss_lora_dpo.pt"
    best_chrf_path = out_dir / "best_chrf_lora_dpo.pt"
    history = []

    print(f"[setup] train_pairs={train_size} val_pairs={val_size}")
    print(f"[setup] trainable_params={sum(p.numel() for p in trainable):,}")

    for epoch in range(args.epochs):
        policy_model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"dpo-train-{epoch}"):
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
            for batch in tqdm(val_loader, desc=f"dpo-val-{epoch}"):
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
    generate_file(best_chrf_policy, best_chrf_base_args, args, Path(args.dev_prompt_path), best_chrf_dev, seed=args.seed + best_chrf_epoch)
    generate_file(best_chrf_policy, best_chrf_base_args, args, Path(args.test_prompt_path), best_chrf_test, seed=args.seed + 10_000 + best_chrf_epoch)
    best_chrf_score = chrf_score(best_chrf_dev, Path(args.dev_gold_path))

    best_loss_policy, best_loss_base_args = load_policy_from_dpo(best_loss_path, Path(args.sft_checkpoint), device, args)
    best_loss_dev = out_dir / "predictions" / "dev_best_loss.txt"
    best_loss_test = out_dir / "predictions" / "test_best_loss.txt"
    best_loss_epoch = min(history, key=lambda item: item["val_loss"])["epoch"]
    generate_file(best_loss_policy, best_loss_base_args, args, Path(args.dev_prompt_path), best_loss_dev, seed=args.seed + best_loss_epoch)
    generate_file(best_loss_policy, best_loss_base_args, args, Path(args.test_prompt_path), best_loss_test, seed=args.seed + 10_000 + best_loss_epoch)
    best_loss_score = chrf_score(best_loss_dev, Path(args.dev_gold_path))

    summary = {
        "method": "MSK SFT checkpoint + LoRA-DPO",
        "sft_checkpoint": str(args.sft_checkpoint),
        "train_path": str(args.train_path),
        "best_chrf_checkpoint": str(best_chrf_path),
        "best_loss_checkpoint": str(best_loss_path),
        "best_dev_chrf": best_chrf_score,
        "best_loss_dev_chrf": best_loss_score,
        "best_chrf_epoch": best_chrf_epoch,
        "best_loss_epoch": best_loss_epoch,
        "history": history,
        "dev_prediction": str(best_loss_dev),
        "test_prediction": str(best_loss_test),
        "best_chrf_dev_prediction": str(best_chrf_dev),
        "best_chrf_test_prediction": str(best_chrf_test),
        "best_loss_dev_prediction": str(best_loss_dev),
        "best_loss_test_prediction": str(best_loss_test),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# MSK SFT LoRA-DPO Summary",
        "",
        f"- SFT checkpoint: `{args.sft_checkpoint}`",
        f"- train data: `{args.train_path}`",
        f"- best-chrF dev chrF: `{best_chrf_score:.4f}`",
        f"- best-loss dev chrF: `{best_loss_score:.4f}`",
        f"- best chrF epoch: `{summary['best_chrf_epoch']}`",
        f"- best DPO val-loss epoch: `{summary['best_loss_epoch']}`",
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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--num_candidates", type=int, default=2)
    parser.add_argument("--model_score_weight", type=float, default=2.0)
    parser.add_argument("--mbr_weight", type=float, default=4.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.03)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--max_generation_tokens", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    train(get_args())
