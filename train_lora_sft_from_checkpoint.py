#!/usr/bin/env python3
"""Train LoRA adapters with the SFT objective from an existing checkpoint.

This is used for the six-way ablation where LoRA is applied after whichever
non-DPO checkpoint is stronger on dev chrF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import SonnetsDataset
from optimizer import AdamW
from run_msk_sft_lora_dpo import apply_lora_to_policy, chrf_score, generate_file, load_sft_model
from sonnet_generation import prompt_token_lengths, seed_everything, weighted_lm_loss


def save_checkpoint(path: Path, model, optimizer, args, base_args, epoch: int, train_loss: float, dev_chrf: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "args": vars(args),
            "base_args": base_args,
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_chrf": dev_chrf,
        },
        path,
    )


def load_lora_checkpoint(path: Path, init_checkpoint: Path, device: torch.device, args):
    model, base_args = load_sft_model(init_checkpoint, device)
    apply_lora_to_policy(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    saved = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    model = model.to(device)
    return model, base_args


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    train_data = SonnetsDataset(args.train_path)
    loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, collate_fn=train_data.collate_fn)

    model, base_args = load_sft_model(Path(args.init_checkpoint), device)
    apply_lora_to_policy(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    newline_token_id = model.tokenizer.encode("\n")[0]

    best_dev_chrf = -float("inf")
    best_train_loss = float("inf")
    best_path = out_dir / "best_chrf_lora_sft.pt"
    history: list[dict[str, float | int]] = []

    print(f"[setup] init_checkpoint={args.init_checkpoint}")
    print(f"[setup] trainable_params={sum(p.numel() for p in trainable):,}")

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in tqdm(loader, desc=f"lora-sft-train-{epoch}"):
            token_ids = batch["token_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_texts = [train_data.sonnets[i] for i in batch["sent_ids"]]
            prompt_lengths = prompt_token_lengths(model.tokenizer, batch_texts)

            optimizer.zero_grad()
            logits = model(token_ids, attention_mask)
            loss = weighted_lm_loss(logits, token_ids, attention_mask, prompt_lengths, newline_token_id, args)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        epoch_dev = pred_dir / f"dev_epoch{epoch}.txt"
        generate_file(model, base_args, args, Path(args.dev_prompt_path), epoch_dev, seed=args.seed + epoch)
        dev_chrf = chrf_score(epoch_dev, Path(args.dev_gold_path))
        if dev_chrf > best_dev_chrf:
            best_dev_chrf = dev_chrf
            best_train_loss = train_loss
            save_checkpoint(best_path, model, optimizer, args, base_args, epoch, train_loss, dev_chrf)

        row = {"epoch": epoch, "train_loss": train_loss, "dev_chrf": dev_chrf}
        history.append(row)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} dev_chrF={dev_chrf:.4f}")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    best_model, best_base_args = load_lora_checkpoint(best_path, Path(args.init_checkpoint), device, args)
    best_epoch = max(history, key=lambda item: item["dev_chrf"])["epoch"]
    dev_best = pred_dir / "dev_best_chrf.txt"
    test_best = pred_dir / "test_best_chrf.txt"
    generate_file(best_model, best_base_args, args, Path(args.dev_prompt_path), dev_best, seed=args.seed + int(best_epoch))
    generate_file(best_model, best_base_args, args, Path(args.test_prompt_path), test_best, seed=args.seed + 10000 + int(best_epoch))
    final_chrf = chrf_score(dev_best, Path(args.dev_gold_path))

    summary = {
        "method": "LoRA-SFT from selected checkpoint",
        "init_checkpoint": args.init_checkpoint,
        "train_path": args.train_path,
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "best_dev_chrf": final_chrf,
        "dev_prediction": str(dev_best),
        "test_prediction": str(test_best),
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# LoRA-SFT From Selected Checkpoint",
        "",
        f"- init checkpoint: `{args.init_checkpoint}`",
        f"- train data: `{args.train_path}`",
        f"- best epoch: `{best_epoch}`",
        f"- best dev chrF: `{final_chrf:.4f}`",
        f"- dev prediction: `{dev_best}`",
        f"- test prediction: `{test_best}`",
        "",
        "| epoch | train loss | dev chrF |",
        "|---:|---:|---:|",
    ]
    for row in history:
        lines.append(f"| {row['epoch']} | {row['train_loss']:.4f} | {row['dev_chrf']:.4f} |")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_checkpoint", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--dev_prompt_path", required=True)
    parser.add_argument("--dev_gold_path", required=True)
    parser.add_argument("--test_prompt_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--prompt_loss_weight", type=float, default=0.35)
    parser.add_argument("--line_break_loss_weight", type=float, default=1.2)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--model_score_weight", type=float, default=2.0)
    parser.add_argument("--mbr_weight", type=float, default=4.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.03)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--max_generation_tokens", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    train(get_args())
