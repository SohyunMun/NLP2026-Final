#!/usr/bin/env python3
"""Generate sonnets from an MSK sonnet_generation checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ROOT = PROJECT_ROOT.parent
for search_path in (SCRIPT_DIR, ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--output_path", required=True)
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


def main() -> None:
    args = get_args()
    import torch

    from run_msk_sft_lora_dpo import generate_file, load_sft_model
    from sonnet_generation_enhanced import seed_everything

    seed_everything(args.seed)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    model, base_args = load_sft_model(Path(args.checkpoint), device)
    generate_file(
        model=model,
        base_args=base_args,
        args=args,
        prompt_path=Path(args.prompt_path),
        output_path=Path(args.output_path),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
