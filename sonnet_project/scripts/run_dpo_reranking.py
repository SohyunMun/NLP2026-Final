#!/usr/bin/env python3
"""Generate multiple DPO candidates and rerank them with reference-free metrics.

This script does not train a new model. It loads the existing LoRA-DPO policy,
generates several candidates per prompt, scores each candidate without looking
at gold references, and writes the best candidate as the final prediction.
Gold references are only used later by the evaluation script.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ROOT = PROJECT_ROOT.parent
for search_path in (SCRIPT_DIR, ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from datasets import SonnetsDataset
from evaluate_sonnet_metrics import (
    distinct_n,
    imagery_literary_device_score,
    line_count_score,
    line_length_score,
    mattr,
    nonempty_lines,
    prompt_theme_overlap,
    repetition_rate,
    rhyme_scores,
    text_from_lines,
)
from evaluate_sonnet_poemetric import (
    form_accuracy,
    lexical_diversity,
    overall_quality_proxy,
    poemetric_proxy,
)
from run_msk_sft_lora_dpo import apply_lora_to_policy, load_sft_model
from sonnet_generation_enhanced import (
    generate_candidate_sonnet,
    mbr_centrality_score,
    model_sequence_score,
    seed_everything,
)


DEFAULT_OUT_DIR = "sonnet_project/experiments/dpo_reranking"


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def generation_args(base_args: argparse.Namespace, args: argparse.Namespace) -> argparse.Namespace:
    gen_args = copy.deepcopy(base_args)
    gen_args.temperature = args.temperature
    gen_args.top_p = args.top_p
    gen_args.top_k = args.top_k
    gen_args.decoding_strategy = "top_p"
    gen_args.decoding_strategies = args.decoding_strategies
    gen_args.num_beams = args.num_beams
    gen_args.num_candidates = 1
    gen_args.dev_num_candidates = 1
    gen_args.model_score_weight = 0.0
    gen_args.mbr_weight = 0.0
    gen_args.repetition_penalty = args.repetition_penalty
    gen_args.no_repeat_ngram_size = args.no_repeat_ngram_size
    gen_args.target_lines = args.target_lines
    gen_args.max_generation_tokens = args.max_generation_tokens
    return gen_args


def load_dpo_policy(args: argparse.Namespace, device: torch.device):
    model, base_args = load_sft_model(project_path(args.sft_checkpoint), device)
    apply_lora_to_policy(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    saved = torch.load(project_path(args.dpo_checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    model = model.to(device)
    model.eval()
    return model, base_args


def candidate_metrics(
    *,
    candidate: str,
    prompt: str,
    candidates: list[str],
    model,
    args: argparse.Namespace,
) -> dict[str, float]:
    full_lines = nonempty_lines(candidate)
    prompt_lines = nonempty_lines(prompt)[: args.prompt_lines]
    continuation_lines = full_lines[len(prompt_lines):] if full_lines[: len(prompt_lines)] == prompt_lines else full_lines

    full_text = text_from_lines(full_lines)
    continuation_text = text_from_lines(continuation_lines)
    prompt_text = text_from_lines(prompt_lines)
    score_text = continuation_text if args.score_part == "continuation" else full_text

    pair_score, final_couplet_score, _endings = rhyme_scores(full_lines)
    row_mattr = mattr(score_text, args.mattr_window)
    row_distinct_2 = distinct_n([score_text], 2)
    row_repetition = repetition_rate(score_text)
    row_imagery = imagery_literary_device_score(continuation_text, continuation_lines)
    row_theme = prompt_theme_overlap(prompt_text, continuation_text)

    row = {
        "exact_14_lines": 1.0 if len(full_lines) == args.target_lines else 0.0,
        "line_count_score": line_count_score(full_lines, args.target_lines),
        "line_length_score": line_length_score(full_lines),
        "shakespearean_rhyme_pair_score": pair_score,
        "final_couplet_rhyme": final_couplet_score,
    }
    form_score = form_accuracy(row)
    lexical_score = lexical_diversity(row_mattr, row_distinct_2)
    overall_score = overall_quality_proxy(
        form_score=form_score,
        lexical_score=lexical_score,
        repetition_score=row_repetition,
        imagery_score=row_imagery,
    )
    poemetric_score = poemetric_proxy(
        form_score=form_score,
        lexical_score=lexical_score,
        overall_score=overall_score,
        theme_score=row_theme,
    )

    sonnet_pass = (
        row["exact_14_lines"] >= 1.0
        and row["line_length_score"] >= args.sonnet_line_length_threshold
        and pair_score >= args.sonnet_rhyme_threshold
        and final_couplet_score >= args.sonnet_couplet_threshold
        and form_score >= args.sonnet_form_threshold
    )
    model_score = model_sequence_score(model, candidate) if args.model_weight else 0.0
    mbr_score = mbr_centrality_score(candidate, candidates) if args.mbr_weight else 0.0

    rerank_score = (
        args.poemetric_weight * poemetric_score
        + args.form_weight * form_score
        + args.rhyme_weight * pair_score
        + args.couplet_weight * final_couplet_score
        + args.theme_weight * row_theme
        + args.lexical_weight * lexical_score
        + args.non_repetition_weight * (1.0 - row_repetition)
        + args.mbr_weight * mbr_score
        + args.model_weight * model_score
        + (args.sonnet_pass_bonus if sonnet_pass else 0.0)
    )

    return {
        "rerank_score": float(rerank_score),
        "POEMetric_proxy": float(poemetric_score),
        "sonnet_form_accuracy": float(form_score),
        "exact_14_lines": float(row["exact_14_lines"]),
        "line_count_score": float(row["line_count_score"]),
        "line_length_score": float(row["line_length_score"]),
        "shakespearean_rhyme_pair_score": float(pair_score),
        "final_couplet_rhyme": float(final_couplet_score),
        "MATTR": float(row_mattr or 0.0),
        "distinct_2": float(row_distinct_2 or 0.0),
        "lexical_diversity": float(lexical_score),
        "repetition_rate": float(row_repetition),
        "non_repetition": float(1.0 - row_repetition),
        "imagery_literary_device_score": float(row_imagery or 0.0),
        "prompt_continuation_theme_overlap": float(row_theme or 0.0),
        "poemetric_overall_quality_proxy": float(overall_score),
        "sonnet_or_not_bot_pass": 1.0 if sonnet_pass else 0.0,
        "mbr_score": float(mbr_score),
        "model_score": float(model_score),
    }


def generate_candidates_for_prompt(
    model,
    base_args: argparse.Namespace,
    prompt: str,
    args: argparse.Namespace,
) -> list[str]:
    gen_args = generation_args(base_args, args)
    strategies = [item.strip() for item in args.decoding_strategies.split(",") if item.strip()]
    temperature_offsets = [0.0, -0.05, 0.05, 0.10, -0.10, 0.15, -0.15, 0.20, -0.20, 0.25, -0.25, 0.30]
    candidates: list[str] = []
    seen: set[str] = set()

    for idx in range(args.num_candidates):
        temperature = max(0.45, min(1.25, args.temperature + temperature_offsets[idx % len(temperature_offsets)]))
        strategy = strategies[idx % len(strategies)] if strategies else args.decoding_strategy
        candidate = generate_candidate_sonnet(model, prompt, gen_args, temperature, strategy)
        if candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)

    if not candidates:
        candidates.append("\n".join(nonempty_lines(prompt)[: args.prompt_lines]))
    return candidates


def write_numbered_predictions(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("--Generated Sonnets--\n\n")
        for sonnet_id, body in rows:
            handle.write(f"\n{sonnet_id}\n{body.strip()}\n\n")


def write_candidate_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def generate_split(
    *,
    model,
    base_args: argparse.Namespace,
    prompt_path: Path,
    output_path: Path,
    candidates_jsonl_path: Path,
    candidate_metrics_path: Path,
    split_name: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    dataset = SonnetsDataset(str(prompt_path))
    selected_rows: list[tuple[str, str]] = []
    metric_rows: list[dict[str, object]] = []
    selected_metric_rows: list[dict[str, object]] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with candidates_jsonl_path.open("w", encoding="utf-8") as jsonl:
        for row_idx, (sonnet_id, prompt) in enumerate(dataset):
            seed_everything(args.seed + args.seed_stride * row_idx + (0 if split_name == "dev" else 10_000))
            candidates = generate_candidates_for_prompt(model, base_args, prompt, args)
            scored = []
            for candidate_idx, candidate in enumerate(candidates):
                metrics = candidate_metrics(
                    candidate=candidate,
                    prompt=prompt,
                    candidates=candidates,
                    model=model,
                    args=args,
                )
                metric_row = {
                    "split": split_name,
                    "row_index": row_idx,
                    "sonnet_id": sonnet_id,
                    "candidate_index": candidate_idx,
                    "selected": 0,
                    **metrics,
                    "candidate_text": candidate,
                }
                scored.append(metric_row)

            best = max(scored, key=lambda item: float(item["rerank_score"]))
            best["selected"] = 1
            selected_rows.append((sonnet_id, str(best["candidate_text"])))
            selected_metric_rows.append(best)
            metric_rows.extend(scored)

            jsonl.write(
                json.dumps(
                    {
                        "split": split_name,
                        "row_index": row_idx,
                        "sonnet_id": sonnet_id,
                        "prompt": prompt,
                        "selected_candidate_index": best["candidate_index"],
                        "selected_rerank_score": best["rerank_score"],
                        "candidates": [
                            {
                                "candidate_index": item["candidate_index"],
                                "rerank_score": item["rerank_score"],
                                "POEMetric_proxy": item["POEMetric_proxy"],
                                "sonnet_form_accuracy": item["sonnet_form_accuracy"],
                                "repetition_rate": item["repetition_rate"],
                                "prompt_continuation_theme_overlap": item["prompt_continuation_theme_overlap"],
                                "text": item["candidate_text"],
                            }
                            for item in scored
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(
                f"[{split_name}] id={sonnet_id} selected={best['candidate_index']} "
                f"score={float(best['rerank_score']):.4f} poemetric={float(best['POEMetric_proxy']):.4f} "
                f"form={float(best['sonnet_form_accuracy']):.4f}"
            )

    write_numbered_predictions(output_path, selected_rows)
    write_candidate_csv(candidate_metrics_path, metric_rows)
    write_candidate_csv(candidate_metrics_path.with_name(candidate_metrics_path.stem + "_selected.csv"), selected_metric_rows)

    return {
        "split": split_name,
        "prompt_path": str(prompt_path),
        "prediction_path": str(output_path),
        "candidates_jsonl": str(candidates_jsonl_path),
        "candidate_metrics_csv": str(candidate_metrics_path),
        "num_prompts": len(selected_rows),
        "num_candidates_requested": args.num_candidates,
        "avg_unique_candidates": sum(1 for row in metric_rows) / max(1, len(selected_rows)),
    }


def write_summary(path: Path, args: argparse.Namespace, split_summaries: list[dict[str, object]]) -> None:
    lines = [
        "# DPO + Reranking Experiment",
        "",
        "This experiment loads the best DPO policy and applies reference-free reranking.",
        "Gold references are not used during candidate selection.",
        "",
        "## Model",
        "",
        f"- SFT base checkpoint: `{args.sft_checkpoint}`",
        f"- DPO checkpoint: `{args.dpo_checkpoint}`",
        "",
        "## Reranking",
        "",
        f"- candidates per prompt: `{args.num_candidates}`",
        f"- decoding strategies: `{args.decoding_strategies}`",
        f"- temperature: `{args.temperature}`",
        f"- top_p: `{args.top_p}`",
        f"- top_k: `{args.top_k}`",
        f"- repetition penalty: `{args.repetition_penalty}`",
        f"- no repeat ngram size: `{args.no_repeat_ngram_size}`",
        "",
        "Rerank score:",
        "",
        "```text",
        "poemetric_weight * POEMetric",
        "+ form_weight * form_accuracy",
        "+ rhyme_weight * Shakespearean rhyme pair score",
        "+ couplet_weight * final couplet rhyme",
        "+ theme_weight * prompt-continuation theme overlap",
        "+ lexical_weight * lexical diversity",
        "+ non_repetition_weight * non-repetition",
        "+ mbr_weight * candidate centrality",
        "+ model_weight * model log-likelihood proxy",
        "+ sonnet_pass_bonus",
        "```",
        "",
        "Weights:",
        "",
        f"- poemetric_weight: `{args.poemetric_weight}`",
        f"- form_weight: `{args.form_weight}`",
        f"- rhyme_weight: `{args.rhyme_weight}`",
        f"- couplet_weight: `{args.couplet_weight}`",
        f"- theme_weight: `{args.theme_weight}`",
        f"- lexical_weight: `{args.lexical_weight}`",
        f"- non_repetition_weight: `{args.non_repetition_weight}`",
        f"- mbr_weight: `{args.mbr_weight}`",
        f"- model_weight: `{args.model_weight}`",
        f"- sonnet_pass_bonus: `{args.sonnet_pass_bonus}`",
        "",
        "## Outputs",
        "",
        "| split | prompts | prediction | candidates | metrics |",
        "|---|---:|---|---|---|",
    ]
    for item in split_summaries:
        lines.append(
            f"| {item['split']} | {item['num_prompts']} | `{item['prediction_path']}` | "
            f"`{item['candidates_jsonl']}` | `{item['candidate_metrics_csv']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DPO candidate generation followed by reference-free reranking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sft_checkpoint",
        default="sonnet_project/experiments/sixway_ablation/dapt_sft_intermediate/best_10-1e-05-sonnet.pt",
    )
    parser.add_argument(
        "--dpo_checkpoint",
        default="sonnet_project/experiments/sixway_ablation/dapt_sft_lora_dpo_best_chrf/best_chrf_lora_dpo_form_rhyme.pt",
    )
    parser.add_argument("--dev_prompt_path", default="sonnet_project/data/strict_497/dev_prompts_12.txt")
    parser.add_argument("--test_prompt_path", default="sonnet_project/data/strict_497/test_prompts_12.txt")
    parser.add_argument("--output_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--seed_stride", type=int, default=100)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_candidates", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.90)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--top_k", type=int, default=60)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--decoding_strategy", default="top_p")
    parser.add_argument("--decoding_strategies", default="top_p,top_k,beam")
    parser.add_argument("--repetition_penalty", type=float, default=1.08)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--max_generation_tokens", type=int, default=135)
    parser.add_argument("--target_lines", type=int, default=14)
    parser.add_argument("--prompt_lines", type=int, default=3)
    parser.add_argument("--score_part", choices=["full", "continuation"], default="full")
    parser.add_argument("--mattr_window", type=int, default=50)
    parser.add_argument("--sonnet_form_threshold", type=float, default=0.70)
    parser.add_argument("--sonnet_line_length_threshold", type=float, default=0.50)
    parser.add_argument("--sonnet_rhyme_threshold", type=float, default=0.25)
    parser.add_argument("--sonnet_couplet_threshold", type=float, default=0.25)
    parser.add_argument("--poemetric_weight", type=float, default=0.55)
    parser.add_argument("--form_weight", type=float, default=0.15)
    parser.add_argument("--rhyme_weight", type=float, default=0.20)
    parser.add_argument("--couplet_weight", type=float, default=0.10)
    parser.add_argument("--theme_weight", type=float, default=0.10)
    parser.add_argument("--lexical_weight", type=float, default=0.05)
    parser.add_argument("--non_repetition_weight", type=float, default=0.05)
    parser.add_argument("--mbr_weight", type=float, default=0.05)
    parser.add_argument("--model_weight", type=float, default=0.0)
    parser.add_argument("--sonnet_pass_bonus", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, base_args = load_dpo_policy(args, device)
    split_summaries = []
    split_summaries.append(
        generate_split(
            model=model,
            base_args=base_args,
            prompt_path=project_path(args.dev_prompt_path),
            output_path=out_dir / "predictions" / "dev_reranked.txt",
            candidates_jsonl_path=out_dir / "candidates" / "dev_candidates.jsonl",
            candidate_metrics_path=out_dir / "candidate_metrics" / "dev_candidates.csv",
            split_name="dev",
            args=args,
        )
    )
    split_summaries.append(
        generate_split(
            model=model,
            base_args=base_args,
            prompt_path=project_path(args.test_prompt_path),
            output_path=out_dir / "predictions" / "test_reranked.txt",
            candidates_jsonl_path=out_dir / "candidates" / "test_candidates.jsonl",
            candidate_metrics_path=out_dir / "candidate_metrics" / "test_candidates.csv",
            split_name="test",
            args=args,
        )
    )

    summary = {
        "method": "DPO + reference-free reranking",
        "device": str(device),
        "args": vars(args),
        "splits": split_summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary(out_dir / "SUMMARY.md", args, split_summaries)
    print(f"Wrote reranked predictions and summary to {out_dir}")


if __name__ == "__main__":
    main()
