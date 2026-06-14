#!/usr/bin/env python3
"""Train a chrF predictor reranker on train candidates and apply it to dev/test.

This is the fair, deployable variant of chrF-based reranking:

1. Generate candidates from train prompts.
2. Use train gold sonnets to compute each candidate's chrF label.
3. Train a small ridge-regression predictor from reference-free candidate
   features to chrF.
4. Apply the predictor to dev/test candidates without using dev/test gold.

The script never uses test gold, and dev gold is only used later by the
separate evaluation script.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ROOT = PROJECT_ROOT.parent
for search_path in (SCRIPT_DIR, ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from evaluate_sonnet_metrics import load_numbered_blocks, nonempty_lines
from evaluate_sonnet_poemetric import corpus_chrf_default, sentence_chrf_default
from run_dpo_reranking import (
    candidate_metrics,
    generate_candidates_for_prompt,
    load_dpo_policy,
    write_candidate_csv,
    write_numbered_predictions,
)
from sonnet_generation_enhanced import seed_everything


DEFAULT_OUT_DIR = "sonnet_project/experiments/train_chrf_reranking"

FEATURE_COLUMNS = [
    "rerank_score",
    "POEMetric_proxy",
    "sonnet_form_accuracy",
    "exact_14_lines",
    "line_count_score",
    "line_length_score",
    "shakespearean_rhyme_pair_score",
    "final_couplet_rhyme",
    "MATTR",
    "distinct_2",
    "lexical_diversity",
    "repetition_rate",
    "non_repetition",
    "imagery_literary_device_score",
    "prompt_continuation_theme_overlap",
    "poemetric_overall_quality_proxy",
    "sonnet_or_not_bot_pass",
    "mbr_score",
]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def first_lines(text: str, count: int = 3) -> str:
    return "\n".join(nonempty_lines(text)[:count])


def read_sonnets(path: Path, max_examples: int = 0) -> list[tuple[str, str]]:
    blocks = load_numbered_blocks(path)
    if max_examples and max_examples > 0:
        blocks = blocks[:max_examples]
    return blocks


def read_candidate_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            clean: dict[str, object] = dict(row)
            clean["row_index"] = int(row["row_index"])
            clean["candidate_index"] = int(row["candidate_index"])
            for column in FEATURE_COLUMNS + ["candidate_chrf", "predicted_chrf"]:
                if column in clean:
                    clean[column] = to_float(clean[column])
            rows.append(clean)
        return rows


def read_jsonl_candidates(path: Path) -> dict[int, list[dict[str, object]]]:
    if not path.exists():
        return {}
    by_row: dict[int, list[dict[str, object]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            rows = item.get("rows", [])
            if rows:
                by_row[int(item["row_index"])] = rows
    return by_row


def flatten_by_row(by_row: dict[int, list[dict[str, object]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_index in sorted(by_row):
        rows.extend(by_row[row_index])
    return rows


@torch.no_grad()
def train_candidate_suffix(args: argparse.Namespace) -> str:
    if args.train_num_shards > 1:
        return f"_shard{args.train_shard_index}of{args.train_num_shards}"
    return ""


def generate_train_candidates(model, base_args, args: argparse.Namespace) -> list[dict[str, object]]:
    out_dir = project_path(args.output_dir)
    candidate_dir = out_dir / "candidate_metrics"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    suffix = train_candidate_suffix(args)
    jsonl_path = candidate_dir / f"train_candidates{suffix}.jsonl"
    csv_path = candidate_dir / f"train_candidates{suffix}.csv"

    if csv_path.exists() and not args.force_train_candidates:
        print(f"[reuse] train candidates: {relative(csv_path)}")
        return read_candidate_csv(csv_path)

    all_sonnets = read_sonnets(project_path(args.train_path), args.max_train_examples)
    sonnets = [
        (row_index, sonnet_id, gold_text)
        for row_index, (sonnet_id, gold_text) in enumerate(all_sonnets)
        if row_index % args.train_num_shards == args.train_shard_index
    ]
    existing = read_jsonl_candidates(jsonl_path) if not args.force_train_candidates else {}
    if args.force_train_candidates and jsonl_path.exists():
        jsonl_path.unlink()

    with jsonl_path.open("a", encoding="utf-8") as jsonl:
        for done_count, (row_index, sonnet_id, gold_text) in enumerate(sonnets, start=1):
            if row_index in existing:
                continue
            seed_everything(args.seed + args.seed_stride * row_index)
            prompt = first_lines(gold_text, args.prompt_lines)
            candidates = generate_candidates_for_prompt(model, base_args, prompt, args)
            rows = []
            for candidate_index, candidate in enumerate(candidates):
                metrics = candidate_metrics(
                    candidate=candidate,
                    prompt=prompt,
                    candidates=candidates,
                    model=model,
                    args=args,
                )
                row = {
                    "split": "train",
                    "row_index": row_index,
                    "sonnet_id": sonnet_id,
                    "candidate_index": candidate_index,
                    **metrics,
                    "candidate_chrf": sentence_chrf_default(candidate, gold_text),
                    "candidate_text": candidate,
                }
                rows.append(row)
            jsonl.write(
                json.dumps(
                    {
                        "row_index": row_index,
                        "sonnet_id": sonnet_id,
                        "prompt": prompt,
                        "rows": rows,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            jsonl.flush()
            print(
                f"[train] shard={args.train_shard_index}/{args.train_num_shards} "
                f"item={done_count}/{len(sonnets)} row={row_index} candidates={len(rows)} "
                f"best_train_chrf={max(to_float(row['candidate_chrf']) for row in rows):.4f}"
            )

    all_rows = flatten_by_row(read_jsonl_candidates(jsonl_path))
    write_candidate_csv(csv_path, all_rows)
    return all_rows


def load_train_candidate_inputs(args: argparse.Namespace) -> list[dict[str, object]]:
    if not args.train_candidate_csvs:
        return []
    rows: list[dict[str, object]] = []
    for item in args.train_candidate_csvs.split(","):
        item = item.strip()
        if not item:
            continue
        rows.extend(read_candidate_csv(project_path(item)))
    rows.sort(key=lambda row: (int(row["row_index"]), int(row["candidate_index"])))
    return rows


def build_matrix(rows: list[dict[str, object]], feature_columns: list[str]) -> np.ndarray:
    return np.asarray(
        [[to_float(row.get(column)) for column in feature_columns] for row in rows],
        dtype=np.float64,
    )


def standardize_train(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return (X - mean) / std, mean, std


def apply_standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def fit_ridge(rows: list[dict[str, object]], feature_columns: list[str], alpha: float) -> dict[str, object]:
    X = build_matrix(rows, feature_columns)
    y = np.asarray([to_float(row.get("candidate_chrf")) for row in rows], dtype=np.float64)
    Xs, mean, std = standardize_train(X)
    X_aug = np.concatenate([np.ones((Xs.shape[0], 1)), Xs], axis=1)
    penalty = np.eye(X_aug.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(X_aug.T @ X_aug + alpha * penalty, X_aug.T @ y)
    pred = X_aug @ weights
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))
    return {
        "feature_columns": feature_columns,
        "alpha": alpha,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": weights.tolist(),
        "train_candidate_rmse": rmse,
        "train_candidate_mae": mae,
    }


def predict_rows(rows: list[dict[str, object]], model_info: dict[str, object]) -> np.ndarray:
    feature_columns = list(model_info["feature_columns"])
    X = build_matrix(rows, feature_columns)
    mean = np.asarray(model_info["mean"], dtype=np.float64)
    std = np.asarray(model_info["std"], dtype=np.float64)
    weights = np.asarray(model_info["weights"], dtype=np.float64)
    Xs = apply_standardize(X, mean, std)
    X_aug = np.concatenate([np.ones((Xs.shape[0], 1)), Xs], axis=1)
    return X_aug @ weights


def group_by_row(rows: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["row_index"]), []).append(row)
    for group in grouped.values():
        group.sort(key=lambda item: int(item["candidate_index"]))
    return dict(sorted(grouped.items()))


def select_predictions(
    rows: list[dict[str, object]],
    model_info: dict[str, object],
    output_path: Path,
    selected_csv_path: Path,
) -> list[dict[str, object]]:
    predictions = predict_rows(rows, model_info)
    scored_rows = []
    for row, score in zip(rows, predictions):
        new_row = dict(row)
        new_row["predicted_chrf"] = float(score)
        new_row["selected"] = 0
        scored_rows.append(new_row)

    selected_rows = []
    prediction_rows = []
    for row_index, group in group_by_row(scored_rows).items():
        best = max(
            group,
            key=lambda row: (
                to_float(row.get("predicted_chrf")),
                to_float(row.get("POEMetric_proxy")),
                -int(row["candidate_index"]),
            ),
        )
        best["selected"] = 1
        selected_rows.append(best)
        prediction_rows.append((str(row_index), str(best["candidate_text"])))

    write_numbered_predictions(output_path, prediction_rows)
    write_candidate_csv(selected_csv_path, selected_rows)
    write_candidate_csv(selected_csv_path.with_name(selected_csv_path.stem + "_all_scored.csv"), scored_rows)
    return selected_rows


def train_validation_summary(
    rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    grouped = group_by_row(rows)
    row_indices = list(grouped)
    rng = random.Random(args.seed)
    rng.shuffle(row_indices)
    val_size = max(1, int(len(row_indices) * args.val_ratio))
    val_indices = set(row_indices[:val_size])
    train_rows = [row for row in rows if int(row["row_index"]) not in val_indices]
    val_rows = [row for row in rows if int(row["row_index"]) in val_indices]
    model_info = fit_ridge(train_rows, FEATURE_COLUMNS, args.ridge_alpha)
    val_preds = predict_rows(val_rows, model_info)
    y_val = np.asarray([to_float(row.get("candidate_chrf")) for row in val_rows], dtype=np.float64)
    candidate_rmse = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))
    candidate_mae = float(np.mean(np.abs(val_preds - y_val)))

    selected = []
    for _row_index, group in group_by_row(val_rows).items():
        pred = predict_rows(group, model_info)
        best_idx = int(np.argmax(pred))
        selected.append(group[best_idx])

    return {
        "validation_groups": len(val_indices),
        "train_groups": len(row_indices) - len(val_indices),
        "candidate_rmse": candidate_rmse,
        "candidate_mae": candidate_mae,
        "mean_selected_candidate_chrf": statistics.mean(to_float(row.get("candidate_chrf")) for row in selected),
    }


def write_summary(
    path: Path,
    args: argparse.Namespace,
    model_info: dict[str, object],
    train_rows: list[dict[str, object]],
    train_val: dict[str, object],
    dev_selected: list[dict[str, object]],
    test_selected: list[dict[str, object]],
) -> None:
    out_dir = project_path(args.output_dir)
    dev_pred_path = out_dir / "predictions" / "dev_train_chrf_reranked.txt"
    test_pred_path = out_dir / "predictions" / "test_train_chrf_reranked.txt"
    dev_csv_path = out_dir / "candidate_metrics" / "dev_train_chrf_selected.csv"
    test_csv_path = out_dir / "candidate_metrics" / "test_train_chrf_selected.csv"
    weights = model_info["weights"]
    feature_rows = []
    for column, weight in zip(model_info["feature_columns"], weights[1:]):
        feature_rows.append((column, float(weight)))
    feature_rows.sort(key=lambda item: abs(item[1]), reverse=True)

    lines = [
        "# Train-chrF Predictor Reranking",
        "",
        "이 실험은 train gold만 사용해 후보의 chrF를 예측하는 ridge-regression reranker를 학습하고, dev/test에서는 gold 없이 feature만으로 후보를 선택한 결과이다.",
        "",
        "## Fairness",
        "",
        "- Reranker label: train candidate vs train gold `chrF`.",
        "- Dev/test candidate selection: gold reference 미사용.",
        "- Dev gold는 최종 평가에서만 사용.",
        "- Test gold는 없으므로 사용하지 않음.",
        "",
        "## Setup",
        "",
        f"- train data: `{args.train_path}`",
        f"- max train examples: `{args.max_train_examples or 'all'}`",
        f"- train candidates: `{args.num_candidates}` per prompt",
        f"- decoding strategies: `{args.decoding_strategies}`",
        f"- ridge alpha: `{args.ridge_alpha}`",
        f"- train candidate rows: `{len(train_rows)}`",
        f"- train candidate RMSE: `{model_info['train_candidate_rmse']:.4f}`",
        f"- train candidate MAE: `{model_info['train_candidate_mae']:.4f}`",
        "",
        "## Internal Train Validation",
        "",
        f"- validation groups: `{train_val['validation_groups']}`",
        f"- candidate RMSE: `{train_val['candidate_rmse']:.4f}`",
        f"- candidate MAE: `{train_val['candidate_mae']:.4f}`",
        f"- mean selected candidate chrF: `{train_val['mean_selected_candidate_chrf']:.4f}`",
        "",
        "## Selected Outputs",
        "",
        "| split | selected rows | prediction | selected metrics |",
        "|---|---:|---|---|",
        "| dev | {dev_count} | `{dev_pred}` | `{dev_csv}` |".format(
            dev_count=len(dev_selected),
            dev_pred=relative(dev_pred_path),
            dev_csv=relative(dev_csv_path),
        ),
        "| test | {test_count} | `{test_pred}` | `{test_csv}` |".format(
            test_count=len(test_selected),
            test_pred=relative(test_pred_path),
            test_csv=relative(test_csv_path),
        ),
        "",
        "## Largest Learned Weights",
        "",
        "| rank | feature | standardized ridge weight |",
        "|---:|---|---:|",
    ]
    for rank, (column, weight) in enumerate(feature_rows[:12], start=1):
        lines.append(f"| {rank} | `{column}` | {weight:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a train-gold chrF predictor reranker and apply it without dev/test gold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train_path", default="sonnet_project/data/strict_497/train_official_131_plus_extra_497_total_628.txt")
    parser.add_argument("--dev_candidates", default="sonnet_project/experiments/dpo_reranking/candidate_metrics/dev_candidates.csv")
    parser.add_argument("--test_candidates", default="sonnet_project/experiments/dpo_reranking/candidate_metrics/test_candidates.csv")
    parser.add_argument("--output_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max_train_examples", type=int, default=0, help="0 means all train examples.")
    parser.add_argument("--force_train_candidates", action="store_true")
    parser.add_argument("--only_generate_train_candidates", action="store_true")
    parser.add_argument("--train_num_shards", type=int, default=1)
    parser.add_argument("--train_shard_index", type=int, default=0)
    parser.add_argument("--train_candidate_csvs", default="", help="Comma-separated train candidate CSVs to reuse for fitting.")
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--seed_stride", type=int, default=100)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--sft_checkpoint", default="sonnet_project/experiments/sixway_ablation/dapt_sft_intermediate/best_10-1e-05-sonnet.pt")
    parser.add_argument("--dpo_checkpoint", default="sonnet_project/experiments/sixway_ablation/dapt_sft_lora_dpo_best_chrf/best_chrf_lora_dpo_form_rhyme.pt")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.90)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--top_k", type=int, default=60)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--decoding_strategy", default="top_p")
    parser.add_argument("--decoding_strategies", default="top_p,top_k")
    parser.add_argument("--repetition_penalty", type=float, default=1.08)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--max_generation_tokens", type=int, default=120)
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
    if args.train_num_shards < 1:
        raise SystemExit("--train_num_shards must be >= 1")
    if not (0 <= args.train_shard_index < args.train_num_shards):
        raise SystemExit("--train_shard_index must be in [0, train_num_shards)")
    seed_everything(args.seed)
    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    train_rows = load_train_candidate_inputs(args)
    if train_rows:
        print(f"[reuse] train candidate rows from inputs: {len(train_rows)}")
        model = None
    else:
        model, base_args = load_dpo_policy(args, device)
        train_rows = generate_train_candidates(model, base_args, args)
    if args.only_generate_train_candidates:
        print("[done] generated train candidates only")
        return
    train_val = train_validation_summary(train_rows, args)
    model_info = fit_ridge(train_rows, FEATURE_COLUMNS, args.ridge_alpha)

    (out_dir / "model.json").write_text(json.dumps(model_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "train_validation.json").write_text(json.dumps(train_val, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    dev_rows = read_candidate_csv(project_path(args.dev_candidates))
    test_rows = read_candidate_csv(project_path(args.test_candidates))
    dev_selected = select_predictions(
        dev_rows,
        model_info,
        out_dir / "predictions" / "dev_train_chrf_reranked.txt",
        out_dir / "candidate_metrics" / "dev_train_chrf_selected.csv",
    )
    test_selected = select_predictions(
        test_rows,
        model_info,
        out_dir / "predictions" / "test_train_chrf_reranked.txt",
        out_dir / "candidate_metrics" / "test_train_chrf_selected.csv",
    )
    summary = {
        "method": "train-gold chrF predictor reranking",
        "device": str(device),
        "args": vars(args),
        "model": model_info,
        "train_validation": train_val,
        "outputs": {
            "dev_prediction": relative(out_dir / "predictions" / "dev_train_chrf_reranked.txt"),
            "test_prediction": relative(out_dir / "predictions" / "test_train_chrf_reranked.txt"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary(out_dir / "SUMMARY.md", args, model_info, train_rows, train_val, dev_selected, test_selected)
    print(f"Wrote train-chrF reranking outputs to {relative(out_dir)}")


if __name__ == "__main__":
    main()
