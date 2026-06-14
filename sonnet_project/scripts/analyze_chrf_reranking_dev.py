#!/usr/bin/env python3
"""Dev-only chrF reranking analysis for DPO candidates.

This script is intentionally limited to dev data because chrF requires gold
references. It writes two outputs:

1. oracle_chrf: chooses the highest-chrF candidate for each dev prompt.
   This is an upper-bound diagnostic, not a deployable inference method.
2. loo_chrf_tuned: leave-one-out selection. For each held-out dev prompt, it
   chooses a reference-free ranking recipe using chrF on the other dev prompts,
   then applies that recipe to the held-out prompt without using its gold.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ROOT = PROJECT_ROOT.parent
for search_path in (SCRIPT_DIR, ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from evaluate_sonnet_metrics import align_blocks, load_numbered_blocks, text_from_lines
from evaluate_sonnet_poemetric import corpus_chrf_default, sentence_chrf_default


DEFAULT_CANDIDATE_CSV = "sonnet_project/experiments/dpo_reranking/candidate_metrics/dev_candidates.csv"
DEFAULT_GOLD = "sonnet_project/data/strict_497/dev_gold_12.txt"
DEFAULT_OUT_DIR = "sonnet_project/experiments/chrf_reranking_dev"


FEATURES = [
    "rerank_score",
    "POEMetric_proxy",
    "sonnet_form_accuracy",
    "line_length_score",
    "shakespearean_rhyme_pair_score",
    "final_couplet_rhyme",
    "lexical_diversity",
    "non_repetition",
    "prompt_continuation_theme_overlap",
    "mbr_score",
]


RECIPES: dict[str, dict[str, float]] = {
    "poemetric_rerank_score": {"rerank_score": 1.0},
    "poemetric_only": {"POEMetric_proxy": 1.0},
    "form_only": {"sonnet_form_accuracy": 1.0},
    "rhyme_only": {"shakespearean_rhyme_pair_score": 1.0, "final_couplet_rhyme": 0.5},
    "theme_only": {"prompt_continuation_theme_overlap": 1.0},
    "lexical_only": {"lexical_diversity": 1.0},
    "non_repetition_only": {"non_repetition": 1.0},
    "mbr_only": {"mbr_score": 1.0},
    "content_proxy": {
        "prompt_continuation_theme_overlap": 0.40,
        "lexical_diversity": 0.25,
        "POEMetric_proxy": 0.20,
        "non_repetition": 0.15,
    },
    "form_rhyme_proxy": {
        "sonnet_form_accuracy": 0.40,
        "shakespearean_rhyme_pair_score": 0.25,
        "final_couplet_rhyme": 0.15,
        "line_length_score": 0.10,
        "non_repetition": 0.10,
    },
    "balanced_proxy": {
        "POEMetric_proxy": 0.40,
        "sonnet_form_accuracy": 0.20,
        "prompt_continuation_theme_overlap": 0.15,
        "lexical_diversity": 0.15,
        "non_repetition": 0.10,
    },
    "chrf_proxy_light": {
        "prompt_continuation_theme_overlap": 0.30,
        "POEMetric_proxy": 0.25,
        "lexical_diversity": 0.20,
        "line_length_score": 0.15,
        "non_repetition": 0.10,
    },
}


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def read_candidates(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            clean: dict[str, object] = dict(row)
            clean["row_index"] = int(row["row_index"])
            clean["candidate_index"] = int(row["candidate_index"])
            for feature in FEATURES:
                clean[feature] = to_float(row.get(feature))
            rows.append(clean)
        return rows


def load_gold_texts(path: Path, count: int) -> list[str]:
    pseudo_predictions = [(str(idx), "") for idx in range(count)]
    gold_blocks = load_numbered_blocks(path)
    aligned = align_blocks(pseudo_predictions, gold_blocks, "index")
    return [block[1] if block else "" for block in aligned]


def attach_chrf(rows: list[dict[str, object]], gold_texts: list[str]) -> None:
    for row in rows:
        row_index = int(row["row_index"])
        candidate_text = str(row["candidate_text"])
        gold_text = gold_texts[row_index]
        row["candidate_chrf"] = sentence_chrf_default(candidate_text, gold_text) if gold_text else 0.0


def group_by_row(rows: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["row_index"])].append(row)
    for group in grouped.values():
        group.sort(key=lambda item: int(item["candidate_index"]))
    return dict(sorted(grouped.items()))


def score_recipe(row: dict[str, object], recipe: dict[str, float]) -> float:
    return sum(weight * to_float(row.get(feature)) for feature, weight in recipe.items())


def select_by_recipe(group: list[dict[str, object]], recipe_name: str) -> dict[str, object]:
    if recipe_name.startswith("candidate_"):
        wanted = int(recipe_name.split("_", 1)[1])
        exact = [row for row in group if int(row["candidate_index"]) == wanted]
        return exact[0] if exact else group[0]

    recipe = RECIPES[recipe_name]
    return max(
        group,
        key=lambda row: (
            score_recipe(row, recipe),
            to_float(row.get("POEMetric_proxy")),
            -int(row["candidate_index"]),
        ),
    )


def corpus_chrf_for_selection(
    grouped: dict[int, list[dict[str, object]]],
    gold_texts: list[str],
    row_indices: list[int],
    recipe_name: str,
) -> float:
    predictions = [str(select_by_recipe(grouped[idx], recipe_name)["candidate_text"]) for idx in row_indices]
    references = [gold_texts[idx] for idx in row_indices]
    return corpus_chrf_default(predictions, references)


def write_numbered_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("--Generated Sonnets--\n\n")
        for row in rows:
            handle.write(f"\n{row['row_index']}\n{str(row['candidate_text']).strip()}\n\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    candidate_path = project_path(args.candidates)
    gold_path = project_path(args.gold)
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_candidates(candidate_path)
    max_row_index = max(int(row["row_index"]) for row in rows)
    gold_texts = load_gold_texts(gold_path, max_row_index + 1)
    attach_chrf(rows, gold_texts)
    grouped = group_by_row(rows)
    row_indices = sorted(grouped)

    candidate_recipe_names = [
        f"candidate_{idx}"
        for idx in sorted({int(row["candidate_index"]) for row in rows})
    ]
    recipe_names = list(RECIPES) + candidate_recipe_names

    oracle_rows = []
    for row_index in row_indices:
        best = max(
            grouped[row_index],
            key=lambda row: (to_float(row.get("candidate_chrf")), to_float(row.get("POEMetric_proxy"))),
        )
        oracle_rows.append({**best, "selection_method": "oracle_chrf"})

    loo_rows = []
    fold_rows = []
    for heldout_idx in row_indices:
        train_indices = [idx for idx in row_indices if idx != heldout_idx]
        recipe_scores = []
        for recipe_name in recipe_names:
            train_chrf = corpus_chrf_for_selection(grouped, gold_texts, train_indices, recipe_name)
            recipe_scores.append((train_chrf, recipe_name))
        best_train_chrf, best_recipe = max(recipe_scores, key=lambda item: (item[0], item[1]))
        selected = select_by_recipe(grouped[heldout_idx], best_recipe)
        loo_rows.append(
            {
                **selected,
                "selection_method": "leave_one_out_chrf_tuned",
                "selected_recipe": best_recipe,
                "train_chrf": best_train_chrf,
            }
        )
        fold_rows.append(
            {
                "heldout_row_index": heldout_idx,
                "selected_recipe": best_recipe,
                "train_chrf": best_train_chrf,
                "heldout_candidate_index": selected["candidate_index"],
                "heldout_candidate_chrf": selected["candidate_chrf"],
            }
        )

    recipe_summary_rows = []
    for recipe_name in recipe_names:
        selected_rows = [select_by_recipe(grouped[idx], recipe_name) for idx in row_indices]
        predictions = [str(row["candidate_text"]) for row in selected_rows]
        refs = [gold_texts[idx] for idx in row_indices]
        recipe_summary_rows.append(
            {
                "recipe": recipe_name,
                "dev_corpus_chrf": corpus_chrf_default(predictions, refs),
                "mean_candidate_chrf": statistics.mean(to_float(row["candidate_chrf"]) for row in selected_rows),
            }
        )
    recipe_summary_rows.sort(key=lambda row: float(row["dev_corpus_chrf"]), reverse=True)

    oracle_prediction = out_dir / "predictions" / "dev_oracle_chrf.txt"
    loo_prediction = out_dir / "predictions" / "dev_loo_chrf_tuned.txt"
    write_numbered_predictions(oracle_prediction, oracle_rows)
    write_numbered_predictions(loo_prediction, loo_rows)
    write_csv(out_dir / "candidate_metrics_with_chrf.csv", rows)
    write_csv(out_dir / "oracle_chrf_selected.csv", oracle_rows)
    write_csv(out_dir / "loo_chrf_tuned_selected.csv", loo_rows)
    write_csv(out_dir / "loo_folds.csv", fold_rows)
    write_csv(out_dir / "recipe_summary.csv", recipe_summary_rows)

    summary = {
        "method": "dev-only chrF reranking analysis",
        "candidates": relative(candidate_path),
        "gold": relative(gold_path),
        "notes": [
            "oracle_chrf uses each dev prompt's gold reference and is only an upper-bound diagnostic.",
            "leave_one_out_chrf_tuned does not use the held-out prompt's gold when selecting its candidate.",
            "No test gold is used and no test chrF reranking is performed.",
        ],
        "oracle_prediction": relative(oracle_prediction),
        "loo_prediction": relative(loo_prediction),
        "best_full_dev_recipe_by_chrf": recipe_summary_rows[0],
        "folds": fold_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(out_dir / "SUMMARY.md", summary, recipe_summary_rows, oracle_rows, loo_rows)
    return summary


def write_markdown(
    path: Path,
    summary: dict[str, object],
    recipe_summary_rows: list[dict[str, object]],
    oracle_rows: list[dict[str, object]],
    loo_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Dev-only chrF Reranking Analysis",
        "",
        "이 실험은 `chrF` gold reference가 있는 dev set에서만 수행한 분석이다.",
        "test gold는 없고, test 결과를 보고 설정을 바꾸는 것을 피하기 위해 test에는 적용하지 않았다.",
        "",
        "## Fairness Policy",
        "",
        "- `oracle_chrf`: 각 dev prompt의 gold를 보고 최고 chrF 후보를 선택함. 최종 성능이 아니라 후보군 상한선 확인용.",
        "- `leave_one_out_chrf_tuned`: held-out prompt를 하나 뺀 나머지 dev prompt에서 chrF가 가장 높은 reference-free recipe를 고른 뒤 held-out prompt에 적용함.",
        "- 따라서 `leave_one_out_chrf_tuned`도 작은 dev set에 대한 분석용이며, test 성능 주장에는 사용하지 않음.",
        "",
        "## Outputs",
        "",
        f"- oracle prediction: `{summary['oracle_prediction']}`",
        f"- leave-one-out prediction: `{summary['loo_prediction']}`",
        "- candidate metrics with chrF: `sonnet_project/experiments/chrf_reranking_dev/candidate_metrics_with_chrf.csv`",
        "- fold decisions: `sonnet_project/experiments/chrf_reranking_dev/loo_folds.csv`",
        "- recipe summary: `sonnet_project/experiments/chrf_reranking_dev/recipe_summary.csv`",
        "",
        "## Best Fixed Recipes on Full Dev",
        "",
        "| rank | recipe | dev corpus chrF | mean selected-candidate chrF |",
        "|---:|---|---:|---:|",
    ]
    for rank, row in enumerate(recipe_summary_rows[:10], start=1):
        lines.append(
            f"| {rank} | `{row['recipe']}` | {float(row['dev_corpus_chrf']):.4f} | "
            f"{float(row['mean_candidate_chrf']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Selection Summary",
            "",
            f"- oracle mean candidate chrF: `{statistics.mean(to_float(row['candidate_chrf']) for row in oracle_rows):.4f}`",
            f"- leave-one-out mean candidate chrF: `{statistics.mean(to_float(row['candidate_chrf']) for row in loo_rows):.4f}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze DPO dev candidates with chrF-based oracle and leave-one-out reranking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--gold", default=DEFAULT_GOLD)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run_analysis(get_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
