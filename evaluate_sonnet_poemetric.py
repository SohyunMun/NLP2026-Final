#!/usr/bin/env python3
"""Evaluate sonnet generations with chrF, Sonnet-or-Not, and POEMetric proxies.

This script is intentionally CPU-only. It reads generated text files, gold
references, and prompts; it does not load generation models or touch CUDA.

The implementation keeps chrF as sacreBLEU's default character F-score, while
Sonnet-or-Not and POEMetric are implemented as transparent rule-based proxies:

* Sonnet-or-Not proxy: 14-line structure, line length, and Shakespearean rhyme.
* POEMetric proxy: form accuracy, lexical diversity, non-repetition,
  imagery/literary-device lexicon use, and prompt/theme overlap.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from evaluate_sonnet_metrics import (
    CHRF,
    align_blocks,
    corpus_repetition_rate,
    distinct_n,
    fallback_corpus_chrf,
    fallback_sentence_chrf,
    imagery_literary_device_score,
    line_count_score,
    line_length_score,
    load_numbered_blocks,
    mattr,
    mean,
    prompt_theme_overlap,
    repetition_rate,
    rhyme_scores,
    rounded,
    split_gold,
    split_prediction,
    text_from_lines,
    write_csv,
)


FORM_WEIGHTS = {
    "exact_14_lines": 0.35,
    "line_length_score": 0.20,
    "shakespearean_rhyme_pair_score": 0.30,
    "final_couplet_rhyme": 0.15,
}

OVERALL_QUALITY_WEIGHTS = {
    "form_accuracy": 0.35,
    "lexical_diversity": 0.25,
    "non_repetition": 0.25,
    "imagery_literary_device_score": 0.15,
}

POEMETRIC_WEIGHTS = {
    "form_accuracy": 0.30,
    "lexical_diversity": 0.25,
    "overall_quality_proxy": 0.30,
    "theme_overlap": 0.15,
}


def safe_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return clean or "model"


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.stem, path
    name, path = spec.split("=", 1)
    return safe_name(name), Path(path)


def weighted_sum(values: dict[str, float | None], weights: dict[str, float]) -> float:
    score = 0.0
    for key, weight in weights.items():
        value = values.get(key)
        score += weight * (float(value) if value is not None else 0.0)
    return score


def corpus_chrf_default(preds: list[str], refs: list[str]) -> float:
    """sacreBLEU default chrF, not chrF++."""
    if not preds:
        return 0.0
    if CHRF is not None:
        return CHRF().corpus_score(preds, [refs]).score
    return fallback_corpus_chrf(preds, refs)


def sentence_chrf_default(pred: str, ref: str) -> float:
    if CHRF is not None:
        return CHRF().sentence_score(pred, [ref]).score
    return fallback_sentence_chrf(pred, ref)


def corpus_chrfpp(preds: list[str], refs: list[str]) -> float:
    """Optional chrF++-style score with word_order=2."""
    if not preds:
        return 0.0
    if CHRF is not None:
        return CHRF(word_order=2).corpus_score(preds, [refs]).score
    return fallback_corpus_chrf(preds, refs)


def sentence_chrfpp(pred: str, ref: str) -> float:
    if CHRF is not None:
        return CHRF(word_order=2).sentence_score(pred, [ref]).score
    return fallback_sentence_chrf(pred, ref)


def form_accuracy(row: dict[str, float | int | str | None]) -> float:
    values = {
        "exact_14_lines": float(row.get("exact_14_lines") or 0.0),
        "line_length_score": float(row.get("line_length_score") or 0.0),
        "shakespearean_rhyme_pair_score": float(row.get("shakespearean_rhyme_pair_score") or 0.0),
        "final_couplet_rhyme": float(row.get("final_couplet_rhyme") or 0.0),
    }
    return weighted_sum(values, FORM_WEIGHTS)


def lexical_diversity(mattr_score: float | None, distinct_2_score: float | None) -> float:
    return 0.50 * (mattr_score or 0.0) + 0.50 * (distinct_2_score or 0.0)


def overall_quality_proxy(
    form_score: float,
    lexical_score: float,
    repetition_score: float,
    imagery_score: float | None,
) -> float:
    return weighted_sum(
        {
            "form_accuracy": form_score,
            "lexical_diversity": lexical_score,
            "non_repetition": 1.0 - repetition_score,
            "imagery_literary_device_score": imagery_score or 0.0,
        },
        OVERALL_QUALITY_WEIGHTS,
    )


def poemetric_proxy(
    form_score: float,
    lexical_score: float,
    overall_score: float,
    theme_score: float | None,
) -> float:
    return weighted_sum(
        {
            "form_accuracy": form_score,
            "lexical_diversity": lexical_score,
            "overall_quality_proxy": overall_score,
            "theme_overlap": theme_score or 0.0,
        },
        POEMETRIC_WEIGHTS,
    )


def sonnet_or_not_pass(
    exact_14: float,
    line_length: float,
    rhyme_pair: float,
    final_couplet: float,
    form_score: float,
    args: argparse.Namespace,
) -> float:
    passed = (
        exact_14 >= 1.0
        and line_length >= args.sonnet_line_length_threshold
        and rhyme_pair >= args.sonnet_rhyme_threshold
        and final_couplet >= args.sonnet_couplet_threshold
        and form_score >= args.sonnet_form_threshold
    )
    return 1.0 if passed else 0.0


def score_one_run(
    name: str,
    pred_path: Path,
    gold_path: Path | None,
    prompt_path: Path | None,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    prediction_blocks = load_numbered_blocks(pred_path)
    gold_blocks = load_numbered_blocks(gold_path) if gold_path else None
    prompt_blocks = load_numbered_blocks(prompt_path) if prompt_path else None

    aligned_gold = align_blocks(prediction_blocks, gold_blocks, args.align)
    aligned_prompts = align_blocks(prediction_blocks, prompt_blocks, args.align)

    rows: list[dict[str, object]] = []
    scored_preds: list[str] = []
    scored_refs: list[str] = []
    scored_texts_for_diversity: list[str] = []

    for idx, (pred_id, pred_body) in enumerate(prediction_blocks):
        gold_block = aligned_gold[idx]
        prompt_block = aligned_prompts[idx]
        gold_body = gold_block[1] if gold_block else None
        prompt_body = prompt_block[1] if prompt_block else None

        full_lines, continuation_lines, prompt_lines, _predicted_prompt_lines = split_prediction(
            pred_body=pred_body,
            prompt_body=prompt_body,
            prompt_line_count=args.prompt_lines,
            prediction_mode=args.prediction_mode,
        )
        gold_full_lines, gold_continuation_lines = split_gold(gold_body, args.prompt_lines)

        full_text = text_from_lines(full_lines)
        continuation_text = text_from_lines(continuation_lines)
        prompt_text = text_from_lines(prompt_lines)
        score_text = continuation_text if args.score_part == "continuation" else full_text
        scored_texts_for_diversity.append(score_text)

        gold_score_text = None
        if gold_body is not None:
            if args.score_part == "continuation" and gold_continuation_lines is not None:
                gold_score_text = text_from_lines(gold_continuation_lines)
            elif gold_full_lines is not None:
                gold_score_text = text_from_lines(gold_full_lines)

        pair_score, final_couplet_score, endings = rhyme_scores(full_lines)
        row_mattr = mattr(score_text, args.mattr_window)
        row_distinct_1 = distinct_n([score_text], 1)
        row_distinct_2 = distinct_n([score_text], 2)
        row_repetition = repetition_rate(score_text)
        row_imagery = imagery_literary_device_score(continuation_text, continuation_lines)
        row_theme = prompt_theme_overlap(prompt_text, continuation_text)

        row: dict[str, object] = {
            "model_name": name,
            "row_index": idx,
            "pred_id": pred_id,
            "gold_id": gold_block[0] if gold_block else "",
            "prompt_id": prompt_block[0] if prompt_block else "",
            "line_count": len(full_lines),
            "exact_14_lines": 1.0 if len(full_lines) == args.target_lines else 0.0,
            "line_count_score": line_count_score(full_lines, args.target_lines),
            "line_length_score": line_length_score(full_lines),
            "shakespearean_rhyme_pair_score": pair_score,
            "final_couplet_rhyme": final_couplet_score,
            "MATTR": row_mattr,
            "distinct_1": row_distinct_1,
            "distinct_2": row_distinct_2,
            "lexical_diversity": lexical_diversity(row_mattr, row_distinct_2),
            "repetition_rate": row_repetition,
            "non_repetition": 1.0 - row_repetition,
            "prompt_continuation_theme_overlap": row_theme,
            "imagery_literary_device_score": row_imagery,
            "ending_words": endings,
        }
        row["sonnet_form_accuracy"] = form_accuracy(row)
        row["sonnet_or_not_bot_pass"] = sonnet_or_not_pass(
            exact_14=float(row["exact_14_lines"]),
            line_length=float(row["line_length_score"]),
            rhyme_pair=float(row["shakespearean_rhyme_pair_score"]),
            final_couplet=float(row["final_couplet_rhyme"]),
            form_score=float(row["sonnet_form_accuracy"]),
            args=args,
        )
        row["poemetric_overall_quality_proxy"] = overall_quality_proxy(
            form_score=float(row["sonnet_form_accuracy"]),
            lexical_score=float(row["lexical_diversity"]),
            repetition_score=row_repetition,
            imagery_score=row_imagery,
        )
        row["POEMetric_proxy"] = poemetric_proxy(
            form_score=float(row["sonnet_form_accuracy"]),
            lexical_score=float(row["lexical_diversity"]),
            overall_score=float(row["poemetric_overall_quality_proxy"]),
            theme_score=row_theme,
        )

        if gold_score_text is not None:
            row["chrF"] = sentence_chrf_default(score_text, gold_score_text)
            if args.include_chrfpp:
                row["chrF++"] = sentence_chrfpp(score_text, gold_score_text)
            scored_preds.append(score_text)
            scored_refs.append(gold_score_text)
        else:
            row["chrF"] = None
            if args.include_chrfpp:
                row["chrF++"] = None

        rows.append(row)

    summary_mattr = mean(row.get("MATTR") for row in rows)
    summary_distinct_1 = distinct_n(scored_texts_for_diversity, 1)
    summary_distinct_2 = distinct_n(scored_texts_for_diversity, 2)
    summary_repetition = corpus_repetition_rate(scored_texts_for_diversity)
    summary_form = mean(row.get("sonnet_form_accuracy") for row in rows) or 0.0
    summary_lexical = lexical_diversity(summary_mattr, summary_distinct_2)
    summary_imagery = mean(row.get("imagery_literary_device_score") for row in rows)
    summary_theme = mean(row.get("prompt_continuation_theme_overlap") for row in rows)
    summary_overall = overall_quality_proxy(
        form_score=summary_form,
        lexical_score=summary_lexical,
        repetition_score=summary_repetition,
        imagery_score=summary_imagery,
    )
    summary_poemetric = poemetric_proxy(
        form_score=summary_form,
        lexical_score=summary_lexical,
        overall_score=summary_overall,
        theme_score=summary_theme,
    )

    summary: dict[str, object] = {
        "model_name": name,
        "prediction_file": str(pred_path.resolve()),
        "gold_file": str(gold_path.resolve()) if gold_path else "",
        "prompt_file": str(prompt_path.resolve()) if prompt_path else "",
        "count": len(rows),
        "score_part": args.score_part,
        "prediction_mode": args.prediction_mode,
        "chrF": corpus_chrf_default(scored_preds, scored_refs) if scored_preds else None,
        "sonnet_or_not_bot_pass_rate": mean(row.get("sonnet_or_not_bot_pass") for row in rows),
        "sonnet_form_accuracy": summary_form,
        "exact_14_lines_rate": mean(row.get("exact_14_lines") for row in rows),
        "line_count_score": mean(row.get("line_count_score") for row in rows),
        "line_length_score": mean(row.get("line_length_score") for row in rows),
        "shakespearean_rhyme_pair_score": mean(row.get("shakespearean_rhyme_pair_score") for row in rows),
        "final_couplet_rhyme": mean(row.get("final_couplet_rhyme") for row in rows),
        "MATTR": summary_mattr,
        "distinct_1": summary_distinct_1,
        "distinct_2": summary_distinct_2,
        "lexical_diversity": summary_lexical,
        "repetition_rate": summary_repetition,
        "non_repetition": 1.0 - summary_repetition,
        "imagery_literary_device_score": summary_imagery,
        "prompt_continuation_theme_overlap": summary_theme,
        "poemetric_overall_quality_proxy": summary_overall,
        "POEMetric_proxy": summary_poemetric,
    }
    if args.include_chrfpp:
        summary["chrF++"] = corpus_chrfpp(scored_preds, scored_refs) if scored_preds else None

    return rows, summary


def write_markdown(path: Path, summaries: list[dict[str, object]], args: argparse.Namespace) -> None:
    sort_key = "POEMetric_proxy" if args.rank_by == "poemetric" else "chrF"

    def score_for_sort(row: dict[str, object]) -> float:
        value = row.get(sort_key)
        return float(value) if value is not None else -1.0

    ranked = sorted(summaries, key=score_for_sort, reverse=True)
    lines = [
        "# Sonnet Evaluation: chrF / Sonnet-or-Not / POEMetric",
        "",
        f"- rank_by: `{sort_key}`",
        f"- score_part: `{args.score_part}`",
        f"- prediction_mode: `{args.prediction_mode}`",
        "",
        "| rank | model | chrF | Sonnet-or-Not pass | form accuracy | lexical diversity | overall quality | theme | POEMetric |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            "| {rank} | `{model}` | {chrf} | {pass_rate} | {form} | {lexical} | {overall} | {theme} | {poemetric} |".format(
                rank=rank,
                model=row["model_name"],
                chrf=format_value(row.get("chrF")),
                pass_rate=format_value(row.get("sonnet_or_not_bot_pass_rate")),
                form=format_value(row.get("sonnet_form_accuracy")),
                lexical=format_value(row.get("lexical_diversity")),
                overall=format_value(row.get("poemetric_overall_quality_proxy")),
                theme=format_value(row.get("prompt_continuation_theme_overlap")),
                poemetric=format_value(row.get("POEMetric_proxy")),
            )
        )

    lines.extend(
        [
            "",
            "## Metric Definitions",
            "",
            "- `chrF`: sacreBLEU default chrF. Gold reference가 있을 때만 계산됨.",
            "- `Sonnet-or-Not pass`: exact 14 lines, line length, rhyme, final couplet, form threshold를 모두 만족한 비율.",
            "- `form accuracy`: 0.35 exact14 + 0.20 line length + 0.30 Shakespearean rhyme pairs + 0.15 final couplet rhyme.",
            "- `lexical diversity`: 0.50 MATTR + 0.50 distinct-2.",
            "- `overall quality`: 0.35 form + 0.25 lexical diversity + 0.25 non-repetition + 0.15 imagery/literary-device proxy.",
            "- `POEMetric`: 0.30 form + 0.25 lexical diversity + 0.30 overall quality + 0.15 prompt/theme overlap.",
            "",
            "## Thresholds",
            "",
            f"- Sonnet-or-Not form threshold: `{args.sonnet_form_threshold}`",
            f"- line length threshold: `{args.sonnet_line_length_threshold}`",
            f"- rhyme-pair threshold: `{args.sonnet_rhyme_threshold}`",
            f"- final-couplet threshold: `{args.sonnet_couplet_threshold}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_value(value: object, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def json_ready(row: dict[str, object]) -> dict[str, object]:
    clean: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, float):
            clean[key] = rounded(value)
        else:
            clean[key] = value
    return clean


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated sonnets with chrF, Sonnet-or-Not, and POEMetric proxies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pred", default="", help="Single prediction file in numbered-block format.")
    parser.add_argument("--name", default="model", help="Run name for --pred.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Named prediction file as NAME=PATH. Can be repeated for multi-run comparison.",
    )
    parser.add_argument("--gold", default="", help="Gold reference file. Omit when references are unavailable.")
    parser.add_argument("--prompts", default="", help="Prompt file used for theme overlap and continuation reconstruction.")
    parser.add_argument("--out_dir", default="experiments/sonnet_poemetric_eval", help="Output directory.")
    parser.add_argument("--align", choices=["auto", "index", "id"], default="auto", help="How prediction blocks align to gold/prompts.")
    parser.add_argument("--prediction_mode", choices=["full", "continuation"], default="full", help="Whether predictions already include prompt lines.")
    parser.add_argument("--score_part", choices=["full", "continuation"], default="full", help="Text span used for chrF/diversity.")
    parser.add_argument("--target_lines", type=int, default=14)
    parser.add_argument("--prompt_lines", type=int, default=3)
    parser.add_argument("--mattr_window", type=int, default=50)
    parser.add_argument("--rank_by", choices=["poemetric", "chrf"], default="poemetric")
    parser.add_argument("--include_chrfpp", action="store_true", help="Also report chrF++ style score with word_order=2.")
    parser.add_argument("--sonnet_form_threshold", type=float, default=0.70)
    parser.add_argument("--sonnet_line_length_threshold", type=float, default=0.50)
    parser.add_argument("--sonnet_rhyme_threshold", type=float, default=0.25)
    parser.add_argument("--sonnet_couplet_threshold", type=float, default=0.25)
    args = parser.parse_args()

    if not args.run and not args.pred:
        parser.error("Provide --pred or at least one --run NAME=PATH.")
    return args


def main() -> None:
    args = get_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[str, Path]] = []
    if args.pred:
        runs.append((safe_name(args.name), Path(args.pred)))
    for spec in args.run:
        runs.append(parse_run_spec(spec))

    gold_path = Path(args.gold).resolve() if args.gold else None
    prompt_path = Path(args.prompts).resolve() if args.prompts else None
    if gold_path and not gold_path.exists():
        raise SystemExit(f"Gold file not found: {gold_path}")
    if prompt_path and not prompt_path.exists():
        raise SystemExit(f"Prompt file not found: {prompt_path}")

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for name, pred_path in runs:
        if not pred_path.exists():
            raise SystemExit(f"Prediction file not found: {pred_path}")
        rows, summary = score_one_run(
            name=name,
            pred_path=pred_path.resolve(),
            gold_path=gold_path,
            prompt_path=prompt_path,
            args=args,
        )
        all_rows.extend(rows)
        summaries.append(summary)

        write_csv(out_dir / f"{safe_name(name)}_per_sonnet_metrics.csv", rows)
        write_csv(out_dir / f"{safe_name(name)}_summary_metrics.csv", [summary])
        (out_dir / f"{safe_name(name)}_summary_metrics.json").write_text(
            json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    write_csv(out_dir / "all_per_sonnet_metrics.csv", all_rows)
    write_csv(out_dir / "all_summary_metrics.csv", summaries)
    (out_dir / "all_summary_metrics.json").write_text(
        json.dumps([json_ready(row) for row in summaries], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(out_dir / "all_summary_metrics.md", summaries, args)

    print(f"Wrote per-sonnet metrics: {out_dir / 'all_per_sonnet_metrics.csv'}")
    print(f"Wrote summary CSV: {out_dir / 'all_summary_metrics.csv'}")
    print(f"Wrote summary JSON: {out_dir / 'all_summary_metrics.json'}")
    print(f"Wrote summary Markdown: {out_dir / 'all_summary_metrics.md'}")


if __name__ == "__main__":
    main()
