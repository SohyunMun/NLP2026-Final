#!/usr/bin/env python3
"""Run the requested six-way sonnet generation ablation.

Definitions used in this runner:
1. base_basic: GPT-2 full fine-tuning with a plain LM loss on official train.
2. base_plus_extra: same baseline on official train + strict extra 497.
3. sft_plus_extra: prompt-focused SFT on official train + strict extra 497.
4. dapt_plus_extra: short domain-adaptive pretraining on the same extra set,
   evaluated directly without task SFT.
5. selected_lora_plus_extra: LoRA-SFT initialized from whichever of #3/#4 has
   better dev chrF.
6. dapt_sft_lora_dpo_best_chrf: DAPT -> SFT -> form/rhyme LoRA-DPO, selecting
   the DPO checkpoint with best dev chrF.

All evaluation uses evaluate_sonnet_poemetric.py:
chrF, Sonnet-or-Not pass rate, form accuracy, lexical diversity,
overall-quality proxy, Theme, and POEMetric proxy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ROOT = PROJECT_ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYTHON = Path(os.environ.get("PYTHON", sys.executable))
BASIC = PROJECT_ROOT / "data" / "basic"
STRICT = PROJECT_ROOT / "data" / "strict_497"
OUT_ROOT = PROJECT_ROOT / "experiments" / "sixway_ablation"

BASIC_TRAIN = BASIC / "train_131.txt"
TRAIN_EXTRA = STRICT / "train_official_131_plus_extra_497_total_628.txt"
DEV_PROMPT = STRICT / "dev_prompts_12.txt"
DEV_GOLD = STRICT / "dev_gold_12.txt"
TEST_PROMPT = STRICT / "test_prompts_12.txt"


def run_command(name: str, cmd: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {name}")
    print(f"[log] {log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        result = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}. Check {log_path}")


def env_for_gpu(gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    return env


def checkpoint_for(out_dir: Path, epochs: int, lr: str) -> Path:
    return out_dir / f"best_{epochs}-{lr}-sonnet.pt"


def run_sonnet_generation_stage(
    name: str,
    train_path: Path,
    out_dir: Path,
    gpu: str,
    epochs: int,
    lr: str,
    prompt_loss_weight: str,
    line_break_loss_weight: str,
    num_candidates: str,
    model_score_weight: str,
    mbr_weight: str,
    max_generation_tokens: str = "120",
    init_checkpoint: Path | None = None,
    force: bool = False,
) -> dict[str, str]:
    pred_dir = out_dir / "predictions"
    dev_pred = pred_dir / "dev.txt"
    test_pred = pred_dir / "test.txt"
    ckpt = checkpoint_for(out_dir, epochs, lr)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    if force or not ckpt.exists() or not dev_pred.exists():
        cmd = [
            str(PYTHON),
            str(ROOT / "sonnet_generation.py"),
            "--epochs", str(epochs),
            "--batch_size", "8",
            "--lr", lr,
            "--weight_decay", "0.01",
            "--eval_every", "1",
            "--patience", str(epochs),
            "--selection_metric", "loss",
            "--skip_epoch_sample",
            "--num_candidates", num_candidates,
            "--dev_num_candidates", "1",
            "--max_generation_tokens", max_generation_tokens,
            "--model_score_weight", model_score_weight,
            "--mbr_weight", mbr_weight,
            "--prompt_loss_weight", prompt_loss_weight,
            "--line_break_loss_weight", line_break_loss_weight,
            "--sonnet_path", str(train_path),
            "--held_out_sonnet_path", str(DEV_PROMPT),
            "--sonnet_out", str(dev_pred),
            "--model_size", "gpt2",
            "--use_gpu",
            "--dev_sonnet_path", str(DEV_PROMPT),
            "--dev_gold_path", str(DEV_GOLD),
        ]
        if init_checkpoint is not None:
            cmd.extend(["--init_checkpoint_path", str(init_checkpoint)])
        run_command(name, cmd, out_dir, out_dir / "train.log", env_for_gpu(gpu))
    else:
        print(f"[skip] {name}: {ckpt}")

    if force or not test_pred.exists():
        cmd = [
            str(PYTHON),
            str(SCRIPT_DIR / "generate_sonnet_checkpoint.py"),
            "--checkpoint", str(ckpt),
            "--prompt_path", str(TEST_PROMPT),
            "--output_path", str(test_pred),
            "--seed", "21711",
            "--num_candidates", num_candidates,
            "--max_generation_tokens", max_generation_tokens,
            "--model_score_weight", model_score_weight,
            "--mbr_weight", mbr_weight,
            "--use_gpu",
        ]
        run_command(f"{name}_test_generation", cmd, ROOT, out_dir / "test_generation.log", env_for_gpu(gpu))
    else:
        print(f"[skip] {name}_test_generation: {test_pred}")

    return {
        "name": name,
        "checkpoint": str(ckpt),
        "dev_prediction": str(dev_pred),
        "test_prediction": str(test_pred),
    }


def run_lora_stage(name: str, init_checkpoint: Path, out_dir: Path, gpu: str, force: bool = False) -> dict[str, str]:
    dev_pred = out_dir / "predictions" / "dev_best_chrf.txt"
    test_pred = out_dir / "predictions" / "test_best_chrf.txt"
    ckpt = out_dir / "best_chrf_lora_sft.pt"
    if force or not ckpt.exists() or not dev_pred.exists() or not test_pred.exists():
        cmd = [
            str(PYTHON),
            str(SCRIPT_DIR / "train_lora_sft_from_checkpoint.py"),
            "--init_checkpoint", str(init_checkpoint),
            "--train_path", str(TRAIN_EXTRA),
            "--dev_prompt_path", str(DEV_PROMPT),
            "--dev_gold_path", str(DEV_GOLD),
            "--test_prompt_path", str(TEST_PROMPT),
            "--output_dir", str(out_dir),
            "--epochs", "10",
            "--batch_size", "8",
            "--lr", "1.5e-4",
            "--lora_r", "8",
            "--lora_alpha", "16",
            "--num_candidates", "4",
            "--max_generation_tokens", "120",
            "--use_gpu",
        ]
        run_command(name, cmd, ROOT, out_dir / "train.log", env_for_gpu(gpu))
    else:
        print(f"[skip] {name}: {ckpt}")
    return {
        "name": name,
        "checkpoint": str(ckpt),
        "dev_prediction": str(dev_pred),
        "test_prediction": str(test_pred),
    }


def run_form_dpo_stage(name: str, sft_checkpoint: Path, out_dir: Path, gpu: str, force: bool = False) -> dict[str, str]:
    dev_pred = out_dir / "predictions" / "dev_best_chrf.txt"
    test_pred = out_dir / "predictions" / "test_best_chrf.txt"
    ckpt = out_dir / "best_chrf_lora_dpo_form_rhyme.pt"
    if force or not ckpt.exists() or not dev_pred.exists() or not test_pred.exists():
        cmd = [
            str(PYTHON),
            str(SCRIPT_DIR / "run_msk_sft_lora_dpo_form_rhyme.py"),
            "--sft_checkpoint", str(sft_checkpoint),
            "--train_path", str(TRAIN_EXTRA),
            "--dev_prompt_path", str(DEV_PROMPT),
            "--dev_gold_path", str(DEV_GOLD),
            "--test_prompt_path", str(TEST_PROMPT),
            "--output_dir", str(out_dir),
            "--epochs", "10",
            "--batch_size", "8",
            "--lr", "1.5e-4",
            "--beta", "0.05",
            "--lora_r", "8",
            "--lora_alpha", "16",
            "--include_short_form_rejects",
            "--num_candidates", "6",
            "--max_generation_tokens", "130",
            "--use_gpu",
        ]
        run_command(name, cmd, ROOT, out_dir / "train.log", env_for_gpu(gpu))
    else:
        print(f"[skip] {name}: {ckpt}")
    return {
        "name": name,
        "checkpoint": str(ckpt),
        "dev_prediction": str(dev_pred),
        "test_prediction": str(test_pred),
    }


def eval_multi(split: str, runs: list[dict[str, str]]) -> Path:
    out_dir = OUT_ROOT / "poemetric_eval" / split
    cmd = [
        str(PYTHON),
        str(SCRIPT_DIR / "evaluate_sonnet_poemetric.py"),
        "--prompts", str(DEV_PROMPT if split == "dev" else TEST_PROMPT),
        "--out_dir", str(out_dir),
        "--rank_by", "poemetric",
    ]
    if split == "dev":
        cmd.extend(["--gold", str(DEV_GOLD)])
    for run in runs:
        cmd.extend(["--run", f"{run['name']}={run[f'{split}_prediction']}"])
    run_command(f"eval_{split}", cmd, ROOT, out_dir / "eval.log", os.environ.copy())
    return out_dir / "all_summary_metrics.json"


def load_rows(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def fmt(value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def metric_by_name(path: Path, key: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in load_rows(path):
        value = row.get(key)
        if isinstance(value, (int, float)):
            result[str(row.get("model_name"))] = float(value)
    return result


def write_summary(dev_eval: Path, test_eval: Path, runs: list[dict[str, str]], selected_source: str) -> None:
    run_meta = {run["name"]: run for run in runs}
    lines = [
        "# Six-Way Sonnet Ablation Summary",
        "",
        "## Data",
        "",
        f"- basic train: `{BASIC_TRAIN}`",
        f"- plus-extra train: `{TRAIN_EXTRA}`",
        f"- dev prompts/gold: `{DEV_PROMPT}`, `{DEV_GOLD}`",
        f"- test prompts: `{TEST_PROMPT}`",
        "- test gold is unavailable; test chrF is intentionally blank.",
        "",
        "## Run Definitions",
        "",
        "| run | definition | checkpoint |",
        "|---|---|---|",
    ]
    definitions = {
        "base_basic": "plain GPT-2 full fine-tuning on official 131 train",
        "base_plus_extra": "plain GPT-2 full fine-tuning on official 131 + strict extra 497",
        "sft_plus_extra": "prompt-focused SFT on official 131 + strict extra 497",
        "dapt_plus_extra": "DAPT-only checkpoint on official 131 + strict extra 497, evaluated directly",
        "selected_lora_plus_extra": f"LoRA-SFT from better non-DPO checkpoint: {selected_source}",
        "dapt_sft_lora_dpo_best_chrf": "DAPT -> SFT -> form/rhyme LoRA-DPO, best dev chrF checkpoint",
    }
    for name, desc in definitions.items():
        lines.append(f"| `{name}` | {desc} | `{run_meta.get(name, {}).get('checkpoint', '')}` |")

    lines.extend([
        "",
        "## Dev Evaluation",
        "",
        "| model | chrF | Sonnet-or-Not | form | lexical diversity | overall quality | Theme | POEMetric |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in load_rows(dev_eval):
        lines.append(
            "| {name} | {chrf} | {sonnet} | {form} | {lexical} | {overall} | {theme} | {poemetric} |".format(
                name=row.get("model_name"),
                chrf=fmt(row.get("chrF")),
                sonnet=fmt(row.get("sonnet_or_not_bot_pass_rate")),
                form=fmt(row.get("sonnet_form_accuracy")),
                lexical=fmt(row.get("lexical_diversity")),
                overall=fmt(row.get("poemetric_overall_quality_proxy")),
                theme=fmt(row.get("prompt_continuation_theme_overlap")),
                poemetric=fmt(row.get("POEMetric_proxy")),
            )
        )

    lines.extend([
        "",
        "## Test Evaluation",
        "",
        "| model | chrF | Sonnet-or-Not | form | lexical diversity | overall quality | Theme | POEMetric |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in load_rows(test_eval):
        lines.append(
            "| {name} | {chrf} | {sonnet} | {form} | {lexical} | {overall} | {theme} | {poemetric} |".format(
                name=row.get("model_name"),
                chrf=fmt(row.get("chrF")),
                sonnet=fmt(row.get("sonnet_or_not_bot_pass_rate")),
                form=fmt(row.get("sonnet_form_accuracy")),
                lexical=fmt(row.get("lexical_diversity")),
                overall=fmt(row.get("poemetric_overall_quality_proxy")),
                theme=fmt(row.get("prompt_continuation_theme_overlap")),
                poemetric=fmt(row.get("POEMetric_proxy")),
            )
        )

    (OUT_ROOT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_ROOT / "runs.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()

    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpus) < 4:
        raise ValueError("--gpus must provide four GPU ids, e.g. 0,1,2,3")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    first_wave_specs = [
        (
            "base_basic",
            BASIC_TRAIN,
            OUT_ROOT / "base_basic",
            gpus[0],
            10,
            "2e-05",
            "1.0",
            "1.0",
            "1",
            "0.0",
            "0.0",
        ),
        (
            "base_plus_extra",
            TRAIN_EXTRA,
            OUT_ROOT / "base_plus_extra",
            gpus[1],
            10,
            "2e-05",
            "1.0",
            "1.0",
            "1",
            "0.0",
            "0.0",
        ),
        (
            "sft_plus_extra",
            TRAIN_EXTRA,
            OUT_ROOT / "sft_plus_extra",
            gpus[2],
            10,
            "1e-05",
            "0.35",
            "1.2",
            "2",
            "2.0",
            "4.0",
        ),
        (
            "dapt_plus_extra",
            TRAIN_EXTRA,
            OUT_ROOT / "dapt_plus_extra",
            gpus[3],
            3,
            "5e-06",
            "1.0",
            "1.0",
            "2",
            "1.0",
            "2.0",
        ),
    ]

    runs: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_sonnet_generation_stage, *spec, force=args.force)
            for spec in first_wave_specs
        ]
        for future in as_completed(futures):
            runs.append(future.result())

    interim_dev_eval = eval_multi("dev", [run for run in runs if run["name"] in {"sft_plus_extra", "dapt_plus_extra"}])
    dev_chrf = metric_by_name(interim_dev_eval, "chrF")
    selected_source = "sft_plus_extra" if dev_chrf.get("sft_plus_extra", -1.0) >= dev_chrf.get("dapt_plus_extra", -1.0) else "dapt_plus_extra"
    selected_checkpoint = Path(next(run["checkpoint"] for run in runs if run["name"] == selected_source))

    selected_lora_future = None
    dapt_sft_future = None
    dapt_checkpoint = Path(next(run["checkpoint"] for run in runs if run["name"] == "dapt_plus_extra"))

    with ThreadPoolExecutor(max_workers=2) as executor:
        selected_lora_future = executor.submit(
            run_lora_stage,
            "selected_lora_plus_extra",
            selected_checkpoint,
            OUT_ROOT / "selected_lora_plus_extra",
            gpus[0],
            args.force,
        )
        dapt_sft_future = executor.submit(
            run_sonnet_generation_stage,
            "dapt_sft_intermediate",
            TRAIN_EXTRA,
            OUT_ROOT / "dapt_sft_intermediate",
            gpus[2],
            10,
            "1e-05",
            "0.35",
            "1.2",
            "2",
            "2.0",
            "4.0",
            "120",
            dapt_checkpoint,
            args.force,
        )
        lora_run = selected_lora_future.result()
        dapt_sft_run = dapt_sft_future.result()
        runs.append(lora_run)

    dpo_run = run_form_dpo_stage(
        "dapt_sft_lora_dpo_best_chrf",
        Path(dapt_sft_run["checkpoint"]),
        OUT_ROOT / "dapt_sft_lora_dpo_best_chrf",
        gpus[3],
        args.force,
    )
    runs.append(dpo_run)

    ordered_names = [
        "base_basic",
        "base_plus_extra",
        "sft_plus_extra",
        "dapt_plus_extra",
        "selected_lora_plus_extra",
        "dapt_sft_lora_dpo_best_chrf",
    ]
    ordered_runs = [next(run for run in runs if run["name"] == name) for name in ordered_names]
    dev_eval = eval_multi("dev", ordered_runs)
    test_eval = eval_multi("test", ordered_runs)
    write_summary(dev_eval, test_eval, ordered_runs, selected_source)
    print(f"[done] {OUT_ROOT / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
