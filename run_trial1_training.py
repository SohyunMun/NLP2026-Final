#!/usr/bin/env python3
"""Run first-trial sonnet-generation training on aligned MSK data groups.

This runner intentionally uses the existing `busi2` environment and never
modifies Python, PyTorch, CUDA, or system GPU packages.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from sacrebleu.metrics import CHRF


MSK_ROOT = Path("/home/msko021220/nlp2026-final-MSK")
SHM_ROOT = Path("/home/msko021220/nlp2026-final-SHM")
HUJ_ROOT = Path("/home/msko021220/nlp2026-final-HUJ")
PYTHON = Path("/home/msko021220/.conda/envs/busi2/bin/python")
DATA_ROOT = MSK_ROOT / "sonnet_data"
EXP_ROOT = MSK_ROOT / "experiments" / "trial1"
FINETUNE_EPOCHS = "10"
DAPT_EPOCHS = "10"
PATIENCE = "10"


@dataclass(frozen=True)
class DataGroup:
    name: str
    train: Path
    dev_prompt: Path
    dev_gold: Path
    test_prompt: Path
    dapt_corpus: Path | None = None


GROUPS = {
    "basic": DataGroup(
        name="basic",
        train=DATA_ROOT / "basic" / "train_131.txt",
        dev_prompt=DATA_ROOT / "basic" / "dev_prompts_12.txt",
        dev_gold=DATA_ROOT / "basic" / "dev_gold_12.txt",
        test_prompt=DATA_ROOT / "basic" / "test_prompts_12.txt",
    ),
    "basic_plus_extra": DataGroup(
        name="basic_plus_extra",
        train=DATA_ROOT / "strict_497" / "train_official_131_plus_extra_497_total_628.txt",
        dev_prompt=DATA_ROOT / "strict_497" / "dev_prompts_12.txt",
        dev_gold=DATA_ROOT / "strict_497" / "dev_gold_12.txt",
        test_prompt=DATA_ROOT / "strict_497" / "test_prompts_12.txt",
        dapt_corpus=DATA_ROOT / "strict_497" / "train_official_131_plus_extra_497_total_628.txt",
    ),
}


@dataclass(frozen=True)
class RunSpec:
    name: str
    group: str
    family: str
    script: Path
    args: tuple[str, ...]
    batch_size: int
    lr: str


def make_runs() -> list[RunSpec]:
    runs: list[RunSpec] = []

    for group in ("basic", "basic_plus_extra"):
        runs.append(RunSpec(
            name=f"{group}__msk_sft_weighted_mbr",
            group=group,
            family="MSK SFT + weighted loss + MBR/rerank generation",
            script=MSK_ROOT / "sonnet_generation.py",
            args=(
                "--epochs", FINETUNE_EPOCHS,
                "--batch_size", "8",
                "--lr", "1e-5",
                "--weight_decay", "0.01",
                "--eval_every", "1",
                "--patience", PATIENCE,
                "--selection_metric", "loss",
                "--num_candidates", "2",
                "--dev_num_candidates", "1",
                "--max_generation_tokens", "120",
                "--model_score_weight", "2.0",
                "--mbr_weight", "4.0",
            ),
            batch_size=8,
            lr="1e-5",
        ))
        runs.append(RunSpec(
            name=f"{group}__shm_baseline_full_ft",
            group=group,
            family="SHM baseline full fine-tuning",
            script=SHM_ROOT / "sonnet_generation.py",
            args=(
                "--variation", "baseline",
                "--epochs", FINETUNE_EPOCHS,
                "--batch_size", "8",
                "--lr", "2e-5",
                "--patience", PATIENCE,
            ),
            batch_size=8,
            lr="2e-5",
        ))

    runs.extend([
        RunSpec(
            name="basic_plus_extra__shm_lora",
            group="basic_plus_extra",
            family="SHM LoRA fine-tuning",
            script=SHM_ROOT / "sonnet_generation.py",
            args=(
                "--variation", "lora",
                "--epochs", FINETUNE_EPOCHS,
                "--batch_size", "8",
                "--lr", "1e-4",
                "--patience", PATIENCE,
                "--lora_r", "8",
                "--lora_alpha", "16",
            ),
            batch_size=8,
            lr="1e-4",
        ),
        RunSpec(
            name="basic_plus_extra__shm_dapt_lora",
            group="basic_plus_extra",
            family="SHM DAPT + LoRA fine-tuning",
            script=SHM_ROOT / "sonnet_generation.py",
            args=(
                "--variation", "dapt_lora",
                "--pretrain_epochs", DAPT_EPOCHS,
                "--pretrain_lr", "1e-5",
                "--pretrain_corpus_path", str(GROUPS["basic_plus_extra"].dapt_corpus),
                "--chunk_size", "256",
                "--epochs", FINETUNE_EPOCHS,
                "--batch_size", "8",
                "--lr", "1e-4",
                "--patience", PATIENCE,
                "--lora_r", "8",
                "--lora_alpha", "16",
            ),
            batch_size=8,
            lr="1e-4",
        ),
        RunSpec(
            name="basic_plus_extra__shm_prefix",
            group="basic_plus_extra",
            family="SHM DAPT + LoRA + Prefix tuning",
            script=SHM_ROOT / "sonnet_generation.py",
            args=(
                "--variation", "prefix",
                "--pretrain_epochs", DAPT_EPOCHS,
                "--pretrain_lr", "1e-5",
                "--pretrain_corpus_path", str(GROUPS["basic_plus_extra"].dapt_corpus),
                "--chunk_size", "256",
                "--epochs", FINETUNE_EPOCHS,
                "--batch_size", "8",
                "--lr", "5e-5",
                "--patience", PATIENCE,
                "--lora_r", "8",
                "--lora_alpha", "16",
                "--prefix_len", "20",
            ),
            batch_size=8,
            lr="5e-5",
        ),
        RunSpec(
            name="basic__huj_baseline_guided",
            group="basic",
            family="HUJ guided baseline",
            script=HUJ_ROOT / "sonnet_baseline.py",
            args=(
                "--epochs", FINETUNE_EPOCHS,
                "--batch_size", "8",
                "--lr", "1e-5",
                "--patience", PATIENCE,
            ),
            batch_size=8,
            lr="1e-5",
        ),
        RunSpec(
            name="basic__huj_dpo",
            group="basic",
            family="HUJ DPO",
            script=HUJ_ROOT / "sonnet_dpo.py",
            args=(
                "--epochs", FINETUNE_EPOCHS,
                "--batch_size", "4",
                "--lr", "5e-6",
                "--dpo_beta", "0.1",
                "--patience", PATIENCE,
            ),
            batch_size=4,
            lr="5e-6",
        ),
        RunSpec(
            name="basic_plus_extra__huj_lora_dpo",
            group="basic_plus_extra",
            family="HUJ LoRA + DPO",
            script=HUJ_ROOT / "sonnet_lora_dpo.py",
            args=(
                "--epochs", FINETUNE_EPOCHS,
                "--batch_size", "4",
                "--lr", "1e-4",
                "--lora_r", "8",
                "--lora_alpha", "16",
                "--dpo_beta", "0.1",
                "--patience", PATIENCE,
            ),
            batch_size=4,
            lr="1e-4",
        ),
        RunSpec(
            name="basic_plus_extra__huj_peft_dpo",
            group="basic_plus_extra",
            family="HUJ LoRA + Prefix PEFT + DPO",
            script=HUJ_ROOT / "sonnet_peft_dpo.py",
            args=(
                "--epochs", FINETUNE_EPOCHS,
                "--batch_size", "4",
                "--lr", "5e-5",
                "--lora_r", "8",
                "--lora_alpha", "16",
                "--prefix_len", "8",
                "--dpo_beta", "0.1",
                "--patience", PATIENCE,
            ),
            batch_size=4,
            lr="5e-5",
        ),
    ])

    return runs


def split_numbered_blocks(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\d+\s*\n", text)
    return [block.strip() for block in blocks[1:] if block.strip()]


def evaluate_chrf(prediction_path: Path, gold_path: Path) -> float | None:
    if not prediction_path.exists() or not gold_path.exists():
        return None
    preds = split_numbered_blocks(prediction_path.read_text(encoding="utf-8", errors="ignore"))
    golds = split_numbered_blocks(gold_path.read_text(encoding="utf-8", errors="ignore"))
    max_len = min(len(preds), len(golds))
    if max_len == 0:
        return None
    return float(CHRF().corpus_score(preds[:max_len], [golds[:max_len]]).score)


def find_prediction_path(run_dir: Path) -> Path:
    preferred = run_dir / "predictions" / "dev_generated_sonnets.txt"
    if preferred.exists():
        return preferred
    candidates = sorted((run_dir / "predictions").glob("*generated_sonnets.txt"))
    return candidates[0] if candidates else preferred


def parse_best_loss(log_path: Path) -> tuple[int | None, float | None, str | None]:
    if not log_path.exists():
        return None, None, None

    val_records: list[tuple[int, float]] = []
    train_records: list[tuple[int, float]] = []
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    number = r"([0-9]+(?:\.[0-9]+)?)"

    for match in re.finditer(r"Epoch\s+(\d+):.*?val loss\s*::\s*" + number, text):
        val_records.append((int(match.group(1)), float(match.group(2))))
    for match in re.finditer(r"Epoch\s+(\d+):.*?Val Loss\s*::\s*" + number, text):
        val_records.append((int(match.group(1)), float(match.group(2))))

    for match in re.finditer(r"Epoch\s+(\d+):\s*train loss\s*::\s*" + number, text):
        train_records.append((int(match.group(1)), float(match.group(2))))
    for match in re.finditer(r"Epoch\s+(\d+):\s*train loss\s*=\s*" + number, text):
        train_records.append((int(match.group(1)), float(match.group(2))))

    if val_records:
        epoch, loss = min(val_records, key=lambda item: item[1])
        return epoch, loss, "val"
    if train_records:
        epoch, loss = min(train_records, key=lambda item: item[1])
        return epoch, loss, "train"
    return None, None, None


def ensure_data_link(run_dir: Path, group: DataGroup) -> None:
    data_dir = run_dir / "data"
    if data_dir.is_symlink():
        data_dir.unlink()
    elif data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.symlink_to(group.dev_gold.parent, target_is_directory=True)


def command_for_run(spec: RunSpec, run_dir: Path, use_gpu: bool) -> list[str]:
    group = GROUPS[spec.group]
    out_path = run_dir / "predictions" / "dev_generated_sonnets.txt"
    base = [str(PYTHON), str(spec.script)]

    common = [
        "--sonnet_path", str(group.train),
        "--held_out_sonnet_path", str(group.dev_prompt),
        "--sonnet_out", str(out_path),
        "--model_size", "gpt2",
    ]
    if use_gpu:
        common.append("--use_gpu")

    if spec.script == MSK_ROOT / "sonnet_generation.py":
        common.extend([
            "--dev_sonnet_path", str(group.dev_prompt),
            "--dev_gold_path", str(group.dev_gold),
        ])
    elif spec.script == SHM_ROOT / "sonnet_generation.py":
        common.extend([
            "--dev_sonnet_path", str(group.dev_prompt),
            "--run_name", spec.name,
            "--milestone_model_dir", str(run_dir / "models"),
            "--milestone_pred_dir", str(run_dir / "predictions"),
        ])
    else:
        common.extend([
            "--gold_sonnet_path", str(group.dev_gold),
        ])

    return base + list(spec.args) + common


def write_run_metadata(spec: RunSpec, run_dir: Path, cmd: list[str], gpu: str) -> None:
    group = GROUPS[spec.group]
    metadata = {
        "name": spec.name,
        "group": spec.group,
        "family": spec.family,
        "script": str(spec.script),
        "python": str(PYTHON),
        "gpu": gpu,
        "train": str(group.train),
        "dev_prompt": str(group.dev_prompt),
        "dev_gold": str(group.dev_gold),
        "test_prompt": str(group.test_prompt),
        "dapt_corpus": str(group.dapt_corpus) if group.dapt_corpus else None,
        "batch_size": spec.batch_size,
        "lr": spec.lr,
        "command": cmd,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_one(spec: RunSpec, gpu: str, use_gpu: bool) -> dict[str, object]:
    run_dir = EXP_ROOT / spec.name
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    ensure_data_link(run_dir, GROUPS[spec.group])

    cmd = command_for_run(spec, run_dir, use_gpu)
    write_run_metadata(spec, run_dir, cmd, gpu)
    log_path = run_dir / "train.log"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu

    start = time.time()
    print(f"[start] {spec.name}")
    print(f"        family={spec.family}")
    print(f"        log={log_path}")

    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(cmd, cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT)

    elapsed = time.time() - start
    pred_path = find_prediction_path(run_dir)
    chrf = evaluate_chrf(pred_path, GROUPS[spec.group].dev_gold)
    best_epoch, best_loss, loss_source = parse_best_loss(log_path)
    status = "ok" if proc.returncode == 0 else "failed"
    print(
        f"[done]  {spec.name} status={status} seconds={elapsed:.1f} "
        f"best_{loss_source}_loss={best_loss} epoch={best_epoch} chrf={chrf}"
    )

    return {
        "name": spec.name,
        "group": spec.group,
        "family": spec.family,
        "status": status,
        "returncode": proc.returncode,
        "seconds": round(elapsed, 1),
        "chrf": chrf,
        "run_dir": str(run_dir),
        "log": str(log_path),
        "prediction": str(pred_path),
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "loss_source": loss_source,
    }


def write_summary(rows: list[dict[str, object]]) -> None:
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Trial 1 Training Summary",
        "",
        "| run | group | method | status | selected epoch | loss source | best loss | chrF | seconds |",
        "|---|---|---|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        chrf = row["chrf"]
        chrf_text = "N/A" if chrf is None else f"{float(chrf):.4f}"
        best_epoch = row.get("best_epoch")
        best_loss = row.get("best_loss")
        loss_source = row.get("loss_source") or "N/A"
        epoch_text = "N/A" if best_epoch is None else str(best_epoch)
        loss_text = "N/A" if best_loss is None else f"{float(best_loss):.4f}"
        seconds = row.get("seconds")
        seconds_text = "N/A" if seconds in (None, 0) else str(seconds)
        lines.append(
            f"| {row['name']} | {row['group']} | {row['family']} | "
            f"{row['status']} | {epoch_text} | {loss_source} | {loss_text} | {chrf_text} | {seconds_text} |"
        )
    (EXP_ROOT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_existing_rows() -> list[dict[str, object]]:
    summary_path = EXP_ROOT / "summary.json"
    if not summary_path.exists():
        return []
    try:
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return rows if isinstance(rows, list) else []


def upsert_row(rows: list[dict[str, object]], row: dict[str, object]) -> list[dict[str, object]]:
    merged = [existing for existing in rows if existing.get("name") != row.get("name")]
    merged.append(row)
    merged.sort(key=lambda item: item["name"])
    return merged


def refresh_existing_summary() -> int:
    summary_path = EXP_ROOT / "summary.json"
    if not summary_path.exists():
        print(f"Summary not found: {summary_path}", file=sys.stderr)
        return 2

    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    for row in rows:
        run_dir = Path(str(row["run_dir"]))
        group_name = str(row["group"])
        if group_name not in GROUPS:
            continue
        pred_path = find_prediction_path(run_dir)
        row["prediction"] = str(pred_path)
        row["chrf"] = evaluate_chrf(pred_path, GROUPS[group_name].dev_gold)
        if row.get("status") == "exception" and pred_path.exists():
            row["status"] = "ok"
            row["returncode"] = 0
            if row.get("seconds") == 0:
                row["seconds"] = None
        best_epoch, best_loss, loss_source = parse_best_loss(Path(str(row["log"])))
        row["best_epoch"] = best_epoch
        row["best_loss"] = best_loss
        row["loss_source"] = loss_source
    rows.sort(key=lambda item: item["name"])
    write_summary(rows)
    print(EXP_ROOT / "SUMMARY.md")
    return 0


def main() -> int:
    global EXP_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU ids for parallel execution, e.g. 0,1,2,3.")
    parser.add_argument("--jobs_per_gpu", type=int, default=1, help="Concurrent jobs assigned to each listed GPU.")
    parser.add_argument("--experiment_name", default="trial1", help="Subdirectory under experiments/.")
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="Run only selected run names.")
    parser.add_argument("--list", action="store_true", help="List available runs and exit.")
    parser.add_argument("--summarize_only", action="store_true", help="Refresh summary from existing outputs.")
    parser.add_argument("--stop_on_error", action="store_true")
    args = parser.parse_args()

    runs = make_runs()
    if args.list:
        for spec in runs:
            print(f"{spec.name}\t{spec.group}\t{spec.family}")
        return 0

    if args.only:
        selected = set(args.only)
        runs = [spec for spec in runs if spec.name in selected]
        missing = selected - {spec.name for spec in runs}
        if missing:
            print(f"Unknown run names: {sorted(missing)}", file=sys.stderr)
            return 2

    if not PYTHON.exists():
        print(f"Python environment not found: {PYTHON}", file=sys.stderr)
        return 2

    EXP_ROOT = MSK_ROOT / "experiments" / args.experiment_name
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        return refresh_existing_summary()

    rows: list[dict[str, object]] = load_existing_rows()

    gpu_ids = [args.gpu]
    if args.gpus:
        gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if args.no_gpu:
        gpu_ids = ["cpu"]
    slots = []
    for gpu_id in gpu_ids:
        slots.extend([gpu_id] * max(1, args.jobs_per_gpu))
    if not slots:
        slots = ["0"]

    print(f"[trial] experiment={EXP_ROOT}")
    print(f"[trial] parallel_slots={slots}")

    if len(slots) == 1:
        for spec in runs:
            try:
                row = run_one(spec, gpu=slots[0], use_gpu=not args.no_gpu)
                rows = upsert_row(rows, row)
                write_summary(rows)
                if row["status"] != "ok" and args.stop_on_error:
                    return int(row["returncode"])
            except Exception as exc:
                row = exception_row(spec, exc)
                rows = upsert_row(rows, row)
                write_summary(rows)
                print(f"[error] {spec.name}: {exc}")
                if args.stop_on_error:
                    return 1
        return 0

    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        future_to_spec = {}
        for idx, spec in enumerate(runs):
            gpu_id = slots[idx % len(slots)]
            future = executor.submit(run_one, spec, gpu_id, not args.no_gpu)
            future_to_spec[future] = spec

        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                row = future.result()
            except Exception as exc:
                row = exception_row(spec, exc)
                print(f"[error] {spec.name}: {exc}")
            rows = upsert_row(rows, row)
            write_summary(rows)
            if row["status"] != "ok" and args.stop_on_error:
                return int(row["returncode"] or 1)
    return 0


def exception_row(spec: RunSpec, exc: Exception) -> dict[str, object]:
    return {
        "name": spec.name,
        "group": spec.group,
        "family": spec.family,
        "status": "exception",
        "returncode": None,
        "seconds": 0,
        "chrf": None,
        "run_dir": str(EXP_ROOT / spec.name),
        "log": str(EXP_ROOT / spec.name / "train.log"),
        "prediction": str(EXP_ROOT / spec.name / "predictions" / "dev_generated_sonnets.txt"),
        "error": repr(exc),
        "best_epoch": None,
        "best_loss": None,
        "loss_source": None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
