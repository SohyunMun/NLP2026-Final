#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def default_project_root() -> Path:
    # Local layout: NLP2026-Final/github_ready/code/<script>.py
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "sonnet_generation.py").exists():
        return candidate
    return Path.cwd()


def parse_common_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--project_root", type=Path, default=default_project_root())
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value. Use empty string for CPU.")
    parser.add_argument("--no_gpu", action="store_true", help="Do not pass --use_gpu.")
    parser.add_argument("--dry_run", action="store_true", help="Print command without running it.")
    parser.add_argument("--skip_training", action="store_true",
                        help="For reranking scripts, reuse the listed checkpoint instead of retraining it first.")
    return parser.parse_args()


def run_python(project_root: Path, script: str, args: list[str], gpu: str = "0", no_gpu: bool = False, dry_run: bool = False):
    cmd = [sys.executable, script]
    if not no_gpu:
        cmd.append("--use_gpu")
    cmd.extend(args)
    printable = " ".join(cmd)
    print(f"[run] cd {project_root}")
    print(f"[run] CUDA_VISIBLE_DEVICES={gpu!r} {printable}")
    if dry_run:
        return
    env = None
    if gpu != "":
        import os
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
    subprocess.run(cmd, cwd=project_root, env=env, check=True)
