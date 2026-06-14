#!/usr/bin/env python3
"""Sync the canonical strict-497 sonnet data into legacy trial1-style paths.

The canonical sonnet-generation data lives under `sonnet_project/data/`. This
helper writes a compatibility copy under `sonnet_project/legacy_trial1_data/`
without depending on the removed legacy 519-extra data.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ROOT = PROJECT_ROOT.parent
SONNET_DATA = PROJECT_ROOT / "data"
TRIAL_DATA = PROJECT_ROOT / "legacy_trial1_data"

COPY_MAP = {
    SONNET_DATA / "basic/train_131.txt": TRIAL_DATA / "basic/sonnets_train.txt",
    SONNET_DATA / "basic/dev_prompts_12.txt": TRIAL_DATA / "basic/sonnets_held_out_dev.txt",
    SONNET_DATA / "basic/dev_gold_12.txt": TRIAL_DATA / "basic/TRUE_sonnets_held_out_dev.txt",
    SONNET_DATA / "basic/test_prompts_12.txt": TRIAL_DATA / "basic/sonnets_held_out.txt",
    SONNET_DATA / "strict_497/official_train_131.txt": TRIAL_DATA / "basic_plus_extra_strict/official_sonnets_train.txt",
    SONNET_DATA / "strict_497/extra_train_strict_497.txt": TRIAL_DATA / "extra_strict/poetryeval_poemetric_sonnets_strict_497.txt",
    SONNET_DATA / "strict_497/train_official_131_plus_extra_497_total_628.txt": TRIAL_DATA / "basic_plus_extra_strict/sonnets_train_plus_extra_strict_497.txt",
    SONNET_DATA / "strict_497/dev_prompts_12.txt": TRIAL_DATA / "basic_plus_extra_strict/sonnets_held_out_dev.txt",
    SONNET_DATA / "strict_497/dev_gold_12.txt": TRIAL_DATA / "basic_plus_extra_strict/TRUE_sonnets_held_out_dev.txt",
    SONNET_DATA / "strict_497/test_prompts_12.txt": TRIAL_DATA / "basic_plus_extra_strict/sonnets_held_out.txt",
}


def count_numbered_blocks(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return len(re.findall(r"(?m)^\s*\d+\s*$", text))


def main() -> None:
    if not SONNET_DATA.exists():
        raise FileNotFoundError(f"Missing canonical data directory: {SONNET_DATA}")

    for src, dst in COPY_MAP.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing source file: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    manifest_src = SONNET_DATA / "docs/strict_497_manifest.md"
    manifest_dst = TRIAL_DATA / "MANIFEST_STRICT_497.md"
    if manifest_src.exists():
        shutil.copy2(manifest_src, manifest_dst)

    print("Synced canonical sonnet data into trial1_data compatibility paths.")
    for src in sorted(COPY_MAP):
        print(f"{src.relative_to(ROOT)}: {count_numbered_blocks(src)}")


if __name__ == "__main__":
    main()
