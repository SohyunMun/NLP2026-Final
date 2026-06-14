#!/usr/bin/env bash
set -euo pipefail

# Run the official baseline sonnet generation entry.
# This intentionally calls the root sonnet_generation.py file.
# The root file keeps the starter-code command-line interface unchanged.
#
# Usage:
#   bash task_scripts/02_run_sonnet_generation_baseline.sh
#   bash task_scripts/02_run_sonnet_generation_baseline.sh --use_gpu
#   bash task_scripts/02_run_sonnet_generation_baseline.sh --dry-run --use_gpu
#
# Any extra arguments are forwarded to sonnet_generation.py.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO_ROOT"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '1,15p' "$0"
  exit 0
fi

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

echo "[run] Official baseline sonnet generation"
echo "[cmd] $PYTHON_BIN sonnet_generation.py $*"
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
"$PYTHON_BIN" sonnet_generation.py "$@"

echo "[done] Baseline sonnet generation finished."
