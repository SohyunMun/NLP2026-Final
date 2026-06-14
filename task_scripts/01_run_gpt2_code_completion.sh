#!/usr/bin/env bash
set -euo pipefail

# Run checks for "GPT-2 code completion".
# Usage:
#   bash task_scripts/01_run_gpt2_code_completion.sh
#   bash task_scripts/01_run_gpt2_code_completion.sh --full-classifier
#   bash task_scripts/01_run_gpt2_code_completion.sh --dry-run
#
# The default mode runs the official lightweight implementation checks:
# optimizer_test.py and sanity_check.py.
# The optional --full-classifier mode also runs classifier.py with its own
# default command-line options.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO_ROOT"

RUN_CLASSIFIER=0
DRY_RUN=0

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '1,16p' "$0"
  exit 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

if [[ "${1:-}" == "--full-classifier" ]]; then
  RUN_CLASSIFIER=1
  shift
fi

echo "[1/3] Running optimizer_test.py"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[cmd] $PYTHON_BIN optimizer_test.py"
else
"$PYTHON_BIN" optimizer_test.py
fi

echo "[2/3] Running sanity_check.py"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[cmd] $PYTHON_BIN sanity_check.py"
else
"$PYTHON_BIN" sanity_check.py
fi

if [[ "$RUN_CLASSIFIER" == "1" ]]; then
  echo "[3/3] Running classifier.py with official default arguments"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[cmd] $PYTHON_BIN classifier.py $*"
  else
  "$PYTHON_BIN" classifier.py "$@"
  fi
else
  echo "[3/3] Skipping classifier.py full training."
  echo "      To run it, use: bash task_scripts/01_run_gpt2_code_completion.sh --full-classifier"
fi

echo "[done] GPT-2 code completion checks finished."
