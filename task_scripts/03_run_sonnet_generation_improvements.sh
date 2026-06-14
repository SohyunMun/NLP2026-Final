#!/usr/bin/env bash
set -euo pipefail

# Run the separated sonnet-generation improvement workflow.
# This uses sonnet_project/scripts instead of changing the official root
# sonnet_generation.py command-line interface.
#
# Usage:
#   bash task_scripts/03_run_sonnet_generation_improvements.sh
#   bash task_scripts/03_run_sonnet_generation_improvements.sh --dry-run
#
# Optional environment variables:
#   GPUS=0,1,2,3     GPU ids for the six-way runner.
#   FORCE=1          Re-run stages even if outputs already exist.
#   PYTHON_BIN=...   Python executable to use.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-0,1,2,3}"
FORCE="${FORCE:-0}"

cd "$REPO_ROOT"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '1,18p' "$0"
  exit 0
fi

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

cmd=(
  "$PYTHON_BIN"
  "sonnet_project/scripts/run_sixway_sonnet_ablation.py"
  "--gpus"
  "$GPUS"
)

if [[ "$FORCE" == "1" ]]; then
  cmd+=("--force")
fi

echo "[run] Sonnet generation improvement workflow"
echo "[cmd] ${cmd[*]}"
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
"${cmd[@]}"

echo "[done] Improvement workflow finished."
