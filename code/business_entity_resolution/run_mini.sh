#!/usr/bin/env bash
# Run the full mini end-to-end cycle (15 realistic records), resumable.
#
# Usage:
#   bash run_mini.sh
#   VALIDATOR=/path/to/validate_submission.py bash run_mini.sh
#
# Requires setup.sh to have been run and the virtualenv activated.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CONFIG="${CONFIG:-configs/mini.json}"
VALIDATOR="${VALIDATOR:-}"

run() { echo; echo "+ python -m business_entity_resolution $*"; python -m business_entity_resolution "$@"; }

run make-mini            --config "$CONFIG"
run prepare              --config "$CONFIG" --split both
run make-splits          --config "$CONFIG" --folds 5
run generate-candidates  --config "$CONFIG" --split train
run generate-candidates  --config "$CONFIG" --split test
run embed                --config "$CONFIG" --split both
run build-features       --config "$CONFIG" --split train
run build-features       --config "$CONFIG" --split test
run train                --config "$CONFIG" --model lightgbm --all-training-data
run tune-decision        --config "$CONFIG"
run evaluate             --config "$CONFIG"
run infer                --config "$CONFIG" --model lightgbm

if [ -n "$VALIDATOR" ]; then
  run preflight --config "$CONFIG" --official-validator "$VALIDATOR" --check-ids
fi

echo
echo "Mini cycle complete. Outputs:"
ls -la output/mini/ 2>/dev/null || true
