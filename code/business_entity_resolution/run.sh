#!/usr/bin/env bash
# ONE command to run the whole Business Entity Resolution pipeline.
#
#   bash run.sh mini      # 15 realistic records (fast end-to-end check)
#   bash run.sh full      # the real dataset
#
# Optional environment overrides:
#   SKIP_SETUP=1        skip dependency installation (already provisioned)
#   SKIP_EMBED=1        skip the BGE-M3 embedding stage
#   VALIDATOR=/path     run the official validate_submission.py at the end
#   PYTHON=python3.11   choose a different interpreter for the venv
#
# It provisions a virtualenv, installs dependencies, then calls
# run_pipeline.py which runs every stage (resumable).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PROFILE="${1:-mini}"
if [ "$PROFILE" != "mini" ] && [ "$PROFILE" != "full" ]; then
  echo "Usage: bash run.sh [mini|full]" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ "${SKIP_SETUP:-0}" != "1" ]; then
  echo "==> Provisioning environment (set SKIP_SETUP=1 to skip)"
  if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip wheel setuptools
  python -m pip install -r requirements.txt -r requirements-remote.txt
else
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

# Keep artifacts on disk between runs so crashes resume instead of restarting.
export PYTHONPATH="${HERE}/src:${PYTHONPATH:-}"

EMBED_FLAG=""
if [ "${SKIP_EMBED:-0}" = "1" ]; then
  EMBED_FLAG="--skip-embeddings"
fi

VALIDATOR_FLAG=""
if [ -n "${VALIDATOR:-}" ]; then
  VALIDATOR_FLAG="--validator ${VALIDATOR}"
fi

# shellcheck disable=SC2086
python run_pipeline.py --profile "$PROFILE" $EMBED_FLAG $VALIDATOR_FLAG
