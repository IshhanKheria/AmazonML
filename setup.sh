#!/usr/bin/env bash
# Provision the Business Entity Resolution pipeline on a fresh Linux machine.
#
# Usage:
#   bash setup.sh                # CPU-only or auto-detected CUDA
#   CUDA=cu121 bash setup.sh     # force a specific torch CUDA wheel channel
#   NO_EMBED=1 bash setup.sh     # skip torch/BGE-M3 (text features only)
#
# The script is idempotent: re-running it reuses the virtualenv and only
# installs what is missing.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "==> Checking Python"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"; print("Python", sys.version.split()[0])'

echo "==> Creating virtualenv at $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip tooling"
python -m pip install --upgrade pip wheel setuptools

echo "==> Installing core + dev requirements"
python -m pip install -r requirements.txt -r requirements-dev.txt

echo "==> Installing LightGBM backend"
python -m pip install -r requirements-remote.txt

if [ "${NO_EMBED:-0}" != "1" ]; then
  echo "==> Installing PyTorch"
  if [ -n "${CUDA:-}" ]; then
    python -m pip install torch --index-url "https://download.pytorch.org/whl/${CUDA}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    echo "    nvidia-smi detected; installing CUDA 12.1 wheels"
    python -m pip install torch --index-url "https://download.pytorch.org/whl/cu121"
  else
    echo "    no GPU detected; installing CPU wheels"
    python -m pip install torch --index-url "https://download.pytorch.org/whl/cpu"
  fi
  echo "==> Installing BGE-M3 runtime (transformers / FlagEmbedding)"
  python -m pip install "transformers==4.46.3" "FlagEmbedding==1.3.4"
else
  echo "==> Skipping torch/BGE-M3 (NO_EMBED=1)"
fi

echo "==> Installing the package in editable mode"
python -m pip install -e .

echo "==> Sanity check"
python - <<'PY'
import importlib
missing = []
for name in ["numpy", "pandas", "sklearn", "rapidfuzz", "pyarrow", "psutil", "lightgbm"]:
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{name}: {exc}")
try:
    import torch, transformers  # noqa: F401
    gpu = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    torch {torch.__version__} (device={gpu})")
except Exception:
    print("    embedding backend unavailable (pipeline will run without BGE-M3 features)")
if missing:
    raise SystemExit("Missing required packages: " + "; ".join(missing))
print("    core dependencies OK")
PY

cat <<'EOF'

==> Setup complete. Activate the environment and set paths before running:

    source .venv/bin/activate
    export BER_DATA_ROOT=/path/to/dataset
    export BER_ARTIFACT_ROOT=/path/to/artifacts
    export BER_OUTPUT_ROOT=/path/to/output
    export BER_THREADS=$(nproc)

Then run the mini cycle (see README.md) or the full pipeline with
configs/remote_full.json. Every long stage resumes automatically if it stops.
EOF
