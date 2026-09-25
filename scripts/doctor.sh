#!/usr/bin/env bash
# Environment and data health check. Non-zero exit means something is broken.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
DATA="${BER_DATA_ROOT:-$ROOT/dataset}"
CANONICAL="$ROOT/6ab10eb3b23ba_student_resource/student_resource/dataset"

FAIL=0
WARN=0
ok()   { echo "  ok    $*"; }
warn() { echo "  warn  $*"; WARN=$((WARN + 1)); }
bad()  { echo "  FAIL  $*"; FAIL=$((FAIL + 1)); }

echo "==> Python"
if command -v python3 >/dev/null 2>&1; then
  if python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    ok "python3 $(python3 -c 'import platform; print(platform.python_version())')"
  else
    bad "python3 is older than 3.10"
  fi
else
  bad "python3 not found"
fi

echo "==> git-lfs"
if command -v git-lfs >/dev/null 2>&1; then
  ok "$(git-lfs version)"
else
  bad "git-lfs not installed (needed for 'make data')"
fi

echo "==> virtualenv"
if [ -x "$PY" ]; then
  ok "$("$PY" -c 'import sys; print(sys.version.split()[0])') ($VENV)"
  if "$PY" - <<'PY' >/dev/null 2>&1
import importlib
for name in ("numpy", "pandas", "sklearn", "rapidfuzz", "pyarrow"):
    importlib.import_module(name)
PY
  then
    ok "core dependencies importable"
  else
    bad "core dependencies missing; run 'make setup'"
  fi
  "$PY" - <<'PY' >/dev/null 2>&1 && ok "lightgbm available" || warn "lightgbm missing (requirements-remote.txt)"
import lightgbm  # noqa: F401
PY
  "$PY" - <<'PY' >/dev/null 2>&1 && ok "embedding backend available (torch)" || warn "torch/BGE-M3 unavailable (text features only)"
import torch  # noqa: F401
PY
else
  bad "virtualenv missing at $VENV; run 'make setup'"
fi

echo "==> config & package"
if PYTHONPATH="$ROOT/src" "$PY" -c "from business_entity_resolution.config import ProjectConfig; ProjectConfig.load('$ROOT/configs/remote_full.json')" >/dev/null 2>&1; then
  ok "package imports and configs parse"
else
  warn "could not import package / parse config (venv may be absent)"
fi

echo "==> dataset"
check_data() {
  local label="$1" dir="$2"
  local train="$dir/train/train_source1.tsv"
  if [ ! -f "$train" ]; then
    warn "$label: no train_source1.tsv at $dir"
    return
  fi
  if head -c 40 "$train" 2>/dev/null | grep -q 'version https://git-lfs'; then
    warn "$label: Git-LFS pointer stubs at $dir (run 'make data')"
  else
    ok "$label: real data at $dir"
  fi
}
check_data "working root" "$DATA"
[ "$CANONICAL" != "$DATA" ] && check_data "original bundle" "$CANONICAL"

echo
if [ "$FAIL" -gt 0 ]; then
  echo "doctor: $FAIL failure(s), $WARN warning(s)."
  exit 1
fi
echo "doctor: all checks passed ($WARN warning(s))."
