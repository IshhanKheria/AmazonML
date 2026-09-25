#!/usr/bin/env bash
# Fetch the real challenge TSVs from Git LFS into the working data root.
#
# Fetches origin/main into dataset/{train,test}/ and verifies the result is real
# data rather than a Git-LFS pointer stub. Re-running is cheap: git-lfs caches
# objects, so only the smudge step repeats.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
TARGET="${BER_DATA_ROOT:-$ROOT/dataset}"
SRC_PREFIX="6ab10eb3b23ba_student_resource/student_resource/dataset"

FILES=(
  train/train_source1.tsv
  train/train_source2.tsv
  train/train_source3.tsv
  train/train_ground_truth.tsv
  test/test_source1.tsv
  test/test_source2.tsv
  test/test_source3.tsv
)

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs is not installed. Install it (e.g. 'sudo pacman -S git-lfs') and retry." >&2
  exit 1
fi

echo "==> Enabling git-lfs"
git -C "$ROOT" lfs install --local
echo "==> Fetching LFS objects (origin main)"
git -C "$ROOT" lfs fetch origin main --include="$SRC_PREFIX/**"

mkdir -p "$TARGET/train" "$TARGET/test"
for rel in "${FILES[@]}"; do
  dest="$TARGET/$rel"
  echo "==> Smudging $rel"
  # `git show` emits the pointer blob; `git lfs smudge` converts it to content.
  git -C "$ROOT" show "origin/main:$SRC_PREFIX/$rel" | git -C "$ROOT" lfs smudge > "$dest"
done

failed=0
for rel in "${FILES[@]}"; do
  path="$TARGET/$rel"
  if [ ! -f "$path" ] || head -c 40 "$path" | grep -q 'version https://git-lfs'; then
    echo "  FAIL  $rel" >&2
    failed=$((failed + 1))
  else
    echo "  ok    $rel  ($(du -h "$path" | cut -f1))"
  fi
done
if [ "$failed" -ne 0 ]; then
  echo "Dataset verification failed: $failed file(s) still missing or pointer stubs." >&2
  exit 1
fi
echo "Dataset ready at $TARGET"
