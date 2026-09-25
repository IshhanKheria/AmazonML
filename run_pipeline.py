#!/usr/bin/env python3
"""Single entry point to run the entire Business Entity Resolution pipeline.

You run this one file; it runs every stage in order and resumes automatically
if anything stops. Choose the profile with a flag:

    python run_pipeline.py --profile mini     # 15 realistic records
    python run_pipeline.py --profile full     # the real dataset

Environment (all optional; sensible defaults shown):

    BER_DATA_ROOT      dataset location        (mini: dataset/mini, full: dataset)
    BER_ARTIFACT_ROOT  intermediate artifacts   (default: artifacts/<profile>)
    BER_OUTPUT_ROOT    final TSV output dir     (default: output/<profile>)
    BER_THREADS        worker threads           (default: all CPUs - 1)

Everything is idempotent: re-running skips completed work. Per-stage logs go to
<artifact_root>/<run_id>/logs/. The final files are
output/<profile>/matching_results.tsv and candidate_pairs.tsv.

Optionally validate against the official validator:

    python run_pipeline.py --profile mini --validator /path/to/validate_submission.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Make the package importable without an editable install.
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from business_entity_resolution.cli import main as cli_main  # noqa: E402

PROFILES = {
    "mini": {
        "config": "configs/mini.json",
        "data_root": "dataset/mini",
        "artifact_root": "artifacts/mini",
        "output_root": "output/mini",
        "folds": 5,
        "model": "lightgbm",
    },
    "full": {
        "config": "configs/remote_full.json",
        "data_root": "dataset",
        "artifact_root": "artifacts/full",
        "output_root": "output",
        "folds": 10,
        "model": "lightgbm",
    },
}


def _sh(args: list[str]) -> int:
    print(f"\n$ {' '.join(args)}", flush=True)
    return int(cli_main(args))


def run_profile(profile: str, validator: str | None, skip_embeddings: bool) -> int:
    spec = PROFILES[profile]
    config = spec["config"]

    # Resolve the data root. Mini lives in the package; full may live in the
    # repo-root dataset/ or the original student_resource bundle, so auto-detect.
    if profile == "full" and "BER_DATA_ROOT" not in os.environ:
        real = _find_real_dataset()
        if real is None:
            print("Could not find the real dataset. Set BER_DATA_ROOT (or "
                  "BER_SOURCE_DATA_ROOT) to the folder containing train/ and test/ "
                  "TSVs, or run 'make data'.", file=sys.stderr)
            return 2
        os.environ["BER_DATA_ROOT"] = str(real)
    else:
        os.environ.setdefault("BER_DATA_ROOT", str((PACKAGE_ROOT / spec["data_root"]).resolve()))
    os.environ.setdefault("BER_ARTIFACT_ROOT", str((PACKAGE_ROOT / spec["artifact_root"]).resolve()))
    os.environ.setdefault("BER_OUTPUT_ROOT", str((PACKAGE_ROOT / spec["output_root"]).resolve()))
    if "BER_THREADS" not in os.environ:
        os.environ["BER_THREADS"] = str(max(1, (os.cpu_count() or 2) - 1))

    print("=" * 72)
    print(f"Business Entity Resolution — profile={profile}")
    print(f"  data_root     = {os.environ['BER_DATA_ROOT']}")
    print(f"  artifact_root = {os.environ['BER_ARTIFACT_ROOT']}")
    print(f"  output_root   = {os.environ['BER_OUTPUT_ROOT']}")
    print(f"  threads       = {os.environ['BER_THREADS']}")
    print("=" * 72)

    started = time.time()
    steps = [
        ["prepare", "--config", config, "--split", "both"],
        ["make-splits", "--config", config, "--folds", str(spec["folds"])],
        ["generate-candidates", "--config", config, "--split", "train"],
        ["generate-candidates", "--config", config, "--split", "test"],
    ]
    if not skip_embeddings:
        steps.append(["embed", "--config", config, "--split", "both"])
    steps += [
        ["build-features", "--config", config, "--split", "train"],
        ["build-features", "--config", config, "--split", "test"],
        # Honest tuning: train without fold 0, score/evaluate only fold 0, so the
        # threshold is chosen on data the model never saw.
        ["train", "--config", config, "--model", spec["model"], "--validation-fold", "0"],
        ["score", "--config", config, "--split", "train", "--model", spec["model"], "--validation-fold", "0"],
        ["tune-decision", "--config", config, "--validation-fold", "0"],
        ["evaluate", "--config", config],
        # Final model on all training data, then inference. infer reuses the
        # tuned threshold persisted by tune-decision.
        ["train", "--config", config, "--model", spec["model"], "--all-training-data"],
        ["infer", "--config", config, "--model", spec["model"]],
    ]
    if validator:
        steps.append(["preflight", "--config", config, "--official-validator", validator, "--check-ids"])

    for index, step in enumerate(steps, start=1):
        print(f"\n### Step {index}/{len(steps)}: {step[0]}")
        code = _sh(step)
        if code != 0:
            print(f"\nStage '{step[0]}' exited with code {code}. Stopping.")
            print("Fix the issue and re-run the same command — completed work is reused.")
            return code

    elapsed = time.time() - started
    print("\n" + "=" * 72)
    print(f"Pipeline complete in {elapsed/60:.1f} min.")
    print(f"Outputs: {os.environ['BER_OUTPUT_ROOT']}/matching_results.tsv")
    print(f"         {os.environ['BER_OUTPUT_ROOT']}/candidate_pairs.tsv")
    print("=" * 72)
    return 0


def _looks_like_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return handle.readline().startswith("version https://git-lfs")
    except OSError:
        return False


def _find_real_dataset() -> Path | None:
    """Locate the real challenge dataset (train/ and test/ TSVs).

    Candidate folders holding only Git-LFS pointer stubs are skipped so the
    pipeline fails with an actionable message instead of a schema error.
    """
    candidates = [
        Path(os.environ["BER_SOURCE_DATA_ROOT"]) if os.environ.get("BER_SOURCE_DATA_ROOT") else None,
        PACKAGE_ROOT / "dataset",
        PACKAGE_ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset",
        Path.cwd() / "dataset",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        train = candidate / "train" / "train_source1.tsv"
        if train.is_file() and not _looks_like_lfs_pointer(train):
            return candidate.resolve()
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full BER pipeline end-to-end.")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True,
                        help="mini = 15 realistic records; full = the real dataset")
    parser.add_argument("--validator", default=None,
                        help="Path to utils/validate_submission.py to run at the end.")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip the BGE-M3 embedding stage (faster, no torch needed).")
    args = parser.parse_args()

    if args.profile == "mini" and not (PACKAGE_ROOT / "dataset/mini/train/train_source1.tsv").exists():
        # The mini set is sampled from the real files; build it first from the
        # REAL dataset (not dataset/mini, which does not exist yet).
        real_root = _find_real_dataset()
        if real_root is None:
            print("Could not find the real dataset. Run 'make data', or set "
                  "BER_DATA_ROOT/BER_SOURCE_DATA_ROOT to the folder containing "
                  "train/ and test/ TSVs.", file=sys.stderr)
            return 2
        print(f"Mini dataset not found; building 15 records from {real_root} ...")
        code = _sh(["make-mini", "--config", PROFILES["mini"]["config"],
                    "--mini-root", str((PACKAGE_ROOT / "dataset/mini").resolve()),
                    "--source-root", str(real_root), "--count", "15"])
        if code != 0:
            return code

    return run_profile(args.profile, args.validator, args.skip_embeddings)


if __name__ == "__main__":
    raise SystemExit(main())
