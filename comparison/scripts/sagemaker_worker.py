"""Train-only SageMaker Processing worker for branch resource pilots.

The worker intentionally never mounts or reads the competition test split.  It
creates a small source-1/ground-truth pilot while retaining the complete target
catalogues, then executes the branch's native pipeline stages unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_runtime(branch_root: Path, log_dir: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "-r",
        str(branch_root / "requirements.txt"),
        "-r",
        str(branch_root / "requirements-remote.txt"),
    ]
    with (log_dir / "pip-install.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def locate_dataset_root(mounted_root: Path) -> Path:
    candidates = [mounted_root, mounted_root / "dataset"]
    for candidate in candidates:
        if (candidate / "train" / "train_source1.tsv").is_file():
            return candidate
    found = list(mounted_root.rglob("train_source1.tsv"))
    if len(found) == 1 and found[0].parent.name == "train":
        return found[0].parent.parent
    raise FileNotFoundError(f"Could not locate train dataset beneath {mounted_root}")


def build_pilot_dataset(full_root: Path, pilot_root: Path, pilot_rows: int) -> dict[str, Any]:
    import pandas as pd

    source_dir = full_root / "train"
    target_dir = pilot_root / "train"
    target_dir.mkdir(parents=True, exist_ok=True)

    s1_path = source_dir / "train_source1.tsv"
    gt_path = source_dir / "train_ground_truth.tsv"
    s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False, nrows=pilot_rows)
    if len(s1) != pilot_rows:
        raise RuntimeError(f"Requested {pilot_rows} source-1 rows, found {len(s1)}")
    id_column = "entity_id" if "entity_id" in s1.columns else s1.columns[0]
    selected_ids = set(s1[id_column].astype(str))
    s1.to_csv(target_dir / s1_path.name, sep="\t", index=False)

    selected_chunks = []
    for chunk in pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, chunksize=250_000):
        selected = chunk.loc[chunk["source1_entity_id"].astype(str).isin(selected_ids)]
        if not selected.empty:
            selected_chunks.append(selected)
    if not selected_chunks:
        raise RuntimeError("Pilot ground-truth selection was empty")
    ground_truth = pd.concat(selected_chunks, ignore_index=True)
    ground_truth.to_csv(target_dir / gt_path.name, sep="\t", index=False)

    for filename in ("train_source2.tsv", "train_source3.tsv"):
        source = source_dir / filename
        destination = target_dir / filename
        try:
            os.symlink(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    return {
        "pilot_source1_rows": int(len(s1)),
        "pilot_ground_truth_rows": int(len(ground_truth)),
        "pilot_unique_source1_ids": int(len(selected_ids)),
        "source1_sha256": sha256(target_dir / s1_path.name),
        "ground_truth_sha256": sha256(target_dir / gt_path.name),
    }


def run_stage(
    name: str,
    arguments: list[str],
    branch_root: Path,
    config_path: Path,
    log_dir: Path,
) -> dict[str, Any]:
    import psutil

    command = [
        sys.executable,
        "-m",
        "business_entity_resolution",
        *arguments,
        "--config",
        str(config_path),
        "--force",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(branch_root / "src")
    started = time.time()
    peak_rss = 0
    log_path = log_dir / f"{name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=branch_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        root = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                descendants = root.children(recursive=True)
                peak_rss = max(peak_rss, root.memory_info().rss + sum(p.memory_info().rss for p in descendants))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(2)
        return_code = process.wait()
    finished = time.time()
    result = {
        "name": name,
        "command": command,
        "started_at_epoch": started,
        "finished_at_epoch": finished,
        "elapsed_seconds": finished - started,
        "peak_rss_bytes": peak_rss,
        "return_code": return_code,
        "log": str(log_path),
    }
    if return_code:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def export_selected_artifacts(artifact_root: Path, export_root: Path) -> list[str]:
    selected: list[str] = []
    patterns = ("*.json", "*.joblib", "*.txt", "*.log")
    for pattern in patterns:
        for source in artifact_root.rglob(pattern):
            if source.is_file():
                relative = source.relative_to(artifact_root)
                destination = export_root / "artifacts" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                selected.append(relative.as_posix())
    return sorted(set(selected))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("/opt/ml/processing/work"))
    parser.add_argument("--export-root", type=Path, default=Path("/opt/ml/processing/output"))
    parser.add_argument("--branch-name", required=True)
    parser.add_argument("--branch-sha", required=True)
    parser.add_argument("--pilot-rows", type=int, default=10_000)
    args = parser.parse_args()

    args.work_root.mkdir(parents=True, exist_ok=True)
    args.export_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.export_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    full_data_root = locate_dataset_root(args.data_root)
    pilot_data_root = args.work_root / "pilot-dataset"
    pilot_summary = build_pilot_dataset(full_data_root, pilot_data_root, args.pilot_rows)
    install_runtime(args.branch_root, log_dir)

    base_config = json.loads((args.branch_root / "configs" / "remote_full.json").read_text(encoding="utf-8"))
    artifact_root = args.work_root / "artifacts"
    output_root = args.work_root / "outputs"
    base_config.update(
        {
            "data_root": str(pilot_data_root),
            "artifact_root": str(artifact_root),
            "output_root": str(output_root),
            "run_id": "sagemaker-pilot",
            "seed": 2026,
            "threads": max(1, int(os.cpu_count() or 1)),
            "n_shards": 16,
            "max_rows": None,
        }
    )
    base_config.setdefault("resources", {})["enable_embeddings"] = "false"
    base_config.setdefault("embeddings", {})["enable"] = "false"
    config_path = args.work_root / "pilot-config.json"
    config_path.write_text(json.dumps(base_config, indent=2, sort_keys=True), encoding="utf-8")

    stages = []
    stage_specs = [
        ("prepare", ["prepare", "--split", "train"]),
        ("make_splits", ["make-splits", "--folds", "10"]),
        ("generate_candidates", ["generate-candidates", "--split", "train"]),
        ("build_features", ["build-features", "--split", "train", "--no-embeddings"]),
        ("train", ["train", "--model", "lightgbm", "--validation-fold", "0"]),
    ]
    try:
        for name, arguments in stage_specs:
            stages.append(run_stage(name, arguments, args.branch_root, config_path, log_dir))
        status = "COMPLETED"
        error = None
    except Exception as exc:  # preserve the evidence before returning failure
        status = "FAILED"
        error = repr(exc)

    selected = export_selected_artifacts(artifact_root, args.export_root)
    report = {
        "status": status,
        "error": error,
        "branch": args.branch_name,
        "branch_sha": args.branch_sha,
        "worker_sha256": sha256(Path(__file__)),
        "full_data_root": str(full_data_root),
        "pilot": pilot_summary,
        "stages": stages,
        "selected_artifacts": selected,
        "elapsed_seconds": time.time() - started,
        "cpu_count": os.cpu_count(),
        "python": sys.version,
        "test_data_accessed": False,
    }
    (args.export_root / "pilot_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    shutil.copy2(config_path, args.export_root / "pilot-config.json")
    if error:
        print(json.dumps(report, indent=2), flush=True)
        return 1
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
