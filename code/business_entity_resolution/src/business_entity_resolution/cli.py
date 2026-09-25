"""Command-line interface for local smoke work and remote full-scale stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd

from .artifacts import read_frame, write_frame, write_manifest
from .audit import audit_dataset
from .blocking import CandidateGenerator
from .config import ProjectConfig
from .data import iter_tsv, load_ground_truth, load_ground_truth_for_ids, source_path
from .decisions import apply_thresholds, score_quantile_thresholds, sweep_thresholds
from .error_analysis import analyze_errors
from .features import build_pair_features, feature_matrix
from .labels import ground_truth_sets, label_candidates
from .metrics import candidate_metrics, evaluate_entity_sets, sets_from_long
from .models import DeterministicScorer, SGDPairModel
from .models.lightgbm_model import LightGBMPairModel
from .normalize import normalize_records
from .pipeline import run_smoke
from .preflight import preflight_outputs, run_official_validator
from .schemas import SOURCE_COLUMNS
from .splits import assign_s1_folds, select_pair_fold
from .submission import write_submission_outputs


def _json_print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _load_config(args: argparse.Namespace) -> ProjectConfig:
    overrides = {"max_rows": getattr(args, "max_rows", None)}
    return ProjectConfig.load(args.config, overrides)


def _prepared_dir(config: ProjectConfig, split: str, source: int) -> Path:
    return config.artifact_dir("prepared") / f"{split}_source{source}"


def _read_prepared(config: ProjectConfig, split: str, source: int) -> pd.DataFrame:
    directory = _prepared_dir(config, split, source)
    paths = sorted(directory.glob("part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"prepared data not found: {directory}; run prepare first")
    return pd.concat([read_frame(path) for path in paths], ignore_index=True)


def _training_truth(config: ProjectConfig, source1_ids=None) -> pd.DataFrame:
    if source1_ids is not None:
        return load_ground_truth_for_ids(config.data_root, source1_ids, config.batch_size)
    return load_ground_truth(config.data_root)


def command_audit(args: argparse.Namespace) -> int:
    config = _load_config(args)
    started = time.time()
    report = audit_dataset(config.data_root, config.batch_size, config.max_rows)
    out = config.artifact_dir("audit") / "audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_manifest(out.with_name("manifest.json"), stage="audit", config=config.as_dict(), rows=sum(v["rows"] for v in report["sources"].values()), schema=[], metrics=report, started_at=started)
    _json_print({"audit": str(out), "bounded": config.max_rows is not None})
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    config = _load_config(args)
    splits = (args.split,) if args.split in {"train", "test"} else ("train", "test")
    for split in splits:
        for source in (1, 2, 3):
            output_dir = _prepared_dir(config, split, source)
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = parts = 0
            for parts, chunk in enumerate(iter_tsv(source_path(config.data_root, split, source), SOURCE_COLUMNS, config.batch_size, config.max_rows), start=1):
                normalized = normalize_records(chunk)
                write_frame(normalized, output_dir / f"part-{parts:05d}.parquet")
                rows += len(normalized)
            write_manifest(output_dir / "manifest.json", stage=f"prepare-{split}-source{source}", config=config.as_dict(), rows=rows, schema=list(normalized.columns) if rows else [])
            print(f"prepared {split} source{source}: {rows} rows in {parts} part(s)")
    return 0


def command_make_splits(args: argparse.Namespace) -> int:
    config = _load_config(args)
    s1 = _read_prepared(config, "train", 1)
    gt = _training_truth(config, s1["entity_id"])
    folds = assign_s1_folds(s1, gt, n_folds=args.folds, seed=config.seed)
    out = write_frame(folds, config.artifact_dir("splits") / "s1_folds.parquet")
    write_manifest(out.with_name("manifest.json"), stage="make-splits", config=config.as_dict(), rows=len(folds), schema=list(folds.columns))
    print(out)
    return 0


def command_generate_candidates(args: argparse.Namespace) -> int:
    config = _load_config(args)
    source1 = _read_prepared(config, args.split, 1)
    frames = []
    for source in (2, 3):
        target = _read_prepared(config, args.split, source)
        frames.append(CandidateGenerator(config.blocking, include_tfidf=not args.no_tfidf).fit(target).transform(source1))
    candidates = pd.concat(frames, ignore_index=True).sort_values(["source1_entity_id", "candidate_source", "retrieval_score", "candidate_entity_id"], ascending=[True, True, False, True], kind="mergesort")
    out = write_frame(candidates, config.artifact_dir("candidates") / f"{args.split}.parquet")
    metrics = {}
    if args.split == "train":
        truth = ground_truth_sets(_training_truth(config, source1["entity_id"]))
        universe_size = sum(len(_read_prepared(config, "train", source)) for source in (2, 3))
        metrics = candidate_metrics(truth, sets_from_long(candidates, "candidate_entity_id"), universe_size)
    write_manifest(out.with_name(f"{args.split}.manifest.json"), stage=f"candidates-{args.split}", config=config.as_dict(), rows=len(candidates), schema=list(candidates.columns), metrics=metrics)
    if metrics:
        _json_print(metrics)
    print(out)
    return 0


def command_build_features(args: argparse.Namespace) -> int:
    config = _load_config(args)
    candidates = read_frame(config.artifact_dir("candidates") / f"{args.split}.parquet")
    source1 = _read_prepared(config, args.split, 1)
    targets = pd.concat([_read_prepared(config, args.split, 2), _read_prepared(config, args.split, 3)], ignore_index=True)
    if args.split == "train":
        truth = ground_truth_sets(_training_truth(config, source1["entity_id"]))
        # Preserve the exact blocker output. Adding missed positives here would leak
        # blocking failures into held-out matching evaluation.
        candidates = label_candidates(candidates, truth)
    features = build_pair_features(candidates, source1, targets)
    out = write_frame(features, config.artifact_dir("features") / f"{args.split}.parquet")
    write_manifest(out.with_name(f"{args.split}.manifest.json"), stage=f"features-{args.split}", config=config.as_dict(), rows=len(features), schema=list(features.columns))
    print(out)
    return 0


def _new_model(config: ProjectConfig, kind: str):
    params = dict(config.model.get("params", {}))
    if kind == "deterministic":
        return DeterministicScorer()
    if kind == "sgd":
        return SGDPairModel(seed=config.seed, **params)
    if kind == "lightgbm":
        return LightGBMPairModel(seed=config.seed, threads=config.threads, **params)
    raise ValueError(f"unknown model kind: {kind}")


def _load_model(kind: str, path: Path):
    if kind == "deterministic":
        return DeterministicScorer.load(path)
    if kind == "sgd":
        return SGDPairModel.load(path)
    if kind == "lightgbm":
        return LightGBMPairModel.load(path)
    raise ValueError(f"unknown model kind: {kind}")


def _default_model_path(config: ProjectConfig, kind: str) -> Path:
    suffixes = {"deterministic": ".json", "sgd": ".joblib", "lightgbm": ".txt"}
    path = config.artifact_dir("models") / f"{kind}{suffixes[kind]}"
    if not path.is_file():
        raise FileNotFoundError(
            f"model artifact not found: {path}. Run the train command or pass --model-path."
        )
    return path


def command_train(args: argparse.Namespace) -> int:
    config = _load_config(args)
    features = read_frame(config.artifact_dir("features") / "train.parquet")
    if "label" not in features:
        raise ValueError("training features are unlabeled")
    if not args.all_training_data:
        folds = read_frame(config.artifact_dir("splits") / "s1_folds.parquet")
        features = select_pair_fold(features, folds, args.validation_fold, validation=False)
    model = _new_model(config, args.model)
    X, y = feature_matrix(features), features["label"].to_numpy(dtype=np.int8)
    weights = features["sample_weight"].to_numpy(dtype=np.float32) if "sample_weight" in features else None
    model.fit(X, y, sample_weight=weights)
    suffix = ".txt" if args.model == "lightgbm" else ".joblib" if args.model == "sgd" else ".json"
    path = config.artifact_dir("models") / f"{args.model}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    write_manifest(path.with_suffix(path.suffix + ".manifest.json"), stage=f"train-{args.model}", config=config.as_dict(), rows=len(features), schema=list(features.columns), metrics={"positive_rows": int(y.sum()), "validation_fold": None if args.all_training_data else args.validation_fold, "all_training_data": bool(args.all_training_data)})
    print(path)
    return 0


def command_score(args: argparse.Namespace) -> int:
    config = _load_config(args)
    features = read_frame(config.artifact_dir("features") / f"{args.split}.parquet")
    if args.split == "train" and not args.all_training_data:
        folds = read_frame(config.artifact_dir("splits") / "s1_folds.parquet")
        features = select_pair_fold(features, folds, args.validation_fold, validation=True)
    model_path = Path(args.model_path) if args.model_path else _default_model_path(config, args.model)
    model = _load_model(args.model, model_path)
    scored = features[["source1_entity_id", "candidate_entity_id", "candidate_source"]].copy()
    scored["score"] = model.predict_scores(feature_matrix(features))
    if "label" in features:
        scored["label"] = features["label"]
    out = write_frame(scored, config.artifact_dir("scores") / f"{args.split}.parquet")
    print(out)
    return 0


def command_tune_decision(args: argparse.Namespace) -> int:
    config = _load_config(args)
    scored = read_frame(config.artifact_dir("scores") / "train.parquet")
    scored_ids = set(scored["source1_entity_id"].astype(str))
    truth = ground_truth_sets(_training_truth(config, scored_ids))
    thresholds = score_quantile_thresholds(scored, args.threshold_count)
    report = sweep_thresholds(scored, truth, thresholds)
    out = write_frame(report, config.artifact_dir("decisions") / "threshold_sweep.parquet")
    print(report.head(10).to_string(index=False)); print(out)
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    config = _load_config(args)
    scored = read_frame(config.artifact_dir("scores") / args.score_file)
    scored_ids = set(scored["source1_entity_id"].astype(str))
    truth = ground_truth_sets(_training_truth(config, scored_ids))
    threshold = float(args.threshold if args.threshold is not None else config.model.get("threshold", 0.8))
    predictions = apply_thresholds(scored, truth.keys(), threshold, config.model.get("source_thresholds", {}))
    _json_print(evaluate_entity_sets(truth, predictions))
    return 0


def command_analyze_errors(args: argparse.Namespace) -> int:
    config = _load_config(args)
    scored = read_frame(config.artifact_dir("scores") / args.score_file)
    candidates = read_frame(config.artifact_dir("candidates") / "train.parquet")
    scored_ids = set(scored["source1_entity_id"].astype(str))
    truth = ground_truth_sets(_training_truth(config, scored_ids))
    threshold = float(args.threshold if args.threshold is not None else config.model.get("threshold", 0.8))
    predictions = apply_thresholds(scored, truth.keys(), threshold)
    candidate_sets = sets_from_long(candidates.loc[candidates["source1_entity_id"].isin(scored_ids)], "candidate_entity_id")
    errors = analyze_errors(truth, candidate_sets, predictions, scored)
    out = write_frame(errors, config.artifact_dir("reports") / "errors.parquet")
    print(errors["error_type"].value_counts().to_string() if not errors.empty else "no errors"); print(out)
    return 0


def command_infer(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if not (config.artifact_dir("candidates") / "test.parquet").exists():
        command_generate_candidates(argparse.Namespace(**{**vars(args), "split": "test"}))
    if not (config.artifact_dir("features") / "test.parquet").exists():
        command_build_features(argparse.Namespace(**{**vars(args), "split": "test"}))
    features = read_frame(config.artifact_dir("features") / "test.parquet")
    candidates = read_frame(config.artifact_dir("candidates") / "test.parquet")
    source1 = _read_prepared(config, "test", 1)
    model_path = Path(args.model_path) if args.model_path else _default_model_path(config, args.model)
    model = _load_model(args.model, model_path)
    scored = features[["source1_entity_id", "candidate_entity_id", "candidate_source"]].copy()
    scored["score"] = model.predict_scores(feature_matrix(features))
    threshold = float(args.threshold if args.threshold is not None else config.model.get("threshold", 0.8))
    predictions = apply_thresholds(scored, source1["entity_id"], threshold, config.model.get("source_thresholds", {}))
    candidate_sets = sets_from_long(candidates, "candidate_entity_id")
    matching, candidate = write_submission_outputs(config.output_root, source1["entity_id"], predictions, candidate_sets)
    print(matching); print(candidate)
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    config = _load_config(args)
    matching, candidate = config.output_root / "matching_results.tsv", config.output_root / "candidate_pairs.tsv"
    test_dir = config.data_root / "test"
    errors = preflight_outputs(matching, candidate, test_dir)
    if errors:
        for error in errors: print(f"ERROR: {error}")
        return 1
    print("Internal preflight: PASS")
    if args.official_validator:
        result = run_official_validator(args.official_validator, matching, candidate, test_dir, check_ids=args.check_ids)
        print(result.stdout); print(result.stderr, file=sys.stderr)
        return result.returncode
    return 0


def command_package(args: argparse.Namespace) -> int:
    config = _load_config(args)
    documentation = Path(args.documentation).resolve()
    if not documentation.is_file():
        raise FileNotFoundError(f"completed competition documentation not found: {documentation}")
    package_root = config.artifact_dir("package") / args.team_name
    package_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(config.output_root, package_root / "output", dirs_exist_ok=True)
    project_root = Path(__file__).resolve().parents[2]
    shutil.copytree(project_root, package_root / "code" / "business_entity_resolution", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    shutil.copy2(documentation, package_root / "Documentation_template.md")
    archive = shutil.make_archive(str(package_root), "zip", package_root)
    print(archive)
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    config = _load_config(args)
    report = run_smoke(config.output_root, config.seed)
    _json_print(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Amazon ML 2026 business entity resolution pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(name: str, function):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--max-rows", type=int)
        command.set_defaults(function=function)
        return command

    add_common("audit", command_audit)
    prepare = add_common("prepare", command_prepare); prepare.add_argument("--split", choices=["train", "test", "both"], default="both")
    splits = add_common("make-splits", command_make_splits); splits.add_argument("--folds", type=int, default=10)
    candidates = add_common("generate-candidates", command_generate_candidates); candidates.add_argument("--split", choices=["train", "test"], required=True); candidates.add_argument("--no-tfidf", action="store_true")
    features = add_common("build-features", command_build_features); features.add_argument("--split", choices=["train", "test"], required=True)
    train = add_common("train", command_train); train.add_argument("--model", choices=["deterministic", "sgd", "lightgbm"], default="sgd"); train.add_argument("--validation-fold", type=int, default=0); train.add_argument("--all-training-data", action="store_true")
    score = add_common("score", command_score); score.add_argument("--split", choices=["train", "test"], required=True); score.add_argument("--model", choices=["deterministic", "sgd", "lightgbm"], default="sgd"); score.add_argument("--model-path"); score.add_argument("--validation-fold", type=int, default=0); score.add_argument("--all-training-data", action="store_true")
    tune = add_common("tune-decision", command_tune_decision); tune.add_argument("--threshold-count", type=int, default=101)
    evaluate = add_common("evaluate", command_evaluate); evaluate.add_argument("--score-file", default="train.parquet"); evaluate.add_argument("--threshold", type=float)
    errors = add_common("analyze-errors", command_analyze_errors); errors.add_argument("--score-file", default="train.parquet"); errors.add_argument("--threshold", type=float)
    infer = add_common("infer", command_infer); infer.add_argument("--model", choices=["deterministic", "sgd", "lightgbm"], default="sgd"); infer.add_argument("--model-path"); infer.add_argument("--threshold", type=float); infer.add_argument("--no-tfidf", action="store_true")
    preflight = add_common("preflight", command_preflight); preflight.add_argument("--official-validator"); preflight.add_argument("--check-ids", action="store_true")
    package = add_common("package", command_package); package.add_argument("--team-name", required=True); package.add_argument("--documentation", required=True)
    add_common("smoke", command_smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
