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
from .checkpoint import ShardStore, input_fingerprint, shard_for_id
from .config import ProjectConfig
from .data import iter_tsv, load_ground_truth, load_ground_truth_for_ids, source_path
from .decisions import apply_thresholds, apply_thresholds_with_france, score_quantile_thresholds, sweep_thresholds
from .embeddings import embed_split, load_embedding_vectors
from .error_analysis import analyze_errors
from .features import build_pair_features, feature_matrix
from .labels import ground_truth_sets, label_candidates
from .logging import stage_logger
from .metrics import candidate_metrics, evaluate_entity_sets, sets_from_long
from .mini import build_mini
from .models import DeterministicScorer, SGDPairModel
from .models.lightgbm_model import LightGBMPairModel
from .normalize import normalize_records
from .pipeline import run_smoke
from .preflight import preflight_outputs, run_official_validator
from .resources import run_with_oom_backoff
from .schemas import SOURCE_COLUMNS
from .splits import assign_s1_folds, select_pair_fold
from .submission import write_submission_outputs


def _json_print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _force(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "force", False))


def _load_config(args: argparse.Namespace) -> ProjectConfig:
    overrides = {"max_rows": getattr(args, "max_rows", None)}
    return ProjectConfig.load(args.config, overrides)


def _prepared_dir(config: ProjectConfig, split: str, source: int) -> Path:
    return config.artifact_dir("prepared") / f"{split}_source{source}"


def _read_prepared(config: ProjectConfig, split: str, source: int) -> pd.DataFrame:
    """Read all prepared shards for a split/source, supporting legacy part naming."""
    directory = _prepared_dir(config, split, source)
    paths = sorted(directory.glob("part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"prepared data not found: {directory}; run prepare first")
    return pd.concat([read_frame(path) for path in paths], ignore_index=True)


def _prepared_shard_store(config: ProjectConfig, split: str, source: int) -> ShardStore:
    directory = _prepared_dir(config, split, source)
    fingerprint = input_fingerprint([source_path(config.data_root, split, source)])
    return ShardStore(directory, fingerprint)


def _read_prepared_part(config: ProjectConfig, split: str, source: int, key: int) -> pd.DataFrame:
    directory = _prepared_dir(config, split, source)
    path = directory / "parts" / f"part-{int(key):05d}.parquet"
    return read_frame(path)


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
    """Normalize raw TSVs into sharded parquet, resumable at chunk granularity.

    Each raw ``iter_tsv`` chunk is normalized and written as an interim part
    under ``parts/`` (bounded to one chunk of memory). On completion the interim
    parts are consolidated into S1-ID-range shards. If the process dies, already
    committed chunks are skipped on the next run.
    """
    config = _load_config(args)
    plan = config.resource_plan()
    splits = (args.split,) if args.split in {"train", "test"} else ("train", "test")
    force = _force(args)
    for split in splits:
        for source in (1, 2, 3):
            path = source_path(config.data_root, split, source)
            store = _prepared_shard_store(config, split, source)
            logger = stage_logger(config, f"prepare-{split}-source{source}")
            logger.stage_start("prepare", split=split, source=source, shards=config.n_shards)
            logger.resource_plan(plan)
            if force:
                for stale in store.parts_dir.glob("chunk-*"):
                    stale.unlink()
                store.stage_manifest_path().unlink(missing_ok=True)
            if store.is_complete():
                logger.info("stage already complete; skipping", rows=store.total_rows())
                logger.stage_end("prepare", split=split, source=source, skipped=True)
                continue

            # Interim, resumable chunk files (independent of final shard layout).
            chunk_dir = store.parts_dir / "chunks"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            total_rows = 0
            written_chunks = 0
            for sequence, chunk in enumerate(iter_tsv(path, SOURCE_COLUMNS, config.batch_size, config.max_rows), start=1):
                chunk_path = chunk_dir / f"chunk-{sequence:07d}.parquet"
                if not chunk_path.is_file():
                    normalized = normalize_records(chunk)
                    write_frame(normalized, chunk_path)
                total_rows += len(chunk)
                written_chunks = sequence
                logger.progress("prepare", min(total_rows, config.max_rows or total_rows), config.max_rows or total_rows)
            logger.info("raw chunks written", chunks=written_chunks, rows_seen=total_rows)

            # Consolidate interim chunks into final S1-ID-range shards.
            buffer: dict[int, list[pd.DataFrame]] = {}
            pending = set(range(config.n_shards))
            flushed = 0
            for chunk_path in sorted(chunk_dir.glob("chunk-*.parquet")):
                frame = read_frame(chunk_path)
                keys = frame["entity_id"].map(lambda value: shard_for_id(value, config.n_shards))
                for key, group in frame.groupby(keys, sort=False):
                    buffer.setdefault(int(key), []).append(group)
                for key in list(buffer):
                    if int(sum(len(f) for f in buffer[key])) >= max(1, plan.chunk_pairs):
                        store.commit(key, pd.concat(buffer.pop(key), ignore_index=True))
                        pending.discard(key)
                        flushed += 1
                        logger.progress("consolidate", flushed, config.n_shards, extra={"pending_shards": len(pending)})
            for key in list(buffer):
                store.commit(key, pd.concat(buffer.pop(key), ignore_index=True))
                pending.discard(key)
            for key in sorted(pending):
                store.commit(key, store.read_all().iloc[0:0])
            rows = store.total_rows()
            schema = list(_read_prepared(config, split, source).columns)
            store.mark_complete(rows, schema, extra={"split": split, "source": source})
            write_manifest(store.stage_manifest_path(), stage=f"prepare-{split}-source{source}", config=config.as_dict(), rows=rows, schema=schema)
            logger.info("prepared", split=split, source=source, rows=rows)
            logger.stage_end("prepare", split=split, source=source, rows=rows)
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
    """Generate the exact last-stage candidate set, sharded and resumable.

    Blockers are fit once per split against all Source 2/3 targets, then Source 1
    is transformed shard-by-shard so memory stays bounded. Each shard is
    committed atomically, so a crash resumes without refitting prior shards.
    """
    config = _load_config(args)
    plan = config.resource_plan()
    logger = stage_logger(config, f"candidates-{args.split}")
    source1 = _read_prepared(config, args.split, 1)
    targets = [(_read_prepared(config, args.split, source), source) for source in (2, 3)]
    fingerprint = input_fingerprint(
        [source_path(config.data_root, args.split, source) for source in (1, 2, 3)]
    ) + ":" + json.dumps(config.blocking, sort_keys=True) + f":tfidf={not args.no_tfidf}"
    store = ShardStore(config.artifact_dir("candidates") / args.split, fingerprint)
    if _force(args):
        for stale in store.parts_dir.glob("part-*"):
            stale.unlink()
        store.stage_manifest_path().unlink(missing_ok=True)
    logger.stage_start("candidates", split=args.split, shards=config.n_shards)
    logger.resource_plan(plan)

    if not store.is_complete():
        keys = source1["entity_id"].map(lambda value: shard_for_id(value, config.n_shards))
        buckets: dict[int, pd.DataFrame] = {int(key): group for key, group in source1.groupby(keys, sort=False)}
        pending = store.pending_keys(range(config.n_shards))
        # Fit each blocker once per target source (the expensive step), then reuse
        # across S1 shards. Refitting per shard would be prohibitive.
        generators = [
            CandidateGenerator(config.blocking, include_tfidf=not args.no_tfidf).fit(target)
            for target, _source in targets
        ]
        for index, key in enumerate(pending, start=1):
            shard_s1 = buckets.get(int(key), source1.iloc[0:0])
            frames = [generator.transform(shard_s1) for generator in generators]
            candidates = pd.concat(frames, ignore_index=True)
            if not candidates.empty:
                candidates = candidates.sort_values(
                    ["source1_entity_id", "candidate_source", "retrieval_score", "candidate_entity_id"],
                    ascending=[True, True, False, True],
                    kind="mergesort",
                ).reset_index(drop=True)
            store.commit(int(key), candidates)
            logger.progress("candidates", index, len(pending), extra={"shard": int(key), "rows": len(candidates)})
        rows = store.total_rows()
        schema = ["source1_entity_id", "candidate_entity_id", "candidate_source", "reason_bits", "reason_mask", "retrieval_score", "retrieval_rank"]
        store.mark_complete(rows, schema, extra={"split": args.split})
    else:
        logger.info("stage already complete; skipping", rows=store.total_rows())

    candidates = store.read_all()
    metrics: dict[str, object] = {}
    if args.split == "train" and not candidates.empty:
        truth = ground_truth_sets(_training_truth(config, source1["entity_id"]))
        universe_size = sum(len(frame) for frame, _source in targets)
        metrics = candidate_metrics(truth, sets_from_long(candidates, "candidate_entity_id"), universe_size)
        for name, value in metrics.items():
            logger.metric(f"candidate_{name}", value)
    write_manifest(store.stage_manifest_path(), stage=f"candidates-{args.split}", config=config.as_dict(), rows=len(candidates), schema=list(candidates.columns), metrics=metrics)
    logger.stage_end("candidates", split=args.split, rows=len(candidates))
    if metrics:
        _json_print(metrics)
    print(store.stage_manifest_path())
    return 0


def command_make_mini(args: argparse.Namespace) -> int:
    """Sample a realistic mini dataset from the real files (schema-identical)."""
    config = _load_config(args)
    source_root = Path(args.source_root) if args.source_root else config.data_root
    mini_root = Path(args.mini_root)
    if not mini_root.is_absolute():
        mini_root = (Path.cwd() / mini_root).resolve()
    summary = build_mini(source_root, mini_root, count=args.count, seed=config.seed)
    _json_print({"mini_root": str(mini_root), **summary})
    return 0


def command_embed(args: argparse.Namespace) -> int:
    """Compute and cache BGE-M3 name/address vectors (resumable, device-aware)."""
    config = _load_config(args)
    splits = (args.split,) if args.split in {"train", "test"} else ("train", "test")
    for split in splits:
        logger = stage_logger(config, f"embed-{split}")
        logger.stage_start("embed", split=split)
        summary = embed_split(config, split, logger)
        for name, value in summary.items():
            logger.metric(f"embed_{name}", value)
        logger.stage_end("embed", split=split, **{k: str(v) for k, v in summary.items()})
    return 0


def command_build_features(args: argparse.Namespace) -> int:
    """Build pair features shard-by-shard, resumable and OOM-adaptive.

    Reads the committed candidate shards, joins each against prepared S1/target
    rows, and commits a feature parquet per candidate shard. Optional BGE-M3
    cosine features are added when embeddings are enabled.
    """
    config = _load_config(args)
    plan = config.resource_plan()
    logger = stage_logger(config, f"features-{args.split}")
    candidate_store = ShardStore(
        config.artifact_dir("candidates") / args.split,
        input_fingerprint([source_path(config.data_root, args.split, source) for source in (1, 2, 3)]),
    )
    source1 = _read_prepared(config, args.split, 1)
    targets = pd.concat([_read_prepared(config, args.split, 2), _read_prepared(config, args.split, 3)], ignore_index=True)
    truth = ground_truth_sets(_training_truth(config, source1["entity_id"])) if args.split == "train" else {}

    embed_names, embed_addrs = (None, None) if getattr(args, "no_embeddings", False) else _load_embedding_vectors(config, args.split)
    embeddings_used = embed_names is not None

    fingerprint = input_fingerprint(
        [source_path(config.data_root, args.split, source) for source in (1, 2, 3)]
    ) + f":embeddings={embeddings_used}"
    store = ShardStore(config.artifact_dir("features") / args.split, fingerprint)
    if _force(args):
        for stale in store.parts_dir.glob("part-*"):
            stale.unlink()
        store.stage_manifest_path().unlink(missing_ok=True)
    logger.stage_start("features", split=args.split, shards=config.n_shards, embeddings=embeddings_used)
    logger.resource_plan(plan)

    if not store.is_complete():
        part_paths = list(candidate_store.iter_parts())
        for index, part_path in enumerate(part_paths, start=1):
            key = int(part_path.stem.split("-")[1])
            if key in store.completed_keys():
                continue

            def _build(chunk_size: int, _path=part_path, _key=key) -> pd.DataFrame:
                candidates = read_frame(_path)
                if args.split == "train":
                    candidates = label_candidates(candidates, truth)
                features = build_pair_features(
                    candidates,
                    source1,
                    targets,
                    embed_names=embed_names,
                    embed_addrs=embed_addrs,
                )
                return features

            features = run_with_oom_backoff(
                _build,
                plan.chunk_pairs,
                on_retry=lambda size, exc: logger.event("oom_backoff", stage="features", new_chunk=size, error=type(exc).__name__),
            )
            store.commit(key, features)
            logger.progress("features", index, len(part_paths), extra={"shard": key, "rows": len(features)})
        rows = store.total_rows()
        sample = store.read_all()
        store.mark_complete(rows, list(sample.columns) if rows else [], extra={"split": args.split})
    else:
        logger.info("stage already complete; skipping", rows=store.total_rows())

    logger.stage_end("features", split=args.split, rows=store.total_rows())
    print(store.stage_manifest_path())
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
    logger = stage_logger(config, f"train-{args.model}")
    logger.stage_start("train", model=args.model, all_training_data=bool(args.all_training_data))
    store = ShardStore(
        config.artifact_dir("features") / "train",
        input_fingerprint([source_path(config.data_root, "train", source) for source in (1, 2, 3)]),
    )
    features = store.read_all()
    if features.empty:
        raise FileNotFoundError("no training features found; run build-features first")
    if "label" not in features:
        raise ValueError("training features are unlabeled")
    if not args.all_training_data:
        folds = read_frame(config.artifact_dir("splits") / "s1_folds.parquet")
        features = select_pair_fold(features, folds, args.validation_fold, validation=False)
    logger.info("training data ready", rows=len(features), positives=int(features["label"].sum()))
    model = _new_model(config, args.model)
    X, y = feature_matrix(features), features["label"].to_numpy(dtype=np.int8)
    weights = features["sample_weight"].to_numpy(dtype=np.float32) if "sample_weight" in features else None

    def _fit(_size: int):
        return model.fit(X, y, sample_weight=weights)

    run_with_oom_backoff(
        _fit,
        max(1, len(features)),
        on_retry=lambda size, exc: logger.event("oom_backoff", stage="train", error=type(exc).__name__),
    )
    suffix = ".txt" if args.model == "lightgbm" else ".joblib" if args.model == "sgd" else ".json"
    path = config.artifact_dir("models") / f"{args.model}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    write_manifest(path.with_suffix(path.suffix + ".manifest.json"), stage=f"train-{args.model}", config=config.as_dict(), rows=len(features), schema=list(features.columns), metrics={"positive_rows": int(y.sum()), "validation_fold": None if args.all_training_data else args.validation_fold, "all_training_data": bool(args.all_training_data)})
    logger.stage_end("train", rows=len(features), positives=int(y.sum()))
    print(path)
    return 0


def _feature_store(config: ProjectConfig, split: str) -> ShardStore:
    return ShardStore(
        config.artifact_dir("features") / split,
        input_fingerprint([source_path(config.data_root, split, source) for source in (1, 2, 3)]),
    )


def _candidate_store(config: ProjectConfig, split: str) -> ShardStore:
    return ShardStore(
        config.artifact_dir("candidates") / split,
        input_fingerprint([source_path(config.data_root, split, source) for source in (1, 2, 3)]),
    )


def command_score(args: argparse.Namespace) -> int:
    config = _load_config(args)
    features = _feature_store(config, args.split).read_all()
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
    logger = stage_logger(config, "tune-decision")
    scored = read_frame(config.artifact_dir("scores") / "train.parquet")
    scored_ids = set(scored["source1_entity_id"].astype(str))
    truth = ground_truth_sets(_training_truth(config, scored_ids))
    thresholds = score_quantile_thresholds(scored, args.threshold_count)
    report = sweep_thresholds(scored, truth, thresholds)
    out = write_frame(report, config.artifact_dir("decisions") / "threshold_sweep.parquet")
    for rank, row in report.head(10).iterrows():
        logger.metric("threshold_sweep", float(row["macro_f0_5"]), rank=int(rank), threshold=float(row["threshold"]), precision=float(row["macro_precision"]), recall=float(row["macro_recall"]))
    print(report.head(10).to_string(index=False)); print(out)
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    config = _load_config(args)
    logger = stage_logger(config, "evaluate")
    scored = read_frame(config.artifact_dir("scores") / args.score_file)
    scored_ids = set(scored["source1_entity_id"].astype(str))
    truth = ground_truth_sets(_training_truth(config, scored_ids))
    threshold = float(args.threshold if args.threshold is not None else config.model.get("threshold", 0.8))
    source1 = _read_prepared(config, "train", 1)
    predictions = apply_thresholds_with_france(
        scored,
        truth.keys(),
        threshold,
        source1=source1,
        france_margin=float((config.model or {}).get("france_margin", 0.05)),
        source_thresholds=config.model.get("source_thresholds", {}),
    )
    metrics = evaluate_entity_sets(truth, predictions)
    for name, value in metrics.items():
        logger.metric(name, value)
    _json_print(metrics)
    return 0


def command_analyze_errors(args: argparse.Namespace) -> int:
    config = _load_config(args)
    scored = read_frame(config.artifact_dir("scores") / args.score_file)
    candidates = _candidate_store(config, "train").read_all()
    scored_ids = set(scored["source1_entity_id"].astype(str))
    truth = ground_truth_sets(_training_truth(config, scored_ids))
    threshold = float(args.threshold if args.threshold is not None else config.model.get("threshold", 0.8))
    predictions = apply_thresholds(scored, truth.keys(), threshold)
    candidate_sets = sets_from_long(candidates.loc[candidates["source1_entity_id"].isin(scored_ids)], "candidate_entity_id")
    errors = analyze_errors(truth, candidate_sets, predictions, scored)
    out = write_frame(errors, config.artifact_dir("reports") / "errors.parquet")
    print(errors["error_type"].value_counts().to_string() if not errors.empty else "no errors"); print(out)
    return 0


def _test_target_ids(config: ProjectConfig, split: str) -> set[str]:
    """All valid Source 2/3 IDs for a split, used by the pre-write format gate."""
    ids: set[str] = set()
    for source in (2, 3):
        frame = _read_prepared(config, split, source)
        ids.update(frame["entity_id"].astype(str))
    return ids


def command_infer(args: argparse.Namespace) -> int:
    config = _load_config(args)
    logger = stage_logger(config, "infer")
    logger.stage_start("infer", model=args.model)
    if not _candidate_store(config, "test").is_complete():
        command_generate_candidates(argparse.Namespace(**{**vars(args), "split": "test"}))
    if not _feature_store(config, "test").is_complete():
        command_build_features(argparse.Namespace(**{**vars(args), "split": "test"}))
    features = _feature_store(config, "test").read_all()
    candidates = _candidate_store(config, "test").read_all()
    source1 = _read_prepared(config, "test", 1)
    model_path = Path(args.model_path) if args.model_path else _default_model_path(config, args.model)
    model = _load_model(args.model, model_path)
    scored = features[["source1_entity_id", "candidate_entity_id", "candidate_source"]].copy()
    scored["score"] = model.predict_scores(feature_matrix(features))
    # Attach the signals the France rule needs without rescoring.
    for signal in ("name_core_exact", "numeric_overlap"):
        if signal in features:
            scored[signal] = features[signal].to_numpy()
    threshold = float(args.threshold if args.threshold is not None else config.model.get("threshold", 0.8))
    predictions = apply_thresholds_with_france(
        scored,
        source1["entity_id"],
        threshold,
        source1=source1,
        france_margin=float((config.model or {}).get("france_margin", 0.05)),
        source_thresholds=config.model.get("source_thresholds", {}),
    )
    candidate_sets = sets_from_long(candidates, "candidate_entity_id")
    valid_targets = _test_target_ids(config, "test") if args.split == "test" else None
    matching, candidate = write_submission_outputs(config.output_root, source1["entity_id"], predictions, candidate_sets, valid_targets=valid_targets)
    logger.info("wrote outputs", matching=str(matching), candidate=str(candidate), entities=len(predictions))
    logger.stage_end("infer")
    print(matching); print(candidate)
    return 0


def command_write_output(args: argparse.Namespace) -> int:
    """Reformat outputs only from cached scores + candidates (never reruns the model)."""
    config = _load_config(args)
    logger = stage_logger(config, "write-output")
    candidates = _candidate_store(config, args.split).read_all()
    source1 = _read_prepared(config, args.split, 1)
    scored_path = config.artifact_dir("scores") / f"{args.split}.parquet"
    if not scored_path.is_file():
        raise FileNotFoundError(f"cached scores not found: {scored_path}; run score/infer first")
    scored = read_frame(scored_path)
    threshold = float(args.threshold if args.threshold is not None else config.model.get("threshold", 0.8))
    predictions = apply_thresholds_with_france(
        scored,
        source1["entity_id"],
        threshold,
        source1=source1,
        france_margin=float((config.model or {}).get("france_margin", 0.05)),
        source_thresholds=config.model.get("source_thresholds", {}),
    )
    candidate_sets = sets_from_long(candidates, "candidate_entity_id")
    valid_targets = _test_target_ids(config, "test") if args.split == "test" else None
    matching, candidate = write_submission_outputs(config.output_root, source1["entity_id"], predictions, candidate_sets, valid_targets=valid_targets)
    logger.event("outputs_rewritten", matching=str(matching), candidate=str(candidate))
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
        command.add_argument("--force", action="store_true", help="Ignore committed checkpoints and redo the stage.")
        command.set_defaults(function=function)
        return command

    add_common("audit", command_audit)
    prepare = add_common("prepare", command_prepare); prepare.add_argument("--split", choices=["train", "test", "both"], default="both")
    splits = add_common("make-splits", command_make_splits); splits.add_argument("--folds", type=int, default=10)
    candidates = add_common("generate-candidates", command_generate_candidates); candidates.add_argument("--split", choices=["train", "test"], required=True); candidates.add_argument("--no-tfidf", action="store_true")
    embed = add_common("embed", command_embed); embed.add_argument("--split", choices=["train", "test", "both"], default="both")
    features = add_common("build-features", command_build_features); features.add_argument("--split", choices=["train", "test"], required=True); features.add_argument("--no-embeddings", action="store_true")
    train = add_common("train", command_train); train.add_argument("--model", choices=["deterministic", "sgd", "lightgbm"], default="sgd"); train.add_argument("--validation-fold", type=int, default=0); train.add_argument("--all-training-data", action="store_true")
    score = add_common("score", command_score); score.add_argument("--split", choices=["train", "test"], required=True); score.add_argument("--model", choices=["deterministic", "sgd", "lightgbm"], default="sgd"); score.add_argument("--model-path"); score.add_argument("--validation-fold", type=int, default=0); score.add_argument("--all-training-data", action="store_true")
    tune = add_common("tune-decision", command_tune_decision); tune.add_argument("--threshold-count", type=int, default=101)
    evaluate = add_common("evaluate", command_evaluate); evaluate.add_argument("--score-file", default="train.parquet"); evaluate.add_argument("--threshold", type=float)
    errors = add_common("analyze-errors", command_analyze_errors); errors.add_argument("--score-file", default="train.parquet"); errors.add_argument("--threshold", type=float)
    infer = add_common("infer", command_infer); infer.add_argument("--model", choices=["deterministic", "sgd", "lightgbm"], default="sgd"); infer.add_argument("--model-path"); infer.add_argument("--threshold", type=float); infer.add_argument("--no-tfidf", action="store_true")
    write_output = add_common("write-output", command_write_output); write_output.add_argument("--split", choices=["train", "test"], default="test"); write_output.add_argument("--threshold", type=float)
    preflight = add_common("preflight", command_preflight); preflight.add_argument("--official-validator"); preflight.add_argument("--check-ids", action="store_true")
    package = add_common("package", command_package); package.add_argument("--team-name", required=True); package.add_argument("--documentation", required=True)
    mini = add_common("make-mini", command_make_mini); mini.add_argument("--mini-root", default="dataset/mini"); mini.add_argument("--count", type=int, default=15); mini.add_argument("--source-root", default=None)
    add_common("smoke", command_smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
