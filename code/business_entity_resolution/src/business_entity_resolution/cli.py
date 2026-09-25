"""Command-line interface for local smoke work and remote full-scale stages."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import read_frame, write_frame, write_manifest
from .audit import audit_dataset
from .blocking import CandidateGenerator
from .checkpoint import ShardStore, input_fingerprint, shard_for_id, stage_fingerprint
from .config import ProjectConfig
from .data import iter_tsv, load_ground_truth, load_ground_truth_for_ids, source_path
from .decisions import apply_thresholds, score_quantile_thresholds, sweep_thresholds
from .embeddings import embed_split, load_embedding_vectors
from .error_analysis import analyze_errors
from .features import FEATURE_COLUMNS, build_pair_features, feature_matrix
from .labels import ground_truth_sets, label_candidates
from .logging import stage_logger
from .metrics import candidate_metrics_from_frames, evaluate_entity_sets, sets_from_long
from .mini import build_mini
from .models import DeterministicScorer, SGDPairModel
from .models.lightgbm_model import LightGBMPairModel
from .normalize import normalize_records
from .pipeline import run_smoke
from .preflight import preflight_outputs, run_official_validator
from .resources import run_with_oom_backoff
from .sampling import sample_candidate_negatives
from .schemas import SOURCE_COLUMNS
from .splits import assign_s1_folds, select_pair_fold
from .submission import write_submission_outputs, write_submission_outputs_sharded


def _json_print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _force(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "force", False))


def _load_config(args: argparse.Namespace) -> ProjectConfig:
    overrides = {"max_rows": getattr(args, "max_rows", None)}
    return ProjectConfig.load(args.config, overrides)


def _model_kind(config: ProjectConfig, args: argparse.Namespace) -> str:
    """Use an explicit CLI model when supplied, otherwise honor the config."""
    return str(getattr(args, "model", None) or config.model.get("kind", "sgd"))


def _model_params(config: ProjectConfig, kind: str) -> dict[str, object]:
    """Return only parameters intended for the selected model backend."""
    by_kind = config.model.get("params_by_kind", {})
    if isinstance(by_kind, dict) and kind in by_kind:
        return dict(by_kind[kind])
    if str(config.model.get("kind", "sgd")) == kind:
        return dict(config.model.get("params", {}))
    return {}


def _validation_entity_ids(config: ProjectConfig, fold: int) -> list[str]:
    folds = read_frame(config.artifact_dir("splits") / "s1_folds.parquet")
    selected = folds.loc[folds["fold"].eq(int(fold)), "source1_entity_id"].astype(str)
    if selected.empty:
        raise ValueError(f"validation fold {fold} contains no Source 1 entities")
    return selected.tolist()


def _score_path(config: ProjectConfig, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config.artifact_dir("scores") / path


def _decision_threshold(config: ProjectConfig, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    selected = config.artifact_dir("decisions") / "selected_threshold.json"
    if selected.is_file():
        payload = json.loads(selected.read_text(encoding="utf-8"))
        return float(payload["threshold"])
    return float(config.model.get("threshold", 0.8))


def _assert_candidate_feature_alignment(candidates: pd.DataFrame, features: pd.DataFrame) -> None:
    """Ensure candidate_pairs is exactly the set passed to the matcher."""
    columns = ["source1_entity_id", "candidate_entity_id"]
    for label, frame in (("candidates", candidates), ("features", features)):
        if frame.duplicated(columns).any():
            raise ValueError(f"{label} contain duplicate pair keys")
    candidate_keys = pd.MultiIndex.from_frame(candidates[columns])
    feature_keys = pd.MultiIndex.from_frame(features[columns])
    missing_features = candidate_keys.difference(feature_keys)
    extra_features = feature_keys.difference(candidate_keys)
    if len(missing_features) or len(extra_features):
        raise ValueError(
            "candidate/feature pair mismatch: "
            f"missing_features={len(missing_features)}, extra_features={len(extra_features)}; "
            "rebuild features for the current candidate artifact"
        )


def _split_input_fingerprint(config: ProjectConfig, split: str) -> str:
    return input_fingerprint([source_path(config.data_root, split, source) for source in (1, 2, 3)])


def _candidate_fingerprint(config: ProjectConfig, split: str, include_tfidf: bool = True) -> str:
    return (
        _split_input_fingerprint(config, split)
        + ":blocking=v3:"
        + json.dumps(config.blocking, sort_keys=True)
        + f":tfidf={include_tfidf}"
    )


def _feature_fingerprint(
    config: ProjectConfig,
    split: str,
    embeddings_used: bool = False,
    include_tfidf: bool = True,
) -> str:
    return (
        _candidate_fingerprint(config, split, include_tfidf=include_tfidf)
        + f":features=v3:embeddings={embeddings_used}"
    )


def _prepared_dir(config: ProjectConfig, split: str, source: int) -> Path:
    return config.artifact_dir("prepared") / f"{split}_source{source}"


def _read_prepared(
    config: ProjectConfig,
    split: str,
    source: int,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read all prepared shards for a split/source (stored under parts/)."""
    directory = _prepared_dir(config, split, source)
    store = _prepared_shard_store(config, split, source)
    frame = store.read_all(columns=columns)
    if frame.empty and not store.is_complete():
        raise FileNotFoundError(f"prepared data not found: {directory}; run prepare first")
    return frame


def _prepared_shard_store(config: ProjectConfig, split: str, source: int) -> ShardStore:
    directory = _prepared_dir(config, split, source)
    fingerprint = (
        input_fingerprint([source_path(config.data_root, split, source)])
        + f":prepare=v2:max_rows={config.max_rows}:shards={config.n_shards}"
    )
    return ShardStore(directory, fingerprint)


def _read_prepared_part(
    config: ProjectConfig,
    split: str,
    source: int,
    key: int,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    directory = _prepared_dir(config, split, source)
    path = directory / "parts" / f"part-{int(key):05d}.parquet"
    return read_frame(path, columns=columns)

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
            chunk_fingerprint = store.parts_dir / "input.fingerprint"
            recorded_fingerprint = chunk_fingerprint.read_text(encoding="utf-8") if chunk_fingerprint.is_file() else ""
            if force or recorded_fingerprint != store.fingerprint:
                # Prepared chunks are only reusable for the exact same raw
                # input fingerprint. Keeping old chunks after max_rows/input
                # changes previously mixed stale records into a new run.
                shutil.rmtree(store.parts_dir, ignore_errors=True)
                store.parts_dir.mkdir(parents=True, exist_ok=True)
                store.stage_manifest_path().unlink(missing_ok=True)
                chunk_fingerprint.write_text(store.fingerprint, encoding="utf-8")
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

            # Partition interim chunks into small on-disk fragments first, then
            # build each final shard once. The old buffer flush overwrote an
            # earlier shard when that shard appeared in a later raw chunk.
            fragment_dir = store.parts_dir / "fragments"
            shutil.rmtree(fragment_dir, ignore_errors=True)
            fragment_dir.mkdir(parents=True, exist_ok=True)
            for stale in store.parts_dir.glob("part-*.parquet"):
                stale.unlink()
            for stale in store.parts_dir.glob("part-*.manifest.json"):
                stale.unlink()
            for chunk_path in sorted(chunk_dir.glob("chunk-*.parquet")):
                frame = read_frame(chunk_path)
                keys = frame["entity_id"].map(lambda value: shard_for_id(value, config.n_shards))
                for key, group in frame.groupby(keys, sort=False):
                    key_dir = fragment_dir / f"shard-{int(key):05d}"
                    key_dir.mkdir(parents=True, exist_ok=True)
                    write_frame(group, key_dir / chunk_path.name)

            empty = normalize_records(pd.DataFrame(columns=SOURCE_COLUMNS)).iloc[0:0]
            for key in range(config.n_shards):
                fragments = sorted((fragment_dir / f"shard-{key:05d}").glob("chunk-*.parquet"))
                shard = pd.concat([read_frame(item) for item in fragments], ignore_index=True) if fragments else empty
                store.commit(key, shard)
                logger.progress("consolidate", key + 1, config.n_shards)
            shutil.rmtree(fragment_dir, ignore_errors=True)
            rows = store.total_rows()
            schema = list(empty.columns)
            store.mark_complete(rows, schema, extra={"split": split, "source": source})
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
    fingerprint = _candidate_fingerprint(config, args.split, include_tfidf=not args.no_tfidf)
    store = ShardStore(config.artifact_dir("candidates") / args.split, fingerprint)
    schema = ["source1_entity_id", "candidate_entity_id", "candidate_source", "reason_bits", "reason_mask", "retrieval_score", "retrieval_rank"]
    if _force(args):
        shutil.rmtree(store.stage_dir, ignore_errors=True)
        store = ShardStore(config.artifact_dir("candidates") / args.split, fingerprint)
        shutil.rmtree(config.artifact_dir("candidate-components") / args.split, ignore_errors=True)
        store.stage_manifest_path().unlink(missing_ok=True)
    logger.stage_start("candidates", split=args.split, shards=config.n_shards)
    logger.resource_plan(plan)

    if not store.is_complete():
        blocker_columns = [
            "entity_id", "country_norm", "name_canonical", "name_compact", "name_core",
            "name_accent_folded", "name_tokens", "address_tokens", "numeric_tokens",
        ]
        component_root = config.artifact_dir("candidate-components") / args.split
        component_stores: list[ShardStore] = []
        for source in (2, 3):
            component_fingerprint = fingerprint + f":target_source={source}"
            component_store = ShardStore(component_root / f"source{source}", component_fingerprint)
            component_stores.append(component_store)
            if component_store.is_complete():
                continue
            target = _read_prepared(config, args.split, source, columns=blocker_columns)
            generator = CandidateGenerator(config.blocking, include_tfidf=not args.no_tfidf).fit(target)
            for key in component_store.pending_keys(range(config.n_shards)):
                shard_s1 = _read_prepared_part(config, args.split, 1, key, columns=blocker_columns)
                candidates = generator.transform(shard_s1)
                component_store.commit(key, candidates)
                logger.progress(
                    f"candidates-source{source}",
                    key + 1,
                    config.n_shards,
                    extra={"shard": key, "rows": len(candidates)},
                )
            component_store.mark_complete(
                component_store.total_rows(), schema, extra={"split": args.split, "source": source}
            )
            del generator, target
            gc.collect()

        for key in store.pending_keys(range(config.n_shards)):
            frames = [read_frame(component.part_path(key)) for component in component_stores]
            candidates = pd.concat(frames, ignore_index=True)
            if not candidates.empty:
                candidates = candidates.sort_values(
                    ["source1_entity_id", "candidate_source", "retrieval_score", "candidate_entity_id"],
                    ascending=[True, True, False, True],
                    kind="mergesort",
                ).reset_index(drop=True)
            store.commit(key, candidates)
        rows = store.total_rows()
        store.mark_complete(rows, schema, extra={"split": args.split})
        shutil.rmtree(component_root, ignore_errors=True)
    else:
        logger.info("stage already complete; skipping", rows=store.total_rows())

    metrics: dict[str, object] = {}
    if args.split == "train":
        source1_ids = _read_prepared(config, args.split, 1, columns=["entity_id"])["entity_id"]
        truth = ground_truth_sets(_training_truth(config, source1_ids))
        universe_size = sum(
            _prepared_shard_store(config, "train", source).total_rows() for source in (2, 3)
        )
        metrics = candidate_metrics_from_frames(
            truth,
            (read_frame(path) for path in store.iter_parts()),
            universe_size,
        )
        for name, value in metrics.items():
            logger.metric(f"candidate_{name}", value)
    # Keep the ShardStore manifest intact; overwriting it with a generic stage
    # manifest removes the fingerprint required for safe resume checks.
    store.mark_complete(store.total_rows(), schema, extra={"split": args.split, "metrics": metrics})
    logger.stage_end("candidates", split=args.split, rows=store.total_rows())
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
        _candidate_fingerprint(config, args.split, include_tfidf=not getattr(args, "no_tfidf", False)),
    )
    feature_record_columns = [
        "entity_id", "business_name_raw", "business_address_raw", "country_norm",
        "name_canonical", "name_compact", "name_core", "name_token_sorted", "name_accent_folded",
        "address_canonical", "address_compact", "name_tokens", "address_tokens",
        "numeric_tokens", "postal_like_tokens", "missing_address",
    ]
    targets = pd.concat([
        _read_prepared(config, args.split, 2, columns=feature_record_columns),
        _read_prepared(config, args.split, 3, columns=feature_record_columns),
    ], ignore_index=True)
    targets = targets.set_index("entity_id", drop=False, verify_integrity=True)
    if args.split == "train":
        source1_ids = _read_prepared(config, args.split, 1, columns=["entity_id"])["entity_id"]
        truth = ground_truth_sets(_training_truth(config, source1_ids))
    else:
        truth = {}

    embed_names, embed_addrs = (None, None) if getattr(args, "no_embeddings", False) else load_embedding_vectors(config, args.split)
    embeddings_used = embed_names is not None

    fingerprint = _feature_fingerprint(
        config,
        args.split,
        embeddings_used,
        include_tfidf=not getattr(args, "no_tfidf", False),
    )
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
                shard_source1 = _read_prepared_part(
                    config, args.split, 1, _key, columns=feature_record_columns
                )
                if args.split == "train":
                    candidates = label_candidates(candidates, truth)
                pieces = []
                for start in range(0, len(candidates), max(1, chunk_size)):
                    candidate_chunk = candidates.iloc[start : start + chunk_size]
                    target_ids = pd.Index(candidate_chunk["candidate_entity_id"].astype(str).unique())
                    candidate_targets = targets.loc[target_ids]
                    pieces.append(build_pair_features(
                        candidate_chunk,
                        shard_source1,
                        candidate_targets,
                        embed_names=embed_names,
                        embed_addrs=embed_addrs,
                    ))
                if pieces:
                    return pd.concat(pieces, ignore_index=True)
                return build_pair_features(candidates, shard_source1, targets)

            features = run_with_oom_backoff(
                _build,
                plan.chunk_pairs,
                on_retry=lambda size, exc: logger.event("oom_backoff", stage="features", new_chunk=size, error=type(exc).__name__),
            )
            store.commit(key, features)
            logger.progress("features", index, len(part_paths), extra={"shard": key, "rows": len(features)})
        rows = store.total_rows()
        first_part = next(store.iter_parts(), None)
        schema = list(read_frame(first_part).columns) if first_part is not None else []
        store.mark_complete(rows, schema, extra={"split": args.split})
    else:
        logger.info("stage already complete; skipping", rows=store.total_rows())

    logger.stage_end("features", split=args.split, rows=store.total_rows())
    print(store.stage_manifest_path())
    return 0


def _new_model(config: ProjectConfig, kind: str):
    params = _model_params(config, kind)
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
    kind = _model_kind(config, args)
    logger = stage_logger(config, f"train-{kind}")
    logger.stage_start("train", model=kind, all_training_data=bool(args.all_training_data))
    store = _feature_store(config, "train")
    part_paths = list(store.iter_parts())
    if not part_paths:
        raise FileNotFoundError("no training features found; run build-features first")
    folds = None if args.all_training_data else read_frame(config.artifact_dir("splits") / "s1_folds.parquet")
    excluded_folds = set()
    if folds is not None:
        raw_excluded = getattr(args, "exclude_folds", None)
        excluded_folds = (
            {int(value.strip()) for value in raw_excluded.split(",") if value.strip()}
            if raw_excluded
            else {int(args.validation_fold)}
        )
        if not excluded_folds:
            raise ValueError("at least one fold must be excluded for held-out training")
        excluded_ids = set(
            folds.loc[folds["fold"].isin(excluded_folds), "source1_entity_id"].astype(str)
        )
    else:
        excluded_ids = set()
    max_negatives = int(config.training.get("max_negatives_per_entity", 50))

    def _training_part(path: Path) -> pd.DataFrame:
        frame = read_frame(path)
        if "label" not in frame:
            raise ValueError("training features are unlabeled")
        if folds is not None:
            frame = frame.loc[~frame["source1_entity_id"].astype(str).isin(excluded_ids)].copy()
        if frame.empty:
            return frame
        return sample_candidate_negatives(frame, max_negatives, config.seed)

    model = _new_model(config, kind)
    rows = positives = 0
    if kind == "sgd":
        # Two bounded passes keep SGD genuinely out-of-core while ensuring all
        # batches use one stable StandardScaler fitted on training rows only.
        for path in part_paths:
            frame = _training_part(path)
            if not frame.empty:
                model.update_scaler(feature_matrix(frame))
                rows += len(frame)
                positives += int(frame["label"].sum())
        if rows == 0:
            raise ValueError("no training pairs remain after fold selection")
        for path in part_paths:
            frame = _training_part(path)
            if frame.empty:
                continue
            X = feature_matrix(frame)
            y = frame["label"].to_numpy(dtype=np.int8)
            weights = frame["sample_weight"].to_numpy(dtype=np.float32)
            model.partial_fit_scaled(X, y, weights)
    else:
        sampled = [_training_part(path) for path in part_paths]
        sampled = [frame for frame in sampled if not frame.empty]
        if not sampled:
            raise ValueError("no training pairs remain after fold selection")
        features = pd.concat(sampled, ignore_index=True)
        rows, positives = len(features), int(features["label"].sum())
        X, y = feature_matrix(features), features["label"].to_numpy(dtype=np.int8)
        weights = features["sample_weight"].to_numpy(dtype=np.float32)
        if kind == "lightgbm":
            model.fit(X, y, sample_weight=weights, logger=logger)
        else:
            model.fit(X, y, sample_weight=weights)

    logger.info("training data ready", rows=rows, positives=positives)
    suffix = ".txt" if kind == "lightgbm" else ".joblib" if kind == "sgd" else ".json"
    output_name = getattr(args, "output_name", None) or kind
    path = config.artifact_dir("models") / f"{output_name}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    write_manifest(
        path.with_suffix(path.suffix + ".manifest.json"),
        stage=f"train-{kind}",
        config=config.as_dict(),
        rows=rows,
        schema=FEATURE_COLUMNS,
        metrics={
            "positive_rows": positives,
            "validation_fold": None if args.all_training_data else args.validation_fold,
            "excluded_folds": sorted(excluded_folds),
            "all_training_data": bool(args.all_training_data),
            "model_metadata": model.metadata(),
            "max_negatives_per_entity": max_negatives,
        },
    )
    logger.stage_end("train", rows=rows, positives=positives)
    print(path)
    return 0


def _feature_store(config: ProjectConfig, split: str) -> ShardStore:
    directory = config.artifact_dir("features") / split
    return ShardStore(directory, stage_fingerprint(directory))


def _candidate_store(config: ProjectConfig, split: str) -> ShardStore:
    directory = config.artifact_dir("candidates") / split
    return ShardStore(directory, stage_fingerprint(directory))


def command_score(args: argparse.Namespace) -> int:
    config = _load_config(args)
    kind = _model_kind(config, args)
    feature_store = _feature_store(config, args.split)
    candidate_store = _candidate_store(config, args.split)
    folds = None
    if args.split == "train" and not args.all_training_data:
        folds = read_frame(config.artifact_dir("splits") / "s1_folds.parquet")
    model_path = Path(args.model_path) if args.model_path else _default_model_path(config, kind)
    model = _load_model(kind, model_path)
    output_name = getattr(args, "output_name", None) or f"{args.split}.parquet"
    if not output_name.endswith(".parquet"):
        output_name += ".parquet"
    out = config.artifact_dir("scores") / output_name
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    writer = None
    scored_rows = 0
    try:
        for feature_path in feature_store.iter_parts():
            key = int(feature_path.stem.split("-")[1])
            features = read_frame(feature_path)
            candidates = read_frame(candidate_store.part_path(key))
            _assert_candidate_feature_alignment(candidates, features)
            if folds is not None:
                features = select_pair_fold(features, folds, args.validation_fold, validation=True)
            if features.empty:
                continue
            scored = features[["source1_entity_id", "candidate_entity_id", "candidate_source"]].copy()
            scored["score"] = model.predict_scores(feature_matrix(features))
            if "label" in features:
                scored["label"] = features["label"]
            table = pa.Table.from_pandas(scored, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="snappy")
            writer.write_table(table)
            scored_rows += len(scored)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        empty_columns = ["source1_entity_id", "candidate_entity_id", "candidate_source", "score"]
        out = write_frame(pd.DataFrame(columns=empty_columns), out)
    else:
        os.replace(temporary, out)
    write_manifest(
        out.with_suffix(out.suffix + ".manifest.json"),
        stage=f"score-{kind}",
        config=config.as_dict(),
        rows=scored_rows,
        schema=list(scored.columns) if scored_rows else empty_columns,
        inputs=[str(model_path)],
        metrics={"validation_fold": None if args.all_training_data else args.validation_fold},
    )
    print(out)
    return 0


def command_tune_decision(args: argparse.Namespace) -> int:
    config = _load_config(args)
    logger = stage_logger(config, "tune-decision")
    scored = read_frame(_score_path(config, args.score_file))
    validation_ids = _validation_entity_ids(config, args.validation_fold)
    validation_set = set(validation_ids)
    scored = scored.loc[scored["source1_entity_id"].astype(str).isin(validation_set)].copy()
    truth = ground_truth_sets(_training_truth(config, validation_ids))
    thresholds = score_quantile_thresholds(scored, args.threshold_count)
    report = sweep_thresholds(scored, truth, thresholds)
    out = write_frame(report, config.artifact_dir("decisions") / "threshold_sweep.parquet")
    selected = report.iloc[0].to_dict()
    selected.update({"validation_fold": int(args.validation_fold), "score_file": str(args.score_file)})
    selected_path = config.artifact_dir("decisions") / "selected_threshold.json"
    selected_path.write_text(
        json.dumps(
            selected,
            indent=2,
            sort_keys=True,
            default=lambda value: value.item() if hasattr(value, "item") else str(value),
        ),
        encoding="utf-8",
    )
    for rank, row in report.head(10).iterrows():
        logger.metric("threshold_sweep", float(row["macro_f0_5"]), rank=int(rank), threshold=float(row["threshold"]), precision=float(row["macro_precision"]), recall=float(row["macro_recall"]))
    print(report.head(10).to_string(index=False)); print(out)
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    config = _load_config(args)
    logger = stage_logger(config, "evaluate")
    scored = read_frame(_score_path(config, args.score_file))
    validation_ids = _validation_entity_ids(config, args.validation_fold)
    validation_set = set(validation_ids)
    scored = scored.loc[scored["source1_entity_id"].astype(str).isin(validation_set)].copy()
    truth = ground_truth_sets(_training_truth(config, validation_ids))
    threshold = _decision_threshold(config, args.threshold)
    predictions = apply_thresholds(
        scored,
        truth.keys(),
        threshold,
        source_thresholds=config.model.get("source_thresholds", {}),
    )
    metrics = evaluate_entity_sets(truth, predictions)
    for name, value in metrics.items():
        logger.metric(name, value)
    _json_print(metrics)
    return 0


def command_analyze_errors(args: argparse.Namespace) -> int:
    config = _load_config(args)
    scored = read_frame(_score_path(config, args.score_file))
    candidates = _candidate_store(config, "train").read_all()
    validation_ids = _validation_entity_ids(config, args.validation_fold)
    scored_ids = set(validation_ids)
    scored = scored.loc[scored["source1_entity_id"].astype(str).isin(scored_ids)].copy()
    truth = ground_truth_sets(_training_truth(config, validation_ids))
    threshold = _decision_threshold(config, args.threshold)
    predictions = apply_thresholds(scored, truth.keys(), threshold, config.model.get("source_thresholds", {}))
    candidate_sets = sets_from_long(candidates.loc[candidates["source1_entity_id"].isin(scored_ids)], "candidate_entity_id")
    errors = analyze_errors(truth, candidate_sets, predictions, scored)
    out = write_frame(errors, config.artifact_dir("reports") / "errors.parquet")
    print(errors["error_type"].value_counts().to_string() if not errors.empty else "no errors"); print(out)
    return 0


def command_infer(args: argparse.Namespace) -> int:
    config = _load_config(args)
    kind = _model_kind(config, args)
    plan = config.resource_plan()
    logger = stage_logger(config, "infer")
    logger.stage_start("infer", model=kind)
    if not _candidate_store(config, "test").is_complete():
        command_generate_candidates(argparse.Namespace(**{**vars(args), "split": "test"}))
    if not _feature_store(config, "test").is_complete():
        command_build_features(argparse.Namespace(**{**vars(args), "split": "test"}))
    feature_store = _feature_store(config, "test")
    candidate_store = _candidate_store(config, "test")
    model_path = Path(args.model_path) if args.model_path else _default_model_path(config, kind)
    model = _load_model(kind, model_path)
    threshold = _decision_threshold(config, args.threshold)
    score_fingerprint = input_fingerprint(
        [model_path, feature_store.stage_manifest_path(), candidate_store.stage_manifest_path()]
    )
    score_store = ShardStore(config.artifact_dir("scores") / "test-shards", score_fingerprint)
    if _force(args):
        shutil.rmtree(score_store.stage_dir, ignore_errors=True)
        score_store = ShardStore(config.artifact_dir("scores") / "test-shards", score_fingerprint)

    def _output_shards():
        for feature_path in feature_store.iter_parts():
            key = int(feature_path.stem.split("-")[1])
            candidates = read_frame(candidate_store.part_path(key))
            if key in score_store.completed_keys():
                scored = read_frame(score_store.part_path(key))
                _assert_candidate_feature_alignment(candidates, scored)
            else:
                features = read_frame(feature_path)
                _assert_candidate_feature_alignment(candidates, features)
                scored = features[["source1_entity_id", "candidate_entity_id", "candidate_source"]].copy()
                scores = np.empty(len(features), dtype=np.float32)
                for start in range(0, len(features), max(1, plan.chunk_pairs)):
                    stop = min(len(features), start + max(1, plan.chunk_pairs))
                    scores[start:stop] = model.predict_scores(feature_matrix(features.iloc[start:stop]))
                scored["score"] = scores
                score_store.commit(key, scored)
            source1 = _read_prepared_part(
                config, "test", 1, key, columns=["entity_id"]
            )
            predictions = apply_thresholds(
                scored,
                source1["entity_id"],
                threshold,
                source_thresholds=config.model.get("source_thresholds", {}),
            )
            candidate_sets = sets_from_long(candidates, "candidate_entity_id")
            yield source1["entity_id"].astype(str), predictions, candidate_sets

    matching, candidate = write_submission_outputs_sharded(config.output_root, _output_shards())
    sample_part = next(score_store.iter_parts(), None)
    score_schema = list(read_frame(sample_part).columns) if sample_part is not None else []
    score_store.mark_complete(score_store.total_rows(), score_schema, extra={"model": kind})
    logger.info("wrote outputs", matching=str(matching), candidate=str(candidate))
    logger.stage_end("infer")
    print(matching); print(candidate)
    return 0


def command_write_output(args: argparse.Namespace) -> int:
    """Reformat outputs only from cached scores + candidates (never reruns the model)."""
    config = _load_config(args)
    logger = stage_logger(config, "write-output")
    threshold = _decision_threshold(config, args.threshold)
    if args.split == "test":
        score_dir = config.artifact_dir("scores") / "test-shards"
        score_store = ShardStore(score_dir, stage_fingerprint(score_dir))
        candidate_store = _candidate_store(config, "test")
        if not score_store.is_complete():
            raise FileNotFoundError(f"complete cached score shards not found: {score_dir}; run infer first")

        def _output_shards():
            for score_path in score_store.iter_parts():
                key = int(score_path.stem.split("-")[1])
                scored = read_frame(score_path)
                candidates = read_frame(candidate_store.part_path(key))
                _assert_candidate_feature_alignment(candidates, scored)
                source1 = _read_prepared_part(
                    config, "test", 1, key, columns=["entity_id"]
                )
                predictions = apply_thresholds(
                    scored,
                    source1["entity_id"],
                    threshold,
                    source_thresholds=config.model.get("source_thresholds", {}),
                )
                yield source1["entity_id"].astype(str), predictions, sets_from_long(candidates, "candidate_entity_id")

        matching, candidate = write_submission_outputs_sharded(config.output_root, _output_shards())
        logger.event("outputs_rewritten", matching=str(matching), candidate=str(candidate))
        print(matching); print(candidate)
        return 0

    candidates = _candidate_store(config, args.split).read_all()
    source1 = _read_prepared(config, args.split, 1)
    scored_path = config.artifact_dir("scores") / f"{args.split}.parquet"
    if not scored_path.is_file():
        raise FileNotFoundError(f"cached scores not found: {scored_path}; run score first")
    scored = read_frame(scored_path)
    _assert_candidate_feature_alignment(candidates, scored)
    predictions = apply_thresholds(
        scored,
        source1["entity_id"],
        threshold,
        source_thresholds=config.model.get("source_thresholds", {}),
    )
    candidate_sets = sets_from_long(candidates, "candidate_entity_id")
    matching, candidate = write_submission_outputs(config.output_root, source1["entity_id"], predictions, candidate_sets)
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
    matching = config.output_root / "matching_results.tsv"
    candidates = config.output_root / "candidate_pairs.tsv"
    errors = preflight_outputs(matching, candidates, config.data_root / "test")
    if errors:
        raise ValueError("submission preflight failed: " + "; ".join(errors[:10]))
    if args.official_validator:
        result = run_official_validator(
            args.official_validator,
            matching,
            candidates,
            config.data_root / "test",
            check_ids=True,
        )
        if result.returncode:
            raise ValueError(f"official validator failed:\n{result.stdout}\n{result.stderr}")

    package_root = config.artifact_dir("package") / f"{args.team_name}_submission"
    # A package is an immutable snapshot; never merge into stale staging files.
    shutil.rmtree(package_root, ignore_errors=True)
    (package_root / "output").mkdir(parents=True, exist_ok=True)
    shutil.copy2(matching, package_root / "output" / matching.name)
    shutil.copy2(candidates, package_root / "output" / candidates.name)
    project_root = Path(__file__).resolve().parents[2]
    shutil.copytree(project_root, package_root / "code" / "business_entity_resolution", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
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
    train = add_common("train", command_train); train.add_argument("--model", choices=["deterministic", "sgd", "lightgbm"]); train.add_argument("--validation-fold", type=int, default=0); train.add_argument("--exclude-folds", help="Comma-separated folds excluded from training; defaults to --validation-fold."); train.add_argument("--all-training-data", action="store_true"); train.add_argument("--output-name")
    score = add_common("score", command_score); score.add_argument("--split", choices=["train", "test"], required=True); score.add_argument("--model", choices=["deterministic", "sgd", "lightgbm"]); score.add_argument("--model-path"); score.add_argument("--validation-fold", type=int, default=0); score.add_argument("--all-training-data", action="store_true"); score.add_argument("--output-name")
    tune = add_common("tune-decision", command_tune_decision); tune.add_argument("--threshold-count", type=int, default=101); tune.add_argument("--score-file", default="train.parquet"); tune.add_argument("--validation-fold", type=int, default=0)
    evaluate = add_common("evaluate", command_evaluate); evaluate.add_argument("--score-file", default="train.parquet"); evaluate.add_argument("--threshold", type=float); evaluate.add_argument("--validation-fold", type=int, default=0)
    errors = add_common("analyze-errors", command_analyze_errors); errors.add_argument("--score-file", default="train.parquet"); errors.add_argument("--threshold", type=float); errors.add_argument("--validation-fold", type=int, default=0)
    infer = add_common("infer", command_infer); infer.add_argument("--model", choices=["deterministic", "sgd", "lightgbm"]); infer.add_argument("--model-path"); infer.add_argument("--threshold", type=float); infer.add_argument("--no-tfidf", action="store_true")
    write_output = add_common("write-output", command_write_output); write_output.add_argument("--split", choices=["train", "test"], default="test"); write_output.add_argument("--threshold", type=float)
    preflight = add_common("preflight", command_preflight); preflight.add_argument("--official-validator"); preflight.add_argument("--check-ids", action="store_true")
    package = add_common("package", command_package); package.add_argument("--team-name", required=True); package.add_argument("--documentation", required=True); package.add_argument("--official-validator")
    mini = add_common("make-mini", command_make_mini); mini.add_argument("--mini-root", default="dataset/mini"); mini.add_argument("--count", type=int, default=15); mini.add_argument("--source-root", default=None)
    add_common("smoke", command_smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
