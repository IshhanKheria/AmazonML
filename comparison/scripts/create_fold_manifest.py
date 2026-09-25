#!/usr/bin/env python3
"""Create the exact shared entity-fold manifest with bounded memory.

This is a disk-backed implementation of business_entity_resolution.splits.assign_s1_folds:
same strata, BLAKE2b hash, stable `(stratum, hash, entity_id)` ordering, and
round-robin fold assignment. Source 1 and ground truth are known to be aligned,
but alignment is asserted for every streamed chunk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def stable_hash(value: str, seed: int) -> int:
    payload = f"{seed}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def safe_name(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    train = args.data_root / "train"
    source_path = train / "train_source1.tsv"
    truth_path = train / "train_ground_truth.tsv"
    temp_root = args.output.parent / ".fold-parts"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True)
    writers: dict[str, pq.ParquetWriter] = {}
    strata: dict[str, str] = {}
    row_count = 0

    database_path = temp_root / "fold_join.sqlite"
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE truth (entity_id TEXT PRIMARY KEY, cardinality INTEGER, source_pattern TEXT)")
    connection.execute("CREATE TABLE source (entity_id TEXT PRIMARY KEY, country TEXT)")
    truth_rows = 0
    for truth in pd.read_csv(
        truth_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=args.chunk_size,
    ):
        matches = truth["matched_entity_ids"].astype(str)
        cardinality = np.where(matches.eq(""), 0, matches.str.count(",").to_numpy() + 1).astype(np.int16)
        has_s2 = matches.str.contains("S2-", regex=False).to_numpy()
        has_s3 = matches.str.contains("S3-", regex=False).to_numpy()
        source_pattern = np.select(
            [has_s2 & has_s3, has_s2, has_s3],
            ["both", "s2", "s3"],
            default="neither",
        )
        connection.executemany(
            "INSERT INTO truth VALUES (?, ?, ?)",
            zip(truth["source1_entity_id"].astype(str), cardinality.tolist(), source_pattern.tolist()),
        )
        truth_rows += len(truth)
        print(f"loaded {truth_rows:,} ground-truth rows", flush=True)
    connection.commit()

    source_rows = 0
    for source in pd.read_csv(
        source_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id", "country"],
        chunksize=args.chunk_size,
    ):
        connection.executemany(
            "INSERT INTO source VALUES (?, ?)",
            zip(source["entity_id"].astype(str), source["country"].astype(str)),
        )
        source_rows += len(source)
        print(f"loaded {source_rows:,} Source 1 rows", flush=True)
    connection.commit()
    if source_rows != truth_rows:
        raise RuntimeError(f"Source 1/ground-truth row count mismatch: {source_rows} != {truth_rows}")
    joined_rows = connection.execute(
        "SELECT COUNT(*) FROM source JOIN truth USING(entity_id)"
    ).fetchone()[0]
    if joined_rows != source_rows:
        raise RuntimeError(f"Source 1/ground-truth ID set mismatch: joined {joined_rows} of {source_rows}")

    cursor = connection.execute(
        "SELECT source.entity_id, source.country, truth.cardinality, truth.source_pattern "
        "FROM source JOIN truth USING(entity_id)"
    )
    while True:
        records = cursor.fetchmany(args.chunk_size)
        if not records:
            break
        joined = pd.DataFrame.from_records(
            records,
            columns=["source1_entity_id", "country", "cardinality", "source_pattern"],
        )
        source_ids = joined["source1_entity_id"].astype(str).reset_index(drop=True)
        cardinality = joined["cardinality"].to_numpy(dtype=np.int16)
        source_pattern = joined["source_pattern"].astype(str).to_numpy()
        cardinality_bin = np.select(
            [cardinality <= 2, cardinality <= 4],
            [cardinality.astype(str), np.full(len(cardinality), "3-4")],
            default="5+",
        )
        singleton = cardinality == 0
        frame = pd.DataFrame(
            {
                "source1_entity_id": source_ids,
                "cardinality": cardinality,
                "singleton": singleton,
                "source_pattern": source_pattern,
            }
        )
        frame["stratum"] = (
            joined["country"].astype(str).reset_index(drop=True)
            + "|"
            + pd.Series(singleton).astype(str)
            + "|"
            + pd.Series(source_pattern)
            + "|"
            + pd.Series(cardinality_bin)
        )
        frame["_hash"] = [stable_hash(value, args.seed) for value in source_ids]
        for stratum, group in frame.groupby("stratum", sort=False):
            key = safe_name(stratum)
            strata[key] = stratum
            table = pa.Table.from_pandas(group, preserve_index=False)
            if key not in writers:
                writers[key] = pq.ParquetWriter(temp_root / f"{key}.parquet", table.schema)
            writers[key].write_table(table)
        row_count += len(frame)
        print(f"partitioned {row_count:,} entities", flush=True)

    connection.close()

    for writer in writers.values():
        writer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_writer: pq.ParquetWriter | None = None
    fold_counts = np.zeros(args.folds, dtype=np.int64)
    for key in sorted(strata, key=lambda item: strata[item]):
        frame = pq.read_table(temp_root / f"{key}.parquet").to_pandas()
        frame = frame.sort_values(["_hash", "source1_entity_id"], kind="mergesort").reset_index(drop=True)
        frame["fold"] = np.arange(len(frame), dtype=np.int64) % args.folds
        for fold, count in frame["fold"].value_counts().items():
            fold_counts[int(fold)] += int(count)
        frame = frame[["source1_entity_id", "fold", "stratum", "cardinality", "singleton", "source_pattern"]]
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if final_writer is None:
            final_writer = pq.ParquetWriter(args.output, table.schema)
        final_writer.write_table(table)
    if final_writer is None:
        raise RuntimeError("no fold rows were generated")
    final_writer.close()
    shutil.rmtree(temp_root)

    manifest = {
        "path": str(args.output),
        "rows": int(row_count),
        "seed": args.seed,
        "n_folds": args.folds,
        "fold_counts": {str(index): int(value) for index, value in enumerate(fold_counts)},
        "sha256": file_sha256(args.output),
        "algorithm": "stratified stable BLAKE2b round-robin; exact equivalent of assign_s1_folds",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
