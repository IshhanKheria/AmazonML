"""Bounded, chunked source-data audit."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .data import iter_tsv, source_path, ground_truth_path
from .schemas import SOURCE_COLUMNS, GROUND_TRUTH_COLUMNS
from .labels import parse_match_ids


def audit_source(path: str | Path, source: int, batch_size: int = 100_000, max_rows: int | None = None) -> dict[str, Any]:
    prefix = f"S{source}-"
    rows = 0
    blanks = Counter()
    countries = Counter()
    ids: set[str] = set()
    duplicate_ids = bad_ids = 0
    for chunk in iter_tsv(path, SOURCE_COLUMNS, batch_size, max_rows):
        rows += len(chunk)
        countries.update(chunk["country"])
        for column in SOURCE_COLUMNS:
            blanks[column] += int(chunk[column].eq("").sum())
        for entity_id in chunk["entity_id"]:
            bad_ids += int(not (entity_id.startswith(prefix) and entity_id[len(prefix):].isdigit()))
            duplicate_ids += int(entity_id in ids)
            ids.add(entity_id)
    return {
        "path": str(path), "rows": rows, "columns": list(SOURCE_COLUMNS),
        "blank_counts": dict(blanks), "countries": dict(countries),
        "duplicate_entity_ids": duplicate_ids, "bad_entity_ids": bad_ids,
    }


def audit_ground_truth(path: str | Path, batch_size: int = 100_000, max_rows: int | None = None) -> dict[str, Any]:
    rows = positives = singletons = duplicate_s1 = malformed = 0
    ids: set[str] = set()
    cardinality = Counter()
    for chunk in iter_tsv(path, GROUND_TRUTH_COLUMNS, batch_size, max_rows):
        for row in chunk.itertuples(index=False):
            rows += 1
            duplicate_s1 += int(row.source1_entity_id in ids)
            ids.add(row.source1_entity_id)
            try:
                matches = parse_match_ids(row.matched_entity_ids)
            except ValueError:
                malformed += 1
                continue
            cardinality[len(matches)] += 1
            positives += len(matches)
            singletons += int(not matches)
    return {
        "path": str(path), "rows": rows, "columns": list(GROUND_TRUTH_COLUMNS),
        "positive_links": positives, "singletons": singletons,
        "singleton_rate": singletons / rows if rows else 0.0,
        "cardinality": dict(sorted(cardinality.items())),
        "duplicate_source1_ids": duplicate_s1, "malformed_rows": malformed,
    }


def audit_dataset(data_root: str | Path, batch_size: int = 100_000, max_rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"sources": {}}
    for split in ("train", "test"):
        for source in (1, 2, 3):
            key = f"{split}_source{source}"
            result["sources"][key] = audit_source(source_path(data_root, split, source), source, batch_size, max_rows)
    result["ground_truth"] = audit_ground_truth(ground_truth_path(data_root), batch_size, max_rows)
    return result

