"""Ground-truth parsing and pair labeling."""

from __future__ import annotations

from collections.abc import Mapping
import pandas as pd

from .schemas import GROUND_TRUTH_COLUMNS, SchemaError, require_columns


def parse_match_ids(value: str) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    ids = tuple(value.split(","))
    if any(not item.startswith(("S2-", "S3-")) for item in ids):
        raise SchemaError(f"ground truth contains non-S2/S3 ID: {value!r}")
    if len(ids) != len(set(ids)):
        raise SchemaError(f"ground truth contains duplicate target ID: {value!r}")
    return ids


def ground_truth_sets(frame: pd.DataFrame) -> dict[str, frozenset[str]]:
    require_columns(list(frame.columns), GROUND_TRUTH_COLUMNS, "ground truth")
    if frame["source1_entity_id"].duplicated().any():
        raise SchemaError("ground truth contains duplicate source1_entity_id rows")
    if (~frame["source1_entity_id"].str.match(r"^S1-\d+$")).any():
        raise SchemaError("ground truth contains malformed S1 IDs")
    return {
        row.source1_entity_id: frozenset(parse_match_ids(row.matched_entity_ids))
        for row in frame.itertuples(index=False)
    }


def explode_ground_truth(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [
        (source1_id, target_id)
        for source1_id, targets in ground_truth_sets(frame).items()
        for target_id in sorted(targets)
    ]
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])


def label_candidates(candidates: pd.DataFrame, truth: Mapping[str, set[str] | frozenset[str]]) -> pd.DataFrame:
    required = {"source1_entity_id", "candidate_entity_id"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"candidate table missing columns: {sorted(required - set(candidates.columns))}")
    out = candidates.copy()
    out["label"] = [
        int(candidate in truth.get(source1_id, frozenset()))
        for source1_id, candidate in zip(out["source1_entity_id"], out["candidate_entity_id"])
    ]
    return out


def force_add_training_positives(candidates: pd.DataFrame, truth: Mapping[str, set[str] | frozenset[str]]) -> pd.DataFrame:
    existing = set(zip(candidates["source1_entity_id"], candidates["candidate_entity_id"]))
    rows = []
    for source1_id, targets in truth.items():
        for target_id in targets:
            if (source1_id, target_id) not in existing:
                rows.append({
                    "source1_entity_id": source1_id,
                    "candidate_entity_id": target_id,
                    "candidate_source": target_id[:2],
                    "reason_mask": "forced_positive",
                    "retrieval_score": 1.0,
                    "retrieval_rank": 0,
                    "forced_positive": True,
                })
    base = candidates.copy()
    if "forced_positive" not in base:
        base["forced_positive"] = False
    return pd.concat([base, pd.DataFrame(rows)], ignore_index=True, sort=False) if rows else base

