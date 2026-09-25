"""Leakage-safe deterministic Source-1 fold assignment."""

from __future__ import annotations

import hashlib
import pandas as pd

from .labels import parse_match_ids


def stable_hash(value: str, seed: int = 2026) -> int:
    payload = f"{seed}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _cardinality_bin(count: int) -> str:
    if count <= 2:
        return str(count)
    return "3-4" if count <= 4 else "5+"


def assign_s1_folds(source1: pd.DataFrame, ground_truth: pd.DataFrame, n_folds: int = 10, seed: int = 2026) -> pd.DataFrame:
    if n_folds < 3:
        raise ValueError("n_folds must be at least 3")
    gt = ground_truth.copy()
    gt["matches"] = gt["matched_entity_ids"].map(parse_match_ids)
    gt["cardinality"] = gt["matches"].map(len)
    gt["has_s2"] = gt["matches"].map(lambda ids: any(value.startswith("S2-") for value in ids))
    gt["has_s3"] = gt["matches"].map(lambda ids: any(value.startswith("S3-") for value in ids))
    gt["source_pattern"] = [
        "both" if s2 and s3 else "s2" if s2 else "s3" if s3 else "neither"
        for s2, s3 in zip(gt["has_s2"], gt["has_s3"])
    ]
    merged = source1[["entity_id", "country"]].merge(
        gt[["source1_entity_id", "cardinality", "source_pattern"]],
        left_on="entity_id",
        right_on="source1_entity_id",
        how="left",
        validate="one_to_one",
    )
    if merged["cardinality"].isna().any():
        raise ValueError("ground truth does not cover every Source 1 row")
    merged["cardinality"] = merged["cardinality"].astype(int)
    merged["cardinality_bin"] = merged["cardinality"].map(_cardinality_bin)
    merged["singleton"] = merged["cardinality"].eq(0)
    merged["stratum"] = (
        merged["country"].astype(str) + "|" + merged["singleton"].astype(str) + "|"
        + merged["source_pattern"] + "|" + merged["cardinality_bin"]
    )
    merged["_hash"] = merged["entity_id"].map(lambda value: stable_hash(value, seed))
    merged = merged.sort_values(["stratum", "_hash", "entity_id"], kind="mergesort")
    merged["fold"] = merged.groupby("stratum", sort=False).cumcount() % n_folds
    return merged[["entity_id", "fold", "stratum", "cardinality", "singleton", "source_pattern"]].rename(
        columns={"entity_id": "source1_entity_id"}
    )


def select_pair_fold(
    pairs: pd.DataFrame,
    folds: pd.DataFrame,
    validation_fold: int,
    *,
    validation: bool,
) -> pd.DataFrame:
    """Select training or held-out pairs by Source 1 entity, never by pair row."""
    joined = pairs.merge(
        folds[["source1_entity_id", "fold"]],
        on="source1_entity_id",
        how="left",
        validate="many_to_one",
    )
    if joined["fold"].isna().any():
        raise ValueError("some pairs have no Source 1 fold assignment")
    mask = joined["fold"].eq(validation_fold)
    return joined.loc[mask if validation else ~mask].drop(columns="fold").reset_index(drop=True)
