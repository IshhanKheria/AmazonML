"""Deterministic candidate-negative sampling and entity-balanced weights."""

from __future__ import annotations

import pandas as pd

from .splits import stable_hash


def sample_candidate_negatives(frame: pd.DataFrame, max_negatives_per_entity: int = 50, seed: int = 2026) -> pd.DataFrame:
    if "label" not in frame:
        raise ValueError("labeled feature table required")
    positives = frame[frame["label"] == 1]
    negatives = frame[frame["label"] == 0].copy()
    negatives["_random"] = [
        stable_hash(f"{source1_id}|{candidate_id}", seed)
        for source1_id, candidate_id in zip(negatives["source1_entity_id"], negatives["candidate_entity_id"])
    ]
    negatives = negatives.sort_values(
        ["source1_entity_id", "retrieval_score", "_random", "candidate_entity_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    negatives = negatives[negatives.groupby("source1_entity_id").cumcount() < max_negatives_per_entity].drop(columns="_random")
    sampled = pd.concat([positives, negatives], ignore_index=True).sort_values(
        ["source1_entity_id", "candidate_entity_id"], kind="mergesort"
    )
    return add_entity_balanced_weights(sampled)


def add_entity_balanced_weights(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    weights = pd.Series(0.0, index=out.index)
    for _, group in out.groupby("source1_entity_id", sort=False):
        positive = group[group["label"] == 1]
        negative = group[group["label"] == 0]
        if len(positive) and len(negative):
            weights.loc[positive.index] = 0.5 / len(positive)
            weights.loc[negative.index] = 0.5 / len(negative)
        elif len(positive):
            weights.loc[positive.index] = 1.0 / len(positive)
        elif len(negative):
            weights.loc[negative.index] = 1.0 / len(negative)
    out["sample_weight"] = weights.astype("float32")
    return out
