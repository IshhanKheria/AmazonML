"""Hard-negative mining from out-of-fold scored candidates."""

from __future__ import annotations

import pandas as pd


def mine_hard_negatives(scored_labeled: pd.DataFrame, per_entity: int = 10, minimum_score: float = 0.0) -> pd.DataFrame:
    required = {"source1_entity_id", "candidate_entity_id", "label", "score"}
    if not required.issubset(scored_labeled.columns):
        raise ValueError(f"hard-negative table missing: {sorted(required - set(scored_labeled.columns))}")
    negatives = scored_labeled[(scored_labeled["label"] == 0) & (scored_labeled["score"] >= minimum_score)].copy()
    negatives = negatives.sort_values(
        ["source1_entity_id", "score", "candidate_entity_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return negatives[negatives.groupby("source1_entity_id").cumcount() < per_entity].reset_index(drop=True)

