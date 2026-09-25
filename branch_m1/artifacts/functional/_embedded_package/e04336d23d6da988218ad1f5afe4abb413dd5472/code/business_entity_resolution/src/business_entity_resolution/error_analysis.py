"""Stage-aware validation error analysis."""

from __future__ import annotations

from collections.abc import Mapping
import pandas as pd


def analyze_errors(
    truth: Mapping[str, set[str] | frozenset[str]],
    candidate_sets: Mapping[str, set[str] | frozenset[str]],
    predictions: Mapping[str, set[str] | frozenset[str]],
    scored_pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    score_lookup: dict[tuple[str, str], float] = {}
    if scored_pairs is not None and not scored_pairs.empty:
        score_lookup = {
            (str(row.source1_entity_id), str(row.candidate_entity_id)): float(row.score)
            for row in scored_pairs.itertuples(index=False)
        }
    rows: list[dict[str, object]] = []
    for source1_id, targets in truth.items():
        target_set = set(targets)
        candidates = set(candidate_sets.get(source1_id, frozenset()))
        predicted = set(predictions.get(source1_id, frozenset()))
        for target_id in sorted(target_set - candidates):
            rows.append({"source1_entity_id": source1_id, "candidate_entity_id": target_id, "stage": "blocking", "error_type": "missed_true_pair", "score": None})
        for target_id in sorted((target_set & candidates) - predicted):
            rows.append({"source1_entity_id": source1_id, "candidate_entity_id": target_id, "stage": "model_or_decision", "error_type": "false_negative", "score": score_lookup.get((source1_id, target_id))})
        for target_id in sorted(predicted - target_set):
            rows.append({"source1_entity_id": source1_id, "candidate_entity_id": target_id, "stage": "model_or_decision", "error_type": "singleton_false_positive" if not target_set else "false_merge", "score": score_lookup.get((source1_id, target_id))})
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "stage", "error_type", "score"])


def high_confidence_pairs(scored_labeled: pd.DataFrame, correct: bool, limit: int = 100) -> pd.DataFrame:
    required = {"label", "score"}
    if not required.issubset(scored_labeled.columns):
        raise ValueError("scored labeled pairs require label and score")
    subset = scored_labeled[scored_labeled["label"].eq(1 if correct else 0)]
    return subset.nlargest(limit, "score")

