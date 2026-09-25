"""Score-to-match decision rules and threshold optimization."""

from __future__ import annotations

from collections.abc import Mapping, Iterable
import math
import pandas as pd

from .metrics import evaluate_entity_sets


def apply_thresholds(
    scored_pairs: pd.DataFrame,
    all_source1_ids: Iterable[str],
    threshold: float,
    source_thresholds: Mapping[str, float] | None = None,
) -> dict[str, frozenset[str]]:
    source_thresholds = source_thresholds or {}
    predictions: dict[str, set[str]] = {str(entity_id): set() for entity_id in all_source1_ids}
    for row in scored_pairs.itertuples(index=False):
        candidate_id = str(row.candidate_entity_id)
        source = str(getattr(row, "candidate_source", candidate_id[:2]))
        cutoff = float(source_thresholds.get(source, threshold))
        if float(row.score) >= cutoff:
            predictions.setdefault(str(row.source1_entity_id), set()).add(candidate_id)
    return {key: frozenset(value) for key, value in predictions.items()}


def sweep_thresholds(
    scored_pairs: pd.DataFrame,
    truth: Mapping[str, set[str] | frozenset[str]],
    thresholds: Iterable[float],
) -> pd.DataFrame:
    rows = []
    for threshold in sorted(set(float(value) for value in thresholds)):
        predictions = apply_thresholds(scored_pairs, truth.keys(), threshold)
        metrics = evaluate_entity_sets(truth, predictions)
        rows.append({"threshold": threshold, **metrics})
    return pd.DataFrame(rows).sort_values(
        ["macro_f0_5", "macro_precision", "singleton_accuracy", "threshold"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def score_quantile_thresholds(scored_pairs: pd.DataFrame, count: int = 101) -> list[float]:
    if scored_pairs.empty:
        return [1.0]
    quantiles = [index / (count - 1) for index in range(count)]
    observed = [float(value) for value in scored_pairs["score"].quantile(quantiles)]
    maximum = float(scored_pairs["score"].max())
    # Include the all-empty decision, which cannot be represented by an
    # observed maximum because apply_thresholds uses >=.
    observed.extend([0.0, 1.0, math.nextafter(maximum, math.inf)])
    return sorted(set(observed))

