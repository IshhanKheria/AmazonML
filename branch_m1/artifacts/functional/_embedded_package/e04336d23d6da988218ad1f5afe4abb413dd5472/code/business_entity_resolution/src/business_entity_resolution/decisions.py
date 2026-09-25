"""Score-to-match decision rules and threshold optimization."""

from __future__ import annotations

from collections.abc import Mapping, Iterable
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
    return sorted(set(float(value) for value in scored_pairs["score"].quantile(quantiles)))


def apply_thresholds_with_france(
    scored_pairs: pd.DataFrame,
    all_source1_ids: Iterable[str],
    threshold: float,
    *,
    source1: pd.DataFrame | None = None,
    france_margin: float = 0.05,
    high_confidence_name: float = 0.0,
    source_thresholds: Mapping[str, float] | None = None,
) -> dict[str, frozenset[str]]:
    """Apply thresholds with a conservative France margin and a high-confidence rule.

    Training contains no France labels, so the France threshold is *raised* by a
    fixed margin (a heuristic, not a fitted value) to protect precision under
    distribution shift. A high-confidence override keeps pairs whose cleaned name
    is an exact match and which carry numeric address evidence — exploiting the
    observed 79% exact-name overlap in France.
    """
    source_thresholds = source_thresholds or {}
    predictions: dict[str, set[str]] = {str(entity_id): set() for entity_id in all_source1_ids}
    if source1 is None or "country_norm" not in source1.columns:
        return apply_thresholds(scored_pairs, all_source1_ids, threshold, source_thresholds)
    country_by_s1 = dict(zip(source1["entity_id"].astype(str), source1["country_norm"].astype(str)))
    for row in scored_pairs.itertuples(index=False):
        candidate_id = str(row.candidate_entity_id)
        source = str(getattr(row, "candidate_source", candidate_id[:2]))
        cutoff = float(source_thresholds.get(source, threshold))
        country = country_by_s1.get(str(row.source1_entity_id), "")
        if country == "france":
            cutoff += france_margin
            override = (
                float(getattr(row, "name_core_exact", 0.0)) >= 1.0
                and float(getattr(row, "numeric_overlap", 0.0)) >= 1.0
            )
            if float(row.score) < cutoff and not override:
                continue
        elif float(row.score) < cutoff:
            continue
        predictions.setdefault(str(row.source1_entity_id), set()).add(candidate_id)
    return {key: frozenset(value) for key, value in predictions.items()}

