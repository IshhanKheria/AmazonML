"""Score-to-match decision rules and threshold optimization."""

from __future__ import annotations

from collections.abc import Mapping, Iterable
import numpy as np
import pandas as pd

from .metrics import evaluate_entity_sets


def _group_selected(selected: pd.DataFrame) -> dict[str, frozenset[str]]:
    if selected.empty:
        return {}
    grouped = selected.groupby("source1_entity_id", sort=False)["candidate_entity_id"].agg(
        lambda values: frozenset(str(value) for value in values)
    )
    return {str(key): value for key, value in grouped.items()}


def apply_thresholds(
    scored_pairs: pd.DataFrame,
    all_source1_ids: Iterable[str],
    threshold: float,
    source_thresholds: Mapping[str, float] | None = None,
) -> dict[str, frozenset[str]]:
    source_thresholds = source_thresholds or {}
    predictions: dict[str, frozenset[str]] = {str(entity_id): frozenset() for entity_id in all_source1_ids}
    if scored_pairs.empty:
        return predictions
    sources = scored_pairs["candidate_source"].astype(str)
    cutoffs = sources.map(lambda source: float(source_thresholds.get(source, threshold))).to_numpy()
    selected = scored_pairs.loc[scored_pairs["score"].to_numpy() >= cutoffs]
    predictions.update(_group_selected(selected))
    return predictions


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


def tune_source_thresholds(
    scored_pairs: pd.DataFrame,
    truth: Mapping[str, set[str] | frozenset[str]],
    thresholds: Iterable[float],
    *,
    source1: pd.DataFrame | None = None,
    france_margin: float = 0.05,
    max_passes: int = 1,
) -> dict[str, object]:
    """Pick the global threshold, then coordinate-ascent per candidate source.

    The competition metric is macro over Source 1 entities, so thresholds are
    selected by macro F0.5 (never pair accuracy). Returns
    ``{"threshold", "source_thresholds", "macro_f0_5"}``.
    """
    candidates = sorted(set(float(value) for value in thresholds))
    if not candidates:
        return {"threshold": 0.8, "source_thresholds": {}, "macro_f0_5": 0.0}

    def macro(threshold: float, source_thresholds: Mapping[str, float]) -> float:
        if source1 is not None:
            predictions = apply_thresholds_with_france(
                scored_pairs, truth.keys(), threshold,
                source1=source1, france_margin=france_margin, source_thresholds=source_thresholds,
            )
        else:
            predictions = apply_thresholds(scored_pairs, truth.keys(), threshold, source_thresholds)
        return float(evaluate_entity_sets(truth, predictions)["macro_f0_5"])

    best_threshold = max(candidates, key=lambda value: (macro(value, {}), -value))
    source_thresholds: dict[str, float] = {}
    sources = sorted(set(scored_pairs["candidate_source"].astype(str))) if "candidate_source" in scored_pairs.columns else []
    for _ in range(max(1, max_passes)):
        improved = False
        for source in sources:
            current = source_thresholds.get(source)
            best_local, best_local_score = current, macro(best_threshold, source_thresholds)
            for value in candidates:
                trial = {**source_thresholds, source: value}
                candidate_score = macro(best_threshold, trial)
                if candidate_score > best_local_score + 1e-12:
                    best_local, best_local_score = value, candidate_score
            if best_local != current:
                source_thresholds[source] = best_local
                improved = True
        if not improved:
            break
    source_thresholds = {source: value for source, value in source_thresholds.items() if abs(value - best_threshold) > 1e-12}
    return {
        "threshold": float(best_threshold),
        "source_thresholds": source_thresholds,
        "macro_f0_5": macro(best_threshold, source_thresholds),
    }


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
    predictions: dict[str, frozenset[str]] = {str(entity_id): frozenset() for entity_id in all_source1_ids}
    if source1 is None or "country_norm" not in source1.columns:
        return apply_thresholds(scored_pairs, all_source1_ids, threshold, source_thresholds)
    if scored_pairs.empty:
        return predictions
    country_by_s1 = dict(zip(source1["entity_id"].astype(str), source1["country_norm"].astype(str)))
    sources = scored_pairs["candidate_source"].astype(str)
    cutoffs = sources.map(lambda source: float(source_thresholds.get(source, threshold))).to_numpy()
    countries = scored_pairs["source1_entity_id"].astype(str).map(lambda value: country_by_s1.get(value, "")).to_numpy()
    is_france = countries == "france"
    cutoffs = cutoffs + np.where(is_france, float(france_margin), 0.0)
    if "name_core_exact" in scored_pairs.columns:
        name_exact = scored_pairs["name_core_exact"].to_numpy(dtype=float)
    else:
        name_exact = np.zeros(len(scored_pairs))
    if "numeric_overlap" in scored_pairs.columns:
        numeric = scored_pairs["numeric_overlap"].to_numpy(dtype=float)
    else:
        numeric = np.zeros(len(scored_pairs))
    override = (name_exact >= 1.0) & (numeric >= 1.0)
    keep = (scored_pairs["score"].to_numpy() >= cutoffs) | (is_france & override)
    predictions.update(_group_selected(scored_pairs.loc[keep]))
    return predictions

