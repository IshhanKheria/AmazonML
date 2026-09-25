"""Competition-faithful entity-set and candidate metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import numpy as np
import pandas as pd


def entity_scores(truth: set[str] | frozenset[str], prediction: set[str] | frozenset[str], beta: float = 0.5) -> tuple[float, float, float]:
    truth_set, prediction_set = set(truth), set(prediction)
    if not truth_set and not prediction_set:
        return 1.0, 1.0, 1.0
    if not truth_set or not prediction_set:
        return 0.0, 0.0, 0.0
    tp = len(truth_set & prediction_set)
    precision, recall = tp / len(prediction_set), tp / len(truth_set)
    beta2 = beta * beta
    denominator = beta2 * precision + recall
    return precision, recall, 0.0 if denominator == 0 else (1 + beta2) * precision * recall / denominator


def evaluate_entity_sets(
    truth: Mapping[str, set[str] | frozenset[str]],
    predictions: Mapping[str, set[str] | frozenset[str]],
    beta: float = 0.5,
) -> dict[str, float | int]:
    rows: list[tuple[float, float, float]] = []
    false_merges = missed_matches = singleton_correct = singleton_total = 0
    for source1_id, target_truth in truth.items():
        predicted = set(predictions.get(source1_id, frozenset()))
        truth_set = set(target_truth)
        rows.append(entity_scores(truth_set, predicted, beta))
        false_merges += len(predicted - truth_set)
        missed_matches += len(truth_set - predicted)
        if not truth_set:
            singleton_total += 1
            singleton_correct += int(not predicted)
    values = np.asarray(rows, dtype=float)
    return {
        "macro_precision": float(values[:, 0].mean()) if len(values) else 0.0,
        "macro_recall": float(values[:, 1].mean()) if len(values) else 0.0,
        "macro_f0_5": float(values[:, 2].mean()) if len(values) else 0.0,
        "singleton_accuracy": singleton_correct / singleton_total if singleton_total else 0.0,
        "singleton_total": singleton_total,
        "false_merges": false_merges,
        "missed_matches": missed_matches,
        "entity_count": len(rows),
    }


def candidate_metrics(
    truth: Mapping[str, set[str] | frozenset[str]],
    candidates: Mapping[str, set[str] | frozenset[str]],
    target_universe_size: int,
) -> dict[str, float | int]:
    total_true = retrieved_true = non_singletons = any_found = complete = 0
    counts: list[int] = []
    for source1_id, target_truth in truth.items():
        candidate_set, truth_set = set(candidates.get(source1_id, frozenset())), set(target_truth)
        counts.append(len(candidate_set))
        total_true += len(truth_set)
        retrieved_true += len(truth_set & candidate_set)
        if truth_set:
            non_singletons += 1
            any_found += int(bool(truth_set & candidate_set))
            complete += int(truth_set.issubset(candidate_set))
    arr = np.asarray(counts, dtype=np.int64)
    total_pairs, possible = int(arr.sum()), len(truth) * target_universe_size
    return {
        "candidate_recall": retrieved_true / total_true if total_true else 1.0,
        "any_match_entity_recall": any_found / non_singletons if non_singletons else 1.0,
        "complete_entity_recall": complete / non_singletons if non_singletons else 1.0,
        "candidate_count": total_pairs,
        "average_candidates": float(arr.mean()) if len(arr) else 0.0,
        "p50_candidates": float(np.quantile(arr, 0.5)) if len(arr) else 0.0,
        "p95_candidates": float(np.quantile(arr, 0.95)) if len(arr) else 0.0,
        "p99_candidates": float(np.quantile(arr, 0.99)) if len(arr) else 0.0,
        "max_candidates": int(arr.max()) if len(arr) else 0,
        "zero_candidate_rate": float((arr == 0).mean()) if len(arr) else 0.0,
        "reduction_ratio": 1.0 - total_pairs / possible if possible else 1.0,
    }


def candidate_metrics_from_frames(
    truth: Mapping[str, set[str] | frozenset[str]],
    candidate_frames: Iterable[pd.DataFrame],
    target_universe_size: int,
) -> dict[str, float | int]:
    """Calculate exact candidate metrics without materializing all pairs.

    Candidate shards must partition Source 1 IDs, as the production sharding
    scheme does. Entities absent from every shard are included with zero
    candidates, preserving singleton and blocking-miss behavior.
    """
    total_true = sum(len(values) for values in truth.values())
    non_singletons = sum(bool(values) for values in truth.values())
    retrieved_true = any_found = complete = total_pairs = seen_entities = 0
    counts: list[int] = []
    for frame in candidate_frames:
        if frame.empty:
            continue
        for source1_id, group in frame.groupby("source1_entity_id", sort=False):
            candidate_set = set(group["candidate_entity_id"].astype(str))
            truth_set = set(truth.get(str(source1_id), frozenset()))
            count = len(candidate_set)
            counts.append(count)
            seen_entities += 1
            total_pairs += count
            retrieved_true += len(candidate_set & truth_set)
            if truth_set:
                any_found += int(bool(candidate_set & truth_set))
                complete += int(truth_set.issubset(candidate_set))
    counts.extend([0] * max(0, len(truth) - seen_entities))
    arr = np.asarray(counts, dtype=np.int64)
    possible = len(truth) * target_universe_size
    return {
        "candidate_recall": retrieved_true / total_true if total_true else 1.0,
        "any_match_entity_recall": any_found / non_singletons if non_singletons else 1.0,
        "complete_entity_recall": complete / non_singletons if non_singletons else 1.0,
        "candidate_count": total_pairs,
        "average_candidates": float(arr.mean()) if len(arr) else 0.0,
        "p50_candidates": float(np.quantile(arr, 0.5)) if len(arr) else 0.0,
        "p95_candidates": float(np.quantile(arr, 0.95)) if len(arr) else 0.0,
        "p99_candidates": float(np.quantile(arr, 0.99)) if len(arr) else 0.0,
        "max_candidates": int(arr.max()) if len(arr) else 0,
        "zero_candidate_rate": float((arr == 0).mean()) if len(arr) else 0.0,
        "reduction_ratio": 1.0 - total_pairs / possible if possible else 1.0,
    }


def sets_from_long(frame: pd.DataFrame, id_column: str) -> dict[str, frozenset[str]]:
    return {
        source1_id: frozenset(group[id_column].astype(str))
        for source1_id, group in frame.groupby("source1_entity_id", sort=False)
    } if not frame.empty else {}
