"""End-to-end in-memory inference orchestration for one partition/smoke run."""

from __future__ import annotations

from collections.abc import Mapping
import pandas as pd

from .blocking import CandidateGenerator
from .decisions import apply_thresholds
from .features import build_pair_features, feature_matrix
from .metrics import sets_from_long
from .normalize import normalize_records


def generate_candidates_for_sources(source1: pd.DataFrame, source2: pd.DataFrame, source3: pd.DataFrame, blocking_config: dict[str, object], include_tfidf: bool = True) -> pd.DataFrame:
    normalized_s1 = normalize_records(source1)
    frames = []
    for targets in (source2, source3):
        normalized_targets = normalize_records(targets)
        frames.append(CandidateGenerator(blocking_config, include_tfidf=include_tfidf).fit(normalized_targets).transform(normalized_s1))
    if not frames:
        return pd.DataFrame()
    candidates = pd.concat(frames, ignore_index=True)
    return candidates.sort_values(["source1_entity_id", "candidate_source", "retrieval_score", "candidate_entity_id"], ascending=[True, True, False, True], kind="mergesort").reset_index(drop=True)


def score_candidates(candidates: pd.DataFrame, source1: pd.DataFrame, targets: pd.DataFrame, model) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_table = build_pair_features(candidates, normalize_records(source1), normalize_records(targets))
    scored = feature_table[["source1_entity_id", "candidate_entity_id", "candidate_source"]].copy()
    scored["score"] = model.predict_scores(feature_matrix(feature_table))
    return scored, feature_table


def infer_partition(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    model,
    blocking_config: dict[str, object],
    threshold: float,
    source_thresholds: Mapping[str, float] | None = None,
    include_tfidf: bool = True,
) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]], pd.DataFrame, pd.DataFrame]:
    candidates = generate_candidates_for_sources(source1, source2, source3, blocking_config, include_tfidf)
    targets = pd.concat([source2, source3], ignore_index=True)
    scored, features = score_candidates(candidates, source1, targets, model)
    ids = source1["entity_id"].astype(str).tolist()
    predictions = apply_thresholds(scored, ids, threshold, source_thresholds)
    candidate_sets = sets_from_long(candidates, "candidate_entity_id")
    for entity_id in ids:
        candidate_sets.setdefault(entity_id, frozenset())
    return predictions, candidate_sets, scored, features

