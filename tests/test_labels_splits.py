import pandas as pd
import pytest

from business_entity_resolution.labels import ground_truth_sets, parse_match_ids
from business_entity_resolution.splits import assign_s1_folds, select_pair_fold


def test_ground_truth_parsing_and_empty_singleton(smoke_frames):
    _, _, _, gt = smoke_frames
    mapping = ground_truth_sets(gt)
    assert mapping["S1-1"] == {"S2-10", "S3-11"}
    assert mapping["S1-4"] == frozenset()
    with pytest.raises(ValueError):
        parse_match_ids("S2-1,S2-1")


def test_split_is_deterministic(smoke_frames):
    s1, _, _, gt = smoke_frames
    first = assign_s1_folds(s1, gt, n_folds=3, seed=7)
    second = assign_s1_folds(s1, gt, n_folds=3, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert set(first.source1_entity_id) == set(s1.entity_id)


def test_pair_fold_selection_keeps_entities_disjoint():
    pairs = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-3"],
        "candidate_entity_id": ["S2-1", "S3-1", "S2-2", "S2-3"],
    })
    folds = pd.DataFrame({"source1_entity_id": ["S1-1", "S1-2", "S1-3"], "fold": [0, 1, 0]})
    training = select_pair_fold(pairs, folds, 0, validation=False)
    held_out = select_pair_fold(pairs, folds, 0, validation=True)
    assert set(training["source1_entity_id"]) == {"S1-2"}
    assert set(held_out["source1_entity_id"]) == {"S1-1", "S1-3"}
    assert set(training["source1_entity_id"]).isdisjoint(held_out["source1_entity_id"])
