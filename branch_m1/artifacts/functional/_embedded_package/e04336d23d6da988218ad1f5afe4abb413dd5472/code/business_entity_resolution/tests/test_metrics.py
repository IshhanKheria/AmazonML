import pytest

from business_entity_resolution.metrics import candidate_metrics, entity_scores, evaluate_entity_sets


def test_singleton_edge_cases():
    assert entity_scores(set(), set()) == (1.0, 1.0, 1.0)
    assert entity_scores(set(), {"S2-1"}) == (0.0, 0.0, 0.0)
    assert entity_scores({"S2-1"}, set()) == (0.0, 0.0, 0.0)


def test_macro_f05_and_candidate_metrics():
    truth = {"S1-1": {"S2-1", "S3-1"}, "S1-2": set()}
    pred = {"S1-1": {"S2-1", "S2-9"}, "S1-2": set()}
    report = evaluate_entity_sets(truth, pred)
    assert report["macro_f0_5"] == pytest.approx((0.5 + 1.0) / 2)
    candidates = {"S1-1": {"S2-1", "S3-1", "S2-9"}, "S1-2": set()}
    candidate_report = candidate_metrics(truth, candidates, target_universe_size=10)
    assert candidate_report["candidate_recall"] == 1.0
    assert candidate_report["complete_entity_recall"] == 1.0

