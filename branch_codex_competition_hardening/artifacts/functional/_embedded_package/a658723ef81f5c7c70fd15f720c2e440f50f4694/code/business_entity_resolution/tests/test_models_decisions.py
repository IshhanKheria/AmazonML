import numpy as np
import pandas as pd

from business_entity_resolution.decisions import apply_thresholds, score_quantile_thresholds, sweep_thresholds
from business_entity_resolution.models import SGDPairModel


def test_sgd_save_load_and_thresholding(tmp_path):
    X = np.asarray([[0, 0], [1, 1], [0.1, 0], [0.9, 1]], dtype=np.float32)
    y = np.asarray([0, 1, 0, 1], dtype=np.int8)
    model = SGDPairModel(seed=3, max_iter=2000, tol=1e-5).fit(X, y)
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = SGDPairModel.load(path)
    np.testing.assert_allclose(model.predict_scores(X), loaded.predict_scores(X))

    scored = pd.DataFrame([
        ("S1-1", "S2-1", "S2", 0.9), ("S1-1", "S3-1", "S3", 0.4),
    ], columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "score"])
    predictions = apply_thresholds(scored, ["S1-1", "S1-2"], 0.5)
    assert predictions["S1-1"] == {"S2-1"}
    assert predictions["S1-2"] == frozenset()
    report = sweep_thresholds(scored, {"S1-1": {"S2-1"}, "S1-2": set()}, [0.3, 0.5])
    assert report.iloc[0].threshold == 0.5


def test_threshold_grid_includes_all_empty_decision():
    scored = pd.DataFrame([
        ("S1-1", "S2-1", "S2", 0.9),
    ], columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "score"])
    thresholds = score_quantile_thresholds(scored, count=3)
    assert max(thresholds) > 0.9
    predictions = apply_thresholds(scored, ["S1-1"], max(thresholds))
    assert predictions["S1-1"] == frozenset()
