from argparse import Namespace
import json

import pandas as pd

from business_entity_resolution.cli import _new_model, command_tune_decision
from business_entity_resolution.config import ProjectConfig


def _write_config(tmp_path):
    data_root = tmp_path / "dataset"
    artifact_root = tmp_path / "artifacts"
    output_root = tmp_path / "output"
    (data_root / "train").mkdir(parents=True)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "data_root": str(data_root),
        "artifact_root": str(artifact_root),
        "output_root": str(output_root),
        "run_id": "test",
        "model": {
            "kind": "lightgbm",
            "params": {"n_estimators": 20, "num_leaves": 7},
        },
    }), encoding="utf-8")
    return config_path, data_root, artifact_root / "test"


def test_model_override_does_not_receive_other_backend_params(tmp_path):
    config_path, _, _ = _write_config(tmp_path)
    config = ProjectConfig.load(config_path)
    model = _new_model(config, "sgd")
    assert "n_estimators" not in model.params
    assert "num_leaves" not in model.params


def test_tuning_includes_zero_candidate_validation_entities(tmp_path):
    config_path, data_root, run_root = _write_config(tmp_path)
    truth = pd.DataFrame([
        ("S1-1", "S2-1"),
        ("S1-2", ""),
    ], columns=["source1_entity_id", "matched_entity_ids"])
    truth.to_csv(data_root / "train" / "train_ground_truth.tsv", sep="\t", index=False)

    (run_root / "splits").mkdir(parents=True)
    pd.DataFrame([
        ("S1-1", 0), ("S1-2", 0),
    ], columns=["source1_entity_id", "fold"]).to_parquet(run_root / "splits" / "s1_folds.parquet", index=False)
    (run_root / "scores").mkdir(parents=True)
    pd.DataFrame([
        ("S1-1", "S2-1", "S2", 0.9, 1),
    ], columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "score", "label"]).to_parquet(
        run_root / "scores" / "train.parquet", index=False
    )

    args = Namespace(
        config=str(config_path), max_rows=None, force=False,
        score_file="train.parquet", validation_fold=0, threshold_count=5,
    )
    assert command_tune_decision(args) == 0
    report = pd.read_parquet(run_root / "decisions" / "threshold_sweep.parquet")
    assert set(report["entity_count"]) == {2}
    assert (run_root / "decisions" / "selected_threshold.json").is_file()
