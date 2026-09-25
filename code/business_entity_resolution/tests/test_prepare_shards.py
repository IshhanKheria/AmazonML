from argparse import Namespace
import json

import pandas as pd

from business_entity_resolution.cli import (
    _candidate_store,
    _feature_store,
    _load_config,
    _prepared_shard_store,
    command_build_features,
    command_generate_candidates,
    command_prepare,
    command_score,
    command_train,
)


def test_prepare_never_overwrites_earlier_chunk_rows(tmp_path):
    data_root = tmp_path / "dataset"
    train = data_root / "train"
    train.mkdir(parents=True)
    columns = ["entity_id", "business_name", "business_address", "country"]
    for source in (1, 2, 3):
        frame = pd.DataFrame([
            (f"S{source}-{index}", f"Business {index}", f"{index} Main Rd", "US")
            for index in range(1, 8)
        ], columns=columns)
        frame.to_csv(train / f"train_source{source}.tsv", sep="\t", index=False)
    pd.DataFrame([
        (f"S1-{index}", f"S2-{index},S3-{index}") for index in range(1, 8)
    ], columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
        train / "train_ground_truth.tsv", sep="\t", index=False
    )

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "data_root": str(data_root),
        "artifact_root": str(tmp_path / "artifacts"),
        "output_root": str(tmp_path / "output"),
        "run_id": "prepare-test",
        "batch_size": 2,
        "n_shards": 2,
        "resources": {"mode": "manual", "chunk_pairs": 2},
    }), encoding="utf-8")
    args = Namespace(config=str(config_path), max_rows=None, force=False, split="train")
    assert command_prepare(args) == 0

    config = _load_config(args)
    for source in (1, 2, 3):
        store = _prepared_shard_store(config, "train", source)
        assert store.is_complete()
        prepared = store.read_all()
        assert len(prepared) == 7
        assert prepared["entity_id"].nunique() == 7

    candidate_args = Namespace(
        config=str(config_path), max_rows=None, force=False, split="train", no_tfidf=False,
    )
    assert command_generate_candidates(candidate_args) == 0
    candidates = _candidate_store(config, "train").read_all()
    candidate_pairs = set(zip(candidates["source1_entity_id"], candidates["candidate_entity_id"]))
    expected_pairs = {
        (f"S1-{index}", f"S{source}-{index}")
        for index in range(1, 8)
        for source in (2, 3)
    }
    assert expected_pairs.issubset(candidate_pairs)

    feature_args = Namespace(
        config=str(config_path), max_rows=None, force=False, split="train",
        no_tfidf=False, no_embeddings=True,
    )
    assert command_build_features(feature_args) == 0
    features = _feature_store(config, "train").read_all()
    assert len(features) == len(candidates)
    assert int(features["label"].sum()) == 14

    split_dir = config.artifact_dir("splits")
    split_dir.mkdir(parents=True)
    pd.DataFrame([
        (f"S1-{index}", index % 2) for index in range(1, 8)
    ], columns=["source1_entity_id", "fold"]).to_parquet(
        split_dir / "s1_folds.parquet", index=False
    )
    train_args = Namespace(
        config=str(config_path), max_rows=None, force=False, model="sgd",
        validation_fold=0, exclude_folds=None, all_training_data=False,
        output_name="sgd-fold0",
    )
    assert command_train(train_args) == 0
    model_path = config.artifact_dir("models") / "sgd-fold0.joblib"
    assert model_path.is_file()

    score_args = Namespace(
        config=str(config_path), max_rows=None, force=False, split="train",
        model="sgd", model_path=str(model_path), validation_fold=0,
        all_training_data=False, output_name="fold0.parquet",
    )
    assert command_score(score_args) == 0
    scored = pd.read_parquet(config.artifact_dir("scores") / "fold0.parquet")
    assert not scored.empty
    assert set(scored["source1_entity_id"]).issubset({"S1-2", "S1-4", "S1-6"})
