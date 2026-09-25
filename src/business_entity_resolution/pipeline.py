"""Reusable stage functions and bounded synthetic smoke workflow."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .blocking import CandidateGenerator
from .features import build_pair_features, feature_matrix
from .inference import infer_partition
from .labels import force_add_training_positives, ground_truth_sets, label_candidates
from .metrics import evaluate_entity_sets
from .models import SGDPairModel
from .normalize import normalize_records
from .preflight import preflight_outputs
from .submission import write_submission_outputs


def make_smoke_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s1 = pd.DataFrame([
        ("S1-1", "Alpha & Sons Ltd", "12 River Rd, Paris 75001", "France"),
        ("S1-2", "Beta Market", "44 Main Street", "US"),
        ("S1-3", "राम मार्केटिंग प्राइवेट लिमिटेड", "10 MG Road 560001", "India"),
        ("S1-4", "Singleton Works", "9 Quiet Lane", "US"),
    ], columns=["entity_id", "business_name", "business_address", "country"])
    s2 = pd.DataFrame([
        ("S2-10", "Alpha and Sons Limited", "12 River Road Paris 75001", "France"),
        ("S2-20", "Beta Market", "44 Main St", "US"),
        ("S2-30", "Ram Marketing Pvt Ltd", "10 MG Rd 560001", "India"),
        ("S2-40", "Other Shop", "99 Elsewhere", "US"),
    ], columns=s1.columns)
    s3 = pd.DataFrame([
        ("S3-11", "Alpha & Sons", "12 River Rd 75001", "France"),
        ("S3-21", "Beta Markets", "44 Main Street", "US"),
        ("S3-31", "राम मार्केटिंग", "10 MG Road 560001", "India"),
        ("S3-41", "Singleton Worse", "9 Quiet Lane", "US"),
    ], columns=s1.columns)
    truth = pd.DataFrame([
        ("S1-1", "S2-10,S3-11"), ("S1-2", "S2-20,S3-21"),
        ("S1-3", "S2-30,S3-31"), ("S1-4", ""),
    ], columns=["source1_entity_id", "matched_entity_ids"])
    return s1, s2, s3, truth


def run_smoke(output_dir: str | Path, seed: int = 2026) -> dict[str, object]:
    output_dir = Path(output_dir)
    s1, s2, s3, gt = make_smoke_fixture()
    truth = ground_truth_sets(gt)
    ns1, n2, n3 = normalize_records(s1), normalize_records(s2), normalize_records(s3)
    blocking = {"rare_token_max_df": 20, "token_posting_cap": 20, "tfidf_top_k": 5, "per_source_cap": 20, "batch_size": 10}
    candidate_frames = [CandidateGenerator(blocking).fit(target).transform(ns1) for target in (n2, n3)]
    candidates = pd.concat(candidate_frames, ignore_index=True)
    training_candidates = label_candidates(force_add_training_positives(candidates, truth), truth)
    features = build_pair_features(training_candidates, ns1, pd.concat([n2, n3], ignore_index=True))
    X, y = feature_matrix(features), features["label"].to_numpy(dtype=np.int8)
    model = SGDPairModel(seed=seed, max_iter=2000, tol=1e-5).fit(X, y)
    predictions, candidate_sets, _, _ = infer_partition(s1, s2, s3, model, blocking, threshold=0.5)

    test_dir = output_dir / "smoke_test_data"
    test_dir.mkdir(parents=True, exist_ok=True)
    for source, frame in ((1, s1), (2, s2), (3, s3)):
        frame.to_csv(test_dir / f"test_source{source}.tsv", sep="\t", index=False, encoding="utf-8", lineterminator="\n")
    matching, candidate = write_submission_outputs(output_dir, s1["entity_id"], predictions, candidate_sets)
    errors = preflight_outputs(matching, candidate, test_dir)
    if errors:
        raise AssertionError("smoke preflight failed: " + "; ".join(errors))
    return {
        "training_pairs": len(features), "positive_pairs": int(y.sum()),
        "candidate_pairs": int(sum(len(values) for values in candidate_sets.values())),
        "metrics": evaluate_entity_sets(truth, predictions),
        "matching_path": str(matching), "candidate_path": str(candidate),
        "preflight_errors": errors,
    }

