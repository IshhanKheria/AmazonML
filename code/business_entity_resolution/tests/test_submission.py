import pytest
import pandas as pd

from business_entity_resolution.preflight import preflight_outputs
from business_entity_resolution.submission import (
    OutputFormatError,
    write_submission_outputs,
    write_submission_outputs_sharded,
)


def test_submission_writer_and_preflight(tmp_path, smoke_frames):
    s1, s2, s3, _ = smoke_frames
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    for source, frame in ((1, s1), (2, s2), (3, s3)):
        frame.to_csv(test_dir / f"test_source{source}.tsv", sep="\t", index=False)
    predictions = {"S1-1": {"S2-10"}, "S1-2": set(), "S1-3": set(), "S1-4": set()}
    candidates = {"S1-1": {"S2-10", "S3-11"}, "S1-2": set(), "S1-3": set(), "S1-4": set()}
    matching, candidate = write_submission_outputs(tmp_path / "out", s1.entity_id, predictions, candidates)
    assert preflight_outputs(matching, candidate, test_dir) == []


def test_writer_refuses_match_not_in_candidates(tmp_path, smoke_frames):
    s1, s2, s3, _ = smoke_frames
    # A matched ID that never appeared as a candidate must fail before writing,
    # so a pipeline bug cannot silently produce an invalid submission.
    with pytest.raises(OutputFormatError):
        write_submission_outputs(tmp_path / "out", s1.entity_id, {"S1-1": {"S2-10"}}, {})


def test_sharded_writer_preserves_empty_rows_and_candidate_subset(tmp_path):
    shards = [
        (["S1-1"], {"S1-1": {"S2-1"}}, {"S1-1": {"S2-1", "S3-1"}}),
        (["S1-2"], {"S1-2": set()}, {"S1-2": set()}),
    ]
    matching, candidates = write_submission_outputs_sharded(tmp_path, shards)
    matching_frame = pd.read_csv(matching, sep="\t", dtype=str, keep_default_na=False)
    candidate_frame = pd.read_csv(candidates, sep="\t", dtype=str, keep_default_na=False)
    assert matching_frame.to_dict("records") == [
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"},
        {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
    ]
    assert candidate_frame.loc[1, "candidate_entity_ids"] == ""
