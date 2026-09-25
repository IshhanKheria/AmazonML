from business_entity_resolution.preflight import preflight_outputs
from business_entity_resolution.submission import write_submission_outputs


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


def test_preflight_catches_match_not_in_candidates(tmp_path, smoke_frames):
    s1, s2, s3, _ = smoke_frames
    test_dir = tmp_path / "test"; test_dir.mkdir()
    for source, frame in ((1, s1), (2, s2), (3, s3)):
        frame.to_csv(test_dir / f"test_source{source}.tsv", sep="\t", index=False)
    matching, candidate = write_submission_outputs(tmp_path / "out", s1.entity_id, {"S1-1": {"S2-10"}}, {})
    assert any("absent from candidates" in error for error in preflight_outputs(matching, candidate, test_dir))

