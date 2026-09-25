from business_entity_resolution.pipeline import run_smoke


def test_complete_smoke_pipeline(tmp_path):
    report = run_smoke(tmp_path)
    assert report["training_pairs"] > 0
    assert report["positive_pairs"] == 6
    assert report["preflight_errors"] == []

