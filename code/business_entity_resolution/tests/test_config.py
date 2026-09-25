import json

from business_entity_resolution.config import ProjectConfig


def test_config_loads_relative_paths_and_env(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"data_root": "data", "artifact_root": "artifacts", "output_root": "out"}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BER_RUN_ID", "test-run")
    config = ProjectConfig.load(config_path)
    assert config.data_root == tmp_path / "data"
    assert config.run_id == "test-run"

