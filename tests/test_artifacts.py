from business_entity_resolution.artifacts import write_manifest
from business_entity_resolution.checkpoint import ShardStore


def test_write_manifest_preserves_shardstore_fingerprint(tmp_path):
    # Regression: write_manifest used to clobber the manifest written by
    # mark_complete, dropping the fingerprint so is_complete() never held.
    store = ShardStore(tmp_path / "stage", "fingerprint-123")
    store.mark_complete(rows=0, schema=[])
    write_manifest(store.stage_manifest_path(), stage="stage", config={}, rows=0, schema=[])
    assert store.is_complete()
