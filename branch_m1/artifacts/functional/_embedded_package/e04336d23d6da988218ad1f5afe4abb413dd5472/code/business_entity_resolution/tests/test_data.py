import pandas as pd
import pytest

from business_entity_resolution.data import load_ground_truth_for_ids, read_tsv, validate_entity_frame
from business_entity_resolution.schemas import SOURCE_COLUMNS, SchemaError


def test_tsv_loader_preserves_commas_and_empty_address(tmp_path):
    path = tmp_path / "source.tsv"
    path.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS2-1\tCafe\tParis, France\tFrance\nS2-2\tEmpty\t\tAtlantis\n", encoding="utf-8")
    frame = read_tsv(path, SOURCE_COLUMNS)
    assert frame.loc[0, "business_address"] == "Paris, France"
    assert frame.loc[1, "business_address"] == ""
    validate_entity_frame(frame, 2)


def test_wrong_schema_and_duplicate_ids_fail(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("entity_id,business_name,business_address,country\n", encoding="utf-8")
    with pytest.raises(SchemaError):
        read_tsv(path, SOURCE_COLUMNS)
    frame = pd.DataFrame([
        ("S1-1", "A", "X", "US"), ("S1-1", "B", "Y", "US")
    ], columns=SOURCE_COLUMNS)
    with pytest.raises(SchemaError):
        validate_entity_frame(frame, 1)


def test_ground_truth_subset_is_selected_by_id_not_row_position(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    (train / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-99\tS2-99\n"
        "S1-2\t\n"
        "S1-1\tS3-1\n",
        encoding="utf-8",
    )
    selected = load_ground_truth_for_ids(tmp_path, ["S1-1", "S1-2"], batch_size=1)
    assert set(selected["source1_entity_id"]) == {"S1-1", "S1-2"}
