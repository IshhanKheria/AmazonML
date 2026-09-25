"""Schema-safe challenge TSV I/O."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
import pandas as pd

from .schemas import GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS, SOURCE_PREFIXES, SchemaError, require_columns


def source_path(data_root: str | Path, split: str, source: int) -> Path:
    if split not in {"train", "test"}:
        raise ValueError(f"split must be train or test, got {split!r}")
    if source not in {1, 2, 3}:
        raise ValueError(f"source must be 1, 2, or 3, got {source!r}")
    return Path(data_root) / split / f"{split}_source{source}.tsv"


def ground_truth_path(data_root: str | Path) -> Path:
    return Path(data_root) / "train" / "train_ground_truth.tsv"


def count_tsv_rows(path: str | Path) -> int:
    """Count data rows (excluding the header) without parsing, for progress ETA."""
    path = Path(path)
    newlines = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            newlines += block.count(b"\n")
    return max(0, newlines - 1)


def read_tsv(path: str | Path, expected_columns: tuple[str, ...], nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"TSV file not found: {path}")
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        nrows=nrows,
    )
    require_columns(list(frame.columns), expected_columns, str(path))
    return frame


def iter_tsv(
    path: str | Path,
    expected_columns: tuple[str, ...],
    batch_size: int = 100_000,
    max_rows: int | None = None,
) -> Iterator[pd.DataFrame]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"TSV file not found: {path}")
    remaining = max_rows
    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        chunksize=batch_size,
    )
    for chunk in reader:
        require_columns(list(chunk.columns), expected_columns, str(path))
        if remaining is not None:
            chunk = chunk.iloc[:remaining].copy()
            remaining -= len(chunk)
        if not chunk.empty:
            yield chunk
        if remaining is not None and remaining <= 0:
            break


def validate_entity_frame(frame: pd.DataFrame, source: int, label: str = "entity frame") -> None:
    require_columns(list(frame.columns[:4]), SOURCE_COLUMNS, label)
    prefix = SOURCE_PREFIXES[f"source{source}"]
    if frame["entity_id"].eq("").any():
        raise SchemaError(f"{label}: blank entity_id")
    invalid = ~frame["entity_id"].str.match(rf"^{prefix}\d+$")
    if invalid.any():
        example = frame.loc[invalid, "entity_id"].iloc[0]
        raise SchemaError(f"{label}: invalid source-{source} ID {example!r}")
    if frame["entity_id"].duplicated().any():
        raise SchemaError(f"{label}: duplicate entity_id")
    for column in ("business_name", "country"):
        if frame[column].eq("").any():
            raise SchemaError(f"{label}: blank {column}")


def load_source(data_root: str | Path, split: str, source: int, nrows: int | None = None) -> pd.DataFrame:
    frame = read_tsv(source_path(data_root, split, source), SOURCE_COLUMNS, nrows=nrows)
    validate_entity_frame(frame, source, f"{split} source{source}")
    return frame


def load_ground_truth(data_root: str | Path, nrows: int | None = None) -> pd.DataFrame:
    return read_tsv(ground_truth_path(data_root), GROUND_TRUTH_COLUMNS, nrows=nrows)


def load_ground_truth_for_ids(
    data_root: str | Path,
    source1_ids: Iterable[str],
    batch_size: int = 100_000,
) -> pd.DataFrame:
    """Stream ground truth and return rows for an exact Source 1 entity set."""
    wanted = {str(value) for value in source1_ids}
    if not wanted:
        return pd.DataFrame(columns=GROUND_TRUTH_COLUMNS)
    found: list[pd.DataFrame] = []
    for chunk in iter_tsv(ground_truth_path(data_root), GROUND_TRUTH_COLUMNS, batch_size=batch_size):
        selected = chunk.loc[chunk["source1_entity_id"].isin(wanted)]
        if not selected.empty:
            found.append(selected)
            wanted.difference_update(selected["source1_entity_id"].astype(str))
        if not wanted:
            break
    if wanted:
        examples = ", ".join(sorted(wanted)[:5])
        raise SchemaError(f"ground truth is missing {len(wanted)} requested Source 1 IDs; examples: {examples}")
    return pd.concat(found, ignore_index=True)
