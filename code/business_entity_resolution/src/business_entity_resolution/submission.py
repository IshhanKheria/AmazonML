"""Exact challenge output construction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import pandas as pd

from .schemas import CANDIDATE_COLUMNS, MATCHING_COLUMNS


def _joined(values: Iterable[str]) -> str:
    unique = sorted(set(str(value) for value in values), key=lambda value: (value[:2], int(value.split("-", 1)[1])))
    return ",".join(unique)


def result_frame(source1_ids: Iterable[str], mapping: Mapping[str, Iterable[str]], value_column: str) -> pd.DataFrame:
    return pd.DataFrame({
        "source1_entity_id": [str(value) for value in source1_ids],
        value_column: [_joined(mapping.get(str(value), ())) for value in source1_ids],
    })


def write_submission_outputs(
    output_dir: str | Path,
    source1_ids: Iterable[str],
    predictions: Mapping[str, Iterable[str]],
    candidates: Mapping[str, Iterable[str]],
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = [str(value) for value in source1_ids]
    matching = result_frame(ids, predictions, MATCHING_COLUMNS[1])
    candidate = result_frame(ids, candidates, CANDIDATE_COLUMNS[1])
    matching_path, candidate_path = output_dir / "matching_results.tsv", output_dir / "candidate_pairs.tsv"
    matching.to_csv(matching_path, sep="\t", index=False, encoding="utf-8", lineterminator="\n")
    candidate.to_csv(candidate_path, sep="\t", index=False, encoding="utf-8", lineterminator="\n")
    return matching_path, candidate_path

