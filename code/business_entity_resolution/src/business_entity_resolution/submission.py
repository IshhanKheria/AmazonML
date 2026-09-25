"""Exact challenge output construction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import pandas as pd

from .schemas import CANDIDATE_COLUMNS, MATCHING_COLUMNS


class OutputFormatError(ValueError):
    """Raised before writing when a submission file would violate the contract."""


def _joined(values: Iterable[str]) -> str:
    unique = sorted(set(str(value) for value in values), key=lambda value: (value[:2], int(value.split("-", 1)[1])))
    return ",".join(unique)


def result_frame(source1_ids: Iterable[str], mapping: Mapping[str, Iterable[str]], value_column: str) -> pd.DataFrame:
    return pd.DataFrame({
        "source1_entity_id": [str(value) for value in source1_ids],
        value_column: [_joined(mapping.get(str(value), ())) for value in source1_ids],
    })


def _assert_format(
    source1_ids: list[str],
    matching: pd.DataFrame,
    candidate: pd.DataFrame,
    valid_targets: set[str] | None,
) -> None:
    """Validate the in-memory frames against every submission rule.

    Runs *before* writing so a formatting bug fails loudly without touching the
    outputs and without requiring the model to be re-run. ``valid_targets`` is
    optional; when provided, matched/candidate IDs must exist in the test set.
    """
    required = set(source1_ids)
    for frame, value_column, expected in (
        (matching, MATCHING_COLUMNS[1], MATCHING_COLUMNS),
        (candidate, CANDIDATE_COLUMNS[1], CANDIDATE_COLUMNS),
    ):
        if list(frame.columns) != list(expected):
            raise OutputFormatError(f"unexpected columns {list(frame.columns)}; expected {list(expected)}")
        s1 = frame["source1_entity_id"].astype(str)
        if s1.duplicated().any():
            raise OutputFormatError(f"{value_column}: duplicate source1_entity_id rows")
        if set(s1) != required:
            missing = required - set(s1)
            extra = set(s1) - required
            raise OutputFormatError(f"{value_column}: S1 coverage mismatch (missing={len(missing)}, extra={len(extra)})")
        for raw in frame[value_column].astype(str):
            if raw == "":
                continue
            ids = raw.split(",")
            if any(item == "" or item != item.strip() for item in ids):
                raise OutputFormatError(f"{value_column}: whitespace/blank token in ID list {raw!r}")
            if len(ids) != len(set(ids)):
                raise OutputFormatError(f"{value_column}: duplicate ID within a list {raw!r}")
            for item in ids:
                if item.startswith("S1-") or not item.startswith(("S2-", "S3-")):
                    raise OutputFormatError(f"{value_column}: invalid target prefix {item!r}")
                if valid_targets is not None and item not in valid_targets:
                    raise OutputFormatError(f"{value_column}: unknown target ID {item!r}")
    # Every final match must have been a candidate.
    match_map = {row.source1_entity_id: set(str(row.matched_entity_ids).split(",")) - {""} for row in matching.itertuples(index=False)}
    cand_map = {row.source1_entity_id: set(str(row.candidate_entity_ids).split(",")) - {""} for row in candidate.itertuples(index=False)}
    for s1_id, matched in match_map.items():
        absent = matched - cand_map.get(s1_id, set())
        if absent:
            raise OutputFormatError(f"matching: {s1_id} has matches absent from candidates: {sorted(absent)[:3]}")


def write_submission_outputs(
    output_dir: str | Path,
    source1_ids: Iterable[str],
    predictions: Mapping[str, Iterable[str]],
    candidates: Mapping[str, Iterable[str]],
    valid_targets: set[str] | None = None,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = [str(value) for value in source1_ids]
    matching = result_frame(ids, predictions, MATCHING_COLUMNS[1])
    candidate = result_frame(ids, candidates, CANDIDATE_COLUMNS[1])
    _assert_format(ids, matching, candidate, valid_targets)
    matching_path, candidate_path = output_dir / "matching_results.tsv", output_dir / "candidate_pairs.tsv"
    matching.to_csv(matching_path, sep="\t", index=False, encoding="utf-8", lineterminator="\n")
    candidate.to_csv(candidate_path, sep="\t", index=False, encoding="utf-8", lineterminator="\n")
    return matching_path, candidate_path

