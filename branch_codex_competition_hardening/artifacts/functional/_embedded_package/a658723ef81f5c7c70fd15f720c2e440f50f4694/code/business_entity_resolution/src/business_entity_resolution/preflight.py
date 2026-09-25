"""Strict, competition-aware output validation."""

from __future__ import annotations

import csv
from itertools import zip_longest
from pathlib import Path
import subprocess
import sys

from .schemas import CANDIDATE_COLUMNS, MATCHING_COLUMNS


def _read_source_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        return {row[0] for row in reader if row}


def _read_output(path: Path, expected_header: tuple[str, str], valid_s1: set[str], valid_targets: set[str]) -> tuple[dict[str, set[str]], list[str]]:
    errors: list[str] = []
    mapping: dict[str, set[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header != list(expected_header):
            return {}, [f"{path.name}: expected exact header {list(expected_header)}, got {header}"]
        for line_number, row in enumerate(reader, start=2):
            if len(row) != 2:
                errors.append(f"{path.name}:{line_number}: expected exactly two tab-separated columns")
                continue
            source1_id, raw = row
            if source1_id in mapping:
                errors.append(f"{path.name}:{line_number}: duplicate Source 1 row {source1_id}")
            if source1_id not in valid_s1:
                errors.append(f"{path.name}:{line_number}: unknown Source 1 ID {source1_id}")
            values = [] if raw == "" else raw.split(",")
            if raw.casefold() == "nan":
                errors.append(f"{path.name}:{line_number}: literal nan is invalid")
            if len(values) != len(set(values)):
                errors.append(f"{path.name}:{line_number}: duplicate ID in list for {source1_id}")
            for value in values:
                if value.startswith("S1-") or not value.startswith(("S2-", "S3-")):
                    errors.append(f"{path.name}:{line_number}: invalid target prefix {value}")
                elif value not in valid_targets:
                    errors.append(f"{path.name}:{line_number}: unknown target ID {value}")
            mapping[source1_id] = set(values)
    missing = valid_s1 - set(mapping)
    extra = set(mapping) - valid_s1
    if missing:
        errors.append(f"{path.name}: {len(missing)} required Source 1 rows are missing")
    if extra:
        errors.append(f"{path.name}: {len(extra)} unexpected Source 1 rows")
    return mapping, errors


def preflight_outputs(matching_path: str | Path, candidate_path: str | Path, test_dir: str | Path) -> list[str]:
    """Validate both outputs in lockstep without storing candidate mappings.

    The pipeline writes the two TSVs in identical Source 1 order. Requiring that
    order here makes exact match-subset validation streaming and prevents a
    hundreds-of-millions-of-IDs candidate dictionary from exhausting memory.
    """
    matching_path, candidate_path, test_dir = Path(matching_path), Path(candidate_path), Path(test_dir)
    for path in (matching_path, candidate_path):
        if not path.is_file():
            return [f"missing output file: {path}"]
    valid_s1 = _read_source_ids(test_dir / "test_source1.tsv")
    valid_targets = _read_source_ids(test_dir / "test_source2.tsv") | _read_source_ids(test_dir / "test_source3.tsv")
    errors: list[str] = []
    seen: set[str] = set()
    with matching_path.open(encoding="utf-8", newline="") as matching_handle, candidate_path.open(
        encoding="utf-8", newline=""
    ) as candidate_handle:
        matching_reader = csv.reader(matching_handle, delimiter="\t")
        candidate_reader = csv.reader(candidate_handle, delimiter="\t")
        matching_header = next(matching_reader, None)
        candidate_header = next(candidate_reader, None)
        if matching_header != list(MATCHING_COLUMNS):
            errors.append(f"{matching_path.name}: expected exact header {list(MATCHING_COLUMNS)}, got {matching_header}")
        if candidate_header != list(CANDIDATE_COLUMNS):
            errors.append(f"{candidate_path.name}: expected exact header {list(CANDIDATE_COLUMNS)}, got {candidate_header}")
        if errors:
            return errors
        for line_number, pair in enumerate(zip_longest(matching_reader, candidate_reader), start=2):
            matching_row, candidate_row = pair
            if matching_row is None or candidate_row is None:
                errors.append("matching and candidate files have different row counts")
                break
            if len(matching_row) != 2 or len(candidate_row) != 2:
                errors.append(f"line {line_number}: each output must contain exactly two tab-separated columns")
                continue
            source1_id, raw_matches = matching_row
            candidate_source1_id, raw_candidates = candidate_row
            if source1_id != candidate_source1_id:
                errors.append(
                    f"line {line_number}: Source 1 row order differs between outputs "
                    f"({source1_id!r} != {candidate_source1_id!r})"
                )
                continue
            if source1_id in seen:
                errors.append(f"line {line_number}: duplicate Source 1 row {source1_id}")
            seen.add(source1_id)
            if source1_id not in valid_s1:
                errors.append(f"line {line_number}: unknown Source 1 ID {source1_id}")
            matched = [] if raw_matches == "" else raw_matches.split(",")
            candidates = [] if raw_candidates == "" else raw_candidates.split(",")
            for label, raw, values in (
                ("matched_entity_ids", raw_matches, matched),
                ("candidate_entity_ids", raw_candidates, candidates),
            ):
                if raw.casefold() == "nan":
                    errors.append(f"line {line_number}: literal nan is invalid in {label}")
                if len(values) != len(set(values)):
                    errors.append(f"line {line_number}: duplicate ID in {label} for {source1_id}")
                for value in values:
                    if value.startswith("S1-") or not value.startswith(("S2-", "S3-")):
                        errors.append(f"line {line_number}: invalid target prefix {value}")
                    elif value not in valid_targets:
                        errors.append(f"line {line_number}: unknown target ID {value}")
            absent = set(matched) - set(candidates)
            if absent:
                errors.append(f"{source1_id}: {len(absent)} matches are absent from candidates")
    missing = valid_s1 - seen
    if missing:
        errors.append(f"outputs: {len(missing)} required Source 1 rows are missing")
    return errors


def run_official_validator(validator: str | Path, matching: str | Path, candidate: str | Path, test_dir: str | Path, check_ids: bool = True) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(validator), "--matching", str(matching), "--candidate", str(candidate), "--test-dir", str(test_dir)]
    if check_ids:
        command.append("--check-ids")
    return subprocess.run(command, text=True, capture_output=True, check=False)
