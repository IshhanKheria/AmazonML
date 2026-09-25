"""Strict, competition-aware output validation."""

from __future__ import annotations

import csv
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
    matching_path, candidate_path, test_dir = Path(matching_path), Path(candidate_path), Path(test_dir)
    for path in (matching_path, candidate_path):
        if not path.is_file():
            return [f"missing output file: {path}"]
    valid_s1 = _read_source_ids(test_dir / "test_source1.tsv")
    valid_targets = _read_source_ids(test_dir / "test_source2.tsv") | _read_source_ids(test_dir / "test_source3.tsv")
    matched, errors = _read_output(matching_path, MATCHING_COLUMNS, valid_s1, valid_targets)
    candidates, candidate_errors = _read_output(candidate_path, CANDIDATE_COLUMNS, valid_s1, valid_targets)
    errors.extend(candidate_errors)
    for source1_id, values in matched.items():
        missing = values - candidates.get(source1_id, set())
        if missing:
            errors.append(f"{source1_id}: {len(missing)} matches are absent from candidates")
    return errors


def run_official_validator(validator: str | Path, matching: str | Path, candidate: str | Path, test_dir: str | Path, check_ids: bool = True) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(validator), "--matching", str(matching), "--candidate", str(candidate), "--test-dir", str(test_dir)]
    if check_ids:
        command.append("--check-ids")
    return subprocess.run(command, text=True, capture_output=True, check=False)

