"""Shared schemas and validation constants."""

from __future__ import annotations

SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")
MATCHING_COLUMNS = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_COLUMNS = ("source1_entity_id", "candidate_entity_ids")
CANDIDATE_LONG_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "reason_mask",
    "retrieval_score",
    "retrieval_rank",
)
SOURCE_PREFIXES = {"source1": "S1-", "source2": "S2-", "source3": "S3-"}


class SchemaError(ValueError):
    """Raised when a challenge file violates its declared schema."""


def require_columns(actual: list[str] | tuple[str, ...], expected: tuple[str, ...], label: str) -> None:
    if tuple(actual) != expected:
        raise SchemaError(f"{label}: expected columns {list(expected)}, got {list(actual)}")

