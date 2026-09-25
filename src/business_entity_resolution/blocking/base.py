"""Shared blocking primitives."""

from __future__ import annotations

from enum import IntFlag
import pandas as pd


class CandidateReason(IntFlag):
    EXACT_NAME = 1
    EXACT_COMPACT = 2
    EXACT_CORE = 4
    RARE_NAME_TOKEN = 8
    RARE_ADDRESS_TOKEN = 16
    NUMERIC_TOKEN = 32
    TFIDF_NAME = 64
    TFIDF_ADDRESS = 128
    COUNTRY_FALLBACK = 256


REASON_NAMES = {reason.value: reason.name.lower() for reason in CandidateReason}


def reason_text(mask: int) -> str:
    return "|".join(name for value, name in REASON_NAMES.items() if mask & value)


def empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "source1_entity_id", "candidate_entity_id", "candidate_source",
        "reason_bits", "reason_mask", "retrieval_score", "retrieval_rank",
    ])


def finalize_candidates(frames: list[pd.DataFrame], per_source_cap: int | None = None) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return empty_candidates()
    combined = pd.concat(nonempty, ignore_index=True, sort=False)
    combined["reason_bits"] = combined["reason_bits"].astype(int)
    combined["retrieval_score"] = combined["retrieval_score"].astype(float)
    combined["retrieval_rank"] = combined["retrieval_rank"].astype(int)
    grouped = combined.groupby(["source1_entity_id", "candidate_entity_id", "candidate_source"], as_index=False).agg(
        reason_bits=("reason_bits", lambda values: int(pd.Series(values).astype(int).map(int).aggregate(lambda x: 0 if len(x) == 0 else __import__('functools').reduce(lambda a, b: a | b, x)))),
        retrieval_score=("retrieval_score", "max"),
        retrieval_rank=("retrieval_rank", "min"),
    )
    grouped["reason_mask"] = grouped["reason_bits"].map(reason_text)
    grouped = grouped.sort_values(
        ["source1_entity_id", "candidate_source", "retrieval_score", "retrieval_rank", "candidate_entity_id"],
        ascending=[True, True, False, True, True],
        kind="mergesort",
    )
    if per_source_cap is not None:
        grouped = grouped[grouped.groupby(["source1_entity_id", "candidate_source"]).cumcount() < per_source_cap]
    return grouped.reset_index(drop=True)

