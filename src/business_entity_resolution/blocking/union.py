"""Configured union of complementary blockers."""

from __future__ import annotations

import pandas as pd

from .base import CandidateReason, finalize_candidates
from .exact import ExactBlocker
from .tokens import RareTokenBlocker
from .tfidf import TfidfTopKBlocker

# Bump whenever the blocker set or its semantics change so cached candidate /
# feature shards keyed on this version are invalidated instead of reused.
BLOCKING_VERSION = "v2"


class CandidateGenerator:
    def __init__(self, config: dict[str, object] | None = None, include_tfidf: bool = True) -> None:
        self.config = config or {}
        self.include_tfidf = include_tfidf
        self.blockers: list[object] = []

    def fit(self, targets: pd.DataFrame) -> "CandidateGenerator":
        country_primary = bool(self.config.get("country_primary", True))
        country_fallback = bool(self.config.get("country_fallback", True))
        posting_cap = int(self.config.get("token_posting_cap", 250))
        max_df = int(self.config.get("rare_token_max_df", 500))
        self.blockers = [
            ExactBlocker("name_canonical", CandidateReason.EXACT_NAME, country_primary, country_fallback, posting_cap),
            ExactBlocker("name_compact", CandidateReason.EXACT_COMPACT, country_primary, country_fallback, posting_cap),
            ExactBlocker("name_core", CandidateReason.EXACT_CORE, country_primary, country_fallback, posting_cap),
            RareTokenBlocker("name_tokens", CandidateReason.RARE_NAME_TOKEN, max_df, posting_cap, country_primary, country_fallback),
            RareTokenBlocker("address_tokens", CandidateReason.RARE_ADDRESS_TOKEN, max_df, posting_cap, country_primary, country_fallback),
            RareTokenBlocker("numeric_tokens", CandidateReason.NUMERIC_TOKEN, max_df, posting_cap, country_primary, country_fallback),
        ]
        if self.include_tfidf:
            self.blockers.append(TfidfTopKBlocker(
                top_k=int(self.config.get("tfidf_top_k", 20)),
                min_score=float(self.config.get("tfidf_min_score", 0.15)),
                batch_size=int(self.config.get("batch_size", 1000)),
                country_primary=country_primary,
            ))
            self.blockers.append(TfidfTopKBlocker(
                field="address_canonical",
                reason=CandidateReason.TFIDF_ADDRESS,
                top_k=int(self.config.get("tfidf_address_top_k", self.config.get("tfidf_top_k", 20))),
                min_score=float(self.config.get("tfidf_address_min_score", self.config.get("tfidf_min_score", 0.15))),
                batch_size=int(self.config.get("batch_size", 1000)),
                country_primary=country_primary,
            ))
        for blocker in self.blockers:
            blocker.fit(targets)
        return self

    def transform(self, source1: pd.DataFrame) -> pd.DataFrame:
        frames = [blocker.transform(source1) for blocker in self.blockers]
        return finalize_candidates(frames, per_source_cap=int(self.config.get("per_source_cap", 100)))

