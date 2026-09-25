"""Rare-token and numeric-token inverted-index blocking."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import pandas as pd

from .base import CandidateReason, empty_candidates


def _tokens(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(value.split()) if value else ()
    return tuple(value)


class RareTokenBlocker:
    def __init__(
        self,
        field: str,
        reason: CandidateReason,
        max_document_frequency: int = 500,
        posting_cap: int = 250,
        country_primary: bool = True,
        country_fallback: bool = True,
    ) -> None:
        self.field = field
        self.reason = reason
        self.max_df = max_document_frequency
        self.posting_cap = posting_cap
        self.country_primary = country_primary
        self.country_fallback = country_fallback
        self._postings: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._global: dict[str, list[str]] = defaultdict(list)
        self._df: Counter[tuple[str, str]] = Counter()
        self._global_df: Counter[str] = Counter()

    def fit(self, targets: pd.DataFrame) -> "RareTokenBlocker":
        for row in targets.itertuples(index=False):
            country, target_id = str(getattr(row, "country_norm")), str(getattr(row, "entity_id"))
            for token in set(_tokens(getattr(row, self.field))):
                self._df[(country, token)] += 1
                self._global_df[token] += 1
                self._postings[(country, token)].append(target_id)
                self._global[token].append(target_id)
        return self

    def transform(self, queries: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for query in queries.itertuples(index=False):
            country = str(getattr(query, "country_norm"))
            scores: dict[str, float] = defaultdict(float)
            fallback = False
            for token in set(_tokens(getattr(query, self.field))):
                df = self._df.get((country, token), 0)
                postings = self._postings.get((country, token), []) if self.country_primary else self._global.get(token, [])
                if not postings and self.country_fallback:
                    postings, df, fallback = self._global.get(token, []), self._global_df.get(token, 0), True
                if not postings or df > self.max_df:
                    continue
                weight = 1.0 / math.log2(df + 2.0)
                for target_id in postings[: self.posting_cap]:
                    scores[target_id] += weight
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: self.posting_cap]
            for rank, (target_id, score) in enumerate(ranked, start=1):
                bits = int(self.reason) | (int(CandidateReason.COUNTRY_FALLBACK) if fallback else 0)
                rows.append({
                    "source1_entity_id": str(getattr(query, "entity_id")),
                    "candidate_entity_id": target_id,
                    "candidate_source": target_id[:2],
                    "reason_bits": bits,
                    "retrieval_score": float(score),
                    "retrieval_rank": rank,
                })
        return pd.DataFrame(rows) if rows else empty_candidates()
