"""Exact normalized-field blocking."""

from __future__ import annotations

from collections import defaultdict
import pandas as pd

from .base import CandidateReason, empty_candidates


def _tokens(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(value.split()) if value else ()
    return tuple(value)


def _jaccard(left: object, right: object) -> float:
    a, b = set(_tokens(left)), set(_tokens(right))
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a | b else 0.0


class ExactBlocker:
    def __init__(
        self,
        field: str,
        reason: CandidateReason,
        country_primary: bool = True,
        country_fallback: bool = True,
        posting_cap: int = 250,
    ) -> None:
        self.field = field
        self.reason = reason
        self.country_primary = country_primary
        self.country_fallback = country_fallback
        self.posting_cap = posting_cap
        self._country_index: dict[tuple[str, str], list[int]] = defaultdict(list)
        self._global_index: dict[str, list[int]] = defaultdict(list)
        self._targets: pd.DataFrame | None = None

    def fit(self, targets: pd.DataFrame) -> "ExactBlocker":
        required = {"entity_id", "country_norm", self.field}
        if not required.issubset(targets.columns):
            raise ValueError(f"exact blocker missing columns: {sorted(required - set(targets.columns))}")
        self._targets = targets.reset_index(drop=True)
        for index, row in self._targets.iterrows():
            value = str(row[self.field])
            if not value:
                continue
            self._country_index[(str(row["country_norm"]), value)].append(index)
            self._global_index[value].append(index)
        return self

    def transform(self, queries: pd.DataFrame) -> pd.DataFrame:
        if self._targets is None:
            raise RuntimeError("exact blocker must be fit before transform")
        rows: list[dict[str, object]] = []
        for query in queries.itertuples(index=False):
            value = str(getattr(query, self.field))
            country = str(getattr(query, "country_norm"))
            indices = self._country_index.get((country, value), []) if self.country_primary else self._global_index.get(value, [])
            fallback = False
            if not indices and self.country_fallback:
                indices = self._global_index.get(value, [])
                fallback = bool(indices)
            ranked = []
            for index in indices:
                target = self._targets.iloc[index]
                score = _jaccard(getattr(query, "address_tokens", ()), target.get("address_tokens", ()))
                ranked.append((score, str(target["entity_id"])))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            for rank, (score, target_id) in enumerate(ranked[: self.posting_cap], start=1):
                bits = int(self.reason) | (int(CandidateReason.COUNTRY_FALLBACK) if fallback else 0)
                rows.append({
                    "source1_entity_id": str(getattr(query, "entity_id")),
                    "candidate_entity_id": target_id,
                    "candidate_source": target_id[:2],
                    "reason_bits": bits,
                    "retrieval_score": 1.0 + score * 0.01,
                    "retrieval_rank": rank,
                })
        return pd.DataFrame(rows) if rows else empty_candidates()
