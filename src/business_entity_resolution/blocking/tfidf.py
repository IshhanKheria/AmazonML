"""Batched sparse character-TF-IDF top-K retrieval."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.extmath import safe_sparse_dot

from .base import CandidateReason, empty_candidates


class TfidfTopKBlocker:
    def __init__(
        self,
        field: str = "name_canonical",
        top_k: int = 20,
        min_score: float = 0.15,
        batch_size: int = 1000,
        country_primary: bool = True,
        reason: CandidateReason = CandidateReason.TFIDF_NAME,
    ) -> None:
        self.field, self.top_k, self.min_score = field, top_k, min_score
        self.batch_size, self.country_primary, self.reason = batch_size, country_primary, reason
        self.vectorizers: dict[str, TfidfVectorizer] = {}
        self.target_matrices: dict[str, object] = {}
        self.target_ids: dict[str, np.ndarray] = {}

    def fit(self, targets: pd.DataFrame) -> "TfidfTopKBlocker":
        groups = targets.groupby("country_norm", sort=True) if self.country_primary else [("__all__", targets)]
        for country, group in groups:
            docs = group[self.field].astype(str).tolist()
            if not docs or not any(docs):
                continue
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), dtype=np.float32, min_df=1, norm="l2")
            matrix = vectorizer.fit_transform(docs).tocsr()
            key = str(country)
            self.vectorizers[key] = vectorizer
            self.target_matrices[key] = matrix
            self.target_ids[key] = group["entity_id"].astype(str).to_numpy()
        return self

    def transform(self, queries: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        key_series = queries["country_norm"].astype(str) if self.country_primary else pd.Series("__all__", index=queries.index)
        for key, group in queries.groupby(key_series, sort=True):
            key = str(key)
            if key not in self.vectorizers:
                continue
            vectorizer, target_matrix, target_ids = self.vectorizers[key], self.target_matrices[key], self.target_ids[key]
            for start in range(0, len(group), self.batch_size):
                batch = group.iloc[start : start + self.batch_size]
                query_matrix = vectorizer.transform(batch[self.field].astype(str)).tocsr()
                similarities = safe_sparse_dot(query_matrix, target_matrix.T, dense_output=False).tocsr()
                for local_row, query_id in enumerate(batch["entity_id"].astype(str)):
                    begin, end = similarities.indptr[local_row], similarities.indptr[local_row + 1]
                    indices, scores = similarities.indices[begin:end], similarities.data[begin:end]
                    valid = scores >= self.min_score
                    indices, scores = indices[valid], scores[valid]
                    if len(scores) > self.top_k:
                        chosen = np.argpartition(scores, -self.top_k)[-self.top_k:]
                        indices, scores = indices[chosen], scores[chosen]
                    order = np.lexsort((target_ids[indices], -scores)) if len(scores) else []
                    for rank, pos in enumerate(order, start=1):
                        target_id = str(target_ids[indices[pos]])
                        rows.append({
                            "source1_entity_id": query_id,
                            "candidate_entity_id": target_id,
                            "candidate_source": target_id[:2],
                            "reason_bits": int(self.reason),
                            "retrieval_score": float(scores[pos]),
                            "retrieval_rank": rank,
                        })
        return pd.DataFrame(rows) if rows else empty_candidates()

