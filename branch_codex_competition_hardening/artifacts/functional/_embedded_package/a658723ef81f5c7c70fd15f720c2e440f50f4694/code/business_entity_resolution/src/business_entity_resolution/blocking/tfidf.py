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
        target_batch_size: int = 50_000,
        country_primary: bool = True,
        reason: CandidateReason = CandidateReason.TFIDF_NAME,
    ) -> None:
        self.field, self.top_k, self.min_score = field, top_k, min_score
        self.batch_size = batch_size
        self.target_batch_size = target_batch_size
        self.country_primary, self.reason = country_primary, reason
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
                best_indices = [np.empty(0, dtype=np.int64) for _ in range(len(batch))]
                best_scores = [np.empty(0, dtype=np.float32) for _ in range(len(batch))]
                # Multiply against bounded target blocks. Computing query x all
                # targets before top-K could create an enormous sparse matrix
                # for common character n-grams.
                for target_start in range(0, target_matrix.shape[0], self.target_batch_size):
                    target_stop = min(target_matrix.shape[0], target_start + self.target_batch_size)
                    similarities = safe_sparse_dot(
                        query_matrix,
                        target_matrix[target_start:target_stop].T,
                        dense_output=False,
                    ).tocsr()
                    for local_row in range(len(batch)):
                        begin, end = similarities.indptr[local_row], similarities.indptr[local_row + 1]
                        indices = similarities.indices[begin:end].astype(np.int64, copy=False) + target_start
                        scores = similarities.data[begin:end].astype(np.float32, copy=False)
                        valid = scores >= self.min_score
                        if not np.any(valid):
                            continue
                        indices = np.concatenate((best_indices[local_row], indices[valid]))
                        scores = np.concatenate((best_scores[local_row], scores[valid]))
                        if len(scores) > self.top_k:
                            chosen = np.argpartition(scores, -self.top_k)[-self.top_k:]
                            indices, scores = indices[chosen], scores[chosen]
                        best_indices[local_row], best_scores[local_row] = indices, scores
                for local_row, query_id in enumerate(batch["entity_id"].astype(str)):
                    indices, scores = best_indices[local_row], best_scores[local_row]
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
