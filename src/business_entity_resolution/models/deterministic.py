"""Transparent deterministic reference scorer."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from ..features import FEATURE_COLUMNS


def _column(frame: np.ndarray, name: str) -> np.ndarray:
    return frame[:, FEATURE_COLUMNS.index(name)]


class DeterministicScorer:
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None, eval_data=None) -> "DeterministicScorer":
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        # Reference feature by name so column insertions cannot silently break it.
        name_exact = np.maximum.reduce([
            _column(X, "name_canonical_exact"),
            _column(X, "name_compact_exact"),
            _column(X, "name_core_exact"),
        ])
        name_similarity = np.maximum(
            _column(X, "name_token_sort_ratio"),
            _column(X, "name_token_set_ratio"),
        )
        address_similarity = np.maximum.reduce([
            _column(X, "address_canonical_exact"),
            _column(X, "address_ratio"),
            _column(X, "address_token_jaccard"),
        ])
        numeric = np.maximum(_column(X, "numeric_exact"), _column(X, "numeric_jaccard"))
        score = 0.35 * name_exact + 0.30 * name_similarity + 0.25 * address_similarity + 0.10 * numeric
        return np.clip(score, 0.0, 1.0).astype(np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.metadata(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DeterministicScorer":
        json.loads(Path(path).read_text(encoding="utf-8"))
        return cls()

    def metadata(self) -> dict[str, object]:
        return {"kind": "deterministic", "license": "project-code"}

