"""Transparent deterministic reference scorer."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np


class DeterministicScorer:
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None, eval_data=None) -> "DeterministicScorer":
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        # FEATURE_COLUMNS positions: exact/core/ratios plus address and numeric evidence.
        name_exact = np.maximum.reduce([X[:, 1], X[:, 2], X[:, 3]])
        name_similarity = np.maximum(X[:, 5], X[:, 8])
        address_similarity = np.maximum.reduce([X[:, 12], X[:, 14], X[:, 17]])
        numeric = np.maximum(X[:, 19], X[:, 20])
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

