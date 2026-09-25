"""Pair model protocol."""

from __future__ import annotations

from typing import Protocol, Any
import numpy as np


class PairModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None, eval_data: tuple[np.ndarray, np.ndarray] | None = None) -> "PairModel": ...
    def predict_scores(self, X: np.ndarray) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
    def metadata(self) -> dict[str, Any]: ...

