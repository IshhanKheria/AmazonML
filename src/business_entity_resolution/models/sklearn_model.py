"""Incremental scikit-learn logistic pair model."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier


class SGDPairModel:
    def __init__(self, seed: int = 2026, **params: Any) -> None:
        defaults = {"loss": "log_loss", "penalty": "l2", "alpha": 1e-4, "max_iter": 1000, "tol": 1e-4, "random_state": seed}
        defaults.update(params)
        self.estimator = SGDClassifier(**defaults)
        self.params = defaults

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None, eval_data=None, logger=None) -> "SGDPairModel":
        self.estimator.fit(X, y, sample_weight=sample_weight)
        if logger is not None:
            logger.metric("train_iterations", int(getattr(self.estimator, "n_iter_", 0)))
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "SGDPairModel":
        self.estimator.partial_fit(X, y, classes=np.array([0, 1]), sample_weight=sample_weight)
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1].astype(np.float32)

    def save(self, path: str | Path) -> None:
        joblib.dump({"estimator": self.estimator, "params": self.params}, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "SGDPairModel":
        payload = joblib.load(Path(path))
        obj = cls(**payload["params"])
        obj.estimator = payload["estimator"]
        return obj

    def metadata(self) -> dict[str, object]:
        return {"kind": "sgd_logistic", "implementation": "scikit-learn", "params": self.params}

