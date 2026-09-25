"""Incremental scikit-learn logistic pair model."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


class SGDPairModel:
    def __init__(self, seed: int = 2026, **params: Any) -> None:
        defaults = {"loss": "log_loss", "penalty": "l2", "alpha": 1e-4, "max_iter": 1000, "tol": 1e-4, "random_state": seed}
        defaults.update(params)
        self.estimator = SGDClassifier(**defaults)
        self.scaler = StandardScaler()
        self._scaler_fitted = False
        self.params = defaults

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None, eval_data=None, logger=None) -> "SGDPairModel":
        scaled = self.scaler.fit_transform(X)
        self._scaler_fitted = True
        self.estimator.fit(scaled, y, sample_weight=sample_weight)
        if logger is not None:
            logger.metric("train_iterations", int(getattr(self.estimator, "n_iter_", 0)))
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "SGDPairModel":
        self.update_scaler(X)
        self.partial_fit_scaled(X, y, sample_weight)
        return self

    def update_scaler(self, X: np.ndarray) -> None:
        """Update scaling statistics without retaining the training batch."""
        self.scaler.partial_fit(X)
        self._scaler_fitted = True

    def partial_fit_scaled(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        """Train after a separate scaler pass, enabling bounded two-pass SGD."""
        if not self._scaler_fitted:
            raise RuntimeError("update_scaler must be called before partial_fit_scaled")
        scaled = self.scaler.transform(X)
        self.estimator.partial_fit(scaled, y, classes=np.array([0, 1]), sample_weight=sample_weight)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(X) if self._scaler_fitted else X
        return self.estimator.predict_proba(scaled)[:, 1].astype(np.float32)

    def save(self, path: str | Path) -> None:
        joblib.dump({
            "estimator": self.estimator,
            "scaler": self.scaler,
            "scaler_fitted": self._scaler_fitted,
            "params": self.params,
        }, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "SGDPairModel":
        payload = joblib.load(Path(path))
        obj = cls(**payload["params"])
        obj.estimator = payload["estimator"]
        if "scaler" in payload:
            obj.scaler = payload["scaler"]
            obj._scaler_fitted = bool(payload.get("scaler_fitted", True))
        else:
            # Backward compatibility for pre-hardening artifacts.
            obj._scaler_fitted = False
        return obj

    def metadata(self) -> dict[str, object]:
        return {
            "kind": "sgd_logistic",
            "implementation": "scikit-learn",
            "implementation_license": "BSD-3-Clause",
            "scaled": True,
            "params": self.params,
        }

