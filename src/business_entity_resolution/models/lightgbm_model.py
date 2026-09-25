"""Optional MIT-licensed LightGBM pair-model adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import numpy as np


class LightGBMPairModel:
    def __init__(self, seed: int = 2026, threads: int = 1, **params: Any) -> None:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("LightGBM is optional; install requirements-remote.txt") from exc
        defaults = {
            "objective": "binary", "learning_rate": 0.05, "n_estimators": 2000,
            "num_leaves": 63, "min_child_samples": 100, "random_state": seed,
            "n_jobs": threads, "deterministic": True, "force_col_wise": True,
        }
        defaults.update(params)
        self.lgb, self.params = lgb, defaults
        self.estimator = lgb.LGBMClassifier(**defaults)

    def fit(self, X, y, sample_weight=None, eval_data=None, logger=None) -> "LightGBMPairModel":
        kwargs: dict[str, object] = {"sample_weight": sample_weight}
        callbacks: list[object] = []
        if eval_data is not None:
            kwargs.update({"eval_set": [eval_data]})
            callbacks.append(self.lgb.early_stopping(100, first_metric_only=True))
        if logger is not None:
            callbacks.append(self._logging_callback(logger))
        if callbacks:
            kwargs["callbacks"] = callbacks
        self.estimator.fit(X, y, **kwargs)
        return self

    def _logging_callback(self, logger):
        def _callback(environment) -> None:
            results = environment.evaluation_result_list or []
            for _dataset, metric, value, _higher in results:
                logger.metric(f"train_{metric}", float(value), iteration=int(environment.iteration))
        return _callback

    def predict_scores(self, X) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1].astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        self.estimator.booster_.save_model(str(path))
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(self.metadata(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMPairModel":
        path = Path(path)
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
        obj = cls(**metadata["params"])
        booster = obj.lgb.Booster(model_file=str(path))
        obj.estimator._Booster = booster
        obj.estimator.fitted_ = True
        # Restore the sklearn wrapper's feature-count state so predict_proba can
        # validate the input matrix instead of assuming -1 features.
        n_features = int(booster.num_feature())
        obj.estimator._n_features = n_features
        obj.estimator._n_features_ = n_features
        obj.estimator.n_features_in_ = n_features
        obj.estimator._le = None
        try:
            obj.estimator._classes = np.array([0, 1])
        except Exception:  # noqa: BLE001
            pass
        return obj

    def metadata(self) -> dict[str, object]:
        return {"kind": "lightgbm", "implementation_license": "MIT", "params": self.params}

