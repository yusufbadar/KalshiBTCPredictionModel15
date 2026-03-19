"""
XGBoost-based BTC 15-minute direction predictor.

Provides:
  - ``train(X, y)``  – train or retrain the model
  - ``predict(X)``   – return (probability_up, confidence)
  - ``save / load``  – persist to disk
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from loguru import logger

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    logger.warning("xgboost not installed – ML predictor will use fallback")

from kalshi_bot.config import ML, MODEL_DIR


class Predictor:
    """XGBoost binary classifier: 1 = BTC goes UP in next 15 min, 0 = DOWN."""

    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self.is_trained = False
        self.feature_names: list[str] = []
        self.train_accuracy: float = 0.0
        self.n_train_samples: int = 0

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        eval_fraction: float = 0.15,
    ) -> dict:
        """Train the model from scratch.

        Returns dict with metrics (accuracy, logloss, n_samples).
        """
        if xgb is None:
            logger.error("Cannot train: xgboost not installed")
            return {"error": "xgboost not installed"}

        n = len(X)
        if n < ML.min_training_samples:
            logger.warning("Not enough samples ({} < {})", n, ML.min_training_samples)
            return {"error": f"need {ML.min_training_samples} samples, got {n}"}

        split = int(n * (1 - eval_fraction))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self.model = xgb.XGBClassifier(**ML.xgb_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        train_acc = float(np.mean(train_pred == y_train))
        val_acc = float(np.mean(val_pred == y_val))

        self.is_trained = True
        self.train_accuracy = val_acc
        self.n_train_samples = n
        if feature_names:
            self.feature_names = list(feature_names)

        metrics = {
            "n_samples": n,
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "val_size": len(y_val),
        }
        logger.info("Model trained: {}", metrics)
        return metrics

    def predict(self, X: np.ndarray) -> Tuple[float, float]:
        """Return (prob_up, confidence) for a single sample.

        *prob_up*: probability BTC goes up (0.0 – 1.0)
        *confidence*: distance from 0.5 scaled to 0–1 (higher = more certain)

        Falls back to 0.5 / 0.0 if model is not trained.
        """
        if not self.is_trained or self.model is None:
            return 0.5, 0.0

        if X.ndim == 1:
            X = X.reshape(1, -1)

        proba = self.model.predict_proba(X)[0]
        prob_up = float(proba[1]) if len(proba) > 1 else float(proba[0])
        confidence = abs(prob_up - 0.5) * 2.0  # 0..1
        return prob_up, confidence

    def feature_importance(self) -> dict[str, float]:
        """Return feature importances keyed by name."""
        if not self.is_trained or self.model is None:
            return {}
        importances = self.model.feature_importances_
        names = self.feature_names or [f"f{i}" for i in range(len(importances))]
        return dict(sorted(zip(names, map(float, importances)), key=lambda x: -x[1]))

    # ── persistence ───────────────────────────────────────────────

    def save(self, tag: str = "latest") -> Path:
        path = MODEL_DIR / f"predictor_{tag}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "train_accuracy": self.train_accuracy,
                "n_train_samples": self.n_train_samples,
            }, f)
        logger.info("Model saved → {}", path)
        return path

    def load(self, tag: str = "latest") -> bool:
        path = MODEL_DIR / f"predictor_{tag}.pkl"
        if not path.exists():
            logger.warning("No model file at {}", path)
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.feature_names = data.get("feature_names", [])
            self.train_accuracy = data.get("train_accuracy", 0.0)
            self.n_train_samples = data.get("n_train_samples", 0)
            self.is_trained = self.model is not None
            logger.info("Model loaded ← {} (acc={:.2%})", path, self.train_accuracy)
            return True
        except Exception as e:
            logger.error("Model load failed: {}", e)
            return False
