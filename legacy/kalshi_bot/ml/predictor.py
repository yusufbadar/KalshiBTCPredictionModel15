"""
Probability model for BTC 15-minute direction prediction.

Default: logistic regression (stable probabilities, easy calibration).
Optional: small XGBoost (max_depth=2) — only use if it beats logistic
on out-of-sample EV after fees.

All features are standardised internally via StandardScaler.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from kalshi_bot.config import ML, MODEL_DIR

try:
    import xgboost as xgb
except ImportError:
    xgb = None


class Predictor:
    """Binary classifier: 1 = BTC goes UP in next 15 min, 0 = DOWN."""

    def __init__(self, model_type: str | None = None):
        self.model_type = model_type or ML.model_type
        self.model = None
        self.scaler: StandardScaler = StandardScaler()
        self.is_trained: bool = False
        self.feature_names: list[str] = []
        self.train_accuracy: float = 0.0
        self.n_train_samples: int = 0

    # ── training ────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        eval_fraction: float = 0.15,
    ) -> dict:
        n = len(X)
        if n < ML.min_training_samples:
            return {"error": f"need {ML.min_training_samples} samples, got {n}"}

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        split = int(n * (1 - eval_fraction))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)

        if self.model_type == "xgboost" and xgb is not None:
            self.model = xgb.XGBClassifier(**ML.xgb_params)
            self.model.fit(
                X_train_s, y_train,
                eval_set=[(X_val_s, y_val)],
                verbose=False,
            )
        else:
            self.model = LogisticRegression(
                C=1.0, max_iter=1000, solver="lbfgs",
            )
            self.model.fit(X_train_s, y_train)

        train_pred = self.model.predict(X_train_s)
        val_pred = self.model.predict(X_val_s)
        train_acc = float(np.mean(train_pred == y_train))
        val_acc = float(np.mean(val_pred == y_val))

        val_proba = self.model.predict_proba(X_val_s)
        prob_up_val = val_proba[:, 1] if val_proba.shape[1] > 1 else val_proba[:, 0]
        brier = float(np.mean((prob_up_val - y_val) ** 2))

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
            "brier_score": round(brier, 4),
            "model_type": self.model_type,
        }
        logger.info("Model trained: {}", metrics)
        return metrics

    # ── inference ───────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> float:
        """Return P(UP) for a single feature vector (scalar float)."""
        if not self.is_trained or self.model is None:
            return 0.5

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_s = self.scaler.transform(X)
        proba = self.model.predict_proba(X_s)[0]
        return float(proba[1]) if len(proba) > 1 else float(proba[0])

    # ── diagnostics ─────────────────────────────────────────────────

    def feature_importance(self) -> dict[str, float]:
        if not self.is_trained or self.model is None:
            return {}
        if isinstance(self.model, LogisticRegression):
            coefs = np.abs(self.model.coef_[0])
            names = self.feature_names or [f"f{i}" for i in range(len(coefs))]
            return dict(sorted(zip(names, map(float, coefs)), key=lambda x: -x[1]))
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            names = self.feature_names or [f"f{i}" for i in range(len(importances))]
            return dict(sorted(zip(names, map(float, importances)), key=lambda x: -x[1]))
        return {}

    # ── persistence ─────────────────────────────────────────────────

    def save(self, tag: str = "latest") -> Path:
        path = MODEL_DIR / f"predictor_{tag}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "train_accuracy": self.train_accuracy,
                "n_train_samples": self.n_train_samples,
                "model_type": self.model_type,
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
            self.scaler = data.get("scaler", StandardScaler())
            self.feature_names = data.get("feature_names", [])
            self.train_accuracy = data.get("train_accuracy", 0.0)
            self.n_train_samples = data.get("n_train_samples", 0)
            self.model_type = data.get("model_type", "logistic")
            self.is_trained = self.model is not None
            logger.info("Model loaded ← {} (acc={:.2%})", path, self.train_accuracy)
            return True
        except Exception as e:
            logger.error("Model load failed: {}", e)
            return False
