"""
Periodic model retraining using accumulated live trade data merged with
historical bootstrap data.

The retrainer:
  1. Merges historical + live features/labels
  2. Retrains the XGBoost model
  3. Compares new vs old accuracy – rolls back if degraded
  4. Adjusts confidence threshold based on calibration
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from loguru import logger

from kalshi_bot.config import ML
from kalshi_bot.ml.predictor import Predictor
from kalshi_bot.ml.model_store import ModelStore
from kalshi_bot.learning.trade_logger import TradeLogger


class Retrainer:
    """Manages the retrain-evaluate-deploy cycle."""

    def __init__(
        self,
        predictor: Predictor,
        model_store: ModelStore,
        trade_logger: TradeLogger,
    ):
        self.predictor = predictor
        self.store = model_store
        self.logger = trade_logger
        self._last_retrain_ts: float = 0
        self._last_retrain_n: int = 0
        self._historical_X: Optional[np.ndarray] = None
        self._historical_y: Optional[np.ndarray] = None

    def set_historical_data(self, X: np.ndarray, y: np.ndarray):
        """Cache the bootstrap historical dataset."""
        self._historical_X = X
        self._historical_y = y
        logger.info("Historical data cached: {} samples", len(y))

    def should_retrain(self) -> bool:
        n_completed = self.logger.completed_trades
        trades_since = n_completed - self._last_retrain_n
        hours_since = (time.time() - self._last_retrain_ts) / 3600

        if trades_since >= ML.retrain_every_n_trades:
            return True
        if hours_since >= ML.retrain_every_hours and trades_since >= 20:
            return True
        return False

    def retrain(self) -> dict:
        """Run a full retrain cycle. Returns metrics dict."""
        # Gather live data
        live_X, live_y = self.logger.get_training_data()
        logger.info(
            "Retrain: {} live samples, {} historical samples",
            len(live_y),
            len(self._historical_y) if self._historical_y is not None else 0,
        )

        # Merge historical + live
        parts_X, parts_y = [], []
        if self._historical_X is not None and len(self._historical_X):
            parts_X.append(self._historical_X)
            parts_y.append(self._historical_y)
        if len(live_X):
            parts_X.append(live_X)
            parts_y.append(live_y)

        if not parts_X:
            logger.warning("No training data available")
            return {"error": "no data"}

        X = np.vstack(parts_X)
        y = np.concatenate(parts_y)

        # Remember old accuracy for comparison
        old_acc = self.store.latest_accuracy

        # Train
        metrics = self.predictor.train(X, y)
        if "error" in metrics:
            return metrics

        new_acc = metrics.get("val_acc", 0)

        # If new model is significantly worse, roll back
        if old_acc > 0 and new_acc < old_acc - 0.03:
            logger.warning(
                "New model acc {:.2%} < old {:.2%} – rolling back",
                new_acc, old_acc,
            )
            self.store.rollback(self.predictor)
            metrics["rolled_back"] = True
        else:
            self.store.save_version(self.predictor, metrics)
            metrics["rolled_back"] = False

        self._last_retrain_ts = time.time()
        self._last_retrain_n = self.logger.completed_trades

        return metrics
