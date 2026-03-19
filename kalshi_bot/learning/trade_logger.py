"""
Append-only trade log stored as JSONL (one JSON object per line).

Every prediction the bot makes – whether it traded or not – is logged
so the retrainer has a complete feature→outcome dataset.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from kalshi_bot.config import TRADE_LOG_PATH


class TradeRecord:
    """Single prediction + outcome record."""

    def __init__(
        self,
        ts: float,
        market_ticker: str,
        features: list[float],
        feature_names: list[str],
        prob_up: float,
        confidence: float,
        direction: str,          # "up", "down", or "skip"
        bet_dollars: float,
        yes_price: Optional[float],
        outcome: Optional[bool] = None,   # True = won, False = lost, None = pending
        pnl_dollars: Optional[float] = None,
    ):
        self.ts = ts
        self.market_ticker = market_ticker
        self.features = features
        self.feature_names = feature_names
        self.prob_up = prob_up
        self.confidence = confidence
        self.direction = direction
        self.bet_dollars = bet_dollars
        self.yes_price = yes_price
        self.outcome = outcome
        self.pnl_dollars = pnl_dollars

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "market": self.market_ticker,
            "features": self.features,
            "feature_names": self.feature_names,
            "prob_up": round(self.prob_up, 4),
            "confidence": round(self.confidence, 4),
            "direction": self.direction,
            "bet_dollars": self.bet_dollars,
            "yes_price": self.yes_price,
            "outcome": self.outcome,
            "pnl_dollars": self.pnl_dollars,
        }

    @staticmethod
    def from_dict(d: dict) -> "TradeRecord":
        return TradeRecord(**d)


class TradeLogger:
    """Persistent append-only trade log."""

    def __init__(self, path: Path = TRADE_LOG_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: TradeRecord):
        with open(self.path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def read_all(self) -> list[TradeRecord]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(TradeRecord.from_dict(json.loads(line)))
            except Exception:
                continue
        return records

    def update_last_outcome(self, outcome: bool, pnl: float):
        """Update the outcome of the most recent record in the log."""
        records = self.read_all()
        if not records:
            return
        records[-1].outcome = outcome
        records[-1].pnl_dollars = pnl
        self._rewrite(records)

    def _rewrite(self, records: list[TradeRecord]):
        with open(self.path, "w") as f:
            for r in records:
                f.write(json.dumps(r.to_dict()) + "\n")

    def get_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract (X, y) from completed trades for retraining."""
        records = [r for r in self.read_all() if r.outcome is not None and r.features]
        if not records:
            return np.array([]), np.array([])
        X = np.array([r.features for r in records], dtype=np.float32)
        y = np.array([1 if r.outcome else 0 for r in records], dtype=np.int32)
        return X, y

    @property
    def total_trades(self) -> int:
        return sum(1 for r in self.read_all() if r.direction != "skip")

    @property
    def completed_trades(self) -> int:
        return sum(1 for r in self.read_all() if r.outcome is not None)

    def summary(self) -> dict:
        records = self.read_all()
        traded = [r for r in records if r.direction != "skip" and r.outcome is not None]
        wins = [r for r in traded if r.outcome is True]
        total_pnl = sum(r.pnl_dollars or 0 for r in traded)
        return {
            "total_logged": len(records),
            "total_traded": len(traded),
            "wins": len(wins),
            "losses": len(traded) - len(wins),
            "win_rate": len(wins) / len(traded) if traded else 0.0,
            "total_pnl": round(total_pnl, 2),
        }
