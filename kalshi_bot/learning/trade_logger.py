"""
Append-only trade log stored as JSONL. Tracks every trade with full details:
fill prices, fees, order IDs, and accurate P&L.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from kalshi_bot.config import TRADE_LOG_PATH


class TradeRecord:
    """Single trade record with full execution details."""

    def __init__(
        self,
        ts: float,
        market_ticker: str,
        features: list[float] | None = None,
        feature_names: list[str] | None = None,
        prob_up: float = 0.5,
        confidence: float = 0.5,
        direction: str = "skip",
        bet_dollars: float = 0.0,
        yes_price: Optional[float] = None,
        outcome: Optional[bool] = None,
        pnl_dollars: Optional[float] = None,
        order_id: str = "",
        fill_price_cents: int = 0,
        fill_count: int = 0,
        fees_cents: float = 0.0,
        side: str = "",
        entry_type: str = "entry",
        **kwargs,
    ):
        self.ts = ts
        self.market_ticker = market_ticker
        self.features = features or []
        self.feature_names = feature_names or []
        self.prob_up = prob_up
        self.confidence = confidence
        self.direction = direction
        self.bet_dollars = bet_dollars
        self.yes_price = yes_price
        self.outcome = outcome
        self.pnl_dollars = pnl_dollars
        self.order_id = order_id
        self.fill_price_cents = fill_price_cents
        self.fill_count = fill_count
        self.fees_cents = fees_cents
        self.side = side
        self.entry_type = entry_type

    @property
    def time_str(self) -> str:
        return datetime.fromtimestamp(self.ts, tz=timezone.utc).strftime("%H:%M:%S")

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "market": self.market_ticker,
            "prob_up": round(self.prob_up, 4),
            "confidence": round(self.confidence, 4),
            "direction": self.direction,
            "bet_dollars": round(self.bet_dollars, 2),
            "yes_price": self.yes_price,
            "outcome": self.outcome,
            "pnl_dollars": self.pnl_dollars,
            "order_id": self.order_id,
            "fill_price_cents": self.fill_price_cents,
            "fill_count": self.fill_count,
            "fees_cents": round(self.fees_cents, 2),
            "side": self.side,
            "entry_type": self.entry_type,
            "features": self.features,
            "feature_names": self.feature_names,
        }

    @staticmethod
    def from_dict(d: dict) -> "TradeRecord":
        return TradeRecord(
            ts=d.get("ts", 0),
            market_ticker=d.get("market", ""),
            features=d.get("features"),
            feature_names=d.get("feature_names"),
            prob_up=d.get("prob_up", 0.5),
            confidence=d.get("confidence", 0.5),
            direction=d.get("direction", "skip"),
            bet_dollars=d.get("bet_dollars", 0),
            yes_price=d.get("yes_price"),
            outcome=d.get("outcome"),
            pnl_dollars=d.get("pnl_dollars"),
            order_id=d.get("order_id", ""),
            fill_price_cents=d.get("fill_price_cents", 0),
            fill_count=d.get("fill_count", 0),
            fees_cents=d.get("fees_cents", 0),
            side=d.get("side", ""),
            entry_type=d.get("entry_type", "entry"),
        )


class TradeLogger:
    """Persistent append-only trade log with accurate tracking."""

    def __init__(self, path: Path = TRADE_LOG_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: list[TradeRecord] | None = None

    def append(self, record: TradeRecord):
        self._cache = None
        with open(self.path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
        logger.debug(
            "Logged trade: {} {} {} ${:.2f} @ {}c",
            record.entry_type, record.direction, record.market_ticker,
            record.bet_dollars, record.fill_price_cents,
        )

    def read_all(self) -> list[TradeRecord]:
        if self._cache is not None:
            return list(self._cache)
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
        self._cache = records
        return list(records)

    def update_outcome(self, market_ticker: str, outcome: bool, pnl: float):
        """Update the outcome for all pending records matching this market."""
        self._cache = None
        records = self.read_all()
        updated = False
        for r in records:
            if r.market_ticker == market_ticker and r.outcome is None and r.direction != "skip":
                r.outcome = outcome
                r.pnl_dollars = round(pnl, 2)
                updated = True
        if updated:
            self._rewrite(records)
        return updated

    def update_last_outcome(self, outcome: bool, pnl: float):
        self._cache = None
        records = self.read_all()
        if not records:
            return
        records[-1].outcome = outcome
        records[-1].pnl_dollars = pnl
        self._rewrite(records)

    def _rewrite(self, records: list[TradeRecord]):
        self._cache = None
        with open(self.path, "w") as f:
            for r in records:
                f.write(json.dumps(r.to_dict()) + "\n")

    def get_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        records = [r for r in self.read_all()
                   if r.outcome is not None and r.features and r.direction != "skip"]
        if not records:
            return np.array([]), np.array([])
        X = np.array([r.features for r in records], dtype=np.float32)
        y = np.array([1 if r.outcome else 0 for r in records], dtype=np.int32)
        return X, y

    @property
    def total_trades(self) -> int:
        return sum(1 for r in self.read_all() if r.direction not in ("skip", "order_error"))

    @property
    def completed_trades(self) -> int:
        return sum(1 for r in self.read_all() if r.outcome is not None)

    def recent_trades(self, n: int = 20) -> list[TradeRecord]:
        all_recs = self.read_all()
        traded = [r for r in all_recs if r.direction not in ("skip", "order_error")]
        return traded[-n:]

    def summary(self) -> dict:
        records = self.read_all()
        traded = [r for r in records if r.direction not in ("skip", "order_error")]
        settled = [r for r in traded if r.outcome is not None]
        wins = [r for r in settled if r.outcome is True]
        total_pnl = sum(r.pnl_dollars or 0 for r in settled)
        total_fees = sum(r.fees_cents or 0 for r in traded) / 100.0
        return {
            "total_logged": len(records),
            "total_traded": len(traded),
            "total_settled": len(settled),
            "pending": len(traded) - len(settled),
            "wins": len(wins),
            "losses": len(settled) - len(wins),
            "win_rate": len(wins) / len(settled) if settled else 0.0,
            "total_pnl": round(total_pnl, 2),
            "total_fees": round(total_fees, 2),
        }
