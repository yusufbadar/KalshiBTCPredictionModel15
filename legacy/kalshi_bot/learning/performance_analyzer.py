"""
Performance analytics – win rate tracking, drawdown, feature importance
evolution, and confidence calibration.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from kalshi_bot.learning.trade_logger import TradeLogger


class PerformanceAnalyzer:
    """Live analytics computed from the trade log."""

    def __init__(self, trade_logger: TradeLogger, predictor=None):
        self.log = trade_logger
        self.predictor = predictor
        self._recent_outcomes: deque[bool] = deque(maxlen=100)
        self._recent_pnl: deque[float] = deque(maxlen=100)

    def record(self, won: bool, pnl: float):
        self._recent_outcomes.append(won)
        self._recent_pnl.append(pnl)

    # ── aggregate metrics ──────────────────────────────────────────

    @property
    def win_rate_all(self) -> float:
        s = self.log.summary()
        return s["win_rate"]

    @property
    def win_rate_recent(self) -> float:
        if not self._recent_outcomes:
            return 0.0
        return sum(self._recent_outcomes) / len(self._recent_outcomes)

    @property
    def total_pnl(self) -> float:
        return self.log.summary()["total_pnl"]

    @property
    def recent_pnl(self) -> float:
        return sum(self._recent_pnl)

    @property
    def max_drawdown(self) -> float:
        pnls = list(self._recent_pnl)
        if not pnls:
            return 0.0
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        dd = peak - cumulative
        return float(np.max(dd)) if len(dd) else 0.0

    @property
    def sharpe(self) -> float:
        if len(self._recent_pnl) < 5:
            return 0.0
        arr = np.array(list(self._recent_pnl))
        std = float(np.std(arr))
        if std == 0:
            return 0.0
        return float(np.mean(arr)) / std * (96 ** 0.5)  # annualised (96 15-min slots/day)

    @property
    def consecutive_losses(self) -> int:
        count = 0
        for o in reversed(self._recent_outcomes):
            if not o:
                count += 1
            else:
                break
        return count

    @property
    def streak(self) -> int:
        """Positive = winning streak, negative = losing streak."""
        if not self._recent_outcomes:
            return 0
        last = self._recent_outcomes[-1]
        count = 0
        for o in reversed(self._recent_outcomes):
            if o == last:
                count += 1
            else:
                break
        return count if last else -count

    def feature_importance_report(self) -> dict[str, float]:
        if self.predictor is None:
            return {}
        return self.predictor.feature_importance()

    def calibration_offset(self) -> float:
        """How much the model's probabilities are off from observed.

        Positive = model is overconfident on UP; negative = underconfident.
        """
        records = [
            r for r in self.log.read_all()
            if r.outcome is not None and r.direction != "skip"
        ]
        if len(records) < 20:
            return 0.0
        predicted_up = np.mean([r.prob_up for r in records])
        actual_up = np.mean([1 if r.outcome else 0 for r in records])
        return float(predicted_up - actual_up)

    def full_report(self) -> dict:
        return {
            "win_rate_all": round(self.win_rate_all, 4),
            "win_rate_recent": round(self.win_rate_recent, 4),
            "total_pnl": round(self.total_pnl, 2),
            "recent_pnl": round(self.recent_pnl, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "sharpe": round(self.sharpe, 2),
            "streak": self.streak,
            "consecutive_losses": self.consecutive_losses,
            "calibration_offset": round(self.calibration_offset(), 4),
            "top_features": dict(list(self.feature_importance_report().items())[:10]),
        }
