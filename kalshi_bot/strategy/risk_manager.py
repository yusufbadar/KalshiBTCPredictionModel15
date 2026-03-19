"""
Risk manager – protects the bankroll using Kelly-inspired sizing,
daily loss limits, consecutive-loss circuit breakers, and a hard floor.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from loguru import logger

from kalshi_bot.config import TRADING
from kalshi_bot.learning.performance_analyzer import PerformanceAnalyzer


class RiskDecision:
    """Result of a risk check."""
    def __init__(self, allowed: bool, bet_dollars: float, reason: str = ""):
        self.allowed = allowed
        self.bet_dollars = bet_dollars
        self.reason = reason

    def __repr__(self):
        return f"<Risk allowed={self.allowed} bet=${self.bet_dollars} reason={self.reason!r}>"


class RiskManager:
    """Centralized risk gate – every trade must pass through ``check()``."""

    def __init__(self, analyzer: PerformanceAnalyzer):
        self.analyzer = analyzer
        self._daily_loss: float = 0.0
        self._daily_reset_date: str = ""
        self._pause_until: float = 0.0

    def check(self, balance: float, confidence: float) -> RiskDecision:
        """Evaluate whether the bot should trade right now.

        Args:
            balance: current Kalshi balance in dollars.
            confidence: model confidence (0.0 – 1.0).

        Returns:
            RiskDecision with allowed flag and recommended bet size.
        """
        self._maybe_reset_daily()

        # Hard bankroll floor
        if balance < TRADING.bankroll_floor:
            return RiskDecision(False, 0, f"Balance ${balance:.2f} < floor ${TRADING.bankroll_floor}")

        # Daily loss limit
        if self._daily_loss >= TRADING.daily_loss_limit:
            return RiskDecision(False, 0, f"Daily loss ${self._daily_loss:.2f} >= limit ${TRADING.daily_loss_limit}")

        # Consecutive-loss pause
        if time.time() < self._pause_until:
            remaining = int(self._pause_until - time.time())
            return RiskDecision(False, 0, f"Paused for {remaining}s after consecutive losses")

        if self.analyzer.consecutive_losses >= TRADING.consecutive_loss_pause:
            self._pause_until = time.time() + TRADING.pause_duration_minutes * 60
            return RiskDecision(
                False, 0,
                f"{self.analyzer.consecutive_losses} consecutive losses – pausing {TRADING.pause_duration_minutes}min",
            )

        # Confidence gate
        if confidence < TRADING.confidence_threshold:
            return RiskDecision(False, 0, f"Confidence {confidence:.2%} < threshold {TRADING.confidence_threshold:.0%}")

        # Bet sizing (Kelly-inspired: reduce when bankroll is low)
        if balance < TRADING.reduce_bet_threshold:
            bet = TRADING.reduced_bet_size_dollars
        else:
            bet = TRADING.bet_size_dollars

        # Never bet more than we can afford
        bet = min(bet, balance - TRADING.bankroll_floor)
        if bet <= 0:
            return RiskDecision(False, 0, "Bet size would breach floor")

        return RiskDecision(True, bet)

    def record_loss(self, amount: float):
        self._daily_loss += abs(amount)

    def record_win(self, amount: float):
        self._daily_loss -= abs(amount)  # wins reduce daily loss tally
        self._daily_loss = max(0, self._daily_loss)

    def _maybe_reset_daily(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_loss = 0.0
            self._daily_reset_date = today
            logger.info("Daily risk counters reset for {}", today)
