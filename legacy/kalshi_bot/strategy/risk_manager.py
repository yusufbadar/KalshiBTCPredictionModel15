"""
Risk manager — percentage-of-balance sizing.

Bets ``TRADING.bet_fraction`` of the Kalshi API cash balance each trade
(default 9%; override with env ``BET_FRACTION``, e.g. ``0.09``).
"""
from __future__ import annotations

from datetime import datetime, timezone

from kalshi_bot.config import TRADING


class RiskDecision:
    def __init__(self, allowed: bool, bet_dollars: float, reason: str = ""):
        self.allowed = allowed
        self.bet_dollars = bet_dollars
        self.reason = reason


class RiskManager:
    def __init__(self):
        self._daily_loss: float = 0.0
        self._daily_reset_date: str = ""

    def check(self, balance: float, ev_cents: float = 0.0) -> RiskDecision:
        self._maybe_reset_daily()

        if balance < TRADING.bankroll_floor:
            return RiskDecision(
                False, 0,
                f"Balance ${balance:.2f} < floor ${TRADING.bankroll_floor}",
            )

        bet = round(balance * TRADING.bet_fraction, 2)

        available = balance - TRADING.bankroll_floor
        if bet > available:
            bet = round(available, 2)

        if bet < 1.0:
            return RiskDecision(False, 0, "Bet size too small (< $1)")

        return RiskDecision(True, bet)

    def record_loss(self, amount: float):
        self._daily_loss += abs(amount)

    def record_win(self, amount: float):
        self._daily_loss -= abs(amount)
        self._daily_loss = max(0, self._daily_loss)

    def _maybe_reset_daily(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_loss = 0.0
            self._daily_reset_date = today
