"""
Risk manager — gentle EV-based sizing.

Bets every interval.  Size scales with the magnitude of the expected
value in three tiers.  No Kelly.  No confidence thresholds.

  EV < 1.5c  →  minimum bet
  1.5–4c     →  2× minimum
  > 4c       →  3× minimum
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

    def check(self, balance: float, ev_cents: float) -> RiskDecision:
        """Decide bet size based on balance and EV magnitude."""
        self._maybe_reset_daily()

        if balance < TRADING.bankroll_floor:
            return RiskDecision(
                False, 0,
                f"Balance ${balance:.2f} < floor ${TRADING.bankroll_floor}",
            )

        abs_ev = abs(ev_cents)
        if abs_ev >= TRADING.ev_tier_2:
            bet = TRADING.min_bet_size_dollars * TRADING.size_multiplier_3
        elif abs_ev >= TRADING.ev_tier_1:
            bet = TRADING.min_bet_size_dollars * TRADING.size_multiplier_2
        else:
            bet = TRADING.min_bet_size_dollars

        bet = min(bet, TRADING.max_bet_size_dollars)
        bet = min(bet, balance - TRADING.bankroll_floor)

        if bet < TRADING.min_bet_size_dollars:
            if balance >= TRADING.bankroll_floor + TRADING.min_bet_size_dollars:
                bet = TRADING.min_bet_size_dollars
            else:
                return RiskDecision(False, 0, "Balance too low for minimum bet")

        return RiskDecision(True, round(bet, 2))

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
