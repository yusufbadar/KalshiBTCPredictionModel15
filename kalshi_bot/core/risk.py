"""
Pre-trade risk checks and the kill switch.

Every order passes through ``RiskManager.check``. Hard limits:
* daily realised loss limit (UTC day) -> trading halts until the next day
* drawdown stop: equity falls a set fraction below its peak -> kill switch
  until a person releases it
* per-contract and per-settlement-moment exposure caps, each the tighter of a
  dollar amount and a fraction of the account, so position size follows the
  account up and down
* stale index / stale book -> no new orders
* never spend cash the account does not have (orders are sized to the cash
  that is free right now, fee included)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from kalshi_bot.config import RiskParams
from kalshi_bot.core.fees import taker_fee


@dataclass
class RiskVerdict:
    ok: bool
    reason: str = ""
    max_count: float = 0.0


def utc_day_start(ts: float | None = None) -> float:
    d = datetime.fromtimestamp(ts or time.time(), tz=timezone.utc)
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()


def _tighter(dollars: float, pct: float, equity: float) -> float:
    """The tighter of a dollar cap and a fraction-of-equity cap (0 = that cap is off)."""
    caps = []
    if dollars > 0:
        caps.append(dollars)
    if pct > 0:
        caps.append(pct * max(equity, 0.0))
    return min(caps) if caps else float("inf")


class RiskManager:
    def __init__(self, params: RiskParams):
        self.p = params
        self.killed = False
        self.kill_reason = ""
        self.halted_until = 0.0

    def kill(self, reason: str) -> None:
        self.killed = True
        self.kill_reason = reason

    def reset_kill(self) -> None:
        self.killed = False
        self.kill_reason = ""

    # ------------------------------------------------------------- limits
    def market_cap(self, equity: float) -> float:
        return _tighter(self.p.max_notional_per_market, self.p.pct_per_market, equity)

    def close_cap(self, equity: float) -> float:
        return _tighter(self.p.max_notional_per_close, self.p.pct_per_close, equity)

    def daily_loss_limit(self, equity: float, day_pnl: float) -> float:
        """Dollars the account may lose today; the % version is measured
        against the equity the day started with."""
        return _tighter(abs(self.p.max_daily_loss), self.p.daily_loss_pct, equity - day_pnl)

    def drawdown_limit(self, equity: float, drawdown: float) -> float:
        """Dollars of realised loss from the P&L peak that trip the stop:
        a fraction of what the account was worth at that peak."""
        dd = self.p.max_drawdown_pct
        return dd * max(equity + drawdown, 0.0) if dd > 0 else float("inf")

    # -------------------------------------------------------------- check
    def check(self, *, price: float, count: float, balance: float, day_pnl: float,
              committed_market: float, committed_close: float, index_age: float,
              book_age: float, equity: float | None = None,
              drawdown: float = 0.0) -> RiskVerdict:
        """balance: cash free to spend right now.
        equity: account value (cash + money in open positions); defaults to balance.
        drawdown: dollars the realised P&L is below its peak.
        committed_market: dollars already at risk in this contract;
        committed_close: dollars at risk across every contract that settles at
        the same moment (they move together, so their risk adds)."""
        now = time.time()
        equity = balance if equity is None else equity
        if self.killed:
            return RiskVerdict(False, f"kill switch: {self.kill_reason}")
        if drawdown > 0 and drawdown >= self.drawdown_limit(equity, drawdown):
            self.kill(f"drawdown stop: realised P&L is ${drawdown:,.2f} below its peak "
                      f"({100 * drawdown / max(equity + drawdown, 1e-9):.0f}% of the account)")
            return RiskVerdict(False, f"kill switch: {self.kill_reason}")
        if now < self.halted_until:
            return RiskVerdict(False, "daily loss limit reached")
        if day_pnl <= -self.daily_loss_limit(equity, day_pnl):
            self.halted_until = utc_day_start(now) + 86400
            return RiskVerdict(False, "daily loss limit reached")
        if index_age > self.p.stale_index_s:
            return RiskVerdict(False, f"index stale ({index_age:.1f}s)")
        if book_age > self.p.stale_book_s:
            return RiskVerdict(False, f"book stale ({book_age:.1f}s)")
        if balance < self.p.min_balance:
            return RiskVerdict(False, f"balance ${balance:.2f} below minimum")
        room = min(self.market_cap(equity) - committed_market,
                   self.close_cap(equity) - committed_close)
        spend_cap = min(room, balance - self.p.min_balance)
        unit = price + taker_fee(price)              # worst-case cash per contract
        if spend_cap < unit:
            return RiskVerdict(False, "exposure cap" if room < unit else "not enough free cash")
        max_count = min(count, spend_cap / unit)
        if max_count < 1.0:
            return RiskVerdict(False, "size below 1 contract")
        # whole contracts only: valid on every market and trivially reconcilable
        return RiskVerdict(True, "", float(int(max_count + 1e-6)))
