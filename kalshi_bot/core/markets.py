"""
Tracks the live KXBTC15M contract (a new one opens every quarter hour).

Upcoming contracts are listed ahead of time (status ``initialized``), so the
clock keeps the next hour of contracts in memory and switches the instant a
contract's open time passes - it does not wait for the listing to show the
new market as active. The strike of a new contract is the previous
contract's settlement average (mean of the BRTI prints stamped
open-60 .. open-1), so it can be computed locally the moment the market
opens; the API value replaces it once published.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from kalshi_bot.api.rest import KalshiREST, fnum
from kalshi_bot.config import SERIES_TICKER


def _ts(s: str | None) -> float:
    if not s:
        return 0.0
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


@dataclass
class Contract:
    ticker: str
    event_ticker: str
    open_ts: int
    close_ts: int
    strike: float | None          # floor_strike: 60 s BRTI average at open
    strike_source: str            # 'api' | 'local' | ''
    status: str
    exchange_index: int
    tick_ranges: list             # [(start, end, step)] from price_ranges
    strike_type: str = "greater_or_equal"   # KXBTC15M: S >= K; KXBTCD ladder: S > K
    series: str = ""
    decimals: int = 2             # precision the settlement average is rounded to
    asset: str = "BTC"

    @property
    def seconds_left(self) -> float:
        return self.close_ts - time.time()

    @property
    def is_quarter_hour(self) -> bool:
        """A 15-minute up/down market (as opposed to an hourly strike)."""
        return self.close_ts - self.open_ts == 900

    @property
    def threshold(self) -> float | None:
        """Level the unrounded 60-print average must reach for YES. Settlement
        rounds the average (to cents for BTC), so with half a rounding unit h:
        S >= K  <=>  avg >= K - h   and   S > K  <=>  avg >= K + h."""
        if self.strike is None:
            return None
        h = 0.5 * 10.0 ** (-self.decimals)
        return self.strike + (h if self.strike_type == "greater" else -h)

    def tick_at(self, price: float, side: str = "down") -> float:
        for lo, hi, step in self.tick_ranges:
            if side == "down" and lo - 1e-9 <= price < hi - 1e-9:
                return step
            if side == "up" and lo + 1e-9 < price <= hi + 1e-9:
                return step
        return 0.01

    def snap(self, price: float, direction: str) -> float:
        """Round a price onto the market's tick grid ('down' or 'up')."""
        step = self.tick_at(price, direction)
        n = price / step
        n = math.floor(n + 1e-9) if direction == "down" else math.ceil(n - 1e-9)
        return round(min(max(n * step, 0.001), 0.999), 4)

    @classmethod
    def from_api(cls, m: dict) -> "Contract":
        ranges = [(fnum(r["start"]), fnum(r["end"]), fnum(r["step"]))
                  for r in (m.get("price_ranges") or [])] or [(0.0, 1.0, 0.01)]
        strike = m.get("floor_strike")
        has = strike not in (None, "")
        return cls(ticker=m["ticker"], event_ticker=m.get("event_ticker", ""),
                   open_ts=int(_ts(m.get("open_time"))), close_ts=int(_ts(m.get("close_time"))),
                   strike=float(strike) if has else None, strike_source="api" if has else "",
                   status=m.get("status", ""), exchange_index=int(m.get("exchange_index", 0) or 0),
                   tick_ranges=ranges, strike_type=m.get("strike_type") or "greater_or_equal",
                   series=(m.get("event_ticker", "").split("-")[0] if m.get("event_ticker") else ""))


class MarketClock:
    def __init__(self, rest: KalshiREST, authed: bool = True, series: str = SERIES_TICKER,
                 decimals: int = 2, asset: str = "BTC"):
        self.rest = rest
        self.authed = authed
        self.series = series
        self.decimals = decimals
        self.asset = asset
        self.contracts: dict[str, Contract] = {}
        self.current: Contract | None = None
        self._last_list = 0.0
        self._retry_at = 0.0
        self._last_detail = 0.0

    async def refresh(self, index=None) -> None:
        now = index.exchange_time() if index is not None and index.last_rx else time.time()
        if (now - self._last_list > 60 or not self._upcoming_after(now)) and now >= self._retry_at:
            await self._list(now)
        live = [c for c in self.contracts.values() if c.open_ts <= now < c.close_ts]
        new = min(live, key=lambda c: c.close_ts) if live else None
        if new is not None and (self.current is None or new.ticker != self.current.ticker):
            self.current = new
        elif new is None:
            self.current = None
        c = self.current
        if c is None:
            return
        if c.strike is None and index is not None:
            local = index.settlement_avg(c.open_ts)    # mean of prints open-60 .. open-1
            if local is not None:
                c.strike, c.strike_source = local, "local"
        # confirm status/strike from the API shortly after the open
        if (c.strike_source != "api" or c.status not in ("active", "open")) and now - self._last_detail > 2:
            self._last_detail = now
            try:
                m = await self.rest.get(f"/markets/{c.ticker}", auth=self.authed)
                fresh = Contract.from_api(m.get("market", m))
                c.status = fresh.status
                c.tick_ranges = fresh.tick_ranges
                if fresh.strike is not None:
                    if c.strike_source == "local" and abs(fresh.strike - c.strike) > 0.011:
                        logger.warning("local strike {} != published {}", c.strike, fresh.strike)
                    c.strike, c.strike_source = fresh.strike, "api"
            except Exception as e:  # noqa: BLE001
                logger.debug("market detail refresh failed: {}", e)

    def _upcoming_after(self, now: float) -> bool:
        return any(c.open_ts > now for c in self.contracts.values())

    async def _list(self, now: float) -> None:
        try:
            d = await self.rest.get("/markets", {"series_ticker": self.series, "limit": 20,
                                                 "min_close_ts": int(now), "max_close_ts": int(now) + 3700},
                                    auth=self.authed)
        except Exception as e:  # noqa: BLE001
            logger.warning("market list failed: {}", e)
            self._retry_at = now + 2.0
            return
        self._last_list = now
        for m in d.get("markets", []):
            c = Contract.from_api(m)
            if c.close_ts - c.open_ts != 900:
                continue
            c.decimals, c.asset = self.decimals, self.asset
            old = self.contracts.get(c.ticker)
            if old is not None and old.strike is not None and c.strike is None:
                c.strike, c.strike_source = old.strike, old.strike_source
            self.contracts[c.ticker] = c
        for tk in [tk for tk, c in self.contracts.items() if c.close_ts < now - 3600]:
            self.contracts.pop(tk, None)


class HourlyLadder:
    """The hourly BTC above/below strike ladder (KXBTCD). Every strike of the
    event closing at the top of the hour settles on the same 60-print BRTI
    average as the KXBTC15M contract closing then, so the same lock-in model
    prices all of them."""

    def __init__(self, rest: KalshiREST, authed: bool = True, series: str = "KXBTCD",
                 decimals: int = 2, asset: str = "BTC"):
        self.rest = rest
        self.authed = authed
        self.series = series
        self.decimals = decimals
        self.asset = asset
        self.contracts: dict[str, Contract] = {}
        self._last_list = 0.0
        self._retry_at = 0.0

    def next_close(self, now: float) -> int | None:
        closes = [c.close_ts for c in self.contracts.values() if c.close_ts > now]
        return min(closes) if closes else None

    async def refresh(self, now: float) -> None:
        nc = self.next_close(now)
        due = now - self._last_list > 300 or nc is None or (nc - now < 200 and now - self._last_list > 60)
        if not due or now < self._retry_at:
            return
        try:
            out, cursor = [], None
            for _ in range(5):
                params = {"series_ticker": self.series, "limit": 1000, "min_close_ts": int(now),
                          "max_close_ts": int(now) + 3700}
                if cursor:
                    params["cursor"] = cursor
                d = await self.rest.get("/markets", params, auth=self.authed)
                out += d.get("markets", [])
                cursor = d.get("cursor")
                if not cursor:
                    break
        except Exception as e:  # noqa: BLE001
            logger.warning("hourly ladder list failed: {}", e)
            self._retry_at = now + 5.0
            return
        self._last_list = now
        fresh = {}
        for m in out:
            c = Contract.from_api(m)
            c.decimals, c.asset = self.decimals, self.asset
            if c.strike is not None and c.close_ts > now:
                fresh[c.ticker] = c
        self.contracts = fresh

    def near_money(self, now: float, x: float, band: float, window_s: float) -> list[Contract]:
        """Strikes of the next hourly close within ``band`` dollars of the index,
        once that close is less than ``window_s`` seconds away."""
        nc = self.next_close(now)
        if nc is None or nc - now > window_s:
            return []
        cs = [c for c in self.contracts.values() if c.close_ts == nc and abs(c.strike - x) <= band]
        return sorted(cs, key=lambda c: abs(c.strike - x))
