"""
Discover and track KXBTC15M (BTC Up/Down 15-minute) markets on Kalshi.

The series ticker is ``KXBTC15M``.  Individual market tickers look like
``KXBTC15M-26MAR1915`` (= the 19:15 UTC window on 26 Mar 2026).

This module finds the *current* open market, determines when it closes,
and calculates optimal entry timing.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from kalshi_bot.kalshi.client import KalshiClient
from kalshi_bot.config import KALSHI_SERIES_TICKER


class MarketInfo:
    """Lightweight container for a single KXBTC15M market."""

    def __init__(self, raw: dict):
        self.raw = raw
        self.ticker: str = raw["ticker"]
        self.title: str = raw.get("title", "")
        self.status: str = raw.get("status", "unknown")
        self.yes_bid: float = raw.get("yes_bid", 0) / 100 if raw.get("yes_bid") else 0
        self.yes_ask: float = raw.get("yes_ask", 0) / 100 if raw.get("yes_ask") else 0
        self.no_bid: float = raw.get("no_bid", 0) / 100 if raw.get("no_bid") else 0
        self.no_ask: float = raw.get("no_ask", 0) / 100 if raw.get("no_ask") else 0
        self.volume: int = raw.get("volume", 0)
        self.open_interest: int = raw.get("open_interest", 0)

        self.open_time: Optional[datetime] = _parse_ts(raw.get("open_time"))
        self.close_time: Optional[datetime] = _parse_ts(raw.get("close_time"))
        self.expiration_time: Optional[datetime] = _parse_ts(raw.get("expiration_time"))

    @property
    def seconds_until_close(self) -> float:
        if not self.close_time:
            return float("inf")
        return (self.close_time - datetime.now(timezone.utc)).total_seconds()

    @property
    def is_open(self) -> bool:
        return self.status in ("open", "active")

    @property
    def mid_price(self) -> float:
        if self.yes_bid and self.yes_ask:
            return (self.yes_bid + self.yes_ask) / 2
        return self.yes_bid or self.yes_ask or 0.5

    def __repr__(self):
        return (
            f"<Market {self.ticker}  status={self.status}  "
            f"yes_bid={self.yes_bid:.2f}  close_in={self.seconds_until_close:.0f}s>"
        )


def _parse_ts(val) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return datetime.fromtimestamp(val, tz=timezone.utc)
    try:
        s = str(val).replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


class MarketDiscovery:
    """Finds and tracks KXBTC15M markets."""

    def __init__(self, client: KalshiClient):
        self.client = client
        self._cache: list[MarketInfo] = []
        self._last_fetch: float = 0

    def fetch_open_markets(self, force: bool = False) -> list[MarketInfo]:
        """Query Kalshi for open KXBTC15M markets.

        Results are cached for 30 s unless *force* is ``True``.
        """
        if not force and (time.time() - self._last_fetch < 30) and self._cache:
            return self._cache

        try:
            resp = self.client.get_markets(
                series_ticker=KALSHI_SERIES_TICKER,
                status="open",
                limit=40,
            )
            markets_raw = resp.get("markets", [])
            markets = [MarketInfo(m) for m in markets_raw]
            markets.sort(key=lambda m: m.close_time or datetime.max.replace(tzinfo=timezone.utc))
            self._cache = markets
            self._last_fetch = time.time()
            logger.info("Fetched {} open KXBTC15M markets", len(markets))
            return markets
        except Exception as e:
            logger.error("Failed to fetch markets: {}", e)
            return self._cache

    def get_current_market(self) -> Optional[MarketInfo]:
        """Return the market that is closest to closing (next to settle)."""
        markets = self.fetch_open_markets()
        now = datetime.now(timezone.utc)
        for m in markets:
            if m.is_open and m.close_time and m.close_time > now:
                return m
        return None

    def get_next_market(self) -> Optional[MarketInfo]:
        """Return the market that will close *after* the current one."""
        markets = self.fetch_open_markets()
        now = datetime.now(timezone.utc)
        future = [m for m in markets if m.is_open and m.close_time and m.close_time > now]
        if len(future) >= 2:
            return future[1]
        return None
