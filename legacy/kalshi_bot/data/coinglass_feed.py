"""
Coinglass-style funding rate + liquidation data.

These are the two primary signals the ascetic0x trader used to turn $12 → $100K:

  FUNDING RATE:
    Positive → longs pay shorts → too many longs → crowded, fragile, bearish signal
    Negative → shorts pay longs → too many shorts → crowded, fragile, bullish signal
    Extreme values (top/bottom 20% of 30-day range) are actionable.

  LIQUIDATIONS:
    High short liquidations → shorts already squeezed → upside fuel exhausted → bearish
    High long liquidations  → longs already flushed  → selling pressure done  → bullish
    We compare the ratio of long-vs-short liquidations over 24h.

Data sources:
  - Binance futures API (free, no key) for funding rates
  - Coinglass open API for aggregated liquidation data
  - Fallback: Binance long/short ratio as proxy for liquidation pressure
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional

import httpx
from loguru import logger

CACHE_TTL = 120  # refresh every 2 minutes

# Binance futures endpoints (US-accessible via futures.binance.com proxy sites)
BINANCE_FAPI = "https://fapi.binance.com"
COINGLASS_API = "https://open-api.coinglass.com/public/v2"


class FundingLiquidationFeed:
    """Fetches funding rates and liquidation data for BTC."""

    def __init__(self):
        self._http: Optional[httpx.AsyncClient] = None
        self._cache: Optional[dict] = None
        self._cache_ts: float = 0
        self._funding_history: deque[float] = deque(maxlen=720)  # ~30 days at 1h

    async def connect(self) -> bool:
        self._http = httpx.AsyncClient(timeout=15.0)
        data = await self.fetch()
        if data:
            logger.info(
                "Funding/Liquidation feed connected (rate={:.4%})",
                data.get("funding_rate", 0),
            )
            return True
        logger.warning("Funding/Liquidation feed: partial connect (will retry)")
        return True  # non-fatal; we run without it if needed

    async def close(self):
        if self._http:
            await self._http.aclose()

    async def fetch(self, force: bool = False) -> Optional[dict]:
        if not force and self._cache and (time.time() - self._cache_ts < CACHE_TTL):
            return self._cache

        result = {}

        # 1) Funding rate from Binance Futures
        funding = await self._fetch_funding_rate()
        if funding is not None:
            result.update(funding)
            self._funding_history.append(funding["funding_rate"])

        # 2) Long/Short ratio from Binance Futures (proxy for liquidation pressure)
        ls_ratio = await self._fetch_long_short_ratio()
        if ls_ratio is not None:
            result.update(ls_ratio)

        # 3) Liquidation data from Coinglass (best effort)
        liqs = await self._fetch_liquidations()
        if liqs is not None:
            result.update(liqs)

        # 4) Compute derived signals
        if self._funding_history:
            hist = list(self._funding_history)
            result["funding_percentile"] = self._percentile_rank(
                hist, result.get("funding_rate", 0)
            )
        else:
            result["funding_percentile"] = 50.0

        result["funding_extreme"] = self._classify_funding(
            result.get("funding_rate", 0),
            result.get("funding_percentile", 50),
        )

        if result:
            self._cache = result
            self._cache_ts = time.time()

        return result if result else None

    # ── Binance Futures: funding rate ─────────────────────────────

    async def _fetch_funding_rate(self) -> Optional[dict]:
        """Current funding rate for BTCUSDT perpetual."""
        try:
            r = await self._http.get(
                f"{BINANCE_FAPI}/fapi/v1/premiumIndex",
                params={"symbol": "BTCUSDT"},
            )
            r.raise_for_status()
            d = r.json()
            rate = float(d.get("lastFundingRate", 0))
            mark = float(d.get("markPrice", 0))
            return {
                "funding_rate": rate,
                "mark_price": mark,
                "next_funding_ts": int(d.get("nextFundingTime", 0)) / 1000,
            }
        except Exception as e:
            logger.debug("Binance funding rate failed: {}", e)
            return None

    # ── Binance Futures: top-trader long/short ratio ──────────────

    async def _fetch_long_short_ratio(self) -> Optional[dict]:
        """Top trader long/short ratio – a proxy for how crowded each side is."""
        try:
            r = await self._http.get(
                f"{BINANCE_FAPI}/futures/data/topLongShortAccountRatio",
                params={"symbol": "BTCUSDT", "period": "15m", "limit": 1},
            )
            r.raise_for_status()
            data = r.json()
            if data:
                entry = data[0] if isinstance(data, list) else data
                long_pct = float(entry.get("longAccount", 0.5))
                short_pct = float(entry.get("shortAccount", 0.5))
                ratio = float(entry.get("longShortRatio", 1.0))
                return {
                    "long_account_pct": long_pct,
                    "short_account_pct": short_pct,
                    "long_short_ratio": ratio,
                }
        except Exception as e:
            logger.debug("Binance L/S ratio failed: {}", e)
        return None

    # ── Coinglass: aggregated liquidation data ────────────────────

    async def _fetch_liquidations(self) -> Optional[dict]:
        """Aggregated liquidation data from Coinglass (open/free endpoint)."""
        try:
            r = await self._http.get(
                f"{COINGLASS_API}/liquidation_chart",
                params={"symbol": "BTC", "time_type": 2},  # 24h
                headers={"accept": "application/json"},
            )
            if r.status_code == 200:
                body = r.json()
                data = body.get("data", {})
                long_liq = float(data.get("longVolUsd", 0))
                short_liq = float(data.get("shortVolUsd", 0))
                total = long_liq + short_liq
                liq_ratio = short_liq / long_liq if long_liq > 0 else 1.0
                return {
                    "liq_long_24h": long_liq,
                    "liq_short_24h": short_liq,
                    "liq_total_24h": total,
                    "liq_short_long_ratio": round(liq_ratio, 4),
                }
        except Exception as e:
            logger.debug("Coinglass liquidation fetch failed: {}", e)

        # Fallback: use Binance open interest change as rough proxy
        return await self._fetch_oi_change()

    async def _fetch_oi_change(self) -> Optional[dict]:
        """Binance open interest as fallback liquidation proxy."""
        try:
            r = await self._http.get(
                f"{BINANCE_FAPI}/futures/data/openInterestHist",
                params={"symbol": "BTCUSDT", "period": "1h", "limit": 24},
            )
            r.raise_for_status()
            data = r.json()
            if len(data) >= 2:
                oi_now = float(data[-1].get("sumOpenInterestValue", 0))
                oi_24h_ago = float(data[0].get("sumOpenInterestValue", 0))
                oi_change = (oi_now - oi_24h_ago) / oi_24h_ago if oi_24h_ago else 0
                return {
                    "oi_current": oi_now,
                    "oi_24h_change": round(oi_change, 4),
                    "liq_long_24h": 0,
                    "liq_short_24h": 0,
                    "liq_total_24h": 0,
                    "liq_short_long_ratio": 1.0,
                }
        except Exception as e:
            logger.debug("Binance OI fallback failed: {}", e)
        return None

    # ── derived signal helpers ────────────────────────────────────

    @staticmethod
    def _percentile_rank(history: list[float], value: float) -> float:
        if not history:
            return 50.0
        below = sum(1 for h in history if h < value)
        return 100.0 * below / len(history)

    @staticmethod
    def _classify_funding(rate: float, percentile: float) -> str:
        """Classify funding as a directional signal.

        Returns:
            'bullish'  – funding extremely negative (shorts overcrowded)
            'bearish'  – funding extremely positive (longs overcrowded)
            'neutral'  – funding is in normal range
        """
        if percentile >= 80 or rate > 0.0005:
            return "bearish"   # longs overcrowded
        if percentile <= 20 or rate < -0.0005:
            return "bullish"   # shorts overcrowded
        return "neutral"
