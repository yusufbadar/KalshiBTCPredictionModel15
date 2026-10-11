"""
Coinbase data feed – BTC-USD spot price, candles, and 24h stats.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import httpx
from loguru import logger

from kalshi_bot.config import COINBASE_PRODUCT


class CoinbaseFeed:
    BASE_URL = "https://api.exchange.coinbase.com"

    def __init__(self, product: str = COINBASE_PRODUCT):
        self.product = product
        self._http: Optional[httpx.AsyncClient] = None
        self.last_price: Optional[Decimal] = None

    async def connect(self) -> bool:
        try:
            self._http = httpx.AsyncClient(
                base_url=self.BASE_URL, timeout=15.0,
                headers={"Accept": "application/json"},
            )
            r = await self._http.get(f"/products/{self.product}")
            r.raise_for_status()
            logger.info("Coinbase feed connected ({})", self.product)
            return True
        except Exception as e:
            logger.error("Coinbase connect failed: {}", e)
            return False

    async def close(self):
        if self._http:
            await self._http.aclose()

    async def get_price(self) -> Optional[Decimal]:
        try:
            r = await self._http.get(f"/products/{self.product}/ticker")
            r.raise_for_status()
            price = Decimal(r.json()["price"])
            self.last_price = price
            return price
        except Exception as e:
            logger.warning("Coinbase price fetch failed: {}", e)
            return self.last_price

    async def get_candles(self, granularity: int = 60, limit: int = 100) -> list[dict]:
        """Return OHLCV candles. *granularity* in seconds (60=1m, 300=5m, 900=15m)."""
        try:
            r = await self._http.get(
                f"/products/{self.product}/candles",
                params={"granularity": granularity},
            )
            r.raise_for_status()
            out = []
            for c in r.json()[:limit]:
                out.append({
                    "ts": datetime.fromtimestamp(c[0], tz=timezone.utc),
                    "open": float(c[3]), "high": float(c[2]),
                    "low": float(c[1]), "close": float(c[4]),
                    "volume": float(c[5]),
                })
            out.sort(key=lambda x: x["ts"])
            return out
        except Exception as e:
            logger.warning("Coinbase candles failed: {}", e)
            return []

    async def get_24h_stats(self) -> Optional[dict]:
        try:
            r = await self._http.get(f"/products/{self.product}/stats")
            r.raise_for_status()
            d = r.json()
            return {
                "open": float(d["open"]), "high": float(d["high"]),
                "low": float(d["low"]), "volume": float(d["volume"]),
                "last": float(d["last"]),
            }
        except Exception as e:
            logger.warning("Coinbase 24h stats failed: {}", e)
            return None
