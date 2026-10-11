"""
Binance REST + WebSocket feed for real-time BTC-USDT data.

For the bot's main loop we use **REST klines** (simpler, no persistent WS needed).
A lightweight WS helper is included for optional real-time streaming.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Optional, Callable

import httpx
import websockets
from loguru import logger

from kalshi_bot.config import BINANCE_SYMBOL

BINANCE_ENDPOINTS = [
    "https://api.binance.us/api/v3",
    "https://api.binance.com/api/v3",
]
BINANCE_WS = "wss://stream.binance.us:9443/ws"


class BinanceFeed:
    """REST-first Binance data source (tries Binance.US first, then global)."""

    def __init__(self, symbol: str = BINANCE_SYMBOL):
        self.symbol = symbol.upper()
        self._http: Optional[httpx.AsyncClient] = None
        self._base: str = BINANCE_ENDPOINTS[0]
        self.last_price: Optional[float] = None

    async def connect(self) -> bool:
        self._http = httpx.AsyncClient(timeout=15.0)
        for base in BINANCE_ENDPOINTS:
            try:
                r = await self._http.get(
                    f"{base}/ticker/price", params={"symbol": self.symbol}
                )
                r.raise_for_status()
                self.last_price = float(r.json()["price"])
                self._base = base
                logger.info("Binance feed connected via {} ({})", base, self.symbol)
                return True
            except Exception as e:
                logger.warning("Binance {} failed: {}", base, e)
        logger.error("All Binance endpoints failed")
        return False

    async def close(self):
        if self._http:
            await self._http.aclose()

    async def get_price(self) -> Optional[float]:
        try:
            r = await self._http.get(
                f"{self._base}/ticker/price", params={"symbol": self.symbol}
            )
            r.raise_for_status()
            self.last_price = float(r.json()["price"])
            return self.last_price
        except Exception as e:
            logger.warning("Binance price failed: {}", e)
            return self.last_price

    async def get_klines(
        self, interval: str = "1m", limit: int = 100
    ) -> list[dict]:
        """Fetch klines (OHLCV). *interval*: 1m, 5m, 15m, 1h, …"""
        try:
            r = await self._http.get(
                f"{self._base}/klines",
                params={"symbol": self.symbol, "interval": interval, "limit": limit},
            )
            r.raise_for_status()
            out = []
            for k in r.json():
                out.append({
                    "ts": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                    "open": float(k[1]), "high": float(k[2]),
                    "low": float(k[3]), "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_ts": datetime.fromtimestamp(k[6] / 1000, tz=timezone.utc),
                    "trades": int(k[8]),
                })
            return out
        except Exception as e:
            logger.warning("Binance klines failed: {}", e)
            return []

    async def get_24h_ticker(self) -> Optional[dict]:
        try:
            r = await self._http.get(
                f"{self._base}/ticker/24hr", params={"symbol": self.symbol}
            )
            r.raise_for_status()
            d = r.json()
            return {
                "price_change_pct": float(d["priceChangePercent"]),
                "high": float(d["highPrice"]),
                "low": float(d["lowPrice"]),
                "volume": float(d["volume"]),
                "quote_volume": float(d["quoteVolume"]),
                "count": int(d["count"]),
            }
        except Exception as e:
            logger.warning("Binance 24h ticker failed: {}", e)
            return None


class BinanceWSStream:
    """Optional real-time websocket streamer."""

    def __init__(self, symbol: str = BINANCE_SYMBOL):
        self.symbol = symbol.lower()
        self._ws = None
        self._running = False
        self.last_price: Optional[float] = None

    async def stream_ticker(self, callback: Callable):
        url = f"{BINANCE_WS}/{self.symbol}@ticker"
        self._running = True
        try:
            async with websockets.connect(url) as ws:
                self._ws = ws
                while self._running:
                    msg = json.loads(await ws.recv())
                    self.last_price = float(msg["c"])
                    await callback({
                        "price": self.last_price,
                        "ts": datetime.fromtimestamp(msg["E"] / 1000, tz=timezone.utc),
                        "change_pct": float(msg["P"]),
                        "volume": float(msg["v"]),
                    })
        except Exception as e:
            logger.warning("Binance WS closed: {}", e)
        finally:
            self._running = False

    def stop(self):
        self._running = False
