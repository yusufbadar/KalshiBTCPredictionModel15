"""
Crypto Fear & Greed Index from alternative.me (free, no API key).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import httpx
from loguru import logger

from kalshi_bot.config import FEAR_GREED_URL


class FearGreedFeed:
    def __init__(self):
        self._http: Optional[httpx.AsyncClient] = None
        self.last_value: Optional[int] = None
        self.last_classification: Optional[str] = None

    async def connect(self) -> bool:
        try:
            self._http = httpx.AsyncClient(timeout=15.0)
            data = await self.fetch()
            logger.info("Fear & Greed feed connected (value={})", data.get("value"))
            return data is not None
        except Exception as e:
            logger.error("Fear & Greed connect failed: {}", e)
            return False

    async def close(self):
        if self._http:
            await self._http.aclose()

    async def fetch(self) -> Optional[dict]:
        try:
            r = await self._http.get(FEAR_GREED_URL)
            r.raise_for_status()
            entry = r.json()["data"][0]
            self.last_value = int(entry["value"])
            self.last_classification = entry["value_classification"]
            return {
                "value": self.last_value,
                "classification": self.last_classification,
                "ts": datetime.fromtimestamp(int(entry["timestamp"]), tz=timezone.utc),
            }
        except Exception as e:
            logger.warning("Fear & Greed fetch failed: {}", e)
            if self.last_value is not None:
                return {"value": self.last_value, "classification": self.last_classification, "ts": None}
            return None
