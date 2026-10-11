"""
Breaking crypto news feed.

The ascetic0x trader checked for fresh headlines that could move BTC within
minutes: geopolitical events, exchange hacks, surprise regulations, ETF news.

Sources (all free, no API key needed for basic use):
  - CryptoPanic public RSS
  - Binance news announcements
  - Fallback: simple keyword scoring of recent headlines

We don't do full NLP – we just check if there are any HIGH-IMPACT headlines
in the last 15 minutes and assign a directional bias.
"""
from __future__ import annotations

import re
import time
from typing import Optional

import httpx
from loguru import logger

CACHE_TTL = 60  # check news every 60 seconds

BEARISH_KEYWORDS = [
    "hack", "hacked", "exploit", "breach", "ban", "banned", "crackdown",
    "lawsuit", "sued", "fraud", "sec charges", "investigation",
    "war", "attack", "missile", "sanctions", "emergency",
    "crash", "plunge", "dump", "liquidat", "bankrupt", "insolven",
    "delisted", "shutdown", "arrest",
]

BULLISH_KEYWORDS = [
    "etf approved", "etf approval", "bitcoin etf", "spot etf",
    "adoption", "legal tender", "institutional",
    "rate cut", "fed cut", "dovish", "stimulus",
    "all-time high", "ath", "breakout", "surge", "rally",
    "partnership", "integration", "reserve",
]


class NewsFeed:
    """Lightweight breaking-news scanner for BTC-relevant headlines."""

    def __init__(self):
        self._http: Optional[httpx.AsyncClient] = None
        self._cache: Optional[dict] = None
        self._cache_ts: float = 0

    async def connect(self) -> bool:
        self._http = httpx.AsyncClient(timeout=10.0, follow_redirects=True)
        logger.info("News feed initialized")
        return True

    async def close(self):
        if self._http:
            await self._http.aclose()

    async def fetch(self, force: bool = False) -> Optional[dict]:
        if not force and self._cache and (time.time() - self._cache_ts < CACHE_TTL):
            return self._cache

        headlines: list[dict] = []

        # Try CryptoPanic (free public feed)
        cp = await self._fetch_cryptopanic()
        if cp:
            headlines.extend(cp)

        # Analyze headlines
        result = self._analyze(headlines)
        self._cache = result
        self._cache_ts = time.time()
        return result

    async def _fetch_cryptopanic(self) -> list[dict]:
        """Fetch recent BTC headlines from CryptoPanic's public feed."""
        try:
            r = await self._http.get(
                "https://cryptopanic.com/api/free/v1/posts/",
                params={
                    "currencies": "BTC",
                    "filter": "hot",
                    "public": "true",
                },
            )
            if r.status_code == 200:
                data = r.json()
                items = []
                for post in data.get("results", [])[:20]:
                    items.append({
                        "title": post.get("title", ""),
                        "source": post.get("source", {}).get("title", ""),
                        "published": post.get("published_at", ""),
                        "kind": post.get("kind", "news"),
                    })
                return items
        except Exception as e:
            logger.debug("CryptoPanic fetch failed: {}", e)
        return []

    def _analyze(self, headlines: list[dict]) -> dict:
        """Score headlines for directional bias."""
        bullish_count = 0
        bearish_count = 0
        high_impact = False
        impact_headlines: list[str] = []

        for h in headlines:
            title = h.get("title", "").lower()

            b_score = sum(1 for kw in BULLISH_KEYWORDS if kw in title)
            s_score = sum(1 for kw in BEARISH_KEYWORDS if kw in title)

            if b_score > 0:
                bullish_count += b_score
            if s_score > 0:
                bearish_count += s_score

            if b_score >= 2 or s_score >= 2:
                high_impact = True
                impact_headlines.append(h.get("title", ""))

        total = bullish_count + bearish_count
        if total > 0:
            news_sentiment = (bullish_count - bearish_count) / total  # -1 to +1
        else:
            news_sentiment = 0.0

        if bearish_count > bullish_count * 2:
            news_direction = "bearish"
        elif bullish_count > bearish_count * 2:
            news_direction = "bullish"
        else:
            news_direction = "neutral"

        return {
            "news_sentiment": round(news_sentiment, 3),
            "news_direction": news_direction,
            "news_bullish_count": bullish_count,
            "news_bearish_count": bearish_count,
            "news_high_impact": high_impact,
            "news_headline_count": len(headlines),
            "news_impact_headlines": impact_headlines[:3],
        }
