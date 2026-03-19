"""
Exchange order book analysis – detects bid/ask WALLS on Binance/Coinbase.

This is a core ascetic0x signal:
  - A large BID wall below price = support floor → favors UP
  - A large ASK wall above price = resistance ceiling → favors DOWN
  - No significant walls = neutral

We pull the order book from Binance (deeper, more liquid) and compute:
  - bid_wall_strength: how much larger the biggest bid cluster is vs average
  - ask_wall_strength: how much larger the biggest ask cluster is vs average
  - wall_imbalance: bid_wall_strength - ask_wall_strength (positive = bullish)
  - support_distance_pct: how far the bid wall is from current price
  - resistance_distance_pct: how far the ask wall is from current price
"""
from __future__ import annotations

import time
from typing import Optional

import httpx
from loguru import logger

CACHE_TTL = 30  # refresh every 30 seconds


class ExchangeOrderbookFeed:
    """Detects bid/ask walls on Binance BTC-USDT order book."""

    def __init__(self, wall_threshold: float = 3.0):
        """
        Args:
            wall_threshold: a level is a "wall" if its size >= threshold * average.
        """
        self.wall_threshold = wall_threshold
        self._http: Optional[httpx.AsyncClient] = None
        self._cache: Optional[dict] = None
        self._cache_ts: float = 0

    async def connect(self) -> bool:
        self._http = httpx.AsyncClient(timeout=10.0)
        data = await self.fetch()
        if data:
            logger.info("Exchange orderbook feed connected")
            return True
        logger.warning("Exchange orderbook feed: connect failed (non-fatal)")
        return True

    async def close(self):
        if self._http:
            await self._http.aclose()

    async def fetch(self, force: bool = False) -> Optional[dict]:
        if not force and self._cache and (time.time() - self._cache_ts < CACHE_TTL):
            return self._cache

        # Try Binance.US first, then global
        for base in ["https://api.binance.us", "https://api.binance.com"]:
            try:
                r = await self._http.get(
                    f"{base}/api/v3/depth",
                    params={"symbol": "BTCUSDT", "limit": 100},
                )
                r.raise_for_status()
                data = r.json()
                result = self._analyze(data)
                if result:
                    self._cache = result
                    self._cache_ts = time.time()
                return result
            except Exception as e:
                logger.debug("Exchange OB {} failed: {}", base, e)

        return self._cache

    def _analyze(self, data: dict) -> Optional[dict]:
        bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
        asks = [(float(p), float(q)) for p, q in data.get("asks", [])]

        if not bids or not asks:
            return None

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2

        # Aggregate into price clusters ($50 buckets)
        bucket = 50.0
        bid_clusters = self._cluster(bids, bucket)
        ask_clusters = self._cluster(asks, bucket)

        # Find walls
        bid_sizes = [s for _, s in bid_clusters]
        ask_sizes = [s for _, s in ask_clusters]

        avg_bid = sum(bid_sizes) / len(bid_sizes) if bid_sizes else 1
        avg_ask = sum(ask_sizes) / len(ask_sizes) if ask_sizes else 1

        # Biggest bid/ask cluster
        max_bid_cluster = max(bid_clusters, key=lambda x: x[1]) if bid_clusters else (mid, 0)
        max_ask_cluster = max(ask_clusters, key=lambda x: x[1]) if ask_clusters else (mid, 0)

        bid_wall_strength = max_bid_cluster[1] / avg_bid if avg_bid > 0 else 0
        ask_wall_strength = max_ask_cluster[1] / avg_ask if avg_ask > 0 else 0

        has_bid_wall = bid_wall_strength >= self.wall_threshold
        has_ask_wall = ask_wall_strength >= self.wall_threshold

        support_dist = (mid - max_bid_cluster[0]) / mid if mid > 0 else 0
        resist_dist = (max_ask_cluster[0] - mid) / mid if mid > 0 else 0

        wall_imbalance = bid_wall_strength - ask_wall_strength

        total_bid_vol = sum(bid_sizes)
        total_ask_vol = sum(ask_sizes)
        volume_imbalance = (
            (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
            if (total_bid_vol + total_ask_vol) > 0 else 0
        )

        return {
            "bid_wall_strength": round(bid_wall_strength, 2),
            "ask_wall_strength": round(ask_wall_strength, 2),
            "has_bid_wall": has_bid_wall,
            "has_ask_wall": has_ask_wall,
            "wall_imbalance": round(wall_imbalance, 2),
            "support_distance_pct": round(support_dist * 100, 3),
            "resistance_distance_pct": round(resist_dist * 100, 3),
            "volume_imbalance": round(volume_imbalance, 4),
            "bid_wall_price": max_bid_cluster[0],
            "ask_wall_price": max_ask_cluster[0],
            "mid_price": mid,
        }

    @staticmethod
    def _cluster(levels: list[tuple[float, float]], bucket: float) -> list[tuple[float, float]]:
        """Group order book levels into price buckets and sum quantities."""
        if not levels:
            return []
        clusters: dict[float, float] = {}
        for price, qty in levels:
            key = round(price / bucket) * bucket
            clusters[key] = clusters.get(key, 0) + qty
        return sorted(clusters.items(), key=lambda x: x[0])
