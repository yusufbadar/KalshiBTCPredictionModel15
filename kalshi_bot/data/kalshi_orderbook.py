"""
Kalshi orderbook data feed for KXBTC15M markets.

Pulls the YES / NO bid arrays and computes imbalance, spread, and depth
features that feed into the ML model.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from kalshi_bot.kalshi.client import KalshiClient


class KalshiOrderbookFeed:
    def __init__(self, client: KalshiClient):
        self.client = client

    def fetch(self, ticker: str) -> Optional[dict]:
        """Fetch orderbook and compute derived features.

        Returns dict with:
            yes_bids, no_bids       – raw arrays
            best_yes_bid, best_no_bid
            best_yes_ask (implied), best_no_ask (implied)
            spread, mid_price
            yes_depth, no_depth     – total contracts on each side
            imbalance               – (yes_depth - no_depth) / total
        """
        try:
            raw = self.client.get_market_orderbook(ticker)
            ob = raw.get("orderbook_fp") or raw.get("orderbook", {})

            yes_bids = ob.get("yes_dollars", ob.get("yes", []))
            no_bids = ob.get("no_dollars", ob.get("no", []))

            best_yes_bid = float(yes_bids[-1][0]) if yes_bids else 0.0
            best_no_bid = float(no_bids[-1][0]) if no_bids else 0.0

            best_yes_ask = 1.0 - best_no_bid if best_no_bid else 1.0
            best_no_ask = 1.0 - best_yes_bid if best_yes_bid else 1.0

            spread = best_yes_ask - best_yes_bid if best_yes_bid else 1.0
            mid_price = (best_yes_bid + best_yes_ask) / 2 if best_yes_bid else 0.5

            yes_depth = sum(float(b[1]) for b in yes_bids) if yes_bids else 0.0
            no_depth = sum(float(b[1]) for b in no_bids) if no_bids else 0.0
            total_depth = yes_depth + no_depth
            imbalance = (yes_depth - no_depth) / total_depth if total_depth > 0 else 0.0

            return {
                "best_yes_bid": best_yes_bid,
                "best_no_bid": best_no_bid,
                "best_yes_ask": best_yes_ask,
                "best_no_ask": best_no_ask,
                "spread": round(spread, 4),
                "mid_price": round(mid_price, 4),
                "yes_depth": yes_depth,
                "no_depth": no_depth,
                "imbalance": round(imbalance, 4),
                "yes_levels": len(yes_bids),
                "no_levels": len(no_bids),
            }

        except Exception as e:
            logger.warning("Kalshi orderbook fetch failed for {}: {}", ticker, e)
            return None
