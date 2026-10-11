"""
Simplified feature engineering for EV-based trading.

12 features focused on price action, volatility, and Kalshi market data.
Slow-moving macro indicators (Fear & Greed, Deribit PCR, news, funding,
liquidations) removed — they are too noisy for a 15-minute horizon.

Features
--------
1–4.  Log returns at 1m, 3m, 5m, 15m
5.    Acceleration  (is the recent move speeding up or fading?)
6.    VWAP z-score  (mean-reversion signal over 60-min lookback)
7.    Realized-vol ratio  (short-term vol / long-term vol)
8.    Kalshi midprice  (market-implied probability)
9.    Kalshi spread
10.   Kalshi order-book imbalance
11–12. Hour-of-day (sin / cos)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from loguru import logger

from kalshi_bot.data.data_aggregator import DataSnapshot

FEATURE_NAMES: list[str] = [
    "r1", "r3", "r5", "r15",
    "accel",
    "vwap_z",
    "rv_ratio",
    "kalshi_mid",
    "kalshi_spread",
    "kalshi_imbalance",
    "hour_sin", "hour_cos",
]


class FeatureEngineer:
    """Builds a compact feature vector from a DataSnapshot."""

    def __init__(self, max_history: int = 200):
        self._feature_names = list(FEATURE_NAMES)

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def record_outcome(self, went_up: bool):
        """No-op — kept for API compatibility with historical collector."""

    def build(self, snap: DataSnapshot) -> Optional[np.ndarray]:
        """Build a 12-element feature vector.  Returns None on failure."""
        try:
            closes = [c["close"] for c in snap.candles_1m] if snap.candles_1m else []
            volumes = [c["volume"] for c in snap.candles_1m] if snap.candles_1m else []
            highs = [c["high"] for c in snap.candles_1m] if snap.candles_1m else []
            lows = [c["low"] for c in snap.candles_1m] if snap.candles_1m else []

            if len(closes) < 15:
                return None

            feats: list[float] = []

            r1 = _log_ret(closes, 1)
            r3 = _log_ret(closes, 3)
            r5 = _log_ret(closes, 5)
            r15 = _log_ret(closes, 15)

            feats.append(r1)
            feats.append(r3)
            feats.append(r5)
            feats.append(r15)

            feats.append(r1 - r5 / 5.0)

            feats.append(_vwap_z(closes, volumes, highs, lows))

            feats.append(_rv_ratio(closes))

            ob = snap.kalshi_ob or {}
            feats.append(ob.get("mid_price", 0.5))
            feats.append(ob.get("spread", 0.1))
            feats.append(ob.get("imbalance", 0.0))

            if snap.ts:
                hour = snap.ts.hour + snap.ts.minute / 60.0
            else:
                hour = 12.0
            feats.append(math.sin(2 * math.pi * hour / 24))
            feats.append(math.cos(2 * math.pi * hour / 24))

            return np.array(feats, dtype=np.float32)

        except Exception as e:
            logger.error("Feature build failed: {}", e)
            return None


# ── pure helper functions ───────────────────────────────────────────────


def _log_ret(closes: list[float], periods: int) -> float:
    if len(closes) < periods + 1 or closes[-1 - periods] <= 0:
        return 0.0
    return math.log(closes[-1] / closes[-1 - periods])


def _vwap_z(closes: list, volumes: list, highs: list, lows: list) -> float:
    """Z-score of current price relative to 60-min VWAP."""
    n = min(len(closes), len(volumes), len(highs), len(lows))
    if n < 15:
        return 0.0
    window = min(n, 60)
    cum_vol = 0.0
    cum_vp = 0.0
    for i in range(-window, 0):
        typical = (highs[i] + lows[i] + closes[i]) / 3
        v = max(volumes[i], 1e-9)
        cum_vol += v
        cum_vp += typical * v
    if cum_vol == 0:
        return 0.0
    vwap = cum_vp / cum_vol
    dev_pct = (closes[-1] - vwap) / vwap
    prices = closes[-window:]
    mean_p = sum(prices) / len(prices)
    std_p = (sum((p - mean_p) ** 2 for p in prices) / len(prices)) ** 0.5
    std_pct = std_p / mean_p if mean_p > 0 else 1e-9
    return dev_pct / std_pct if std_pct > 0 else 0.0


def _rv_ratio(closes: list[float]) -> float:
    """Ratio of 5-min realized vol to 60-min realized vol."""
    if len(closes) < 60:
        return 1.0
    short_rets = [
        math.log(closes[i] / closes[i - 1])
        for i in range(-5, 0)
        if closes[i - 1] > 0
    ]
    long_rets = [
        math.log(closes[i] / closes[i - 1])
        for i in range(-60, 0)
        if closes[i - 1] > 0
    ]
    if not short_rets or not long_rets:
        return 1.0
    short_std = (sum(r ** 2 for r in short_rets) / len(short_rets)) ** 0.5
    long_std = (sum(r ** 2 for r in long_rets) / len(long_rets)) ** 0.5
    return short_std / long_std if long_std > 0 else 1.0
