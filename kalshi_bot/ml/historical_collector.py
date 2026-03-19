"""
Bootstrap the ML model by collecting historical 15-minute BTC returns from
Binance and generating labelled training data.

The label is simple:
  1 = price went UP in that 15-min window
  0 = price went DOWN
"""
from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone

import numpy as np
from loguru import logger

from kalshi_bot.data.binance_feed import BinanceFeed
from kalshi_bot.ml.feature_engineer import FeatureEngineer


async def collect_historical_features(
    n_candles_15m: int = 1000,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fetch historical candles and build feature/label arrays.

    Tries Binance first; falls back to Coinbase if Binance is unavailable.
    This is a simplified bootstrap: we only have price data (no orderbook,
    no sentiment) so the corresponding features are filled with defaults.
    The model will retrain once live data accumulates.

    Returns (X, y, feature_names).
    """
    from kalshi_bot.data.coinbase_feed import CoinbaseFeed

    candles_15m: list[dict] = []
    candles_1m: list[dict] = []
    candles_5m: list[dict] = []

    # Try Binance first
    feed = BinanceFeed()
    if await feed.connect():
        candles_15m = await feed.get_klines("15m", limit=min(n_candles_15m, 1000))
        candles_1m = await feed.get_klines("1m", limit=1000)
        candles_5m = await feed.get_klines("5m", limit=500)
        await feed.close()

    # Fallback to Coinbase
    if len(candles_15m) < 50:
        logger.info("Binance unavailable – falling back to Coinbase for historical data")
        cb = CoinbaseFeed()
        if await cb.connect():
            raw_15 = await cb.get_candles(granularity=900, limit=min(n_candles_15m, 300))
            raw_5 = await cb.get_candles(granularity=300, limit=500)
            raw_1 = await cb.get_candles(granularity=60, limit=300)
            await cb.close()
            candles_15m = raw_15
            candles_5m = raw_5
            candles_1m = raw_1

    if len(candles_15m) < 50:
        logger.error("Not enough historical candles ({})", len(candles_15m))
        return np.array([]), np.array([]), []

    from kalshi_bot.data.data_aggregator import DataSnapshot

    eng = FeatureEngineer()
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    # Walk through 15m candles; for each one, build features from the
    # 1m candles that precede it and label from its close vs open.
    for i in range(30, len(candles_15m) - 1):
        c = candles_15m[i]
        ts = c["ts"]

        sub_1m = [k for k in candles_1m if k["ts"] <= ts][-100:]
        sub_5m = [k for k in candles_5m if k["ts"] <= ts][-50:]

        snap = DataSnapshot(
            ts=ts,
            btc_price=c["close"],
            candles_1m=sub_1m,
            candles_5m=sub_5m,
            candles_15m=candles_15m[max(0, i - 30):i + 1],
            stats_24h=None,
            binance_ticker=None,
            fear_greed=None,
            deribit_pcr=None,
            kalshi_ob=None,
            kalshi_mid_price=None,
        )

        features = eng.build(snap)
        if features is None:
            continue

        next_candle = candles_15m[i + 1]
        went_up = next_candle["close"] > next_candle["open"]
        eng.record_outcome(went_up)

        X_list.append(features)
        y_list.append(1 if went_up else 0)

    X = np.vstack(X_list) if X_list else np.array([])
    y = np.array(y_list)
    logger.info("Historical dataset: {} samples, {:.1%} up", len(y), np.mean(y) if len(y) else 0)
    return X, y, eng.feature_names
