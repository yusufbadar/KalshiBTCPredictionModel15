"""
Feature engineering pipeline.

Transforms a ``DataSnapshot`` into a flat numpy feature vector consumed by the
XGBoost classifier.  All features are designed for 15-minute BTC up/down
prediction.

Feature groups
--------------
1. Price returns & momentum   (8 features)
2. Technical indicators       (10 features)
3. Volatility                 (4 features)
4. Kalshi orderbook           (6 features)
5. Sentiment / macro          (4 features)
6. Time / cyclical            (4 features)
7. Lagged outcomes            (4 features)
8. Funding rate (ascetic0x)   (4 features)
9. Liquidations (ascetic0x)   (4 features)
10. Exchange OB walls          (4 features)
11. News sentiment             (3 features)
                              -----------
                              ~55 features total
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
from loguru import logger

from kalshi_bot.data.data_aggregator import DataSnapshot

FEATURE_NAMES: list[str] = []


def _safe(val, default=0.0):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return float(val)


class FeatureEngineer:
    """Stateful feature builder – keeps a rolling buffer for lagged features."""

    def __init__(self, max_history: int = 200):
        self._outcomes: deque[int] = deque(maxlen=max_history)   # 1=up, 0=down
        self._prices: deque[float] = deque(maxlen=max_history)
        self._feature_names: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def record_outcome(self, went_up: bool):
        """Record the outcome of the last 15-min window (call after settlement)."""
        self._outcomes.append(1 if went_up else 0)

    def build(self, snap: DataSnapshot) -> Optional[np.ndarray]:
        """Build a feature vector from *snap*.  Returns ``None`` on failure."""
        try:
            feats: list[float] = []
            names: list[str] = []

            closes_1m = [c["close"] for c in snap.candles_1m] if snap.candles_1m else []
            closes_5m = [c["close"] for c in snap.candles_5m] if snap.candles_5m else []
            closes_15m = [c["close"] for c in snap.candles_15m] if snap.candles_15m else []
            volumes_1m = [c["volume"] for c in snap.candles_1m] if snap.candles_1m else []
            highs_1m = [c["high"] for c in snap.candles_1m] if snap.candles_1m else []
            lows_1m = [c["low"] for c in snap.candles_1m] if snap.candles_1m else []

            if snap.btc_price:
                self._prices.append(snap.btc_price)

            # ── 1. Price returns & momentum ───────────────────────────
            feats.append(_ret(closes_1m, 1));    names.append("ret_1m")
            feats.append(_ret(closes_1m, 5));    names.append("ret_5m")
            feats.append(_ret(closes_1m, 15));   names.append("ret_15m")
            feats.append(_ret(closes_1m, 30));   names.append("ret_30m")
            feats.append(_ret(closes_1m, 60));   names.append("ret_60m")
            feats.append(_log_ret(closes_1m, 1));  names.append("logret_1m")
            feats.append(_log_ret(closes_1m, 15)); names.append("logret_15m")
            feats.append(_momentum(closes_1m, 14)); names.append("momentum_14")

            # ── 2. Technical indicators ───────────────────────────────
            feats.append(_rsi(closes_1m, 14));     names.append("rsi_14")
            feats.append(_rsi(closes_1m, 7));      names.append("rsi_7")
            macd_line, signal_line = _macd(closes_1m)
            feats.append(macd_line);               names.append("macd_line")
            feats.append(signal_line);             names.append("macd_signal")
            feats.append(macd_line - signal_line); names.append("macd_hist")
            feats.append(_bb_pct_b(closes_1m, 20)); names.append("bb_pctb_20")
            feats.append(_ema_cross(closes_1m, 9, 21)); names.append("ema_cross_9_21")
            feats.append(_ema_cross(closes_1m, 5, 13)); names.append("ema_cross_5_13")
            feats.append(_stoch_k(highs_1m, lows_1m, closes_1m, 14)); names.append("stoch_k")
            feats.append(_williams_r(highs_1m, lows_1m, closes_1m, 14)); names.append("williams_r")

            # ── 3. Volatility ─────────────────────────────────────────
            feats.append(_rolling_std(closes_1m, 14));  names.append("vol_14")
            feats.append(_rolling_std(closes_1m, 30));  names.append("vol_30")
            feats.append(_atr(highs_1m, lows_1m, closes_1m, 14)); names.append("atr_14")
            feats.append(_volume_ratio(volumes_1m, 14)); names.append("vol_ratio_14")

            # ── 4. Kalshi orderbook ───────────────────────────────────
            ob = snap.kalshi_ob or {}
            feats.append(_safe(ob.get("imbalance")));    names.append("ob_imbalance")
            feats.append(_safe(ob.get("spread")));       names.append("ob_spread")
            feats.append(_safe(ob.get("mid_price"), 0.5)); names.append("ob_mid_price")
            feats.append(_safe(ob.get("yes_depth")));    names.append("ob_yes_depth")
            feats.append(_safe(ob.get("no_depth")));     names.append("ob_no_depth")
            total_d = _safe(ob.get("yes_depth")) + _safe(ob.get("no_depth"))
            feats.append(total_d); names.append("ob_total_depth")

            # ── 5. Sentiment / macro ──────────────────────────────────
            fg = snap.fear_greed or {}
            feats.append(_safe(fg.get("value"), 50) / 100.0); names.append("fear_greed_norm")
            pcr = snap.deribit_pcr or {}
            feats.append(_safe(pcr.get("short_pcr"), 1.0));   names.append("deribit_short_pcr")
            feats.append(_safe(pcr.get("overall_pcr"), 1.0)); names.append("deribit_overall_pcr")
            bt = snap.binance_ticker or {}
            feats.append(_safe(bt.get("price_change_pct")));  names.append("binance_24h_chg_pct")

            # ── 6. Time / cyclical ────────────────────────────────────
            if snap.ts:
                hour = snap.ts.hour + snap.ts.minute / 60.0
                dow = snap.ts.weekday()
            else:
                hour, dow = 12.0, 2
            feats.append(math.sin(2 * math.pi * hour / 24)); names.append("hour_sin")
            feats.append(math.cos(2 * math.pi * hour / 24)); names.append("hour_cos")
            feats.append(math.sin(2 * math.pi * dow / 7));   names.append("dow_sin")
            feats.append(math.cos(2 * math.pi * dow / 7));   names.append("dow_cos")

            # ── 7. Lagged outcomes ────────────────────────────────────
            oc = list(self._outcomes)
            feats.append(oc[-1] if oc else 0.5);             names.append("prev_outcome")
            feats.append(float(np.mean(oc[-3:])) if len(oc) >= 3 else 0.5)
            names.append("win_rate_last3")
            feats.append(float(np.mean(oc[-10:])) if len(oc) >= 10 else 0.5)
            names.append("win_rate_last10")
            feats.append(_streak(oc)); names.append("streak")

            # ── 8. Funding rate (ascetic0x) ────────────────────────────
            fl = snap.funding_liq or {}
            feats.append(_safe(fl.get("funding_rate"), 0) * 10000)  # scale to basis points
            names.append("funding_rate_bps")
            feats.append(_safe(fl.get("funding_percentile"), 50) / 100.0)
            names.append("funding_percentile")
            extreme = fl.get("funding_extreme", "neutral")
            feats.append(1.0 if extreme == "bearish" else (-1.0 if extreme == "bullish" else 0.0))
            names.append("funding_signal")
            feats.append(_safe(fl.get("long_short_ratio"), 1.0))
            names.append("long_short_ratio")

            # ── 9. Liquidation data (ascetic0x) ────────────────────────
            feats.append(_safe(fl.get("liq_short_long_ratio"), 1.0))
            names.append("liq_short_long_ratio")
            liq_total = _safe(fl.get("liq_total_24h"), 0)
            feats.append(min(liq_total / 1e9, 5.0))  # normalize: $1B = 1.0
            names.append("liq_total_norm")
            feats.append(_safe(fl.get("oi_24h_change"), 0))
            names.append("oi_24h_change")
            long_pct = _safe(fl.get("long_account_pct"), 0.5)
            feats.append(long_pct - 0.5)  # centered: >0 means more longs
            names.append("long_account_bias")

            # ── 10. Exchange order book walls (ascetic0x) ──────────────
            eob = snap.exchange_ob or {}
            feats.append(_safe(eob.get("wall_imbalance"), 0) / 10.0)
            names.append("wall_imbalance")
            feats.append(_safe(eob.get("volume_imbalance"), 0))
            names.append("ob_volume_imbalance")
            feats.append(1.0 if eob.get("has_bid_wall") else 0.0)
            names.append("has_bid_wall")
            feats.append(1.0 if eob.get("has_ask_wall") else 0.0)
            names.append("has_ask_wall")

            # ── 11. News sentiment (ascetic0x) ─────────────────────────
            nw = snap.news or {}
            feats.append(_safe(nw.get("news_sentiment"), 0))
            names.append("news_sentiment")
            feats.append(1.0 if nw.get("news_high_impact") else 0.0)
            names.append("news_high_impact")
            ndir = nw.get("news_direction", "neutral")
            feats.append(1.0 if ndir == "bullish" else (-1.0 if ndir == "bearish" else 0.0))
            names.append("news_direction_signal")

            self._feature_names = names
            return np.array(feats, dtype=np.float32)

        except Exception as e:
            logger.error("Feature build failed: {}", e)
            return None


# ─── helper functions (pure, no side-effects) ──────────────────────────


def _ret(closes: list[float], periods: int) -> float:
    if len(closes) < periods + 1:
        return 0.0
    return (closes[-1] - closes[-1 - periods]) / closes[-1 - periods]


def _log_ret(closes: list[float], periods: int) -> float:
    if len(closes) < periods + 1 or closes[-1 - periods] <= 0:
        return 0.0
    return math.log(closes[-1] / closes[-1 - periods])


def _momentum(closes: list[float], periods: int) -> float:
    if len(closes) < periods + 1:
        return 0.0
    return closes[-1] - closes[-1 - periods]


def _rsi(closes: list[float], period: int) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(-period, 0)]
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _ema(values: list[float], span: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (span + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def _macd(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9):
    if len(closes) < slow:
        return 0.0, 0.0
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = _ema(macd_line, signal)
    return macd_line[-1] if macd_line else 0.0, signal_line[-1] if signal_line else 0.0


def _bb_pct_b(closes: list[float], period: int = 20, num_std: float = 2.0) -> float:
    if len(closes) < period:
        return 0.5
    window = closes[-period:]
    mean = sum(window) / period
    std = (sum((x - mean) ** 2 for x in window) / period) ** 0.5
    if std == 0:
        return 0.5
    upper = mean + num_std * std
    lower = mean - num_std * std
    return (closes[-1] - lower) / (upper - lower)


def _ema_cross(closes: list[float], fast: int, slow: int) -> float:
    if len(closes) < slow:
        return 0.0
    e_fast = _ema(closes, fast)
    e_slow = _ema(closes, slow)
    return e_fast[-1] - e_slow[-1]


def _rolling_std(closes: list[float], period: int) -> float:
    if len(closes) < period:
        return 0.0
    window = closes[-period:]
    mean = sum(window) / period
    return (sum((x - mean) ** 2 for x in window) / period) ** 0.5


def _atr(highs: list, lows: list, closes: list, period: int = 14) -> float:
    n = min(len(highs), len(lows), len(closes))
    if n < period + 1:
        return 0.0
    trs = []
    for i in range(-period, 0):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    return sum(trs) / len(trs)


def _volume_ratio(volumes: list[float], period: int) -> float:
    if len(volumes) < period + 1:
        return 1.0
    recent = sum(volumes[-period:]) / period
    prev = sum(volumes[-2 * period:-period]) / period if len(volumes) >= 2 * period else recent
    return recent / prev if prev > 0 else 1.0


def _stoch_k(highs: list, lows: list, closes: list, period: int = 14) -> float:
    n = min(len(highs), len(lows), len(closes))
    if n < period:
        return 50.0
    h = max(highs[-period:])
    l = min(lows[-period:])
    if h == l:
        return 50.0
    return 100.0 * (closes[-1] - l) / (h - l)


def _williams_r(highs: list, lows: list, closes: list, period: int = 14) -> float:
    n = min(len(highs), len(lows), len(closes))
    if n < period:
        return -50.0
    h = max(highs[-period:])
    l = min(lows[-period:])
    if h == l:
        return -50.0
    return -100.0 * (h - closes[-1]) / (h - l)


def _streak(outcomes: list[int]) -> float:
    if not outcomes:
        return 0.0
    last = outcomes[-1]
    count = 0
    for o in reversed(outcomes):
        if o == last:
            count += 1
        else:
            break
    return float(count) if last == 1 else float(-count)
