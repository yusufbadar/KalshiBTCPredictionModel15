"""
Signal alignment analyzer with research-backed indicators.

Always produces a direction (never "neutral" with no trade). When signals are
weak or conflicting, the alignment_strength is low, which feeds into dynamic
bet sizing rather than skipping.

CHECKLIST (7 signals, each scores ±1):
  1. PRICE ACTION    – trend/stall/reversal via multi-timeframe returns
  2. FUNDING RATE    – overcrowded side detection (top/bottom 20% of range)
  3. LIQUIDATIONS    – squeeze detection (which side got flushed)
  4. ORDER BOOK      – wall presence and volume imbalance
  5. NEWS            – breaking headline sentiment
  6. VWAP DEVIATION  – price vs volume-weighted average (mean-reversion signal)
  7. MICROSTRUCTURE  – tick momentum and rate-of-change in last 2 min

Research basis:
  - VWAP deviation: institutional benchmark; prices revert to VWAP on short
    timeframes (Berkowitz et al. 2005, "Best Execution in the Digital Age")
  - Order flow imbalance: predictive of 1-5 min returns (Cont, Kukanov &
    Stoikov 2014, "The Price Impact of Order Book Events")
  - Volatility regime: high vol → contrarian signals work better; low vol →
    momentum signals work better (Moskowitz, Ooi & Pedersen 2012)
  - Microstructure momentum: rate of price change in the last 60-120s
    predicts next-period direction (Bouchaud et al. 2004)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class SignalVote:
    """One signal's contribution to the overall alignment."""
    name: str
    direction: str        # "bullish", "bearish", or "neutral"
    score: float          # -1.0 to +1.0
    reason: str = ""


@dataclass
class AlignmentResult:
    """Aggregate signal alignment result."""
    votes: list[SignalVote] = field(default_factory=list)
    raw_score: float = 0.0
    aligned_direction: str = "neutral"
    alignment_strength: float = 0.0
    should_trade: bool = True           # always True now
    reason: str = ""

    @property
    def bullish_count(self) -> int:
        return sum(1 for v in self.votes if v.direction == "bullish")

    @property
    def bearish_count(self) -> int:
        return sum(1 for v in self.votes if v.direction == "bearish")


class SignalAnalyzer:
    """Applies expanded signal checklist. Always produces a tradeable direction."""

    def __init__(self, min_alignment: int = 1):
        self.min_alignment = min_alignment

    def analyze(
        self,
        *,
        candles_1m: list[dict],
        funding_data: Optional[dict],
        liquidation_data: Optional[dict],
        exchange_ob: Optional[dict],
        news_data: Optional[dict],
        ml_direction: str = "neutral",
        ml_confidence: float = 0.5,
    ) -> AlignmentResult:
        votes: list[SignalVote] = []

        votes.append(self._score_price_action(candles_1m))
        votes.append(self._score_funding(funding_data))
        votes.append(self._score_liquidations(liquidation_data))
        votes.append(self._score_orderbook_walls(exchange_ob))
        votes.append(self._score_news(news_data))
        votes.append(self._score_vwap_deviation(candles_1m))
        votes.append(self._score_microstructure(candles_1m))

        raw_score = sum(v.score for v in votes)

        if raw_score > 0:
            aligned_dir = "up"
        elif raw_score < 0:
            aligned_dir = "down"
        else:
            # Tiebreaker: trust ML direction, or default to "up"
            aligned_dir = ml_direction if ml_direction != "neutral" else "up"

        max_possible = len(votes)
        alignment_strength = abs(raw_score) / max_possible if max_possible > 0 else 0.1

        # Always trade. Strength just affects bet sizing via confidence.
        should_trade = True
        reason = (
            f"{abs(raw_score):.1f}/{max_possible} signals → "
            f"{aligned_dir.upper()} (strength={alignment_strength:.0%})"
        )

        if ml_direction != "neutral" and ml_direction != aligned_dir:
            alignment_strength *= 0.7
            reason += " [ML disagrees]"

        result = AlignmentResult(
            votes=votes,
            raw_score=raw_score,
            aligned_direction=aligned_dir,
            alignment_strength=alignment_strength,
            should_trade=should_trade,
            reason=reason,
        )

        logger.info(
            "Signal alignment: score={:.1f} dir={} strength={:.0%} | {}",
            raw_score, aligned_dir, alignment_strength, reason,
        )
        for v in votes:
            logger.debug("  {} → {} ({:+.1f}): {}", v.name, v.direction, v.score, v.reason)

        return result

    # ── Original signals (improved) ──────────────────────────────

    def _score_price_action(self, candles: list[dict]) -> SignalVote:
        if not candles or len(candles) < 10:
            return SignalVote("price_action", "neutral", 0, "insufficient data")

        closes = [c["close"] for c in candles]

        ret_5m = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        ret_15m = (closes[-1] - closes[-15]) / closes[-15] if len(closes) >= 15 else 0
        ret_30m = (closes[-1] - closes[-30]) / closes[-30] if len(closes) >= 30 else ret_15m
        ret_60m = (closes[-1] - closes[-60]) / closes[-60] if len(closes) >= 60 else ret_30m

        big_move_up = ret_60m > 0.005
        big_move_down = ret_60m < -0.005
        stalling = abs(ret_5m) < 0.0005

        if big_move_up and stalling:
            return SignalVote(
                "price_action", "bearish", -0.8,
                f"stalling after rise ({ret_60m:+.2%} → {ret_5m:+.3%})"
            )
        if big_move_down and stalling:
            return SignalVote(
                "price_action", "bullish", 0.8,
                f"stalling after drop ({ret_60m:+.2%} → {ret_5m:+.3%})"
            )

        # Multi-timeframe momentum agreement (stronger signal)
        if ret_5m > 0 and ret_15m > 0 and ret_30m > 0.002:
            return SignalVote("price_action", "bullish", 0.9, f"multi-TF uptrend {ret_30m:+.2%}")
        if ret_5m < 0 and ret_15m < 0 and ret_30m < -0.002:
            return SignalVote("price_action", "bearish", -0.9, f"multi-TF downtrend {ret_30m:+.2%}")

        if ret_15m > 0.002 and ret_5m > 0:
            return SignalVote("price_action", "bullish", 0.6, f"uptrend {ret_15m:+.2%}")
        if ret_15m < -0.002 and ret_5m < 0:
            return SignalVote("price_action", "bearish", -0.6, f"downtrend {ret_15m:+.2%}")

        # Weak lean based on recent 5m direction
        if ret_5m > 0.0003:
            return SignalVote("price_action", "bullish", 0.2, f"slight upward {ret_5m:+.3%}")
        if ret_5m < -0.0003:
            return SignalVote("price_action", "bearish", -0.2, f"slight downward {ret_5m:+.3%}")

        return SignalVote("price_action", "neutral", 0, "no clear pattern")

    def _score_funding(self, data: Optional[dict]) -> SignalVote:
        if not data:
            return SignalVote("funding_rate", "neutral", 0, "no data")

        rate = data.get("funding_rate", 0)
        percentile = data.get("funding_percentile", 50)
        extreme = data.get("funding_extreme", "neutral")

        if extreme == "bearish" or percentile >= 80:
            return SignalVote(
                "funding_rate", "bearish", -1.0,
                f"rate={rate:.4%} pctl={percentile:.0f}% – longs overcrowded"
            )
        if extreme == "bullish" or percentile <= 20:
            return SignalVote(
                "funding_rate", "bullish", 1.0,
                f"rate={rate:.4%} pctl={percentile:.0f}% – shorts overcrowded"
            )

        if percentile >= 65:
            return SignalVote("funding_rate", "bearish", -0.4, f"rate slightly elevated {rate:.4%}")
        if percentile <= 35:
            return SignalVote("funding_rate", "bullish", 0.4, f"rate slightly negative {rate:.4%}")

        return SignalVote("funding_rate", "neutral", 0, f"rate={rate:.4%} in normal range")

    def _score_liquidations(self, data: Optional[dict]) -> SignalVote:
        if not data:
            return SignalVote("liquidations", "neutral", 0, "no data")

        long_liq = data.get("liq_long_24h", 0)
        short_liq = data.get("liq_short_24h", 0)
        total = long_liq + short_liq

        if total <= 0:
            ls_ratio = data.get("long_short_ratio", 1.0)
            if ls_ratio > 1.5:
                return SignalVote(
                    "liquidations", "bearish", -0.6,
                    f"L/S ratio {ls_ratio:.2f} – too many longs"
                )
            if ls_ratio < 0.67:
                return SignalVote(
                    "liquidations", "bullish", 0.6,
                    f"L/S ratio {ls_ratio:.2f} – too many shorts"
                )
            return SignalVote("liquidations", "neutral", 0, f"L/S ratio balanced {ls_ratio:.2f}")

        ratio = short_liq / long_liq if long_liq > 0 else 10.0

        if ratio > 2.0:
            return SignalVote(
                "liquidations", "bearish", -1.0,
                f"shorts squeezed (S/L={ratio:.1f}x, ${short_liq/1e6:.0f}M)"
            )
        if ratio < 0.5:
            return SignalVote(
                "liquidations", "bullish", 1.0,
                f"longs flushed (S/L={ratio:.1f}x, ${long_liq/1e6:.0f}M)"
            )
        if ratio > 1.3:
            return SignalVote("liquidations", "bearish", -0.5, f"more shorts liquidated (S/L={ratio:.1f}x)")
        if ratio < 0.77:
            return SignalVote("liquidations", "bullish", 0.5, f"more longs liquidated (S/L={ratio:.1f}x)")

        return SignalVote("liquidations", "neutral", 0, f"balanced liquidations (S/L={ratio:.1f}x)")

    def _score_orderbook_walls(self, ob: Optional[dict]) -> SignalVote:
        if not ob:
            return SignalVote("orderbook_walls", "neutral", 0, "no data")

        has_bid = ob.get("has_bid_wall", False)
        has_ask = ob.get("has_ask_wall", False)
        imbalance = ob.get("wall_imbalance", 0)
        vol_imb = ob.get("volume_imbalance", 0)

        if has_bid and not has_ask:
            return SignalVote(
                "orderbook_walls", "bullish", 1.0,
                f"bid wall at ${ob.get('bid_wall_price', 0):,.0f}, no ask wall"
            )
        if has_ask and not has_bid:
            return SignalVote(
                "orderbook_walls", "bearish", -1.0,
                f"ask wall at ${ob.get('ask_wall_price', 0):,.0f}, no bid wall"
            )
        if has_bid and has_ask:
            if imbalance > 1:
                return SignalVote("orderbook_walls", "bullish", 0.5, f"bid wall stronger ({imbalance:+.1f})")
            if imbalance < -1:
                return SignalVote("orderbook_walls", "bearish", -0.5, f"ask wall stronger ({imbalance:+.1f})")

        if vol_imb > 0.15:
            return SignalVote("orderbook_walls", "bullish", 0.4, f"bid volume dominant ({vol_imb:+.2f})")
        if vol_imb < -0.15:
            return SignalVote("orderbook_walls", "bearish", -0.4, f"ask volume dominant ({vol_imb:+.2f})")

        return SignalVote("orderbook_walls", "neutral", 0, "no significant walls")

    def _score_news(self, data: Optional[dict]) -> SignalVote:
        if not data:
            return SignalVote("news", "neutral", 0, "no data")

        direction = data.get("news_direction", "neutral")
        sentiment = data.get("news_sentiment", 0)
        high_impact = data.get("news_high_impact", False)
        headlines = data.get("news_impact_headlines", [])

        if high_impact:
            if direction == "bullish":
                return SignalVote("news", "bullish", 1.0, f"high-impact bullish: {headlines[0] if headlines else 'n/a'}")
            if direction == "bearish":
                return SignalVote("news", "bearish", -1.0, f"high-impact bearish: {headlines[0] if headlines else 'n/a'}")

        if abs(sentiment) > 0.3:
            s = min(0.6, abs(sentiment))
            if sentiment > 0:
                return SignalVote("news", "bullish", s, f"mildly bullish news ({sentiment:+.2f})")
            return SignalVote("news", "bearish", -s, f"mildly bearish news ({sentiment:+.2f})")

        return SignalVote("news", "neutral", 0, "no breaking news")

    # ── NEW research-backed signals ──────────────────────────────

    def _score_vwap_deviation(self, candles: list[dict]) -> SignalVote:
        """VWAP deviation: price far above VWAP → overbought (bearish mean-reversion),
        price far below VWAP → oversold (bullish mean-reversion).

        On 15-min horizons, VWAP acts as an institutional gravity anchor.
        Deviation > 1 standard deviation is a strong reversion signal.
        """
        if not candles or len(candles) < 15:
            return SignalVote("vwap_deviation", "neutral", 0, "insufficient data")

        recent = candles[-60:] if len(candles) >= 60 else candles

        cum_vol = 0.0
        cum_vp = 0.0
        prices = []
        for c in recent:
            vol = c.get("volume", 1.0)
            typical = (c.get("high", c["close"]) + c.get("low", c["close"]) + c["close"]) / 3
            cum_vol += vol
            cum_vp += typical * vol
            prices.append(c["close"])

        if cum_vol == 0:
            return SignalVote("vwap_deviation", "neutral", 0, "no volume data")

        vwap = cum_vp / cum_vol
        current = prices[-1]
        deviation_pct = (current - vwap) / vwap

        # Standard deviation of prices for threshold
        mean_p = sum(prices) / len(prices)
        std_p = (sum((p - mean_p) ** 2 for p in prices) / len(prices)) ** 0.5
        std_pct = std_p / mean_p if mean_p > 0 else 0.001

        if std_pct == 0:
            return SignalVote("vwap_deviation", "neutral", 0, "zero volatility")

        z_score = deviation_pct / std_pct if std_pct > 0 else 0

        # Strong mean-reversion signal beyond 1.5 std devs
        if z_score > 1.5:
            return SignalVote("vwap_deviation", "bearish", -0.9, f"price {deviation_pct:+.3%} above VWAP (z={z_score:.1f})")
        if z_score < -1.5:
            return SignalVote("vwap_deviation", "bullish", 0.9, f"price {deviation_pct:+.3%} below VWAP (z={z_score:.1f})")

        if z_score > 0.8:
            return SignalVote("vwap_deviation", "bearish", -0.5, f"moderately above VWAP (z={z_score:.1f})")
        if z_score < -0.8:
            return SignalVote("vwap_deviation", "bullish", 0.5, f"moderately below VWAP (z={z_score:.1f})")

        if z_score > 0.3:
            return SignalVote("vwap_deviation", "bearish", -0.2, f"slightly above VWAP (z={z_score:.1f})")
        if z_score < -0.3:
            return SignalVote("vwap_deviation", "bullish", 0.2, f"slightly below VWAP (z={z_score:.1f})")

        return SignalVote("vwap_deviation", "neutral", 0, f"near VWAP (z={z_score:.1f})")

    def _score_microstructure(self, candles: list[dict]) -> SignalVote:
        """Microstructure momentum: rate-of-change and tick direction clustering
        in the last 2-3 minutes.

        Short-term price momentum (60-180s) is predictive of next-period
        direction due to informed order flow (Bouchaud et al. 2004).

        Also detects "acceleration" – whether the price is moving faster in
        the last minute than the previous minute (breakout detection).
        """
        if not candles or len(candles) < 5:
            return SignalVote("microstructure", "neutral", 0, "insufficient data")

        closes = [c["close"] for c in candles]

        # Last 2 minutes rate-of-change
        ret_2m = (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0
        ret_prev_2m = (closes[-2] - closes[-3]) / closes[-3] if len(closes) >= 3 else 0

        # Tick direction: count consecutive same-direction moves (last 5 candles)
        recent = closes[-6:] if len(closes) >= 6 else closes
        upticks = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
        downticks = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])
        total_ticks = max(1, upticks + downticks)

        tick_bias = (upticks - downticks) / total_ticks

        # Acceleration: is the move getting stronger?
        acceleration = ret_2m - ret_prev_2m

        # Combine: momentum + tick clustering + acceleration
        combo_score = (ret_2m * 5000) + (tick_bias * 0.3) + (acceleration * 3000)

        if combo_score > 0.8:
            return SignalVote("microstructure", "bullish", 0.9, f"strong micro-momentum up (ret={ret_2m:+.3%}, ticks={tick_bias:+.1f})")
        if combo_score < -0.8:
            return SignalVote("microstructure", "bearish", -0.9, f"strong micro-momentum down (ret={ret_2m:+.3%}, ticks={tick_bias:+.1f})")

        if combo_score > 0.3:
            return SignalVote("microstructure", "bullish", 0.5, f"moderate micro-momentum up (ret={ret_2m:+.3%})")
        if combo_score < -0.3:
            return SignalVote("microstructure", "bearish", -0.5, f"moderate micro-momentum down (ret={ret_2m:+.3%})")

        if combo_score > 0.1:
            return SignalVote("microstructure", "bullish", 0.2, f"slight micro-momentum up")
        if combo_score < -0.1:
            return SignalVote("microstructure", "bearish", -0.2, f"slight micro-momentum down")

        return SignalVote("microstructure", "neutral", 0, "no clear microstructure signal")
