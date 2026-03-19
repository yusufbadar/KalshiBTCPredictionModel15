"""
Ascetic0x signal alignment analyzer.

Implements the exact checklist from the article – the bot only trades when
multiple independent signals AGREE on direction. This is layered on top of
the ML model: ML provides a probability, signal alignment provides conviction.

CHECKLIST (each item scores ±1 for direction):
  1. PRICE ACTION:  Is BTC trending, stalling, or reversing?
  2. FUNDING RATE:  Is one side overcrowded (top/bottom 20% of 30-day range)?
  3. LIQUIDATIONS:  Has one side been squeezed hard in last 24h?
  4. ORDER BOOK:    Is there a wall supporting the predicted direction?
  5. NEWS:          Any breaking headline confirming or contradicting?

The final "alignment score" (-5 to +5) is combined with the ML prediction.
Positive = bullish, Negative = bearish. Only trade when |score| >= 3.
"""
from __future__ import annotations

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
    raw_score: float = 0.0          # sum of all vote scores
    aligned_direction: str = "neutral"  # "up", "down", or "neutral"
    alignment_strength: float = 0.0  # 0.0 – 1.0
    should_trade: bool = False
    reason: str = ""

    @property
    def bullish_count(self) -> int:
        return sum(1 for v in self.votes if v.direction == "bullish")

    @property
    def bearish_count(self) -> int:
        return sum(1 for v in self.votes if v.direction == "bearish")


class SignalAnalyzer:
    """Applies the ascetic0x signal checklist and produces an alignment score."""

    def __init__(self, min_alignment: int = 3):
        """
        Args:
            min_alignment: minimum |raw_score| (out of 5) to consider signals
                           "aligned" enough to trade. Default 3 means at least
                           3 out of 5 signals must agree.
        """
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
        """Run the full signal checklist.

        Args:
            candles_1m: recent 1-minute candles (list of dicts with 'close').
            funding_data: output from FundingLiquidationFeed.fetch().
            liquidation_data: same dict (contains liq_ fields).
            exchange_ob: output from ExchangeOrderbookFeed.fetch().
            news_data: output from NewsFeed.fetch().
            ml_direction: "up" or "down" from the ML model.
            ml_confidence: ML confidence 0–1.

        Returns:
            AlignmentResult with scoring and trade recommendation.
        """
        votes: list[SignalVote] = []

        # 1) PRICE ACTION
        votes.append(self._score_price_action(candles_1m))

        # 2) FUNDING RATE
        votes.append(self._score_funding(funding_data))

        # 3) LIQUIDATIONS
        votes.append(self._score_liquidations(liquidation_data))

        # 4) ORDER BOOK WALLS
        votes.append(self._score_orderbook_walls(exchange_ob))

        # 5) NEWS
        votes.append(self._score_news(news_data))

        raw_score = sum(v.score for v in votes)

        if raw_score > 0:
            aligned_dir = "up"
        elif raw_score < 0:
            aligned_dir = "down"
        else:
            aligned_dir = "neutral"

        max_possible = len(votes)
        alignment_strength = abs(raw_score) / max_possible if max_possible > 0 else 0

        # Should we trade?
        if abs(raw_score) >= self.min_alignment:
            should_trade = True
            reason = (
                f"{abs(raw_score):.0f}/{max_possible} signals aligned "
                f"→ {aligned_dir.upper()}"
            )
        else:
            should_trade = False
            reason = (
                f"Only {abs(raw_score):.1f}/{max_possible} signals aligned "
                f"(need {self.min_alignment})"
            )

        # Bonus: check if ML agrees with signal alignment
        if should_trade and ml_direction != "neutral":
            ml_agrees = (
                (ml_direction == "up" and aligned_dir == "up") or
                (ml_direction == "down" and aligned_dir == "down")
            )
            if not ml_agrees:
                alignment_strength *= 0.6
                reason += " [ML disagrees – reduced confidence]"

        result = AlignmentResult(
            votes=votes,
            raw_score=raw_score,
            aligned_direction=aligned_dir,
            alignment_strength=alignment_strength,
            should_trade=should_trade,
            reason=reason,
        )

        logger.info(
            "Signal alignment: score={:.1f} dir={} strength={:.0%} trade={} | {}",
            raw_score, aligned_dir, alignment_strength, should_trade, reason,
        )
        for v in votes:
            logger.debug("  {} → {} ({:+.1f}): {}", v.name, v.direction, v.score, v.reason)

        return result

    # ── Individual signal scorers ─────────────────────────────────

    def _score_price_action(self, candles: list[dict]) -> SignalVote:
        """Check recent price action for trend/stall/reversal."""
        if not candles or len(candles) < 30:
            return SignalVote("price_action", "neutral", 0, "insufficient data")

        closes = [c["close"] for c in candles]

        ret_5m = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        ret_30m = (closes[-1] - closes[-30]) / closes[-30] if len(closes) >= 30 else 0
        ret_60m = (closes[-1] - closes[-60]) / closes[-60] if len(closes) >= 60 else ret_30m

        # Check for stalling after a big move (reversal signal)
        big_move_up = ret_60m > 0.005    # >0.5% in last hour
        big_move_down = ret_60m < -0.005
        stalling = abs(ret_5m) < 0.0005  # last 5 min barely moved

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

        # Clear trend
        if ret_30m > 0.003 and ret_5m > 0:
            return SignalVote("price_action", "bullish", 0.6, f"uptrend {ret_30m:+.2%}")
        if ret_30m < -0.003 and ret_5m < 0:
            return SignalVote("price_action", "bearish", -0.6, f"downtrend {ret_30m:+.2%}")

        return SignalVote("price_action", "neutral", 0, "no clear pattern")

    def _score_funding(self, data: Optional[dict]) -> SignalVote:
        """Score funding rate signal."""
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

        # Moderate lean
        if percentile >= 65:
            return SignalVote("funding_rate", "bearish", -0.4, f"rate slightly elevated {rate:.4%}")
        if percentile <= 35:
            return SignalVote("funding_rate", "bullish", 0.4, f"rate slightly negative {rate:.4%}")

        return SignalVote("funding_rate", "neutral", 0, f"rate={rate:.4%} in normal range")

    def _score_liquidations(self, data: Optional[dict]) -> SignalVote:
        """Score liquidation data.

        Key insight from ascetic0x: the side that just got liquidated is
        likely DONE selling/buying. So high short liquidations → bears are
        flushed → less upside fuel → potential reversal DOWN.
        """
        if not data:
            return SignalVote("liquidations", "neutral", 0, "no data")

        long_liq = data.get("liq_long_24h", 0)
        short_liq = data.get("liq_short_24h", 0)
        total = long_liq + short_liq

        if total <= 0:
            # Fallback: use long/short account ratio
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

        # ascetic0x logic: high short liquidations → shorts squeezed → upside fuel gone → DOWN
        if ratio > 2.0:
            return SignalVote(
                "liquidations", "bearish", -1.0,
                f"shorts squeezed hard (S/L={ratio:.1f}x, ${short_liq/1e6:.0f}M)"
            )
        # High long liquidations → longs flushed → selling done → UP
        if ratio < 0.5:
            return SignalVote(
                "liquidations", "bullish", 1.0,
                f"longs flushed (S/L={ratio:.1f}x, ${long_liq/1e6:.0f}M)"
            )

        if ratio > 1.3:
            return SignalVote(
                "liquidations", "bearish", -0.5,
                f"more shorts liquidated (S/L={ratio:.1f}x)"
            )
        if ratio < 0.77:
            return SignalVote(
                "liquidations", "bullish", 0.5,
                f"more longs liquidated (S/L={ratio:.1f}x)"
            )

        return SignalVote("liquidations", "neutral", 0, f"balanced liquidations (S/L={ratio:.1f}x)")

    def _score_orderbook_walls(self, ob: Optional[dict]) -> SignalVote:
        """Score exchange order book wall presence."""
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
                return SignalVote(
                    "orderbook_walls", "bullish", 0.5,
                    f"both walls present, bid wall stronger ({imbalance:+.1f})"
                )
            if imbalance < -1:
                return SignalVote(
                    "orderbook_walls", "bearish", -0.5,
                    f"both walls present, ask wall stronger ({imbalance:+.1f})"
                )

        # Fall back to volume imbalance
        if vol_imb > 0.15:
            return SignalVote(
                "orderbook_walls", "bullish", 0.4,
                f"bid volume dominant ({vol_imb:+.2f})"
            )
        if vol_imb < -0.15:
            return SignalVote(
                "orderbook_walls", "bearish", -0.4,
                f"ask volume dominant ({vol_imb:+.2f})"
            )

        return SignalVote("orderbook_walls", "neutral", 0, "no significant walls")

    def _score_news(self, data: Optional[dict]) -> SignalVote:
        """Score breaking news impact."""
        if not data:
            return SignalVote("news", "neutral", 0, "no data")

        direction = data.get("news_direction", "neutral")
        sentiment = data.get("news_sentiment", 0)
        high_impact = data.get("news_high_impact", False)
        headlines = data.get("news_impact_headlines", [])

        if high_impact:
            if direction == "bullish":
                return SignalVote(
                    "news", "bullish", 1.0,
                    f"high-impact bullish: {headlines[0] if headlines else 'n/a'}"
                )
            if direction == "bearish":
                return SignalVote(
                    "news", "bearish", -1.0,
                    f"high-impact bearish: {headlines[0] if headlines else 'n/a'}"
                )

        if abs(sentiment) > 0.3:
            s = min(0.6, abs(sentiment))
            if sentiment > 0:
                return SignalVote("news", "bullish", s, f"mildly bullish news ({sentiment:+.2f})")
            return SignalVote("news", "bearish", -s, f"mildly bearish news ({sentiment:+.2f})")

        return SignalVote("news", "neutral", 0, "no breaking news")
