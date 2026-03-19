"""
Core trading logic – ties prediction, risk, and execution together.

Implements a dual-layer decision system:
  LAYER 1: XGBoost ML model → probability + confidence
  LAYER 2: Ascetic0x signal alignment → checklist scoring

Only trades when BOTH layers agree (or signal alignment is very strong).

Lifecycle of one 15-minute cycle:
  1. Discover the current KXBTC15M market.
  2. Wait until ~2 min before close (optimal entry window).
  3. Collect data snapshot + build features.
  4. Run ML prediction (Layer 1).
  5. Run signal alignment analysis (Layer 2 – ascetic0x checklist).
  6. Combine both layers → final direction + confidence.
  7. Apply risk checks.
  8. If allowed → place order on Kalshi.
  9. Wait for settlement → log outcome → feed self-learning loop.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from loguru import logger

from kalshi_bot.config import TRADING
from kalshi_bot.kalshi.client import KalshiClient
from kalshi_bot.kalshi.market_discovery import MarketDiscovery, MarketInfo
from kalshi_bot.data.data_aggregator import DataAggregator, DataSnapshot
from kalshi_bot.ml.feature_engineer import FeatureEngineer
from kalshi_bot.ml.predictor import Predictor
from kalshi_bot.strategy.risk_manager import RiskManager
from kalshi_bot.strategy.signal_analyzer import SignalAnalyzer, AlignmentResult
from kalshi_bot.learning.trade_logger import TradeLogger, TradeRecord
from kalshi_bot.learning.performance_analyzer import PerformanceAnalyzer


class TradingEngine:
    """Executes the full trade cycle for one 15-minute market."""

    def __init__(
        self,
        client: KalshiClient,
        discovery: MarketDiscovery,
        aggregator: DataAggregator,
        feature_eng: FeatureEngineer,
        predictor: Predictor,
        risk_mgr: RiskManager,
        trade_logger: TradeLogger,
        analyzer: PerformanceAnalyzer,
        signal_analyzer: Optional[SignalAnalyzer] = None,
    ):
        self.client = client
        self.discovery = discovery
        self.aggregator = aggregator
        self.feature_eng = feature_eng
        self.predictor = predictor
        self.risk = risk_mgr
        self.trade_logger = trade_logger
        self.analyzer = analyzer
        self.signal_analyzer = signal_analyzer or SignalAnalyzer()
        self._last_traded_market: Optional[str] = None

    async def run_cycle(self) -> dict:
        """Execute one complete 15-minute trade cycle.

        Returns a status dict summarizing what happened.
        """
        result = {"action": "skip", "reason": "", "market": None}

        # 1) Find current market
        market = self.discovery.get_current_market()
        if market is None:
            result["reason"] = "no open market found"
            logger.warning("No open KXBTC15M market")
            return result

        result["market"] = market.ticker
        logger.info("Current market: {} (closes in {:.0f}s)", market.ticker, market.seconds_until_close)

        # Avoid trading the same market twice
        if market.ticker == self._last_traded_market:
            result["reason"] = "already traded this market"
            return result

        # 2) Wait for optimal entry window
        entry_window = TRADING.trade_entry_minutes_before_close * 60
        wait_seconds = market.seconds_until_close - entry_window
        if wait_seconds > 0:
            logger.info("Waiting {:.0f}s for entry window...", wait_seconds)
            await asyncio.sleep(min(wait_seconds, 600))

        market = self.discovery.get_current_market()
        if market is None or not market.is_open:
            result["reason"] = "market closed while waiting"
            return result

        # 3) Collect data
        snap = await self.aggregator.snapshot(market_ticker=market.ticker)
        if not snap.is_valid:
            result["reason"] = "data snapshot invalid"
            logger.warning("Data snapshot not valid enough to trade")
            return result

        # 4) LAYER 1 – ML prediction
        features = self.feature_eng.build(snap)
        if features is None:
            result["reason"] = "feature engineering failed"
            return result

        prob_up, ml_confidence = self.predictor.predict(features)
        ml_direction = "up" if prob_up > 0.5 else "down"
        logger.info(
            "ML Layer: direction={} prob_up={:.2%} confidence={:.2%}",
            ml_direction, prob_up, ml_confidence,
        )

        # 5) LAYER 2 – Ascetic0x signal alignment
        alignment = self.signal_analyzer.analyze(
            candles_1m=snap.candles_1m,
            funding_data=snap.funding_liq,
            liquidation_data=snap.funding_liq,  # same dict has liq fields
            exchange_ob=snap.exchange_ob,
            news_data=snap.news,
            ml_direction=ml_direction,
            ml_confidence=ml_confidence,
        )

        result["alignment_score"] = alignment.raw_score
        result["alignment_direction"] = alignment.aligned_direction
        result["alignment_votes"] = [
            {"name": v.name, "dir": v.direction, "score": v.score}
            for v in alignment.votes
        ]

        # 6) Combine both layers → final decision
        final_direction, final_confidence, skip_reason = self._combine_layers(
            ml_direction, ml_confidence, prob_up, alignment,
        )

        if skip_reason:
            result["action"] = "skip"
            result["reason"] = skip_reason
            logger.info("Signal filter: {}", skip_reason)
            self._log_skip(market, features, prob_up, ml_confidence, final_direction, skip_reason)
            return result

        direction = final_direction
        confidence = final_confidence
        side = "yes" if direction == "up" else "no"

        logger.info(
            "FINAL: direction={} confidence={:.2%} (ML={:.2%} + alignment={:.1f})",
            direction, confidence, ml_confidence, alignment.raw_score,
        )

        # 7) Risk check
        balance = self.client.get_balance_dollars()
        risk_decision = self.risk.check(balance, confidence)
        if not risk_decision.allowed:
            result["action"] = "skip"
            result["reason"] = risk_decision.reason
            logger.info("Risk blocked: {}", risk_decision.reason)
            self._log_skip(market, features, prob_up, confidence, direction, risk_decision.reason)
            return result

        bet_dollars = risk_decision.bet_dollars

        # 8) Place order
        ob = snap.kalshi_ob or {}
        if side == "yes":
            price_cents = int(min(99, max(1, (ob.get("best_yes_ask", 0.55)) * 100)))
        else:
            price_cents = int(min(99, max(1, (ob.get("best_no_ask", 0.55)) * 100)))

        cost_per_contract = price_cents / 100.0
        count = max(1, int(bet_dollars / cost_per_contract))

        try:
            order_resp = self.client.create_order(
                ticker=market.ticker,
                side=side,
                action="buy",
                count=count,
                yes_price=price_cents if side == "yes" else None,
                no_price=price_cents if side == "no" else None,
                time_in_force="immediate_or_cancel",
            )
            order = order_resp.get("order", {})
            order_id = order.get("order_id", "unknown")
            status = order.get("status", "unknown")
            logger.info("Order placed: {} status={}", order_id, status)
            result["action"] = "traded"
            result["direction"] = direction
            result["side"] = side
            result["count"] = count
            result["price_cents"] = price_cents
            result["order_id"] = order_id
        except Exception as e:
            logger.error("Order failed: {}", e)
            result["action"] = "order_failed"
            result["reason"] = str(e)
            self._log_skip(market, features, prob_up, confidence, direction, f"order_error: {e}")
            return result

        self._last_traded_market = market.ticker

        # 9) Log the trade
        record = TradeRecord(
            ts=time.time(),
            market_ticker=market.ticker,
            features=features.tolist(),
            feature_names=self.feature_eng.feature_names,
            prob_up=prob_up,
            confidence=confidence,
            direction=direction,
            bet_dollars=bet_dollars,
            yes_price=price_cents / 100.0,
        )
        self.trade_logger.append(record)

        return result

    def _combine_layers(
        self,
        ml_direction: str,
        ml_confidence: float,
        prob_up: float,
        alignment: AlignmentResult,
    ) -> tuple[str, float, Optional[str]]:
        """Combine ML prediction with ascetic0x signal alignment.

        Returns (direction, confidence, skip_reason).
        skip_reason is None if we should trade, or a string explaining why not.
        """
        # If signals aren't aligned enough, skip unless ML is very confident
        if not alignment.should_trade:
            if ml_confidence >= 0.75:
                # ML is very confident – trade anyway but with reduced confidence
                return ml_direction, ml_confidence * 0.8, None
            return ml_direction, ml_confidence, alignment.reason

        # Signals are aligned – check if ML agrees
        if alignment.aligned_direction == ml_direction:
            # Both agree → boost confidence
            boosted = min(0.95, ml_confidence + alignment.alignment_strength * 0.15)
            return ml_direction, boosted, None

        # ML and signals disagree
        if alignment.alignment_strength >= 0.8:
            # Signals are very strong → trust them over ML
            new_conf = 0.55 + alignment.alignment_strength * 0.2
            return alignment.aligned_direction, new_conf, None

        if ml_confidence >= 0.7:
            # ML is fairly confident → trust ML but reduce confidence
            return ml_direction, ml_confidence * 0.7, None

        # Neither is confident enough
        return (
            ml_direction, ml_confidence,
            f"ML ({ml_direction}) and signals ({alignment.aligned_direction}) disagree, neither strong enough",
        )

    async def check_settlement(self, market_ticker: str) -> Optional[dict]:
        """Poll for the settlement of a market and update the trade log.

        Returns the settlement info or None if not yet settled.
        """
        try:
            mkt_data = self.client.get_market(market_ticker)
            market_info = mkt_data.get("market", mkt_data)
            status = market_info.get("status", "")

            if status != "settled":
                return None

            result_str = market_info.get("result", "")
            logger.info("Market {} settled: result={}", market_ticker, result_str)

            records = self.trade_logger.read_all()
            matching = [r for r in records if r.market_ticker == market_ticker and r.outcome is None]
            if not matching:
                return {"settled": True, "result": result_str}

            rec = matching[-1]
            # Determine win/loss
            if rec.direction == "up":
                won = result_str == "yes"
            else:
                won = result_str == "no"

            if won:
                payout_per_contract = 1.0 - (rec.yes_price or 0.5)
                pnl = payout_per_contract * (rec.bet_dollars / (rec.yes_price or 0.5))
            else:
                pnl = -rec.bet_dollars

            self.trade_logger.update_last_outcome(won, round(pnl, 2))
            self.feature_eng.record_outcome(won)
            self.analyzer.record(won, pnl)

            if won:
                self.risk.record_win(abs(pnl))
            else:
                self.risk.record_loss(abs(pnl))

            logger.info(
                "Trade outcome: {} PnL=${:.2f} (cumulative WR={:.1%})",
                "WIN" if won else "LOSS", pnl, self.analyzer.win_rate_all,
            )

            return {"settled": True, "result": result_str, "won": won, "pnl": pnl}

        except Exception as e:
            logger.warning("Settlement check failed for {}: {}", market_ticker, e)
            return None

    def _log_skip(self, market, features, prob_up, confidence, direction, reason):
        record = TradeRecord(
            ts=time.time(),
            market_ticker=market.ticker,
            features=features.tolist() if features is not None else [],
            feature_names=self.feature_eng.feature_names,
            prob_up=prob_up,
            confidence=confidence,
            direction="skip",
            bet_dollars=0,
            yes_price=None,
        )
        self.trade_logger.append(record)
