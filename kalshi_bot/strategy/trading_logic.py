"""
Rule-based trading engine for KXBTC15M markets.

Strategy (pure price action, no ML):
  1. Wait 10 minutes into each 15-minute window (two 5m candles complete)
  2. Both 5m candles GREEN + price above 20 EMA  ->  buy YES
  3. Both 5m candles RED   + price below 20 EMA  ->  buy NO
  4. Mixed / ambiguous  ->  trade the direction of the bigger-body candle
  5. Hold to settlement (~5 minutes remaining)
"""
from __future__ import annotations

import asyncio
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from loguru import logger

from kalshi_bot.config import TRADING
from kalshi_bot.kalshi.client import KalshiClient
from kalshi_bot.kalshi.market_discovery import MarketDiscovery
from kalshi_bot.data.binance_feed import BinanceFeed
from kalshi_bot.data.kalshi_orderbook import KalshiOrderbookFeed
from kalshi_bot.strategy.risk_manager import RiskManager
from kalshi_bot.learning.trade_logger import TradeLogger, TradeRecord
from kalshi_bot.learning.performance_analyzer import PerformanceAnalyzer


def estimate_fee_cents(price_cents: int) -> float:
    """Kalshi quadratic fee estimate per contract."""
    p = min(price_cents, 100 - price_cents)
    return max(1.0, math.ceil(p * p / 10000.0))


def ema_20(closes: list[float]) -> float:
    """20-period exponential moving average (oldest-first list)."""
    period = 20
    if len(closes) < period:
        return sum(closes) / len(closes)
    mult = 2.0 / (period + 1)
    val = sum(closes[:period]) / period
    for price in closes[period:]:
        val = (price - val) * mult + val
    return val


class TradingEngine:
    """Rule-based engine: two 5m candles + 20 EMA -> pick side, hold to settlement."""

    def __init__(
        self,
        client: KalshiClient,
        discovery: MarketDiscovery,
        binance: BinanceFeed,
        kalshi_ob: KalshiOrderbookFeed,
        risk_mgr: RiskManager,
        trade_logger: TradeLogger,
        analyzer: PerformanceAnalyzer,
    ):
        self.client = client
        self.discovery = discovery
        self.binance = binance
        self.kalshi_ob = kalshi_ob
        self.risk = risk_mgr
        self.trade_logger = trade_logger
        self.analyzer = analyzer

        self._last_traded_market: Optional[str] = None
        self.current_position: Optional[dict] = None
        self.activity_log: list[str] = []
        self.last_cycle: dict = {}

    def _log_activity(self, msg: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"{ts} {msg}"
        self.activity_log.append(entry)
        if len(self.activity_log) > 50:
            self.activity_log = self.activity_log[-50:]
        logger.info(msg)

    # ── main entry point ────────────────────────────────────────────

    async def run_cycle(self) -> dict:
        """Execute one rule-based trade for the current 15-min window."""
        result: dict = {"action": "skip", "reason": "", "market": None}

        market = self.discovery.get_current_market()
        if market is None:
            result["reason"] = "no open market found"
            return result

        result["market"] = market.ticker

        if market.ticker == self._last_traded_market:
            result["reason"] = "already traded this market"
            return result

        # Wait until 10 minutes into the window (5 min before close)
        seconds_into_window = 900 - market.seconds_until_close
        wait_for_entry = 600 - seconds_into_window

        if wait_for_entry > 0:
            self._log_activity(
                f"Market: {market.ticker} -- waiting {wait_for_entry:.0f}s "
                f"for two 5m candles..."
            )
            waited = 0
            while waited < wait_for_entry:
                chunk = min(10, wait_for_entry - waited)
                await asyncio.sleep(chunk)
                waited += chunk

        # Refresh after wait
        market = self.discovery.get_current_market()
        if market is None or market.seconds_until_close < 240:
            result["reason"] = "too late to enter after wait"
            return result

        self._log_activity(
            f"Entry window: {market.ticker} ({market.seconds_until_close:.0f}s left)"
        )

        # ── 1. Fetch 5m candles ─────────────────────────────────────
        candles_5m = await self.binance.get_klines("5m", 50)
        if len(candles_5m) < 22:
            result["reason"] = "not enough 5m candle data"
            return result

        # ── 2. Find the two candles inside this window ──────────────
        window_start = market.close_time - timedelta(minutes=15)
        c1_target = window_start
        c2_target = window_start + timedelta(minutes=5)

        candle1, candle2 = None, None
        for c in candles_5m:
            if abs((c["ts"] - c1_target).total_seconds()) < 90:
                candle1 = c
            if abs((c["ts"] - c2_target).total_seconds()) < 90:
                candle2 = c

        if candle1 is None or candle2 is None:
            result["reason"] = "could not find intra-window 5m candles"
            self._log_activity("Missing 5m candles for current window")
            return result

        # ── 3. Compute 20 EMA ───────────────────────────────────────
        trailing_closes = [
            c["close"] for c in candles_5m
            if c["close_ts"] <= candle2["close_ts"]
        ][-22:]
        ema_val = ema_20(trailing_closes)
        current_price = candle2["close"]

        # ── 4. Apply rules ──────────────────────────────────────────
        c1_green = candle1["close"] > candle1["open"]
        c2_green = candle2["close"] > candle2["open"]
        above_ema = current_price > ema_val

        both_green = c1_green and c2_green
        both_red = (not c1_green) and (not c2_green)

        if both_green and above_ema:
            side, rule = "yes", "2xGREEN+aboveEMA"
        elif both_red and (not above_ema):
            side, rule = "no", "2xRED+belowEMA"
        else:
            body1 = abs(candle1["close"] - candle1["open"])
            body2 = abs(candle2["close"] - candle2["open"])
            if body1 >= body2:
                side = "yes" if c1_green else "no"
            else:
                side = "yes" if c2_green else "no"
            rule = "bigger_body"

        direction = "up" if side == "yes" else "down"

        self._log_activity(
            f"C1={'G' if c1_green else 'R'} C2={'G' if c2_green else 'R'} "
            f"EMA={'above' if above_ema else 'below'} -> {side.upper()} ({rule})"
        )

        result.update({
            "candle1_green": c1_green,
            "candle2_green": c2_green,
            "above_ema": above_ema,
            "ema_20": ema_val,
            "current_price": current_price,
            "rule": rule,
            "chosen_side": side,
        })

        # ── 5. Get Kalshi prices ────────────────────────────────────
        ob = self.kalshi_ob.fetch(market.ticker) or {}
        if side == "yes":
            ask_price = ob.get("best_yes_ask", 0.55)
        else:
            ask_price = ob.get("best_no_ask", 0.55)

        price_cents = max(1, min(99, int(round(ask_price * 100))))
        fee_cents = estimate_fee_cents(price_cents)

        # ── 6. Size the bet ─────────────────────────────────────────
        balance = self.client.get_balance_dollars()
        risk_decision = self.risk.check(balance, 2.0)
        if not risk_decision.allowed:
            result["action"] = "skip"
            result["reason"] = risk_decision.reason
            return result

        bet_dollars = risk_decision.bet_dollars
        cost_per = price_cents / 100.0
        count = max(1, int(bet_dollars / cost_per))
        total_fees = fee_cents * count

        # ── 7. Place order (IOC limit) ──────────────────────────────
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

            raw_filled = order.get("count_filled")
            filled = int(raw_filled) if raw_filled is not None else 0

            if filled == 0:
                self._log_activity(
                    f"Order placed but 0 contracts filled (requested {count})"
                )
                result["action"] = "no_fill"
                result["reason"] = "0 contracts filled"
                return result

            actual_cost = round(filled * (price_cents / 100.0), 2)
            actual_fees = round(fee_cents * filled, 2)

            self._log_activity(
                f"FILLED: {filled}x {side.upper()} @ {price_cents}c "
                f"(${actual_cost:.2f}, fee ~{actual_fees:.0f}c)"
            )

            self.current_position = {
                "side": side,
                "direction": direction,
                "entry_price": price_cents,
                "count": filled,
                "bet_dollars": actual_cost,
                "order_id": order_id,
                "entry_time": time.time(),
            }

            result.update({
                "action": "traded",
                "direction": direction,
                "side": side,
                "count": filled,
                "price_cents": price_cents,
                "order_id": order_id,
                "bet_dollars": actual_cost,
                "fees_cents": actual_fees,
            })

        except Exception as e:
            logger.error("Order failed: {}", e)
            result["action"] = "order_failed"
            result["reason"] = str(e)
            self._log_activity(f"ORDER FAILED: {e}")
            return result

        self._last_traded_market = market.ticker

        # ── 8. Log trade ────────────────────────────────────────────
        record = TradeRecord(
            ts=time.time(),
            market_ticker=market.ticker,
            prob_up=0.5,
            confidence=0.0,
            direction=direction,
            bet_dollars=actual_cost,
            yes_price=price_cents / 100.0,
            order_id=order_id,
            fill_price_cents=price_cents,
            fill_count=filled,
            fees_cents=actual_fees,
            side=side,
            entry_type="entry",
        )
        self.trade_logger.append(record)

        self.last_cycle = result
        return result

    # ── settlement ──────────────────────────────────────────────────

    async def check_settlement(self, market_ticker: str) -> Optional[dict]:
        """Check if market has settled and compute accurate P&L."""
        try:
            mkt_data = self.client.get_market(market_ticker)
            market_info = mkt_data.get("market", mkt_data)
            status = market_info.get("status", "")

            if status not in ("settled", "finalized"):
                return None

            result_str = market_info.get("result", "")
            self._log_activity(f"Market {market_ticker} settled: {result_str}")

            records = self.trade_logger.read_all()
            entries = [
                r for r in records
                if r.market_ticker == market_ticker
                and r.outcome is None
                and r.entry_type == "entry"
            ]

            if not entries:
                self.current_position = None
                return {"settled": True, "result": result_str}

            rec = entries[-1]
            won = (
                (rec.direction == "up" and result_str == "yes")
                or (rec.direction == "down" and result_str == "no")
            )

            if won:
                payout_per = 1.0 - (rec.yes_price or 0.5)
                pnl = payout_per * rec.fill_count - (rec.fees_cents / 100.0)
            else:
                pnl = -(rec.fill_count * (rec.yes_price or 0.5)) - (rec.fees_cents / 100.0)

            pnl = round(pnl, 2)

            self.trade_logger.update_outcome(market_ticker, won, pnl)
            self.analyzer.record(won, pnl)

            if won:
                self.risk.record_win(abs(pnl))
            else:
                self.risk.record_loss(abs(pnl))

            self._log_activity(
                f"{'WIN' if won else 'LOSS'}: PnL=${pnl:+.2f} "
                f"(WR={self.analyzer.win_rate_all:.0%})"
            )

            self.current_position = None
            return {"settled": True, "result": result_str, "won": won, "pnl": pnl}

        except Exception as e:
            logger.warning("Settlement check failed: {}", e)
            return None
