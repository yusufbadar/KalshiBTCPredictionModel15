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
from kalshi_bot.kalshi.client import KalshiClient, _fp_dollars, order_filled_count
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

    def _record_entry_fill(
        self,
        market_ticker: str,
        side: str,
        direction: str,
        order_id: str,
        filled: int,
        fill_price_cents: int,
        fees_cents: float,
        result: dict,
    ) -> None:
        """Update position, trade log, and ``result`` from an executed buy."""
        actual_cost = round(filled * (fill_price_cents / 100.0), 2)
        actual_fees = round(fees_cents, 2)
        self._log_activity(
            f"FILLED: {filled}x {side.upper()} @ {fill_price_cents:.0f}c avg "
            f"(${actual_cost:.2f}, fees ~{actual_fees:.0f}c)"
        )
        self.current_position = {
            "side": side,
            "direction": direction,
            "entry_price": fill_price_cents,
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
            "price_cents": fill_price_cents,
            "order_id": order_id,
            "bet_dollars": actual_cost,
            "fees_cents": actual_fees,
        })
        self._last_traded_market = market_ticker
        record = TradeRecord(
            ts=time.time(),
            market_ticker=market_ticker,
            prob_up=0.5,
            confidence=0.0,
            direction=direction,
            bet_dollars=actual_cost,
            yes_price=fill_price_cents / 100.0,
            order_id=order_id,
            fill_price_cents=int(round(fill_price_cents)),
            fill_count=filled,
            fees_cents=actual_fees,
            side=side,
            entry_type="entry",
        )
        self.trade_logger.append(record)

    async def _reconcile_buy_execution(
        self,
        order_id: str,
        ticker: str,
        side: str,
        requested_count: int,
        limit_cents: int,
        order: dict,
    ) -> tuple[int, int, float]:
        """Return ``(filled, fill_price_cents, fees_cents)`` using order, fills, then position."""
        if not order_id or order_id == "unknown":
            return (0, 0, 0.0)

        def from_agg(agg: dict) -> tuple[int, int, float]:
            n = int(agg["count"])
            vwap = float(agg["vwap_dollars"])
            fc = max(0.0, float(agg["fees_dollars"]) * 100.0)
            pc = max(1, min(99, int(round(vwap * 100))))
            return (n, pc, fc)

        agg = self.client.aggregate_buy_fills_for_order(order_id, side)
        if agg:
            return from_agg(agg)

        filled = order_filled_count(order)
        await asyncio.sleep(0.35)
        agg = self.client.aggregate_buy_fills_for_order(order_id, side)
        if agg:
            return from_agg(agg)
        if filled > 0:
            for _ in range(8):
                agg = self.client.aggregate_buy_fills_for_order(order_id, side)
                if agg:
                    return from_agg(agg)
                await asyncio.sleep(0.25)
            fee_est = float(estimate_fee_cents(limit_cents) * filled)
            return (filled, limit_cents, fee_est)

        for _ in range(28):
            if filled <= 0:
                agg = self.client.aggregate_buy_fills_for_order(order_id, side)
                if agg:
                    return from_agg(agg)
            try:
                st = self.client.get_order(order_id)
                od = st.get("order", st)
                filled = order_filled_count(od)
            except Exception as e:
                logger.warning("reconcile get_order: {}", e)
            if filled > 0:
                agg = self.client.aggregate_buy_fills_for_order(order_id, side)
                if agg:
                    return from_agg(agg)
                fee_est = estimate_fee_cents(limit_cents) * filled
                return (filled, limit_cents, fee_est)
            await asyncio.sleep(0.45)

        agg = self.client.aggregate_buy_fills_for_order(order_id, side)
        if agg:
            return from_agg(agg)

        row = self.client.get_market_position_row(ticker)
        if row:
            pc = self.client.position_contracts_for_side(row, side)
            if pc > 0:
                exp = _fp_dollars(row.get("market_exposure_dollars"))
                vwap = exp / pc if pc else 0.0
                price_c = max(1, min(99, int(round(vwap * 100))))
                fees = _fp_dollars(row.get("fees_paid_dollars")) * 100.0
                if fees <= 0:
                    fees = float(estimate_fee_cents(price_c) * pc)
                return (pc, price_c, fees)

        return (0, 0, 0.0)

    async def repair_entry_from_exchange(
        self,
        ticker: str,
        side: str,
        direction: str,
        order_id: str | None,
    ) -> bool:
        """If we have no local position but the exchange shows a fill, sync state."""
        if self.current_position:
            return False
        if order_id and order_id != "unknown":
            for r in self.trade_logger.read_all():
                if r.order_id == order_id and r.entry_type == "entry":
                    return False
            try:
                agg = self.client.aggregate_buy_fills_for_order(order_id, side)
                if agg:
                    n = int(agg["count"])
                    vwap = float(agg["vwap_dollars"])
                    fc = max(0.0, float(agg["fees_dollars"]) * 100.0)
                    pc = max(1, min(99, int(round(vwap * 100))))
                    result = dict(self.last_cycle)
                    result["market"] = ticker
                    self._record_entry_fill(
                        ticker, side, direction, order_id, n, pc, fc, result
                    )
                    self.last_cycle = result
                    self._log_activity(
                        f"Repaired entry from fills API (order {order_id[:12]}…)"
                    )
                    return True
            except Exception as e:
                logger.warning("repair fills: {}", e)

        try:
            row = self.client.get_market_position_row(ticker)
            if not row:
                return False
            pc = self.client.position_contracts_for_side(row, side)
            if pc < 1:
                return False
            for r in reversed(self.trade_logger.read_all()):
                if r.market_ticker != ticker or r.entry_type != "entry":
                    continue
                if r.outcome is None and r.side == side and r.fill_count == pc:
                    return False
            exp = _fp_dollars(row.get("market_exposure_dollars"))
            vwap = exp / pc if pc else 0.0
            price_c = max(1, min(99, int(round(vwap * 100))))
            fees = _fp_dollars(row.get("fees_paid_dollars")) * 100.0
            if fees <= 0:
                fees = float(estimate_fee_cents(price_c) * pc)
            oid = order_id or f"sync-{int(time.time())}"
            result = dict(self.last_cycle)
            result["market"] = ticker
            self._record_entry_fill(
                ticker, side, direction, oid, pc, price_c, fees, result
            )
            self.last_cycle = result
            self._log_activity(
                f"Repaired entry from portfolio position ({pc}x {side.upper()})"
            )
            return True
        except Exception as e:
            logger.warning("repair position: {}", e)
            return False

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
            self.last_cycle = result
            return result

        result["market"] = market.ticker

        if market.ticker == self._last_traded_market:
            result["reason"] = "already traded this market"
            self.last_cycle = result
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
            self.last_cycle = result
            return result

        self._log_activity(
            f"Entry window: {market.ticker} ({market.seconds_until_close:.0f}s left)"
        )

        # ── 1. Fetch 5m candles ─────────────────────────────────────
        candles_5m = await self.binance.get_klines("5m", 50)
        if len(candles_5m) < 22:
            result["reason"] = "not enough 5m candle data"
            self.last_cycle = result
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
            self.last_cycle = result
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

        # ── 5–6. Size from balance % (orderbook fetched in size_buy_for_budget) ──
        balance = self.client.get_balance_dollars()
        risk_decision = self.risk.check(balance, 2.0)
        if not risk_decision.allowed:
            result["action"] = "skip"
            result["reason"] = risk_decision.reason
            self.last_cycle = result
            return result

        bet_dollars = risk_decision.bet_dollars
        pct = TRADING.bet_fraction * 100.0

        sized = self.kalshi_ob.size_buy_for_budget(
            market.ticker, side, bet_dollars
        )
        if not sized:
            result["action"] = "skip"
            result["reason"] = "no orderbook liquidity within risk budget"
            self._log_activity(
                "FOK skip: cannot place at least 1 contract within book + budget"
            )
            self.last_cycle = result
            return result

        count, price_cents = sized
        if count < 1:
            result["action"] = "skip"
            result["reason"] = "no contracts after book sizing"
            self.last_cycle = result
            return result

        notion_est = round(count * (price_cents / 100.0), 2)
        self._log_activity(
            f"Risk {pct:.1f}% → ${bet_dollars:.2f} / ${balance:.2f} balance "
            f"→ {count}x {side.upper()} FOK ≤{price_cents}c (~${notion_est} max notional)"
        )

        # ── 7. Place FOK at swept limit (full size or no trade) ────
        order_resp = None
        order_id = None
        for attempt in range(3):
            try:
                order_resp = self.client.create_order(
                    ticker=market.ticker,
                    side=side,
                    action="buy",
                    count=count,
                    yes_price=price_cents if side == "yes" else None,
                    no_price=price_cents if side == "no" else None,
                    time_in_force="fill_or_kill",
                )
                order = order_resp.get("order", {})
                order_id = order.get("order_id", "unknown")
                break
            except Exception as e:
                self._log_activity(
                    f"Order attempt {attempt+1}/3 failed: {e}"
                )
                if attempt < 2:
                    await asyncio.sleep(3)
                else:
                    result["action"] = "order_failed"
                    result["reason"] = str(e)
                    self.last_cycle = result
                    return result

        self._log_activity(
            f"Order submitted: {count}x {side.upper()} FOK limit {price_cents}c — reconciling…"
        )
        filled, fill_px, fees_cents = await self._reconcile_buy_execution(
            order_id,
            market.ticker,
            side,
            count,
            price_cents,
            order,
        )

        if filled == 0:
            oid_short = order_id if len(order_id) <= 14 else f"{order_id[:12]}…"
            self._log_activity(
                f"No fill confirmed (requested {count} @ {price_cents}c, order {oid_short})"
            )
            result["action"] = "no_fill"
            result["reason"] = "exchange reported no fill (order/fills/position)"
            result["order_id"] = order_id
            self.last_cycle = result
            return result

        if filled < count:
            self._log_activity(
                f"Partial fill confirmed: {filled}/{count} (avg {fill_px}c)"
            )

        self._record_entry_fill(
            market.ticker,
            side,
            direction,
            order_id,
            filled,
            fill_px,
            fees_cents,
            result,
        )
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
