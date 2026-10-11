"""
Order and position management.

Every order is expressed as "buy <outcome> at <price for that outcome>":
buying YES at p is a V2 ``bid`` at p; buying NO at q is a V2 ``ask`` at 1 - q
(selling YES). Kalshi nets YES against NO, and because every YES+NO pair is
worth exactly $1, P&L is  sum over fills of (payoff(outcome) - price - fee).

Fill accounting uses a single source of truth - individual fill records
(WebSocket ``fill`` channel, backfilled from REST ``/portfolio/fills``),
de-duplicated by trade id. Order responses only update order state, which is
what risk/exposure checks use between an order's acknowledgement and the
arrival of its fill records.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from loguru import logger

from kalshi_bot.api.rest import KalshiError, KalshiREST, fnum
from kalshi_bot.core.fees import taker_fee
from kalshi_bot.core.ledger import Ledger


@dataclass
class WorkingOrder:
    client_order_id: str
    ticker: str
    outcome: str                 # 'yes' | 'no'
    price: float                 # limit price of the outcome bought
    count: float
    tif: str
    post_only: bool
    order_id: str | None = None
    filled: float = 0.0          # from order acks/updates
    status: str = "pending"      # pending | resting | executed | canceled | rejected
    ts: float = field(default_factory=time.time)
    reason: str = ""

    @property
    def remaining(self) -> float:
        return max(0.0, self.count - self.filled)

    @property
    def active(self) -> bool:
        return self.status in ("pending", "resting")

    @property
    def book_side(self) -> str:
        return "bid" if self.outcome == "yes" else "ask"

    @property
    def yes_price(self) -> float:
        return self.price if self.outcome == "yes" else round(1.0 - self.price, 4)


@dataclass
class Position:
    ticker: str
    yes: float = 0.0
    no: float = 0.0
    cost: float = 0.0
    fees: float = 0.0

    @property
    def net_yes(self) -> float:
        return self.yes - self.no

    def pnl_if(self, yes_wins: bool) -> float:
        return (self.yes if yes_wins else self.no) - self.cost - self.fees


class Broker:
    """Interface implemented by the Kalshi REST broker and the paper broker."""
    name = "base"

    async def place(self, o: WorkingOrder) -> dict: ...
    async def cancel(self, o: WorkingOrder) -> None: ...
    async def cancel_all(self) -> None: ...
    async def fills(self, ticker: str) -> list[dict]:
        return []


class KalshiBroker(Broker):
    name = "kalshi"

    def __init__(self, rest: KalshiREST):
        self.rest = rest

    async def place(self, o: WorkingOrder) -> dict:
        return await self.rest.create_order(o.ticker, o.book_side, o.yes_price, o.count, tif=o.tif,
                                            post_only=o.post_only, client_order_id=o.client_order_id)

    async def cancel(self, o: WorkingOrder) -> None:
        if o.order_id:
            await self.rest.cancel_order(o.order_id, o.ticker)

    async def cancel_all(self) -> None:
        await self.rest.cancel_all()

    async def fills(self, ticker: str) -> list[dict]:
        out, cursor = [], None
        for _ in range(10):
            params = {"ticker": ticker, "limit": 200}
            if cursor:
                params["cursor"] = cursor
            d = await self.rest.fills(**params)
            out += d.get("fills", [])
            cursor = d.get("cursor")
            if not cursor:
                break
        return out


def fill_from_api(f: dict) -> dict:
    """Normalise a REST fill or WS fill message into our fill record."""
    outcome = f.get("outcome_side") or f.get("purchased_side") or f.get("side")
    yes_px = fnum(f.get("yes_price_dollars"))
    if f.get("no_price_dollars") is not None and outcome == "no":
        price = fnum(f.get("no_price_dollars"))
    else:
        price = yes_px if outcome == "yes" else round(1 - yes_px, 6)
    ts = f.get("ts_ms")
    if ts is not None:
        ts = fnum(ts) / 1000
    elif f.get("created_time"):
        from datetime import datetime
        ts = datetime.fromisoformat(f["created_time"].replace("Z", "+00:00")).timestamp()
    else:
        ts = time.time()
    return {"fill_id": f.get("trade_id") or f.get("fill_id"),
            "ticker": f.get("market_ticker") or f.get("ticker"),
            "order_id": f.get("order_id"), "outcome": outcome, "price": price,
            "count": fnum(f.get("count_fp", f.get("count"))), "fee": fnum(f.get("fee_cost")),
            "is_taker": bool(f.get("is_taker")), "ts": ts}


class OrderManager:
    def __init__(self, broker: Broker, ledger: Ledger, env: str):
        self.broker = broker
        self.ledger = ledger
        self.env = env
        self.orders: dict[str, WorkingOrder] = {}
        self.by_order_id: dict[str, str] = {}
        self.positions: dict[str, Position] = {}
        self.seen_fills: set[str] = set()
        self.listeners: list = []          # callables(event: str, payload: dict)

    # ---------------------------------------------------------------- utils
    def load_open_positions(self) -> int:
        """Rebuild positions for markets that have fills but no settlement yet
        (e.g. after a restart mid-market) so exposure limits stay correct."""
        n = 0
        for tk in self.ledger.unsettled_tickers(self.env):
            p = self.position(tk)
            p.yes = p.no = p.cost = p.fees = 0.0
            for f in self.ledger.fills_for(self.env, tk):
                if f["outcome"] == "yes":
                    p.yes += f["count"]
                else:
                    p.no += f["count"]
                p.cost += f["price"] * f["count"]
                p.fees += f["fee"]
            n += 1
        # fill ids already booked must not be booked twice by a reconcile
        for fid in self.ledger.fill_ids(self.env):
            self.seen_fills.add(fid)
        return n

    def position(self, ticker: str) -> Position:
        if ticker not in self.positions:
            self.positions[ticker] = Position(ticker)
        return self.positions[ticker]

    def working(self, ticker: str | None = None) -> list[WorkingOrder]:
        return [o for o in self.orders.values() if o.active and (ticker is None or o.ticker == ticker)]

    def exposure(self, ticker: str) -> tuple[float, float]:
        """(yes, no) contracts owned or already filled per order acks - whichever
        is larger, so a fill is never missed while its record is in flight."""
        p = self.positions.get(ticker)
        ack_yes = sum(o.filled for o in self.orders.values() if o.ticker == ticker and o.outcome == "yes")
        ack_no = sum(o.filled for o in self.orders.values() if o.ticker == ticker and o.outcome == "no")
        return max(p.yes if p else 0.0, ack_yes), max(p.no if p else 0.0, ack_no)

    def committed_cost(self, ticker: str) -> float:
        """Dollars spent plus dollars reserved by resting orders on this market.
        Fills acknowledged by an order response count immediately (valued at
        the limit price plus its taker fee, an upper bound) even before their
        fill records arrive."""
        p = self.positions.get(ticker)
        booked = (p.cost + p.fees) if p else 0.0
        acked = sum(o.filled * (o.price + taker_fee(o.price)) for o in self.orders.values() if o.ticker == ticker)
        resting = sum(o.remaining * (o.price + taker_fee(o.price)) for o in self.working(ticker))
        return max(booked, acked) + resting

    def _emit(self, event: str, payload: dict) -> None:
        for fn in self.listeners:
            try:
                fn(event, payload)
            except Exception:  # noqa: BLE001
                pass

    # --------------------------------------------------------------- orders
    async def buy(self, ticker: str, outcome: str, price: float, count: float, *,
                  tif: str = "immediate_or_cancel", post_only: bool = False,
                  reason: str = "") -> WorkingOrder:
        o = WorkingOrder(client_order_id=str(uuid.uuid4()), ticker=ticker, outcome=outcome,
                         price=round(price, 4), count=round(count, 2), tif=tif,
                         post_only=post_only, reason=reason)
        self.orders[o.client_order_id] = o
        self.ledger.order(self.env, {"client_order_id": o.client_order_id, "ticker": ticker,
                                     "outcome": outcome, "book_side": o.book_side, "price": o.price,
                                     "count": o.count, "tif": tif, "post_only": post_only,
                                     "status": "pending", "reason": reason})
        try:
            resp = await self.broker.place(o)
        except KalshiError as e:
            o.status = "rejected"
            self.ledger.order_status(o.client_order_id, "rejected")
            self.ledger.event(self.env, "warn", f"order rejected {ticker} {outcome}@{price:.3f}: {e.message}")
            self._emit("reject", {"ticker": ticker, "error": str(e)})
            return o
        o.order_id = resp.get("order_id")
        if o.order_id:
            self.by_order_id[o.order_id] = o.client_order_id
        o.filled = max(o.filled, fnum(resp.get("fill_count")))
        remaining = fnum(resp.get("remaining_count"), o.count - o.filled)
        if o.tif in ("immediate_or_cancel", "fill_or_kill") or remaining <= 1e-9:
            o.status = "executed" if o.filled > 0 else "canceled"
        elif o.status == "pending":
            o.status = "resting"
        self.ledger.order_status(o.client_order_id, o.status, o.filled, o.order_id)
        return o

    async def cancel(self, o: WorkingOrder) -> None:
        if not o.active:
            return
        try:
            await self.broker.cancel(o)
            o.status = "canceled"
        except KalshiError as e:
            if e.status == 404:
                o.status = "canceled"
            else:
                logger.warning("cancel failed {}: {}", o.order_id, e)
                return
        self.ledger.order_status(o.client_order_id, o.status, o.filled)

    async def cancel_ticker(self, ticker: str) -> None:
        await asyncio.gather(*(self.cancel(o) for o in self.working(ticker)))

    async def cancel_everything(self) -> None:
        try:
            await self.broker.cancel_all()
        except Exception as e:  # noqa: BLE001
            logger.warning("cancel_all failed: {}", e)
        for o in self.working():
            o.status = "canceled"
            self.ledger.order_status(o.client_order_id, "canceled", o.filled)

    # ---------------------------------------------------------------- fills
    def book_fill(self, f: dict) -> None:
        fid = f.get("fill_id")
        if not fid or fid in self.seen_fills:
            return
        self.seen_fills.add(fid)
        if not self.ledger.fill(self.env, f):
            return
        p = self.position(f["ticker"])
        if f["outcome"] == "yes":
            p.yes += f["count"]
        else:
            p.no += f["count"]
        p.cost += f["price"] * f["count"]
        p.fees += f["fee"]
        self._emit("fill", f)

    def on_ws_fill(self, msg: dict) -> None:
        # positions come from fill records; cumulative order fill counts come
        # from order acks / user_order updates (see exposure())
        self.book_fill(fill_from_api(msg.get("msg", msg)))

    def on_ws_order(self, msg: dict) -> None:
        m = msg.get("msg", msg)
        cid = m.get("client_order_id") or self.by_order_id.get(m.get("order_id", ""))
        if not cid or cid not in self.orders:
            return
        o = self.orders[cid]
        if m.get("order_id"):
            o.order_id = m["order_id"]
            self.by_order_id[o.order_id] = cid
        if m.get("fill_count_fp") is not None:
            o.filled = max(o.filled, fnum(m.get("fill_count_fp")))
        st = m.get("status")
        if st in ("resting", "canceled", "executed"):
            o.status = st
        self.ledger.order_status(cid, o.status, o.filled, o.order_id)

    async def reconcile_fills(self, ticker: str) -> None:
        try:
            for f in await self.broker.fills(ticker):
                self.book_fill(fill_from_api(f))
        except Exception as e:  # noqa: BLE001
            logger.debug("fill reconcile failed: {}", e)

    # ----------------------------------------------------------- settlement
    def settle(self, ticker: str, result: str) -> dict | None:
        fills = self.ledger.fills_for(self.env, ticker)
        if not fills:
            self.positions.pop(ticker, None)
            return None
        yes_q = sum(f["count"] for f in fills if f["outcome"] == "yes")
        no_q = sum(f["count"] for f in fills if f["outcome"] == "no")
        cost = sum(f["price"] * f["count"] for f in fills)
        fees = sum(f["fee"] for f in fills)
        payout = yes_q if result == "yes" else no_q
        s = {"ticker": ticker, "ts": time.time(), "result": result, "yes_qty": yes_q, "no_qty": no_q,
             "cost": round(cost, 6), "fees": round(fees, 6), "payout": payout,
             "pnl": round(payout - cost - fees, 6)}
        self.ledger.settlement(self.env, s)
        self.positions.pop(ticker, None)
        self._emit("settle", s)
        return s
