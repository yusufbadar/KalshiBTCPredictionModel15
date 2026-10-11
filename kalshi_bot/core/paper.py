"""
Paper trading against *production* KXBTC15M market data (public endpoints,
no orders sent anywhere).

* ``ProductionFeed`` polls the production order book (~2 Hz) and public trade
  tape (~1 Hz) for the current contract.
* ``PaperBroker`` fills simulated orders conservatively:
    - IOC/FOK orders walk the most recent production book snapshot after a
      latency delay (only displayed size, never better than the book);
    - resting orders fill only when public trades print *through* the limit
      (strictly better for the taker), or at the limit once the displayed
      queue ahead of us at placement has traded away.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from loguru import logger

from kalshi_bot.api.rest import KalshiREST, fnum
from kalshi_bot.core.fees import maker_fee, taker_fee
from kalshi_bot.core.orderbook import OrderBook
from kalshi_bot.core.orders import Broker, OrderManager, WorkingOrder

PAPER_LATENCY_S = 0.15


class ProductionFeed:
    """Polls production books (and the primary contract's trade tape) for the
    tickers the engine is watching. Several books are fetched in one batch
    request so watching an hourly strike ladder stays within rate limits."""

    def __init__(self, public: KalshiREST):
        self.public = public
        self.books: dict[str, OrderBook] = {}
        self.tickers: list[str] = []
        self.ticker: str | None = None            # primary contract (trade tape)
        self.trade_cursor_ts: dict[str, float] = {}
        self.trade_listeners: list = []
        self.poll_interval = 0.5                  # engine tightens this near the close
        self._task: asyncio.Task | None = None

    def set_ticker(self, ticker: str | None) -> None:
        self.set_tickers([ticker] if ticker else [])

    def set_tickers(self, tickers: list[str]) -> None:
        self.tickers = list(dict.fromkeys(t for t in tickers if t))
        self.ticker = self.tickers[0] if self.tickers else None
        for tk in self.tickers:
            if tk not in self.books:
                self.books[tk] = OrderBook(tk)
                self.trade_cursor_ts[tk] = time.time()

    async def run(self) -> None:
        last_trades = 0.0
        while True:
            tks = list(self.tickers)
            if tks:
                try:
                    if len(tks) == 1:
                        ob = await self.public.get_orderbook(tks[0], depth=25)
                        self.books[tks[0]].apply_snapshot_rest(ob)
                    else:
                        for i in range(0, len(tks), 100):
                            d = await self.public.get("/markets/orderbooks", {"tickers": tks[i:i + 100]},
                                                      auth=False)
                            for item in d.get("orderbooks", []):
                                b = self.books.get(item.get("ticker"))
                                if b is not None:
                                    b.apply_snapshot_rest(item.get("orderbook_fp") or item.get("orderbook") or {})
                except Exception as e:  # noqa: BLE001
                    logger.debug("paper book poll failed: {}", e)
                if self.ticker and time.time() - last_trades > 1.0:
                    last_trades = time.time()
                    await self._poll_trades(self.ticker)
            await asyncio.sleep(self.poll_interval)

    async def _poll_trades(self, tk: str) -> None:
        since = self.trade_cursor_ts.get(tk, time.time())
        try:
            d = await self.public.get_trades(ticker=tk, limit=1000, min_ts=int(since) - 1)
        except Exception as e:  # noqa: BLE001
            logger.debug("paper trade poll failed: {}", e)
            return
        from datetime import datetime
        new = []
        for t in d.get("trades", []):
            ts = datetime.fromisoformat(t["created_time"].replace("Z", "+00:00")).timestamp()
            if ts <= since:
                continue
            new.append({"ts": ts, "yes_price": fnum(t["yes_price_dollars"]), "count": fnum(t["count_fp"]),
                        "taker_yes": (t.get("taker_outcome_side") or t.get("taker_side")) == "yes"})
        if new:
            new.sort(key=lambda r: r["ts"])
            self.trade_cursor_ts[tk] = new[-1]["ts"]
            for fn in self.trade_listeners:
                fn(tk, new)


@dataclass
class _Resting:
    order: WorkingOrder
    queue_ahead: float
    placed_ts: float = field(default_factory=time.time)


class PaperBroker(Broker):
    name = "paper"

    def __init__(self, feed: ProductionFeed, clock=None):
        self.feed = feed
        self.clock = clock or time.time          # exchange-time function
        self.om: OrderManager | None = None
        self.resting: dict[str, _Resting] = {}
        feed.trade_listeners.append(self._on_trades)

    async def place(self, o: WorkingOrder) -> dict:
        await asyncio.sleep(PAPER_LATENCY_S)
        book = self.feed.books.get(o.ticker)
        oid = "paper-" + uuid.uuid4().hex[:12]
        o.order_id = oid
        if book is None or not book.valid or time.time() - book.updated > 2.0:
            return {"order_id": oid, "fill_count": "0", "remaining_count": "0"}
        # crossing part
        filled, cost = 0.0, 0.0
        if o.outcome == "yes":
            levels = sorted(book.asks.items())
            for k, sz in levels:
                px = k / 10000
                if px > o.price + 1e-9 or filled >= o.count:
                    break
                take = min(sz, o.count - filled)
                filled += take
                cost += take * px
        else:
            levels = sorted(book.bids.items(), reverse=True)
            for k, sz in levels:
                px = 1 - k / 10000
                if px > o.price + 1e-9 or filled >= o.count:
                    break
                take = min(sz, o.count - filled)
                filled += take
                cost += take * px
        if o.post_only and filled > 0:
            return {"order_id": oid, "fill_count": "0", "remaining_count": "0"}
        if filled > 0:
            avg = cost / filled
            self._fill(o, filled, avg, taker=True)
        remaining = o.count - filled
        if remaining > 0 and o.tif == "good_till_canceled":
            ahead = (book.bid_size(o.price) if o.outcome == "yes" else book.ask_size(1 - o.price))
            self.resting[oid] = _Resting(o, queue_ahead=ahead, placed_ts=self.clock())
            return {"order_id": oid, "fill_count": f"{filled:.2f}", "remaining_count": f"{remaining:.2f}"}
        return {"order_id": oid, "fill_count": f"{filled:.2f}", "remaining_count": "0"}

    def _fill(self, o: WorkingOrder, count: float, price: float, taker: bool) -> None:
        if self.om is None:
            return
        fee = taker_fee(price, count) if taker else maker_fee(price, count)
        self.om.book_fill({"fill_id": "pf-" + uuid.uuid4().hex, "ticker": o.ticker, "order_id": o.order_id,
                           "outcome": o.outcome, "price": round(price, 6), "count": count,
                           "fee": round(fee, 6), "is_taker": taker, "ts": time.time()})

    def _on_trades(self, ticker: str, trades: list[dict]) -> None:
        for oid, r in list(self.resting.items()):
            o = r.order
            if o.ticker != ticker or not o.active:
                self.resting.pop(oid, None)
                continue
            for t in trades:
                if t["ts"] < r.placed_ts + PAPER_LATENCY_S:
                    continue
                # our YES bid is hit by taker sells of YES; our NO bid (= YES ask
                # at 1-price) is lifted by taker buys of YES
                if o.outcome == "yes" and not t["taker_yes"]:
                    through = t["yes_price"] < o.price - 1e-9
                    at = abs(t["yes_price"] - o.price) < 1e-9
                elif o.outcome == "no" and t["taker_yes"]:
                    through = t["yes_price"] > (1 - o.price) + 1e-9
                    at = abs(t["yes_price"] - (1 - o.price)) < 1e-9
                else:
                    continue
                qty = 0.0
                if through:
                    qty = min(o.remaining, t["count"])
                elif at:
                    r.queue_ahead -= t["count"]
                    if r.queue_ahead < 0:
                        qty = min(o.remaining, -r.queue_ahead)
                        r.queue_ahead = 0.0
                if qty > 0:
                    o.filled += qty
                    self._fill(o, qty, o.price, taker=False)
                    if o.remaining <= 1e-9:
                        o.status = "executed"
                        self.resting.pop(oid, None)
                        break

    async def cancel(self, o: WorkingOrder) -> None:
        self.resting.pop(o.order_id or "", None)

    async def cancel_all(self) -> None:
        self.resting.clear()
