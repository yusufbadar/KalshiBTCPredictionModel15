"""
Live trading engine (demo / paper / live).

One asyncio task graph:
    ws.run()          settlement-index feeds (+ order books, fills, orders on demo/live)
    feed.run()        production book/trade polling (paper only)
    _loop()           decision loop (10 Hz near a close, 4 Hz otherwise)
    _housekeeping()   balances, settlements, reconciliation, equity samples

Every traded coin has its own ``AssetState`` (index tracker, volatility
estimate, 15-minute market clock, optional hourly strike ladder). All of them
share one order manager, one risk manager and one pool of cash.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime

import numpy as np
from loguru import logger

from kalshi_bot.api.auth import Signer
from kalshi_bot.api.rest import KalshiError, KalshiREST, fnum
from kalshi_bot.api.ws import KalshiWS
from kalshi_bot.config import DB_PATH, PUBLIC_REST, STORAGE_DIR, AssetSpec, Settings, parse_assets, set_settings
from kalshi_bot.core.index import IndexTracker
from kalshi_bot.core.ledger import Ledger
from kalshi_bot.core.lock import InstanceLock
from kalshi_bot.core.markets import Contract, HourlyLadder, MarketClock
from kalshi_bot.core.orderbook import OrderBook
from kalshi_bot.core.orders import KalshiBroker, OrderManager
from kalshi_bot.core.paper import PaperBroker, ProductionFeed
from kalshi_bot.core.risk import RiskManager, utc_day_start
from kalshi_bot.model.fair_value import fair_yes, window_bounds
from kalshi_bot.model.vol import EwmaVol
from kalshi_bot.research.dataset import reconstruct_prints
from kalshi_bot.strategy.base import Intent, TickState
from kalshi_bot.strategy.lockin import LockInStrategy

CRYPTO_SHARD = 2
VOL_INIT_REL = 7e-5          # unseeded sigma: 0.7 bp of price per sqrt(s), ~3x a normal BTC level


class AssetState:
    """Everything the engine tracks for one coin."""

    def __init__(self, spec: AssetSpec, halflife_s: float):
        self.spec = spec
        self.symbol = spec.symbol
        self.index = IndexTracker(decimals=spec.decimals)
        self.vol = EwmaVol(halflife_s=halflife_s, floor=0.0, init_rel=VOL_INIT_REL)
        self.clock: MarketClock | None = None
        self.hourly: HourlyLadder | None = None
        self.watched_hourly: list[Contract] = []
        self.hourly_view: list[dict] = []
        self.last_decision: dict = {}
        self.history: deque = deque(maxlen=1200)      # per-second snapshots for charts


class Engine:
    def __init__(self, settings: Settings):
        self.s = settings
        self.env = settings.env
        self.ledger = Ledger(DB_PATH)
        specs = parse_assets(settings.strategy.assets)
        self.assets: dict[str, AssetState] = {
            sp.symbol: AssetState(sp, settings.strategy.vol_halflife_s) for sp in specs}
        self.primary: AssetState = next(iter(self.assets.values()))
        self._by_index: dict[str, AssetState] = {a.spec.index_id: a for a in self.assets.values()}
        self.risk = RiskManager(settings.risk)
        self.strategy = LockInStrategy(settings.strategy)
        self.running = False
        self.status = "stopped"
        self.error = ""
        self.balance = 0.0
        self.balance_by_shard: dict[int, float] = {}
        self.books: dict[str, OrderBook] = {}
        self.logs: deque = deque(maxlen=300)
        self.settled: set[str] = set()
        self._tasks: list[asyncio.Task] = []
        self.rest: KalshiREST | None = None
        self.public = KalshiREST(PUBLIC_REST, None)
        self.ws: KalshiWS | None = None
        self.feed: ProductionFeed | None = None
        self.om: OrderManager | None = None
        self.started_at = 0.0
        self.trading_active = True           # exchange-wide trading status
        self._rejects: deque = deque(maxlen=5)
        self._pause_until = 0.0
        self._book_tickers: tuple = ()
        self._snap_requested: set[str] = set()
        self._close_of: dict[str, int] = {}          # ticker -> close time of every contract priced
        self._spend: deque = deque()                 # (time, dollars) of fills not yet in the REST balance
        self._announced: set[str] = set()
        self._drawdown = 0.0
        self._dd_logged = False
        self._topup_fails = 0
        self._lock = InstanceLock(STORAGE_DIR / f"engine_{self.env}.lock")

    # ------------------------------------------- single-coin compatibility
    # The first configured coin is the "primary" one: the dashboard's main
    # panel and older call sites address it through these attributes.
    @property
    def index(self) -> IndexTracker:
        return self.primary.index

    @property
    def vol(self) -> EwmaVol:
        return self.primary.vol

    @property
    def clock(self):
        return self.primary.clock

    @clock.setter
    def clock(self, v) -> None:
        self.primary.clock = v

    @property
    def hourly(self):
        return self.primary.hourly

    @hourly.setter
    def hourly(self, v) -> None:
        self.primary.hourly = v

    @property
    def watched_hourly(self) -> list[Contract]:
        return self.primary.watched_hourly

    @property
    def hourly_view(self) -> list[dict]:
        return self.primary.hourly_view

    @property
    def last_decision(self) -> dict:
        return self.primary.last_decision

    @property
    def history(self) -> deque:
        return self.primary.history

    # ------------------------------------------------------------ logging
    def log(self, msg: str, level: str = "info") -> None:
        self.logs.append({"ts": time.time(), "level": level, "msg": msg})
        self.ledger.event(self.env, level, msg)
        getattr(logger, "warning" if level == "warn" else level if level in ("info", "error") else "info")(msg)

    # ---------------------------------------------------------- lifecycle
    async def start(self) -> None:
        if self.running:
            return
        s = self.s
        key_id, key_path = s.creds()
        if not key_id or not key_path:
            which = "live (production)" if s.trading_env == "live" else "demo"
            raise RuntimeError(f"Add your {which} API key id and private key in Settings first.")
        if not self._lock.acquire():
            raise RuntimeError(f"Another copy of the bot is already running in {self.env.upper()} mode on this "
                               f"PC. Stop that one first - two bots on one account would double every position.")
        try:
            signer = Signer(key_id, key_path)
            self.rest = KalshiREST(s.rest_url(), signer)
        except Exception:
            self._lock.release()
            raise
        # validate credentials before anything else
        try:
            asked = time.time()
            bal = await self.rest.balance()
        except KalshiError as e:
            await self.rest.close()
            self._lock.release()
            raise RuntimeError(f"Kalshi rejected the credentials ({e.status}): {e.message}") from e
        except Exception:
            await self.rest.close()
            self._lock.release()
            raise
        self._apply_balance(bal, asked)
        self.ws = KalshiWS(s.ws_url(), signer)
        self.ws.on("cfbenchmarks_value", self._on_index)
        self.ws.on("cfbenchmarks_value_5hz", self._on_index)
        ids = [a.spec.index_id for a in self.assets.values()]
        await self.ws.subscribe("brti", ["cfbenchmarks_value"], index_ids=ids)
        await self.ws.subscribe("brti5", ["cfbenchmarks_value_5hz"], index_ids=ids)
        paper = self.env == "paper"
        market_rest = self.public if paper else self.rest
        if paper:
            self.feed = ProductionFeed(self.public)
            self.books = self.feed.books
            broker = PaperBroker(self.feed, clock=self.index.exchange_time)
            self.om = OrderManager(broker, self.ledger, self.env)
            broker.om = self.om
        else:
            self.om = OrderManager(KalshiBroker(self.rest), self.ledger, self.env)
            self.ws.on("orderbook_snapshot", self._on_book_snapshot)
            self.ws.on("orderbook_delta", self._on_book_delta)
            self.ws.on("fill", lambda m: self.om.on_ws_fill(m))
            self.ws.on("user_order", lambda m: self.om.on_ws_order(m))
            self.ws.on_reconnect(self._on_ws_reconnect)
            await self.ws.subscribe("fills", ["fill"])
            await self.ws.subscribe("orders", ["user_orders"])
        for a in self.assets.values():
            a.clock = MarketClock(market_rest, authed=not paper, series=a.spec.series15,
                                  decimals=a.spec.decimals, asset=a.symbol)
            if s.strategy.trade_hourly and a.spec.hourly:
                a.hourly = HourlyLadder(market_rest, authed=not paper, series=a.spec.hourly,
                                        decimals=a.spec.decimals, asset=a.symbol)
        self.om.listeners.append(self._on_om_event)
        restored = self.om.load_open_positions()
        if restored:
            self.log(f"Restored {restored} open position(s) from the ledger")
        await asyncio.gather(*(self._seed_index_history(a) for a in self.assets.values()))
        self._drawdown = self.ledger.realised_drawdown(self.env, self.s.dd_reset_ts.get(self.env, 0.0))
        if self.s.kill_state.get(self.env):
            self.risk.kill(self.s.kill_state[self.env])
            self.log("Kill switch is still engaged from before the restart - no orders until you release it", "warn")
        self.running = True
        self.started_at = time.time()
        self.status = "running"
        self.error = ""
        self._tasks = [asyncio.create_task(self.ws.run(), name="ws"),
                       asyncio.create_task(self._loop(), name="loop"),
                       asyncio.create_task(self._housekeeping(), name="house")]
        if self.feed:
            self._tasks.append(asyncio.create_task(self.feed.run(), name="feed"))
        eq = self.equity()
        self.log(f"Engine started in {self.env.upper()} mode - {', '.join(self.assets)} - account "
                 f"${eq:,.2f}, free cash ${self.free_cash():,.2f}; limits ${self.risk.market_cap(eq):,.2f} "
                 f"per contract, ${self.risk.close_cap(eq):,.2f} per settlement moment")

    async def stop(self, reason: str = "stopped by user") -> None:
        if not self.running:
            return
        self.running = False
        self.status = "stopping"
        try:
            if self.om:
                await self.om.cancel_everything()
        except Exception as e:  # noqa: BLE001
            self.log(f"cancel on stop failed: {e}", "warn")
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        if self.ws:
            await self.ws.close()
        if self.rest:
            await self.rest.close()
        self._lock.release()
        self.status = "stopped"
        self.log(f"Engine stopped ({reason})")

    # ------------------------------------------------------------ feeds
    def _on_index(self, msg: dict) -> None:
        m = msg.get("msg", {})
        try:
            d = json.loads(m.get("data", "{}"))
            v = float(d.get("value") or m.get("value_usd"))
            t = int(d.get("time") or m.get("source_ts_ms"))
        except (TypeError, ValueError):
            return
        a = self._by_index.get(m.get("index_id") or d.get("id") or self.primary.spec.index_id)
        if a is not None:
            a.index.on_tick(t, v, m.get("received_at"))

    def _book_age(self, book: OrderBook | None) -> float:
        """How stale our view of a book may be. REST-polled (paper) books age
        from their last poll. WebSocket books are pushed on every change, so a
        quiet book is still current while the socket is alive and in sequence:
        they age from the last message on the connection."""
        if book is None or not book.valid:
            return 1e9
        if self.env == "paper" or self.ws is None:
            return time.time() - book.updated if book.updated else 1e9
        if not self.ws.connected.is_set():
            return 1e9
        return time.time() - max(self.ws.last_msg_ts, book.updated)

    def _on_book_snapshot(self, msg: dict) -> None:
        m = msg["msg"]
        b = self.books.setdefault(m["market_ticker"], OrderBook(m["market_ticker"]))
        b.apply_snapshot_ws(m)
        b.seq = msg.get("seq", 0)

    def _on_book_delta(self, msg: dict) -> None:
        m = msg["msg"]
        b = self.books.get(m["market_ticker"])
        if b is None or not b.valid:
            return
        seq = msg.get("seq", 0)
        if b.seq and seq and seq != b.seq + 1:
            b.valid = False
            asyncio.create_task(self.ws.update("book", "get_snapshot", market_tickers=[m["market_ticker"]]))
            return
        b.seq = seq
        b.apply_delta_ws(m)

    async def _on_ws_reconnect(self) -> None:
        for b in self.books.values():
            b.reset()
        self._snap_requested.clear()
        self.log("WebSocket reconnected - order books resynced", "warn")

    def _on_om_event(self, event: str, p: dict) -> None:
        if event == "fill":
            self._rejects.clear()
            if self.env != "paper":
                self._spend.append((time.time(), p["price"] * p["count"] + p["fee"]))
            self.log(f"FILL {p['ticker'][-11:]} {p['outcome'].upper()} {p['count']:.2f} @ {p['price']:.3f}"
                     f"{' (taker)' if p['is_taker'] else ' (maker)'} fee ${p['fee']:.4f}")
        elif event == "settle":
            self.log(f"SETTLED {p['ticker']} -> {p['result'].upper()}  P&L ${p['pnl']:+.2f}")
            self._drawdown = self.ledger.realised_drawdown(self.env, self.s.dd_reset_ts.get(self.env, 0.0))
        elif event == "reject":
            self.log(f"order rejected: {p['error']}", "warn")
            now = time.time()
            if "insufficient" in str(p.get("error", "")).lower():
                # our view of the cash was stale: stop for a moment and re-read it
                self._pause_until = max(self._pause_until, now + 2.0)
                if self.running and self.env != "paper":
                    asyncio.create_task(self._refresh_balance())
                return
            self._rejects.append(now)
            if len(self._rejects) == self._rejects.maxlen and now - self._rejects[0] < 60:
                self._pause_until = now + 60
                self._rejects.clear()
                self.log("5 rejections in a row - pausing new orders for 60 s", "warn")

    async def _seed_index_history(self, a: AssetState) -> None:
        """Seed the last hour of index prints from Kalshi's public event series
        so the volatility estimate is warm at start-up."""
        async def fetch(path, params=None):
            # the trading exchange's own endpoint first (separate rate budget),
            # then the public production endpoint
            last = None
            for attempt in range(6):
                for client, authed in ((self.rest, True), (self.public, False)):
                    if client is None:
                        continue
                    try:
                        return await client.get(path, params, auth=authed)
                    except Exception as e:  # noqa: BLE001
                        last = e
                await asyncio.sleep(1.0 + attempt)
            raise last or RuntimeError("seed fetch failed")

        try:
            d = await fetch("/markets", {"series_ticker": a.spec.series15, "status": "open", "limit": 5})
            ms = sorted((m for m in d.get("markets", []) if m.get("close_time")), key=lambda m: m["close_time"])
            ms = [m for m in ms if datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp()
                  > time.time()]
            if not ms:
                return
            ev = ms[0]["event_ticker"]
            ld = await fetch(f"/live_data/events/{ev}")
            ts = ld["live_data"]["details"]["timeseries"]
            avg = np.array([p["v"] for p in ts], float)
            secs = [p["t"] // 1000 for p in ts]
            if len(avg) >= 600 and all(b - a_ == 1 for a_, b in zip(secs, secs[1:])):
                x = reconstruct_prints(avg)
                for s_, v in zip(secs, x):
                    if s_ not in a.index.prints:
                        a.index.prints[s_] = float(v)
                        a.index.order.append(s_)
                a.vol.warm_start(a.index.prints, secs[-1], span_s=len(secs) - 10)
                self.log(f"Seeded {len(secs)} s of {a.spec.index_id} history, sigma "
                         f"{1e4 * a.vol.sigma / max(float(x[-1]), 1e-12):.2f} bp/sqrt(s)")
        except Exception as e:  # noqa: BLE001
            self.log(f"could not seed {a.symbol} index history: {e}", "warn")

    # ------------------------------------------------------------- money
    def _apply_balance(self, bal: dict, asked_at: float | None = None) -> None:
        self.balance = fnum(bal.get("balance_dollars"), fnum(bal.get("balance")) / 100)
        self.balance_by_shard = {int(b.get("exchange_index", 0)): fnum(b.get("balance"))
                                 for b in bal.get("balance_breakdown", [])}
        # fills booked before this balance was requested are already inside it
        cutoff = (asked_at if asked_at is not None else time.time()) - 0.5
        while self._spend and self._spend[0][0] < cutoff:
            self._spend.popleft()

    async def _refresh_balance(self) -> None:
        if self.rest is None or self.env == "paper":
            return
        try:
            asked = time.time()
            self._apply_balance(await self.rest.balance(), asked)
        except Exception as e:  # noqa: BLE001
            logger.debug("balance refresh failed: {}", e)

    def _pending_spend(self) -> float:
        return sum(amt for _, amt in self._spend)

    def free_cash(self) -> float:
        """Cash that can be spent on crypto contracts right now: the last
        balance Kalshi reported minus fills since it was requested."""
        if self.env == "paper":
            return self.paper_balance()
        return self.balance_by_shard.get(CRYPTO_SHARD, self.balance) - self._pending_spend()

    def trading_balance(self) -> float:
        return self.free_cash()

    def paper_balance(self) -> float:
        return self.s.paper_start + self.ledger.pnl_since(self.env, 0) - self._open_cost()

    def _open_cost(self) -> float:
        if not self.om:
            return 0.0
        return sum(p.cost + p.fees for p in self.om.positions.values())

    def equity(self) -> float:
        """Account value: all cash plus the money sitting in open positions."""
        if self.env == "paper":
            return self.s.paper_start + self.ledger.pnl_since(self.env, 0)
        cash = sum(self.balance_by_shard.values()) if self.balance_by_shard else self.balance
        return cash - self._pending_spend() + self._open_cost()

    def day_pnl(self) -> float:
        return self.ledger.pnl_since(self.env, utc_day_start())

    def engage_kill(self, reason: str) -> None:
        """Manual kill switch. Remembered across restarts, so a bot that is
        set to resume by itself stays blocked until a person releases it."""
        self.risk.kill(reason)
        self.s.kill_state = {**self.s.kill_state, self.env: reason}
        try:
            set_settings(self.s)
        except Exception as e:  # noqa: BLE001
            logger.warning("could not save settings: {}", e)

    def release_kill(self) -> None:
        """Release the kill switch. If the drawdown stop tripped it, the
        drawdown is measured afresh from now."""
        self.risk.reset_kill()
        self.risk.halted_until = 0.0
        self.s.dd_reset_ts = {**self.s.dd_reset_ts, self.env: time.time()}
        self.s.kill_state = {k: v for k, v in self.s.kill_state.items() if k != self.env}
        self._drawdown = 0.0
        self._dd_logged = False
        try:
            set_settings(self.s)
        except Exception as e:  # noqa: BLE001
            logger.warning("could not save settings: {}", e)

    # ------------------------------------------------------------- books
    def _watched(self) -> list[Contract]:
        out: list[Contract] = []
        for a in self.assets.values():
            c = a.clock.current if a.clock else None
            if c is not None:
                out.append(c)
            out += [h for h in a.watched_hourly if c is None or h.ticker != c.ticker]
        return out

    async def _sync_books(self) -> None:
        """Keep order-book subscriptions equal to the contracts being watched."""
        watched = self._watched()
        tickers = tuple(dict.fromkeys(c.ticker for c in watched))
        for c in watched:
            self._close_of[c.ticker] = c.close_ts
        self._announce()
        if tickers == self._book_tickers:
            return
        self._book_tickers = tickers
        if self.env == "paper":
            self.feed.set_tickers(list(tickers))
        elif tickers:
            for tk in tickers:
                self.books.setdefault(tk, OrderBook(tk))
            await self.ws.unsubscribe("book")
            await self.ws.subscribe("book", ["orderbook_delta"], market_tickers=list(tickers), use_yes_price=True)
            self._snap_requested.clear()
        now = time.time()
        for tk in [tk for tk, ts in self._close_of.items() if ts < now - 7200]:
            self._close_of.pop(tk, None)

    def _announce(self) -> None:
        """Log each new contract once, as soon as its strike is known."""
        for a in self.assets.values():
            c = a.clock.current if a.clock else None
            if c is not None and c.strike is not None and c.ticker not in self._announced:
                self._announced.add(c.ticker)
                self.log(f"Tracking {c.ticker}  strike {c.strike}")
            if a.watched_hourly:
                key = f"{a.symbol}-hourly-{a.watched_hourly[0].close_ts}"
                if key not in self._announced:
                    self._announced.add(key)
                    when = time.strftime("%H:%M", time.gmtime(a.watched_hourly[0].close_ts))
                    self.log(f"Watching {len(a.watched_hourly)} hourly {a.symbol} strikes for the {when} UTC close")
        if len(self._announced) > 400:
            self._announced.clear()

    # --------------------------------------------------------------- loop
    def _near_close(self) -> bool:
        for a in self.assets.values():
            c = a.clock.current if a.clock else None
            if a.watched_hourly or (c is not None and c.close_ts - a.index.exchange_time() <= 100):
                return True
        return False

    async def _loop(self) -> None:
        last_hist = 0
        while self.running:
            t0 = time.time()
            try:
                await self._step()
                sec = int(t0)
                if sec != last_hist:
                    last_hist = sec
                    self._snapshot(sec)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self.error = str(e)
                logger.exception("loop error")
                self.log(f"loop error: {e}", "error")
                await asyncio.sleep(1.0)
            # react faster inside the final 90 s, where the lock-in edge lives
            near = self._near_close()
            period = 0.1 if near else 0.25
            if self.feed is not None:
                self.feed.poll_interval = 0.25 if near else 1.0
            await asyncio.sleep(max(0.0, period - (time.time() - t0)))

    async def _step(self) -> None:
        sp = self.s.strategy
        for a in self.assets.values():
            await a.clock.refresh(a.index)
            extra: list[Contract] = []
            if a.hourly is not None and a.index.last_value is not None:
                now = a.index.exchange_time()
                await a.hourly.refresh(now)
                extra = a.hourly.near_money(now, a.index.last_value, sp.hourly_band,
                                            window_s=sp.hourly_max_seconds_left + 60)
            a.watched_hourly = extra
        await self._sync_books()
        plans: list[tuple[float, list[Intent], TickState, AssetState]] = []
        for a in self.assets.values():
            plans += await self._decide_asset(a)
        # cash and the shared per-close cap go to the best expected return per dollar first
        plans.sort(key=lambda t: -t[0])
        for _, intents, st, a in plans:
            await self._execute(intents, st, a)

    async def _decide_asset(self, a: AssetState) -> list:
        """Price every watched contract of one coin; returns the trades wanted."""
        c = a.clock.current if a.clock else None
        now = a.index.exchange_time()              # all window arithmetic on Kalshi's clock
        if c is None:
            a.last_decision = {"note": "waiting for the next market"}
        elif c.strike is None:
            a.last_decision = {"note": "waiting for strike"}
        if a.index.last_value is None:
            a.last_decision = {"note": f"waiting for the {a.spec.index_id} feed"}
            return []
        last_print_sec = a.index.latest_print_sec()
        if last_print_sec is None or int(now) - last_print_sec > 3:
            a.last_decision = {"note": "index gap"}
            return []
        # fold in variance only up to the newest print that has actually arrived
        a.vol.update(last_print_sec, a.index.prints)
        plans = []
        if c is not None and c.strike is not None:
            a.last_decision, plan = await self._decide_contract(a, c, now, last_print_sec)
            if plan:
                plans.append(plan)
        view = []
        for h in a.watched_hourly:
            if c is not None and h.ticker == c.ticker:
                continue
            d, plan = await self._decide_contract(a, h, now, last_print_sec)
            if plan:
                plans.append(plan)
            b = self.books.get(h.ticker)
            yq, nq = self.om.exposure(h.ticker)
            view.append({"ticker": h.ticker, "strike": h.strike, "close_ts": h.close_ts,
                         "p_model": d.get("p_model"), "bid": b.best_bid if b else None,
                         "ask": b.best_ask if b else None, "yes": yq, "no": nq, "note": d.get("note", "")})
        a.hourly_view = view
        return plans

    def _sigma(self, a: AssetState, seconds_left: float) -> float:
        """Volatility handed to the model: the live estimate (floored), times
        the safety factor, times extra caution early in the averaging window."""
        sp = self.s.strategy
        sig = max(a.vol.sigma, sp.vol_floor_rel * (a.index.last_value or 0.0)) * sp.sigma_scale
        if sp.sigma_time_k > 0:
            sig *= 1.0 + sp.sigma_time_k * max(0.0, (seconds_left - 30.0) / 30.0)
        return sig

    async def _decide_contract(self, a: AssetState, c: Contract, now: float, last_print_sec: int):
        """Price one contract on its settlement index. Returns (decision info,
        plan) where plan is (priority, intents, state, asset) or None."""
        self._close_of[c.ticker] = c.close_ts
        if (self.env != "paper" and self.ws is not None and c.close_ts - now <= 90
                and c.ticker not in self._snap_requested):
            self._snap_requested.add(c.ticker)
            await self.ws.update("book", "get_snapshot", market_tickers=[c.ticker])
        book = self.books.get(c.ticker)
        first, last = window_bounds(c.close_ts)
        x_print = a.index.prints[last_print_sec]
        obs_end = min(last_print_sec, last)
        if obs_end >= first:
            psum, pn = a.index.window_sum(first, obs_end)
        else:
            psum, pn = 0.0, 0
        x_now = a.index.last_value
        sigma = self._sigma(a, c.close_ts - now)
        # data-quality gates: official strike only, no holes in the window, warm volatility
        gaps = a.index.missing_in(first, obs_end) if obs_end >= first else 0
        data_ok = c.strike_source == "api" and gaps == 0 and a.vol.ready
        thr = c.threshold
        p_model = fair_yes(x_now, thr, sigma, last_print_sec, c.close_ts, psum, pn, self.s.strategy.tail_nu)
        yes_q, no_q = self.om.exposure(c.ticker)
        st = TickState(now=now, contract=c, seconds_left=c.close_ts - now, x=x_now, x_print=x_print,
                       strike=thr, sigma=sigma, past_sum=psum, past_n=pn, p_model=p_model,
                       book=book, yes_qty=yes_q, no_qty=no_q,
                       working=self.om.working(c.ticker), balance=self.free_cash())
        intents = self.strategy.decide(st)
        note = self.strategy.note
        if not data_ok and intents:
            intents = [i for i in intents if i.kind != "buy"]
            note = ("waiting for the published strike" if c.strike_source != "api"
                    else f"feed missed {gaps} window print(s) - not trading this market" if gaps
                    else "volatility estimate still warming up")
            self.strategy.note = note
        # never leave resting orders into the close
        if c.close_ts - now < 2.0 and self.om.working(c.ticker):
            intents = intents + [Intent("cancel_all")]
        info = {"p_model": p_model, "note": note, "sigma": a.vol.sigma, "x": x_now, "strike": c.strike,
                "past_n": pn, "gaps": gaps}
        if not intents:
            return info, None
        buys = [i for i in intents if i.kind == "buy"]
        prio = max((i.edge / max(i.price, 1e-6) for i in buys), default=0.0)
        return info, (prio, intents, st, a)

    async def _execute(self, intents: list[Intent], st: TickState, a: AssetState | None = None) -> None:
        a = a or self.primary
        c = st.contract
        book = st.book
        for it in intents:
            if it.kind == "cancel":
                await self.om.cancel(it.order)
                continue
            if it.kind == "cancel_all":
                await self.om.cancel_ticker(c.ticker)
                continue
            note = await self._try_buy(it, st, a, book)
            if note:
                d = a.last_decision if (a.clock and a.clock.current is c) else None
                if d is not None:
                    d["note"] = note
                self.strategy.note = note

    async def _try_buy(self, it: Intent, st: TickState, a: AssetState, book: OrderBook | None) -> str:
        """Run one buy through every gate; returns a note when it is blocked."""
        c = st.contract
        if not self.s.strategy.enabled:
            return ""
        if not self.trading_active:
            return "exchange trading is paused"
        if time.time() < self._pause_until:
            return "paused after a rejected order"
        idx_age = time.time() - a.index.last_rx if a.index.last_rx else 1e9
        book_age = self._book_age(book)
        # contracts settling at the same moment move together (same index, or
        # coins that jump together): cap them jointly
        same_close = {tk for tk, ts in self._close_of.items() if ts == c.close_ts} | {c.ticker}
        committed_close = sum(self.om.committed_cost(tk) for tk in same_close)
        v = self.risk.check(price=it.price, count=it.count, balance=self.free_cash(), day_pnl=self.day_pnl(),
                            committed_market=self.om.committed_cost(c.ticker),
                            committed_close=committed_close, index_age=idx_age, book_age=book_age,
                            equity=self.equity(), drawdown=self._drawdown)
        if not v.ok:
            if self.risk.killed and "drawdown" in self.risk.kill_reason and not self._dd_logged:
                self._dd_logged = True
                self.log(f"KILL SWITCH - {self.risk.kill_reason}. Trading stays off until you release it.", "warn")
            return f"risk: {v.reason}"
        price = c.snap(it.price, "down")
        o = await self.om.buy(c.ticker, it.outcome, price, v.max_count, tif=it.tif,
                              post_only=it.post_only, reason=it.reason)
        self.log(f"ORDER {c.ticker} {it.outcome.upper()} {v.max_count:.2f} @ {price:.3f} {it.tif} "
                 f"({it.reason}) -> {o.status}{' filled ' + format(o.filled, '.2f') if o.filled else ''}")
        return ""

    def _snapshot(self, sec: int) -> None:
        for a in self.assets.values():
            c = a.clock.current if a.clock else None
            book = self.books.get(c.ticker) if c else None
            a.history.append({
                "t": sec, "x": a.index.last_value, "strike": c.strike if c else None,
                "p": a.last_decision.get("p_model"),
                "bid": book.best_bid if book else None, "ask": book.best_ask if book else None,
                "ticker": c.ticker if c else None,
            })

    # ------------------------------------------------------- housekeeping
    async def _housekeeping(self) -> None:
        n = 0
        while self.running:
            try:
                if self.env != "paper":
                    await self._refresh_balance()
                    if self.s.shard_topup:
                        await self._maybe_topup()
                await self._check_settlements()
                if n % 6 == 0 and self.env != "paper":
                    for a in self.assets.values():
                        if a.clock and a.clock.current:
                            await self.om.reconcile_fills(a.clock.current.ticker)
                if n % 12 == 0:
                    self.ledger.equity(self.env, self.equity(), self.day_pnl())
                    await self._check_exchange_status()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning("housekeeping: {}", e)
            n += 1
            await asyncio.sleep(5.0)

    async def _check_exchange_status(self) -> None:
        """Weekly maintenance (Thu 3-5 AM ET) and outages pause trading."""
        try:
            client = self.public if self.env == "paper" else self.rest
            st = await client.get("/exchange/status", auth=False)
        except Exception:  # noqa: BLE001
            return
        active = bool(st.get("trading_active", True)) and bool(st.get("exchange_active", True))
        for sh in st.get("exchange_index_statuses") or []:
            if int(sh.get("exchange_index", -1)) == CRYPTO_SHARD:
                active = active and bool(sh.get("trading_active", True))
        if active != self.trading_active:
            self.log("Exchange trading resumed" if active else "Exchange trading is PAUSED - no new orders",
                     "info" if active else "warn")
        self.trading_active = active

    async def _maybe_topup(self) -> None:
        """Kalshi requires collateral on the crypto shard before placing orders:
        keep enough there for one full settlement moment, with a margin."""
        crypto = self.balance_by_shard.get(CRYPTO_SHARD, 0.0)
        default = self.balance_by_shard.get(0, 0.0)
        need = min(self.risk.close_cap(self.equity()) * 1.5 + self.s.risk.min_balance, crypto + default)
        if crypto >= need or default < 1.0:
            return
        amt = round(min(default, need - crypto), 2)
        if amt < 1.0:
            return
        try:
            await self.rest.shard_transfer(amt, 0, CRYPTO_SHARD)
            self.log(f"Moved ${amt:.2f} of collateral to the crypto exchange shard")
            await self._refresh_balance()
        except KalshiError as e:
            self.log(f"shard transfer failed: {e.message}", "warn")
            self._topup_fails += 1
            if self._topup_fails >= 5:
                self.s.shard_topup = False

    async def _check_settlements(self) -> None:
        tickers = set(self.ledger.unsettled_tickers(self.env))
        settled_any = False
        for tk in tickers:
            if tk in self.settled:
                continue
            try:
                m = await (self.public.get(f"/markets/{tk}", auth=False) if self.env == "paper"
                           else self.rest.get(f"/markets/{tk}"))
                m = m.get("market", m)
            except Exception:  # noqa: BLE001
                continue
            if m.get("status") in ("settled", "finalized") and m.get("result") in ("yes", "no"):
                if self.env != "paper":
                    await self.om.reconcile_fills(tk)
                self.om.settle(tk, m["result"])
                self.settled.add(tk)
                settled_any = True
        if settled_any:
            await self._refresh_balance()

    # ------------------------------------------------------------- state
    def _asset_state(self, a: AssetState) -> dict:
        c = a.clock.current if a.clock else None
        book = self.books.get(c.ticker) if c else None
        pos = None
        yes_q = no_q = 0.0
        if c and self.om:
            yes_q, no_q = self.om.exposure(c.ticker)
            p = self.om.positions.get(c.ticker)
            if p:
                pos = {"yes": p.yes, "no": p.no, "cost": p.cost, "fees": p.fees,
                       "pnl_if_yes": p.pnl_if(True), "pnl_if_no": p.pnl_if(False)}
        x = a.index.last_value
        return {
            "symbol": a.symbol, "index_id": a.spec.index_id, "series": a.spec.series15,
            "index": {"value": x, "latency_ms": a.index.latency_ms,
                      "clock_offset_ms": 1000 * a.index.clock_offset,
                      "age": time.time() - a.index.last_rx if a.index.last_rx else None,
                      "sigma": a.vol.sigma, "sigma_bp": (1e4 * a.vol.sigma / x) if x else None,
                      "vol_ready": a.vol.ready},
            "market": None if c is None else {
                "ticker": c.ticker, "open_ts": c.open_ts, "close_ts": c.close_ts, "strike": c.strike,
                "seconds_left": c.close_ts - a.index.exchange_time(),
                "bid": book.best_bid if book else None, "ask": book.best_ask if book else None,
                "bid_size": book.bid_size() if book else 0, "ask_size": book.ask_size() if book else 0,
                "levels": book.levels(6) if book else {"bids": [], "asks": []},
            },
            "decision": a.last_decision,
            "hourly": a.hourly_view,
            "hourly_next_close": (a.hourly.next_close(a.index.exchange_time()) if a.hourly else None),
            "position": pos, "exposure": {"yes": yes_q, "no": no_q},
            "history": list(a.history)[-900:],
        }

    def state(self) -> dict:
        stats = self.ledger.stats(self.env)
        per_asset = [self._asset_state(a) for a in self.assets.values()]
        main = per_asset[0]
        eq = self.equity() if self.running else None
        limits = None
        if eq is not None:
            dpnl = self.day_pnl()
            limits = {"per_market": self.risk.market_cap(eq), "per_close": self.risk.close_cap(eq),
                      "daily_loss": self.risk.daily_loss_limit(eq, dpnl), "drawdown": self._drawdown,
                      "drawdown_limit": self.risk.drawdown_limit(eq, self._drawdown)}
            limits = {k: (None if v == float("inf") else v) for k, v in limits.items()}
        return {
            "env": self.env, "status": self.status, "running": self.running, "error": self.error,
            "uptime": time.time() - self.started_at if self.running else 0,
            "balance": self.free_cash() if self.running else None,
            "equity": eq, "limits": limits,
            "balance_total": self.balance, "day_pnl": self.day_pnl(),
            "killed": self.risk.killed, "kill_reason": self.risk.kill_reason,
            "trading_enabled": self.s.strategy.enabled,
            # the first coin, in the shape the single-coin dashboard used
            "index": main["index"], "market": main["market"], "decision": main["decision"],
            "hourly": main["hourly"], "hourly_next_close": main["hourly_next_close"],
            "position": main["position"], "exposure": main["exposure"], "history": main["history"],
            "assets": per_asset,
            "orders": [asdict(o) for o in (self.om.working() if self.om else [])],
            "logs": list(self.logs)[-80:],
            "stats": stats,
            "settlements": self.ledger.settlements_list(self.env, 50),
            "fills": self.ledger.recent_fills(self.env, 50),
        }
