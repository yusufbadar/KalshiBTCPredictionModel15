"""Engine integration test: simulated BRTI + order book -> strategy -> risk -> orders."""
from __future__ import annotations

import asyncio
import time

import pytest

from kalshi_bot.config import RiskParams, Settings, StrategyParams
from kalshi_bot.core.markets import Contract
from kalshi_bot.core.orderbook import OrderBook
from kalshi_bot.core.orders import Broker, OrderManager
from kalshi_bot.engine import Engine
from kalshi_bot.strategy.lockin import max_price_for_edge


class FakeBroker(Broker):
    def __init__(self):
        self.placed = []

    async def place(self, o):
        self.placed.append(o)
        return {"order_id": f"o{len(self.placed)}", "fill_count": f"{o.count:.2f}", "remaining_count": "0"}

    async def cancel(self, o):
        pass

    async def cancel_all(self):
        pass


class FakeClock:
    def __init__(self, c):
        self.current = c

    async def refresh(self, index=None):
        pass


class FakeFeed:
    def set_ticker(self, tk):
        pass

    def set_tickers(self, tks):
        self.tickers = tks


def _engine(tmp_path, monkeypatch, strike, prints, close_ts, ask=None, bid=None, offset=0.0):
    import kalshi_bot.engine as eng
    monkeypatch.setattr(eng, "DB_PATH", tmp_path / "e.sqlite3")
    s = Settings(env="paper", api_key_id="x", private_key_path="x",
                 strategy=StrategyParams(max_seconds_left=90, p_min=0.9, z_min=2.0, min_edge=0.03,
                                         sigma_scale=1.2, max_price=0.99, max_contracts_per_order=50,
                                         max_position_per_market=300),
                 risk=RiskParams(max_notional_per_market=300, max_daily_loss=300, pct_per_market=0.0,
                                 pct_per_close=0.0, daily_loss_pct=0.0, max_drawdown_pct=0.0),
                 paper_start=1000.0)
    e = Engine(s)
    c = Contract("KXBTC15M-TEST", "EV", close_ts - 900, close_ts, strike, "api", "active", 2,
                 [(0.0, 0.1, 0.001), (0.1, 0.9, 0.01), (0.9, 1.0, 0.001)])
    e.clock = FakeClock(c)
    e.feed = FakeFeed()
    broker = FakeBroker()
    e.om = OrderManager(broker, e.ledger, "paper")
    b = OrderBook(c.ticker)
    b.bids = {} if bid is None else {int(round(bid * 10000)): 500.0}
    b.asks = {} if ask is None else {int(round(ask * 10000)): 500.0}
    b.valid, b.updated = True, time.time()
    e.books[c.ticker] = b
    now_local = time.time()
    for sec, v in prints:
        e.index.on_tick(sec * 1000, v, received_ms=int((now_local - offset) * 1000))
    e.index.clock_offset = offset
    e.index.last_rx = now_local
    e.vol.var = 4.0 ** 2
    e.vol.n = 1000
    e.vol._last_sec = max(sec for sec, _ in prints)
    e._book_tickers = (c.ticker,)
    e.running = True
    return e, broker, c


def test_engine_buys_locked_in_favourite(tmp_path, monkeypatch):
    now = int(time.time())
    close = now + 20                      # 20 s left: 41 of 60 window prints known
    strike = 80_000.0
    prints = [(s, 80_100.0) for s in range(now - 400, now + 1)]   # $100 above the strike all along
    e, broker, c = _engine(tmp_path, monkeypatch, strike, prints, close, ask=0.80, bid=0.79)
    asyncio.run(e._step())
    assert len(broker.placed) == 1, e.strategy.note
    o = broker.placed[0]
    assert o.outcome == "yes"
    p = e.last_decision["p_model"]
    assert p > 0.999
    assert o.price <= max_price_for_edge(p, 0.03) + 1e-9 and o.price >= 0.80
    # 20c edge vs 3c minimum -> edge sizing scales the 50-contract base by the 3x cap
    assert o.count == 150


def test_engine_respects_edge_and_side(tmp_path, monkeypatch):
    now = int(time.time())
    close = now + 20
    prints = [(s, 80_100.0) for s in range(now - 400, now + 1)]
    # favourite (YES) already priced at 0.99 -> no edge, no order
    e, broker, _ = _engine(tmp_path, monkeypatch, 80_000.0, prints, close, ask=0.99, bid=0.985)
    asyncio.run(e._step())
    assert broker.placed == []
    # model says NO is the favourite -> buys NO by hitting the YES bid
    prints = [(s, 79_900.0) for s in range(now - 400, now + 1)]
    e, broker, _ = _engine(tmp_path, monkeypatch, 80_000.0, prints, close, ask=0.25, bid=0.20)
    asyncio.run(e._step())
    assert len(broker.placed) == 1 and broker.placed[0].outcome == "no"
    assert broker.placed[0].price >= 0.80        # NO costs 1 - 0.20


def test_engine_waits_outside_window(tmp_path, monkeypatch):
    now = int(time.time())
    close = now + 400                      # too early for the lock-in
    prints = [(s, 80_300.0) for s in range(now - 400, now + 1)]
    e, broker, _ = _engine(tmp_path, monkeypatch, 80_000.0, prints, close, ask=0.60, bid=0.59)
    asyncio.run(e._step())
    assert broker.placed == []
    assert "waiting" in e.strategy.note


def test_engine_uses_exchange_clock(tmp_path, monkeypatch):
    """Local clock 5 s ahead of the exchange: the engine must still see the
    latest print as current and compute the window on exchange time."""
    now_ex = int(time.time()) - 5
    close = now_ex + 20
    prints = [(s, 80_100.0) for s in range(now_ex - 400, now_ex + 1)]
    e, broker, _ = _engine(tmp_path, monkeypatch, 80_000.0, prints, close, ask=0.80, bid=0.79, offset=5.0)
    asyncio.run(e._step())
    # prints stamped close-60 .. now are known: 41 of 60, 19 still to come
    assert e.last_decision.get("past_n") == 41
    assert len(broker.placed) == 1


def test_engine_skips_market_with_window_gap(tmp_path, monkeypatch):
    now = int(time.time())
    close = now + 20
    prints = [(s, 80_100.0) for s in range(now - 400, now + 1) if s != close - 50]   # one missing print
    e, broker, _ = _engine(tmp_path, monkeypatch, 80_000.0, prints, close, ask=0.80, bid=0.79)
    asyncio.run(e._step())
    assert broker.placed == []
    assert "missed" in e.strategy.note


def test_engine_requires_published_strike(tmp_path, monkeypatch):
    now = int(time.time())
    close = now + 20
    prints = [(s, 80_100.0) for s in range(now - 400, now + 1)]
    e, broker, c = _engine(tmp_path, monkeypatch, 80_000.0, prints, close, ask=0.80, bid=0.79)
    c.strike_source = "local"
    asyncio.run(e._step())
    assert broker.placed == []


# ------------------------------------------------------------ hourly ladder
class FakeLadder:
    def __init__(self, contracts):
        self.cs = contracts

    async def refresh(self, now):
        pass

    def next_close(self, now):
        return min(c.close_ts for c in self.cs)

    def near_money(self, now, x, band, window_s):
        return [c for c in self.cs if abs(c.strike - x) <= band]


def test_contract_threshold_semantics():
    ge = Contract("A", "E", 0, 900, 100.0, "api", "active", 2, [(0, 1, 0.01)])
    gt = Contract("B", "E", 0, 3600, 100.0, "api", "active", 2, [(0, 1, 0.01)], strike_type="greater")
    assert ge.threshold == pytest.approx(99.995)      # S >= 100.00
    assert gt.threshold == pytest.approx(100.005)     # S > 100.00


def test_engine_trades_hourly_strikes_with_joint_cap(tmp_path, monkeypatch):
    now = int(time.time())
    close = now + 20
    prints = [(s, 80_100.0) for s in range(now - 400, now + 1)]
    e, broker, primary = _engine(tmp_path, monkeypatch, 80_000.0, prints, close, ask=0.80, bid=0.79)
    grid = [(0.0, 0.1, 0.01), (0.1, 0.9, 0.01), (0.9, 1.0, 0.01)]
    below = Contract("KXBTCD-T80049.99", "EV", close - 3600, close, 80_049.99, "api", "active", 2, grid,
                     strike_type="greater")     # settlement ~80,100 > strike -> YES locked
    above = Contract("KXBTCD-T80149.99", "EV", close - 3600, close, 80_149.99, "api", "active", 2, grid,
                     strike_type="greater")     # settlement ~80,100 < strike -> NO locked
    for c, bid, ask in ((below, 0.84, 0.85), (above, 0.15, 0.16)):
        b = OrderBook(c.ticker)
        b.bids, b.asks = {int(bid * 10000): 500.0}, {int(ask * 10000): 500.0}
        b.valid, b.updated = True, time.time()
        e.books[c.ticker] = b
    e.hourly = FakeLadder([below, above])
    e.s.strategy.hourly_band = 400.0
    e.s.risk.max_notional_per_market = 300.0
    e.s.risk.max_notional_per_close = 200.0          # joint cap across all three contracts
    e._book_tickers = (primary.ticker, below.ticker, above.ticker)
    asyncio.run(e._step())
    placed = {o.ticker: o for o in broker.placed}
    assert primary.ticker in placed and placed[primary.ticker].outcome == "yes"
    spent = sum(o.count * o.price for o in broker.placed)
    assert spent <= 200.0 + 1e-6                        # the joint cap binds
    if above.ticker in placed:
        assert placed[above.ticker].outcome == "no"
    if below.ticker in placed:
        assert placed[below.ticker].outcome == "yes"
    assert len(e.hourly_view) == 2


class FakeWS:
    def __init__(self):
        self.connected = asyncio.Event()
        self.connected.set()
        self.last_msg_ts = time.time()
        self.updates = []

    async def update(self, name, action, **params):
        self.updates.append((name, action, params))

    def subscribed(self, name):
        return self.connected.is_set()

    async def unsubscribe(self, name):
        pass

    async def subscribe(self, name, channels, **params):
        pass


def test_quiet_websocket_book_is_not_stale(tmp_path, monkeypatch):
    """Demo/live books are pushed on change; a book that has not changed for
    minutes is still current while the socket is alive (regression test)."""
    now = int(time.time())
    close = now + 20
    prints = [(s, 80_100.0) for s in range(now - 400, now + 1)]
    e, broker, c = _engine(tmp_path, monkeypatch, 80_000.0, prints, close, ask=0.80, bid=0.79)
    e.env = "demo"
    e.balance_by_shard = {2: 1000.0}
    e.ws = FakeWS()
    e.books[c.ticker].updated = time.time() - 600      # no delta for 10 minutes
    asyncio.run(e._step())
    assert len(broker.placed) == 1, e.strategy.note
    assert any(a == "get_snapshot" for _, a, _ in e.ws.updates)
    # a dead socket must block trading
    e2, broker2, c2 = _engine(tmp_path / "b", monkeypatch, 80_000.0, prints, close, ask=0.80, bid=0.79)
    e2.env = "demo"
    e2.balance_by_shard = {2: 1000.0}
    e2.ws = FakeWS()
    e2.ws.connected.clear()
    asyncio.run(e2._step())
    assert broker2.placed == []
