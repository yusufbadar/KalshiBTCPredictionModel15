"""Account-relative risk limits, cash-aware sizing, drawdown stop, per-coin
settlement rounding and multi-coin operation."""
from __future__ import annotations

import asyncio
import time

import pytest

from kalshi_bot.config import ASSETS, RiskParams, Settings, StrategyParams, parse_assets
from kalshi_bot.core.fees import taker_fee
from kalshi_bot.core.markets import Contract
from kalshi_bot.core.orderbook import OrderBook
from kalshi_bot.core.orders import Broker, OrderManager
from kalshi_bot.core.risk import RiskManager
from kalshi_bot.engine import Engine
from kalshi_bot.model.vol import EwmaVol

GRID = [(0.0, 0.1, 0.001), (0.1, 0.9, 0.01), (0.9, 1.0, 0.001)]
OK = dict(day_pnl=0.0, committed_market=0.0, committed_close=0.0, index_age=0.1, book_age=0.1)


# ------------------------------------------------------------------ risk
def test_percent_caps_follow_the_account():
    r = RiskManager(RiskParams(max_notional_per_market=100, pct_per_market=0.06,
                               max_notional_per_close=150, pct_per_close=0.10))
    assert r.market_cap(100.0) == pytest.approx(6.0)        # 6% of a $100 account
    assert r.close_cap(100.0) == pytest.approx(10.0)
    assert r.market_cap(5000.0) == pytest.approx(100.0)     # the dollar cap takes over on a big account
    v = r.check(price=0.95, count=50, balance=100.0, equity=100.0, **OK)
    assert v.ok
    assert v.max_count * (0.95 + taker_fee(0.95)) <= 6.0 + 1e-9
    assert v.max_count == 6.0                              # whole contracts


def test_one_engine_per_environment(tmp_path):
    from kalshi_bot.core.lock import InstanceLock
    a, b = InstanceLock(tmp_path / "engine_live.lock"), InstanceLock(tmp_path / "engine_live.lock")
    assert a.acquire()
    assert not b.acquire()                                 # a second bot on the same account is refused
    a.release()
    assert b.acquire()
    b.release()


def test_orders_never_exceed_free_cash_including_fee():
    r = RiskManager(RiskParams(max_notional_per_market=1000, pct_per_market=0.0, max_notional_per_close=1000,
                               pct_per_close=0.0, min_balance=5.0))
    v = r.check(price=0.80, count=500, balance=30.0, equity=400.0, **OK)
    assert v.ok
    assert v.max_count * (0.80 + taker_fee(0.80)) <= 25.0 + 1e-9      # $30 cash - $5 kept back
    v = r.check(price=0.80, count=500, balance=5.5, equity=400.0, **OK)
    assert not v.ok and "cash" in v.reason


def test_joint_cap_and_daily_loss_percent():
    r = RiskManager(RiskParams(max_notional_per_market=0, pct_per_market=0.10, max_notional_per_close=0,
                               pct_per_close=0.15, max_daily_loss=0, daily_loss_pct=0.15))
    base = dict(price=0.9, count=100, balance=200.0, equity=200.0, index_age=0.1, book_age=0.1)
    v = r.check(day_pnl=0.0, committed_market=0.0, committed_close=25.0, **base)
    assert v.ok and v.max_count * (0.9 + taker_fee(0.9)) <= 5.0 + 1e-9   # 15% of 200 = 30, 25 already used
    # lost 15% of the day's starting equity ($230 -> $200 after -$34.5 is beyond 15% of 234.5)
    v = r.check(day_pnl=-36.0, committed_market=0.0, committed_close=0.0, **base)
    assert not v.ok and "daily loss" in v.reason


def test_drawdown_stop_trips_the_kill_switch():
    r = RiskManager(RiskParams(max_drawdown_pct=0.30))
    assert r.check(price=0.9, count=5, balance=80.0, equity=80.0, drawdown=20.0, **OK).ok     # 20% of the peak
    v = r.check(price=0.9, count=5, balance=70.0, equity=70.0, drawdown=30.0, **OK)           # 30% of the peak
    assert not v.ok and r.killed and "drawdown" in r.kill_reason
    r.reset_kill()
    assert r.check(price=0.9, count=5, balance=70.0, equity=70.0, drawdown=0.0, **OK).ok


def test_ledger_realised_drawdown(tmp_path):
    from kalshi_bot.core.ledger import Ledger
    led = Ledger(tmp_path / "l.sqlite3")
    for i, pnl in enumerate([5.0, 3.0, -6.0, 1.0, -4.0]):
        led.settlement("live", {"ticker": f"T{i}", "ts": 1000.0 + i, "result": "yes", "yes_qty": 1, "no_qty": 0,
                                "cost": 1.0, "fees": 0.0, "payout": 1.0, "pnl": pnl})
    assert led.realised_drawdown("live") == pytest.approx(9.0)          # peak +8, now -1
    assert led.realised_drawdown("live", since_ts=1003.0) == pytest.approx(4.0)   # fresh start: +1 then -4
    assert led.realised_drawdown("demo") == 0.0


# ------------------------------------------------------------ conventions
def test_threshold_uses_each_coins_rounding():
    xrp = Contract("KXXRP15M-T", "E", 0, 900, 1.4055, "api", "active", 2, GRID, decimals=4, asset="XRP")
    assert xrp.threshold == pytest.approx(1.40545)                     # avg rounds to 1.4055 from here up
    btc = Contract("KXBTC15M-T", "E", 0, 900, 82954.01, "api", "active", 2, GRID)
    assert btc.threshold == pytest.approx(82954.005)
    assert [a.symbol for a in parse_assets("eth, btc ,nope,ETH")] == ["ETH", "BTC"]
    assert parse_assets("")[0].symbol == "BTC"
    assert ASSETS["DOGE"].index_id == "DOGEUSD_RTI"


def test_vol_estimate_starts_relative_to_price_and_reports_ready():
    v = EwmaVol(halflife_s=300.0, floor=0.0, init_rel=7e-5)
    prints = {s: 2500.0 + 0.05 * ((s * 7919) % 13 - 6) for s in range(1000, 1400)}
    v.update(1010, prints)
    assert 0.05 < v.sigma < 0.5            # anchored near 7e-5 * 2500 = 0.175, not at a BTC-sized $6
    assert not v.ready
    v.update(1399, prints)
    assert v.ready


def test_credentials_are_kept_per_exchange(tmp_path):
    key = tmp_path / "k.key"
    key.write_text("x", encoding="utf-8")
    s = Settings(env="demo", api_key_id="demo-id", private_key_path=str(key))
    assert s.creds() == ("demo-id", str(key)) and s.trading_env == "demo"
    s.env = "paper"
    assert s.trading_env == "demo"                       # paper logs in with the demo key when it has one
    s.env = "live"
    assert s.creds() == ("", "")                         # the demo key is never sent to production
    s.live_api_key_id, s.live_private_key_path = "live-id", str(key)
    assert s.creds() == ("live-id", str(key))
    only_live = Settings(env="paper", live_api_key_id="live-id", live_private_key_path=str(key))
    assert only_live.trading_env == "live"               # paper falls back to the live login (read-only use)


# ---------------------------------------------------------------- engine
class FakeBroker(Broker):
    def __init__(self, reject: str | None = None):
        self.placed, self.reject = [], reject

    async def place(self, o):
        if self.reject:
            from kalshi_bot.api.rest import KalshiError
            raise KalshiError(400, self.reject, self.reject)
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
    def set_tickers(self, tks):
        self.tickers = tks


def _engine(tmp_path, monkeypatch, coins: dict, risk: RiskParams, strategy: StrategyParams | None = None,
            paper_start: float = 100.0, seconds_left: int = 20, broker=None):
    """coins: {symbol: (strike, index level, ask, bid)} - one live contract per coin."""
    import kalshi_bot.engine as eng
    monkeypatch.setattr(eng, "DB_PATH", tmp_path / "e.sqlite3")
    now = int(time.time())
    close = now + seconds_left
    sp = strategy or StrategyParams(max_seconds_left=60, p_min=0.9, z_min=1.5, min_edge=0.02)
    sp.assets = ",".join(coins)
    sp.trade_hourly = False
    s = Settings(env="paper", api_key_id="x", private_key_path="x", strategy=sp, risk=risk,
                 paper_start=paper_start)
    e = Engine(s)
    e.feed = FakeFeed()
    broker = broker or FakeBroker()
    e.om = OrderManager(broker, e.ledger, "paper")
    e.om.listeners.append(e._on_om_event)
    contracts = {}
    for sym, (strike, level, ask, bid) in coins.items():
        a = e.assets[sym]
        c = Contract(f"{a.spec.series15}-TEST", "EV", close - 900, close, strike, "api", "active", 2, GRID,
                     decimals=a.spec.decimals, asset=sym)
        a.clock = FakeClock(c)
        b = OrderBook(c.ticker)
        b.asks = {int(round(ask * 10000)): 500.0}
        b.bids = {int(round(bid * 10000)): 500.0}
        b.valid, b.updated = True, time.time()
        e.books[c.ticker] = b
        for sec in range(now - 400, now + 1):
            a.index.on_tick(sec * 1000, level, received_ms=int(time.time() * 1000))
        a.index.clock_offset = 0.0
        a.vol.var = (2e-5 * level) ** 2
        a.vol._anchored = True
        a.vol.n = 1000
        a.vol._last_sec = now
        contracts[sym] = c
    e._book_tickers = tuple(c.ticker for c in contracts.values())
    e.running = True
    return e, broker, contracts


def test_small_account_orders_are_sized_to_the_percent_cap(tmp_path, monkeypatch):
    risk = RiskParams(pct_per_market=0.06, pct_per_close=0.10)
    e, broker, cs = _engine(tmp_path, monkeypatch, {"BTC": (80_000.0, 80_100.0, 0.80, 0.79)}, risk)
    asyncio.run(e._step())
    assert len(broker.placed) == 1, e.strategy.note
    o = broker.placed[0]
    assert o.count * (o.price + taker_fee(o.price)) <= 6.0 + 1e-6          # 6% of the $100 paper account
    assert o.count >= 5


def test_two_coins_share_cash_and_the_per_close_cap(tmp_path, monkeypatch):
    risk = RiskParams(pct_per_market=0.06, pct_per_close=0.10)
    coins = {"BTC": (80_000.0, 80_100.0, 0.90, 0.89),      # YES locked, offered at 90c
             "ETH": (2_500.0, 2_496.0, 0.30, 0.25)}        # NO locked, NO costs 75c -> bigger edge per dollar
    e, broker, cs = _engine(tmp_path, monkeypatch, coins, risk)
    asyncio.run(e._step())
    placed = {o.ticker: o for o in broker.placed}
    assert set(placed) == {cs["BTC"].ticker, cs["ETH"].ticker}, e.strategy.note
    assert placed[cs["ETH"].ticker].outcome == "no" and placed[cs["BTC"].ticker].outcome == "yes"
    assert broker.placed[0].ticker == cs["ETH"].ticker                      # best return per dollar goes first
    spent = sum(o.count * (o.price + taker_fee(o.price)) for o in broker.placed)
    assert spent <= 10.0 + 1e-6                                             # joint cap: 10% of $100
    eth = placed[cs["ETH"].ticker]
    assert eth.count * (eth.price + taker_fee(eth.price)) <= 6.0 + 1e-6
    assert {a["symbol"] for a in e.state()["assets"]} == {"BTC", "ETH"}


def test_live_cash_is_reduced_by_fills_before_the_next_balance_poll(tmp_path, monkeypatch):
    risk = RiskParams(max_notional_per_market=1000, pct_per_market=0.0, max_notional_per_close=1000,
                      pct_per_close=0.0, max_drawdown_pct=0.0, daily_loss_pct=0.0, min_balance=5.0)
    e, broker, cs = _engine(tmp_path, monkeypatch, {"BTC": (80_000.0, 80_100.0, 0.80, 0.79)}, risk)
    e.env = "demo"

    class WS:
        connected = asyncio.Event()
        last_msg_ts = time.time()

        async def update(self, *a, **k):
            pass

        def subscribed(self, name):
            return True
    WS.connected.set()
    e.ws = WS()
    e._apply_balance({"balance": 6000, "balance_breakdown": [{"exchange_index": 2, "balance": "60.00"}]},
                     asked_at=time.time())
    assert e.free_cash() == pytest.approx(60.0)
    asyncio.run(e._step())
    assert len(broker.placed) == 1
    o = broker.placed[0]
    assert o.count * (o.price + taker_fee(o.price)) <= 55.0 + 1e-6          # sized to the cash, no rejection
    # the fill arrives: cash drops at once, equity is unchanged (cash became a position)
    cost = o.count * 0.80
    e.om.book_fill({"fill_id": "f1", "ticker": o.ticker, "order_id": "o1", "outcome": "yes", "price": 0.80,
                    "count": o.count, "fee": 0.5, "is_taker": True, "ts": time.time()})
    assert e.free_cash() == pytest.approx(60.0 - cost - 0.5)
    assert e.equity() == pytest.approx(60.0)
    # a balance requested *after* the fill already contains it
    e._apply_balance({"balance": 0, "balance_breakdown": [{"exchange_index": 2, "balance": f"{60 - cost - 0.5:.2f}"}]},
                     asked_at=time.time() + 1.0)
    assert e.free_cash() == pytest.approx(60.0 - cost - 0.5, abs=0.01)


def test_insufficient_balance_rejection_pauses_briefly_not_for_a_minute(tmp_path, monkeypatch):
    risk = RiskParams(pct_per_market=0.5, pct_per_close=0.5)
    e, broker, cs = _engine(tmp_path, monkeypatch, {"BTC": (80_000.0, 80_100.0, 0.80, 0.79)}, risk,
                            broker=FakeBroker(reject="insufficient_balance"))
    asyncio.run(e._step())
    assert 0 < e._pause_until - time.time() <= 2.1
    assert len(e._rejects) == 0


def test_no_trading_until_the_volatility_estimate_is_warm(tmp_path, monkeypatch):
    risk = RiskParams(pct_per_market=0.06, pct_per_close=0.10)
    e, broker, cs = _engine(tmp_path, monkeypatch, {"ETH": (2_500.0, 2_504.0, 0.80, 0.79)}, risk)
    e.assets["ETH"].vol.n = 10
    asyncio.run(e._step())
    assert broker.placed == [] and "warming" in e.strategy.note
    e.assets["ETH"].vol.n = 1000
    e.strategy._last_order.clear()
    asyncio.run(e._step())
    assert len(broker.placed) == 1


def test_early_window_caution_lowers_confidence(tmp_path, monkeypatch):
    risk = RiskParams(pct_per_market=0.06, pct_per_close=0.10)
    sp = StrategyParams(max_seconds_left=60, p_min=0.9, z_min=1.5, min_edge=0.02, sigma_time_k=0.5)
    e, _, cs = _engine(tmp_path, monkeypatch, {"BTC": (80_000.0, 80_010.0, 0.80, 0.79)}, risk, strategy=sp,
                       seconds_left=58)
    a = e.assets["BTC"]
    late, early = e._sigma(a, 20.0), e._sigma(a, 60.0)
    assert early == pytest.approx(late * 1.5)
    assert e._sigma(a, 45.0) == pytest.approx(late * 1.25)
