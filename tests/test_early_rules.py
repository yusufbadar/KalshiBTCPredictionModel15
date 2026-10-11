"""The rules added after the first live day (11 Oct 2026):

* 15-minute markets act only in the final 55 seconds;
* while more than 45 seconds remain they need the expected settlement to be at
  least 2 bp of price (about $17 on BTC) from the strike;
* the hourly strike ladder is not affected by either.

Live, all three losses of that day were 15-minute entries at 58-60 s left with
leads of $10-13 that a single one-second index jump overturned."""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pandas as pd
import pytest

import kalshi_bot.config as config
from kalshi_bot.config import RiskParams, Settings, StrategyParams
from kalshi_bot.core.markets import Contract
from kalshi_bot.core.orderbook import OrderBook
from kalshi_bot.core.orders import Broker, OrderManager
from kalshi_bot.engine import Engine
from kalshi_bot.research.lockin import HOUSE_15M, HOUSE_HOURLY, LockinParams, house_for, make_rule
from kalshi_bot.strategy.base import TickState
from kalshi_bot.strategy.lockin import LockInStrategy, expected_settlement

GRID = [(0.0, 0.1, 0.001), (0.1, 0.9, 0.01), (0.9, 1.0, 0.001)]
STRIKE = 80_000.0


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
    def set_tickers(self, tks):
        self.tickers = tks


def _engine(tmp_path, monkeypatch, *, seconds_left: int, lead: float, life_s: int = 900, strike_type="greater_or_equal",
            strategy: StrategyParams | None = None):
    """One BTC contract `seconds_left` before its close, with the index flat at
    strike + lead (so YES is the favourite) and YES offered at 80c."""
    import kalshi_bot.engine as eng
    monkeypatch.setattr(eng, "DB_PATH", tmp_path / "e.sqlite3")
    now = int(time.time())
    close = now + seconds_left
    sp = strategy or StrategyParams()                    # shipped defaults
    sp.trade_hourly = False
    s = Settings(env="paper", api_key_id="x", private_key_path="x", strategy=sp,
                 risk=RiskParams(pct_per_market=0.10, pct_per_close=0.15), paper_start=100.0)
    e = Engine(s)
    e.feed = FakeFeed()
    broker = FakeBroker()
    e.om = OrderManager(broker, e.ledger, "paper")
    a = e.assets["BTC"]
    c = Contract("KXBTC-TEST", "EV", close - life_s, close, STRIKE, "api", "active", 2, GRID, strike_type=strike_type)
    a.clock = FakeClock(c)
    b = OrderBook(c.ticker)
    b.asks, b.bids = {8000: 500.0}, {7900: 500.0}
    b.valid, b.updated = True, time.time()
    e.books[c.ticker] = b
    level = STRIKE + lead
    for sec in range(now - 400, now + 1):
        a.index.on_tick(sec * 1000, level, received_ms=int(time.time() * 1000))
    a.index.clock_offset = 0.0
    a.vol.var = (2e-5 * level) ** 2                      # 0.2 bp per sqrt-second: a quiet market
    a.vol._anchored = True
    a.vol.n = 1000
    a.vol._last_sec = now
    e._book_tickers = (c.ticker,)
    e.running = True
    return e, broker


def test_shipped_defaults():
    sp = StrategyParams()
    assert (sp.max_seconds_left, sp.hourly_max_seconds_left, sp.early_seconds, sp.early_cushion_bp) == (55.0, 60.0, 45.0, 2.0)
    assert (HOUSE_15M.max_seconds_left, HOUSE_15M.early_seconds, HOUSE_15M.early_cushion_bp) == (55, 45, 2.0)
    assert (HOUSE_HOURLY.max_seconds_left, HOUSE_HOURLY.early_cushion_bp) == (60, 0.0)
    assert house_for("KXBTC15M") is HOUSE_15M and house_for("kxeth15m") is HOUSE_15M and house_for("KXBTCD") is HOUSE_HOURLY


def test_quarter_hour_market_waits_for_the_final_55_seconds(tmp_path, monkeypatch):
    e, broker = _engine(tmp_path, monkeypatch, seconds_left=58, lead=100.0)      # a huge lead is still too early
    asyncio.run(e._step())
    assert broker.placed == [] and "final 55s" in e.strategy.note


def test_small_early_lead_is_skipped_but_a_solid_one_trades(tmp_path, monkeypatch):
    # $13 above the strike (1.6 bp) with 52 s left: confident enough for the model, but one jump from flipping
    e, broker = _engine(tmp_path, monkeypatch, seconds_left=52, lead=13.0)
    asyncio.run(e._step())
    assert broker.placed == [], e.strategy.note
    assert "lead is only 1.6 bp" in e.strategy.note
    assert e.last_decision["p_model"] > 0.9                # the model alone would have bought
    # $20 above (2.5 bp) at the same moment
    e, broker = _engine(tmp_path / "b", monkeypatch, seconds_left=52, lead=20.0)
    asyncio.run(e._step())
    assert len(broker.placed) == 1 and broker.placed[0].outcome == "yes", e.strategy.note


def test_the_same_small_lead_trades_once_45_seconds_or_less_remain(tmp_path, monkeypatch):
    e, broker = _engine(tmp_path, monkeypatch, seconds_left=40, lead=13.0)
    asyncio.run(e._step())
    assert len(broker.placed) == 1, e.strategy.note


def test_rule_can_be_switched_off(tmp_path, monkeypatch):
    e, broker = _engine(tmp_path, monkeypatch, seconds_left=52, lead=13.0, strategy=StrategyParams(early_cushion_bp=0.0))
    asyncio.run(e._step())
    assert len(broker.placed) == 1, e.strategy.note


def test_hourly_strikes_keep_the_full_minute_and_need_no_early_lead(tmp_path, monkeypatch):
    # an hourly strike 58 s before its close with the same $13 lead: trades
    e, broker = _engine(tmp_path, monkeypatch, seconds_left=58, lead=13.0, life_s=3600, strike_type="greater")
    asyncio.run(e._step())
    assert len(broker.placed) == 1 and broker.placed[0].outcome == "yes", e.strategy.note
    # ...and it still waits outside its own 60 s window
    e, broker = _engine(tmp_path / "b", monkeypatch, seconds_left=63, lead=100.0, life_s=3600, strike_type="greater")
    asyncio.run(e._step())
    assert broker.placed == [] and "final 60s" in e.strategy.note


def test_expected_settlement_uses_the_prints_already_in_the_window():
    c = Contract("T", "E", 0, 900, 100.0, "api", "active", 2, GRID)
    st = TickState(now=870.0, contract=c, seconds_left=30.0, x=110.0, x_print=110.0, strike=100.0, sigma=1.0,
                   past_sum=31 * 100.0, past_n=31, p_model=0.9, book=None, yes_qty=0.0, no_qty=0.0)
    assert expected_settlement(st) == pytest.approx((31 * 100.0 + 29 * 110.0) / 60.0)
    st.past_n, st.past_sum = 0, 0.0
    assert expected_settlement(st) == 110.0


def test_research_rule_applies_the_same_early_lead():
    """Backtest and live must agree: same window, same lead rule, same boundary at 45 s."""
    rows = []
    for m, lead_bp in ((58, 5.0), (52, 1.6), (52, 2.5), (45, 1.6), (40, 1.6)):
        x = 80_000.0 * (1 + lead_bp * 1e-4)
        pn = 60 - m + 1
        rows.append(dict(ticker=f"T{m}-{lead_bp}", sec=1000 - m, m=m, K=80_000.0, Kt=80_000.0, x=x,
                         past_sum=pn * x, past_n=pn, ask_prev=0.80, ask_age=0.5, bid_prev=0.79, bid_age=0.5,
                         exec_ask=0.80, exec_ask_q=100.0, exec_bid=0.79, exec_bid_q=100.0, y=1, v_mid=1.0))
    d = pd.DataFrame(rows)
    prob, z = np.full(len(d), 0.97), np.full(len(d), 2.0)            # the model is confident on every row
    yes, no, _ = make_rule(HOUSE_15M, prob, z).signal(d)
    fires = dict(zip(d.ticker, ~np.isnan(yes)))
    assert fires == {"T58-5.0": False,      # before the 55 s window
                     "T52-1.6": False,      # early and the lead is under 2 bp
                     "T52-2.5": True,       # early with a solid lead
                     "T45-1.6": True,       # 45 s left: the early rule no longer applies
                     "T40-1.6": True}
    yes_h, _, _ = make_rule(HOUSE_HOURLY, prob, z).signal(d)
    assert (~np.isnan(yes_h)).all()                                  # hourly: 60 s window, no lead rule
    off, _, _ = make_rule(LockinParams(55, 0.9, 1.5, 0.01, 1.0), prob, z).signal(d)
    assert list(~np.isnan(off)) == [False, True, True, True, True]


def test_settings_from_the_previous_version_get_the_new_window(tmp_path, monkeypatch):
    v2 = {"version": 2, "env": "live", "live_unlocked": True, "run_wanted": True,
          "strategy": {"max_seconds_left": 60.0, "min_edge": 0.02, "z_min": 1.75},
          "risk": {"pct_per_market": 0.10, "pct_per_close": 0.15, "daily_loss_pct": 0.30, "max_drawdown_pct": 0.40}}
    s = Settings._from_dict(v2)
    assert s.strategy.max_seconds_left == 55.0                 # re-validated field adopts the new value
    assert (s.strategy.early_cushion_bp, s.strategy.early_seconds, s.strategy.hourly_max_seconds_left) == (2.0, 45.0, 60.0)
    assert (s.strategy.min_edge, s.strategy.z_min) == (0.02, 1.75)          # everything else the user set is kept
    assert (s.risk.pct_per_market, s.risk.pct_per_close, s.risk.daily_loss_pct, s.risk.max_drawdown_pct) == (0.10, 0.15, 0.30, 0.40)
    assert s.env == "live" and s.live_unlocked and s.run_wanted and s.version == config.SETTINGS_VERSION
    # a choice made on the current version is respected
    mine = Settings._from_dict({**v2, "version": config.SETTINGS_VERSION})
    assert mine.strategy.max_seconds_left == 60.0
    # and the migrated settings survive a save / load round trip
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    s.save()
    again = Settings.load()
    assert again.strategy.max_seconds_left == 55.0 and again.version == config.SETTINGS_VERSION and again.env == "live"
