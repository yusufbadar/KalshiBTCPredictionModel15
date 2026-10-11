"""Correctness tests for pricing, reconstruction, book handling, fees and accounting."""
from __future__ import annotations

import asyncio
import math
import time

import numpy as np
import pytest

from kalshi_bot.core.fees import taker_fee
from kalshi_bot.core.index import IndexTracker
from kalshi_bot.core.markets import Contract
from kalshi_bot.core.orderbook import OrderBook
from kalshi_bot.model.fair_value import N_AVG, fair_yes_vec, settlement_zscore
from kalshi_bot.research.dataset import reconstruct_prints
from kalshi_bot.strategy.lockin import max_price_for_edge


# ------------------------------------------------------------------ pricing
def _mc_prob(x_path_known, strike, sigma, t, close, n=200_000, seed=0):
    """Monte Carlo P(mean(prints close-60..close-1) >= strike) given prints up to t."""
    rng = np.random.default_rng(seed)
    first, last = close - 60, close - 1
    steps = last - t
    incs = rng.normal(0, sigma, size=(n, max(steps, 0)))
    paths = x_path_known[t] + np.cumsum(incs, axis=1) if steps > 0 else np.zeros((n, 0))
    known = [x_path_known[s] for s in range(first, min(t, last) + 1)]
    fut_start = max(first, t + 1)
    fut = paths[:, fut_start - (t + 1):] if steps > 0 else np.zeros((n, 0))
    total = sum(known) + fut.sum(axis=1)
    return float(np.mean(total / 60.0 >= strike))


@pytest.mark.parametrize("secs_left", [300, 61, 60, 45, 20, 5, 2])
def test_fair_value_matches_monte_carlo(secs_left):
    rng = np.random.default_rng(1)
    close = 10_000
    sigma = 5.0
    x = {s: 80_000 + 0.0 for s in range(close - 400, close)}
    # a random past path so the in-window part matters
    v = 80_000.0
    for s in range(close - 400, close):
        v += rng.normal(0, sigma)
        x[s] = v
    t = close - secs_left
    strike = x[t] - 3.0
    first = close - 60
    past = [x[s] for s in range(first, t + 1)] if t >= first else []
    p = fair_yes_vec(x[t], strike, sigma, secs_left, sum(past), len(past), nu=math.inf)
    mc = _mc_prob(x, strike, sigma, t, close)
    assert abs(float(p) - mc) < 0.006, (secs_left, float(p), mc)


def test_fair_value_is_certain_after_last_print():
    p = fair_yes_vec(100.0, 99.0, 5.0, 1, 60 * 100.0, 60, nu=5.0)
    assert float(p) == 1.0
    p = fair_yes_vec(100.0, 101.0, 5.0, 1, 60 * 100.0, 60, nu=5.0)
    assert float(p) == 0.0


def test_zscore_sign():
    assert settlement_zscore(100.0, 90.0, 1.0, 300, 0.0, 0) > 0
    assert settlement_zscore(100.0, 110.0, 1.0, 300, 0.0, 0) < 0


# ---------------------------------------------------------- reconstruction
def test_reconstruct_prints_round_trip():
    rng = np.random.default_rng(7)
    x = 60_000 + np.cumsum(rng.normal(0, 4, 3601 + 59))
    x = np.round(x, 2)
    a = np.array([x[i - 59:i + 1].mean() for i in range(59, len(x))])
    xr = reconstruct_prints(a)
    truth = x[59:]
    # per-print error comes from estimating 60 residue offsets from an hour of
    # data (~sigma/sqrt(60)); it is small and unbiased
    err = xr - truth
    assert np.mean(np.abs(err)) < 1.0 and abs(np.mean(err)) < 0.2
    # settlement-style averages must be exact regardless
    close = len(a) - 1
    s_true = truth[close - 60:close].mean()
    s_rec = xr[close - 60:close].mean()
    assert abs(s_true - s_rec) < 1e-6


# ------------------------------------------------------------------- index
def test_index_settlement_window_is_t_minus_60_to_t_minus_1():
    ix = IndexTracker()
    close = 1_000_000
    for s in range(close - 120, close + 1):
        ix.on_tick(s * 1000, float(s - (close - 120)))
    expect = round(np.mean([s - (close - 120) for s in range(close - 60, close)]), 2)
    assert ix.settlement_avg(close) == expect
    s, n = ix.window_sum(close - 60, close - 1)
    assert n == 60


# -------------------------------------------------------------------- book
def test_orderbook_rest_and_ws_conventions():
    b = OrderBook("T")
    b.apply_snapshot_rest({"yes_dollars": [["0.4000", "10"], ["0.4100", "5"]],
                           "no_dollars": [["0.5700", "7"], ["0.5800", "3"]]})
    assert b.best_bid == pytest.approx(0.41)
    assert b.best_ask == pytest.approx(0.42)          # 1 - best NO bid 0.58
    assert b.ask_size() == 3
    w = OrderBook("T")
    w.apply_snapshot_ws({"yes_dollars_fp": [["0.4100", "5.00"]], "no_dollars_fp": [["0.4200", "3.00"]]})
    assert w.best_ask == pytest.approx(0.42)
    w.apply_delta_ws({"side": "no", "price_dollars": "0.4200", "delta_fp": "-3.00"})
    assert w.best_ask is None
    w.apply_delta_ws({"side": "yes", "price_dollars": "0.4300", "delta_fp": "2.00"})
    assert w.best_bid == pytest.approx(0.43)
    filled, cost, worst = b.cost_to_buy_no(6)
    assert filled == 6 and worst == pytest.approx(0.60)


# -------------------------------------------------------------- fees/limits
def test_fee_formula():
    assert taker_fee(0.5) == pytest.approx(0.0175)
    assert taker_fee(0.9, 10) == pytest.approx(0.063)


@pytest.mark.parametrize("p,edge", [(0.99, 0.05), (0.95, 0.02), (0.999, 0.1)])
def test_max_price_for_edge(p, edge):
    L = max_price_for_edge(p, edge)
    assert p - L - taker_fee(L) == pytest.approx(edge, abs=1e-6)


def test_tick_snapping_tapered_grid():
    c = Contract("T", "E", 0, 900, 1.0, "api", "active", 2,
                 [(0.0, 0.1, 0.001), (0.1, 0.9, 0.01), (0.9, 1.0, 0.001)])
    assert c.snap(0.9437, "down") == pytest.approx(0.943)
    assert c.snap(0.555, "down") == pytest.approx(0.55)
    assert c.snap(0.0555, "down") == pytest.approx(0.055)
    assert c.snap(0.555, "up") == pytest.approx(0.56)
    assert c.snap(0.99999, "down") == pytest.approx(0.999)


# ------------------------------------------------------------- accounting
def test_settlement_pnl_accounting(tmp_path):
    from kalshi_bot.core.ledger import Ledger
    from kalshi_bot.core.orders import Broker, OrderManager

    class Fake(Broker):
        async def place(self, o):
            return {"order_id": "o1", "fill_count": "10.00", "remaining_count": "0.00"}

    led = Ledger(tmp_path / "t.sqlite3")
    om = OrderManager(Fake(), led, "test")
    o = asyncio.run(om.buy("MKT", "yes", 0.9, 10, tif="immediate_or_cancel"))
    assert o.status == "executed" and o.filled == 10
    om.book_fill({"fill_id": "f1", "ticker": "MKT", "order_id": "o1", "outcome": "yes",
                  "price": 0.9, "count": 10.0, "fee": 0.063, "is_taker": True, "ts": time.time()})
    om.book_fill({"fill_id": "f1", "ticker": "MKT", "order_id": "o1", "outcome": "yes",
                  "price": 0.9, "count": 10.0, "fee": 0.063, "is_taker": True, "ts": time.time()})  # dup
    s = om.settle("MKT", "yes")
    assert s["payout"] == 10 and s["pnl"] == pytest.approx(10 - 9 - 0.063)
    assert led.stats("test")["pnl"] == pytest.approx(0.937)
