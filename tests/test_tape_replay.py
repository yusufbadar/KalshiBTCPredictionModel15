"""research/tape_replay.py runs the live strategy over a recorded market trade
by trade. These tests use a made-up market to pin down what it books."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from kalshi_bot.config import StrategyParams
from kalshi_bot.core.fees import taker_fee
from kalshi_bot.research.market_data import MarketData
from kalshi_bot.research.tape_replay import replay_market, settle, summary

CLOSE = 1_800_000_900
STRIKE = 80_000.0


def _market(trades, lead=30.0, y=1, ticker="KXBTC15M-T", life=900):
    """Index flat at strike + lead for the last hour; ``trades`` = (seconds before
    the close, YES price, contracts, taker bought YES)."""
    x = np.full(3601, STRIKE + lead)
    tr = sorted(trades, key=lambda r: -r[0])
    return MarketData(ticker=ticker, open_ts=CLOSE - life, close_ts=CLOSE, strike=STRIKE, y=y, t0=CLOSE - 3600,
                      a=x.copy(), x=x,
                      tr_t=np.array([CLOSE - r[0] for r in tr], float), tr_p=np.array([r[1] for r in tr], float),
                      tr_q=np.array([r[2] for r in tr], float), tr_buy=np.array([r[3] for r in tr], bool))


def _quiet(per_second=100.0, start=45.0, seconds=14):
    """Ordinary trading: someone buys YES at 99c every second."""
    return [(start - i, 0.99, per_second, True) for i in range(seconds)]


ON = StrategyParams()
OFF = replace(ON, fall_guard=0.0, sell_guard_x=0.0, guard_hold_s=0.0)


def test_buys_where_another_taker_bought_the_favourite():
    md = _market(_quiet() + [(30.4, 0.93, 400.0, True)])
    res = replay_market(md, "greater_or_equal", {"on": ON, "off": OFF}, equity=100.0, per_market=0.10)
    for name in ("on", "off"):
        r = res[name]
        assert r["yes"] and r["px"] == pytest.approx(0.93) and r["left"] == pytest.approx(30.4)
        (t, n, px, unit), = r["fills"]
        assert n == 10 and unit == pytest.approx(r["limit"] + taker_fee(r["limit"]))     # $10 at the limit price
    R = settle([res["on"]], 100.0, 0.15)
    assert len(R) == 1 and R.pnl.iloc[0] == pytest.approx(10 * (1 - 0.93 - taker_fee(0.93)))
    s = summary(R)
    assert (s["trades"], s["lost"]) == (1, 0)


def test_no_offer_is_assumed_where_nobody_bought():
    # the only prints near the bot's price are takers SELLING the favourite: that shows a bid, not an offer
    md = _market(_quiet() + [(30.4, 0.93, 400.0, False)])
    res = replay_market(md, "greater_or_equal", {"off": OFF})
    assert res["off"] == {}
    # ... and a favourite that is not clear of the line is never bought
    md = _market(_quiet() + [(30.4, 0.93, 400.0, True)], lead=0.2)
    assert replay_market(md, "greater_or_equal", {"off": OFF})["off"] == {}


def test_a_rush_of_selling_is_waited_out():
    burst = [(30.6, 0.95, 2000.0, False)]                      # 2,000 contracts dumped; a normal second here is 100
    md = _market(_quiet() + burst + [(30.4, 0.93, 400.0, True), (27.0, 0.94, 400.0, True)], y=0)
    res = replay_market(md, "greater_or_equal", {"on": ON, "off": OFF})
    assert res["off"]["left"] == pytest.approx(30.4)          # without the checks it buys straight into it
    assert res["on"]["left"] == pytest.approx(27.0) and res["on"]["waits"] >= 1
    assert settle([res["off"]], 100.0, 0.15).pnl.iloc[0] < 0


def test_size_is_limited_by_what_really_traded_and_by_the_shared_cap():
    md = _market(_quiet() + [(30.4, 0.93, 20.0, True)])       # only 20 contracts traded there: the bot gets a quarter
    r = replay_market(md, "greater_or_equal", {"off": OFF})["off"]
    assert r["fills"][0][1] == 5
    a = replay_market(_market(_quiet() + [(30.4, 0.93, 400.0, True)], ticker="KXBTC15M-A"), "greater_or_equal", {"off": OFF})["off"]
    b = replay_market(_market(_quiet() + [(20.4, 0.93, 400.0, True)], ticker="KXBTCD-B", life=3600), "greater", {"off": OFF})["off"]
    R = settle([a, b], 100.0, 0.15).set_index("ticker")       # same closing moment: $15 between them, first come first served
    assert R.loc["KXBTC15M-A", "qty"] == 10 and R.loc["KXBTCD-B", "qty"] == 5
    assert R.cost.sum() <= 15.0
    assert bool(R.loc["KXBTCD-B", "hourly"]) and not bool(R.loc["KXBTC15M-A", "hourly"])
