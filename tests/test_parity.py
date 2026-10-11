"""Research <-> live parity: the live LockInStrategy must take the same trades
the research rule took, on the same inputs (historical panel rows), and the
defaults the bot ships with must be the parameters the report validated.

The panel test is skipped automatically when the cached research panel is not present."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalshi_bot.config import StrategyParams
from kalshi_bot.core.markets import Contract
from kalshi_bot.core.orderbook import OrderBook
from kalshi_bot.research.lockin import HOUSE_HOURLY, VOL_FLOOR_REL, backtest, eff_sigma, model_prob, model_z
from kalshi_bot.research.report import HOUSE
from kalshi_bot.strategy.base import TickState
from kalshi_bot.strategy.lockin import LockInStrategy

PANEL = Path(__file__).resolve().parents[1] / "storage" / "reports" / "panel_kxbtc15m.parquet"


def test_shipped_defaults_are_the_validated_parameters():
    sp = StrategyParams()
    assert sp.max_seconds_left == HOUSE.max_seconds_left
    assert sp.p_min == HOUSE.p_min and sp.z_min == HOUSE.z_min and sp.min_edge == HOUSE.min_edge
    assert sp.sigma_scale == HOUSE.sigma_scale and sp.sigma_time_k == HOUSE.sigma_time_k
    assert sp.tail_nu == HOUSE.nu and sp.max_price == HOUSE.max_price
    assert sp.edge_size_max == HOUSE.edge_size_max
    assert sp.max_contracts_per_order == HOUSE.max_per_order
    assert sp.max_position_per_market == HOUSE.max_per_market
    assert sp.vol_floor_rel == VOL_FLOOR_REL
    assert sp.early_seconds == HOUSE.early_seconds and sp.early_cushion_bp == HOUSE.early_cushion_bp
    # the hourly ladder ships the same rule with its own window and no early-lead requirement
    assert sp.hourly_max_seconds_left == HOUSE_HOURLY.max_seconds_left and HOUSE_HOURLY.early_cushion_bp == 0.0
    from dataclasses import replace
    assert replace(HOUSE_HOURLY, max_seconds_left=HOUSE.max_seconds_left, early_cushion_bp=HOUSE.early_cushion_bp) == HOUSE


@pytest.mark.skipif(not PANEL.exists(), reason="research panel not built")
def test_live_strategy_matches_research_rule():
    panel = pd.read_parquet(PANEL)
    panel = panel[panel.K.notna() & (panel.m <= 60)].reset_index(drop=True)
    days = panel.sec // 86400
    panel = panel[days >= days.max() - 3].reset_index(drop=True)      # last few days keeps it quick
    rp = HOUSE
    prob = model_prob(panel, rp.sigma_scale, rp.nu, time_k=rp.sigma_time_k)
    z = model_z(panel, rp.sigma_scale, time_k=rp.sigma_time_k)
    sig = eff_sigma(panel, rp.sigma_scale, time_k=rp.sigma_time_k)
    tr = backtest(panel, rp, prob, z).trades
    assert len(tr) > 20
    sp = StrategyParams(cooldown_s=0.0)                               # shipped defaults
    idx = panel.set_index(["ticker", "sec"])
    grid = [(0.0, 0.1, 0.001), (0.1, 0.9, 0.01), (0.9, 1.0, 0.001)]
    agree = 0
    for t in tr.itertuples():
        i = idx.index.get_loc((t.ticker, t.sec))
        r = panel.iloc[i]
        close = int(t.sec + r.m)
        c = Contract(t.ticker, "E", close - 900, close, float(r.K), "api", "active", 2, grid)
        b = OrderBook(t.ticker)
        # the research trigger is the last traded price on each side
        b.asks = {int(round(r.ask_prev * 10000)): 1000.0}
        b.bids = {int(round(r.bid_prev * 10000)): 1000.0}
        b.valid, b.updated = True, time.time()
        st = TickState(now=float(t.sec), contract=c, seconds_left=float(r.m), x=float(r.x), x_print=float(r.x),
                       strike=c.threshold, sigma=float(sig[i]), past_sum=float(r.past_sum),
                       past_n=int(r.past_n), p_model=float(prob[i]),
                       book=b, yes_qty=0.0, no_qty=0.0)
        out = LockInStrategy(sp).decide(st)
        if out and out[0].outcome == t.outcome and out[0].price + 1e-9 >= t.price:
            agree += 1
    print(f"live strategy agreed on {agree}/{len(tr)} research trades")
    assert agree / len(tr) >= 0.99, f"live strategy agreed on {agree}/{len(tr)} research trades"
