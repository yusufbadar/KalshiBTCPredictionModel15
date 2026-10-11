"""
Passive (maker) order simulation against the public trade tape.

A resting bid to BUY an outcome at price b is filled by a later public taker
trade that SELLS that outcome at a price strictly worse than b for the
taker (i.e. the trade printed below b): with our bid in the book at b the
incoming sell order would have matched us first, at b. Trades printing at
exactly b are ignored (unknown queue), so fills are conservative. Orders go
live ``latency`` seconds after the decision and cancels take effect
``latency`` seconds after the decision too. Maker fee on KXBTC15M is 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kalshi_bot.research.market_data import MarketData


def simulate_maker_market(md: MarketData, decide_sec: np.ndarray, bid_yes: np.ndarray, bid_no: np.ndarray,
                          size: float = 50.0, max_pos: float = 300.0, latency: float = 0.25,
                          participation: float = 1.0) -> list[tuple]:
    """decide_sec: 1 s grid; bid_yes / bid_no: desired resting limit (NaN = no order)
    for that second. Returns fills (sec, outcome, price, qty)."""
    fills = []
    pos = 0.0
    # map each trade to the order state that was live when it printed
    tt, tp, tq, tb = md.tr_t, md.tr_p, md.tr_q, md.tr_buy
    # order live interval for grid second k: [sec_k + latency, sec_{k+1} + latency)
    start = decide_sec + latency
    idx = np.searchsorted(start, tt, side="right") - 1          # grid row live at each trade
    valid = idx >= 0
    left_yes = {}
    left_no = {}
    for j in np.flatnonzero(valid):
        k = idx[j]
        if k >= len(decide_sec):
            continue
        if pos >= max_pos:
            break
        if not tb[j]:
            # taker sold YES at tp[j] -> hits a YES bid above tp
            b = bid_yes[k]
            if np.isnan(b) or tp[j] >= b - 1e-9:
                continue
            rem = left_yes.get(k, size)
            q = min(rem, tq[j] * participation, max_pos - pos)
            if q <= 0:
                continue
            left_yes[k] = rem - q
            pos += q
            fills.append((decide_sec[k], "yes", float(b), float(q), float(tt[j])))
        else:
            # taker bought YES at tp[j] -> lifts a YES ask (= our NO bid at 1-ask) below tp
            b = bid_no[k]
            if np.isnan(b) or tp[j] <= (1 - b) + 1e-9:
                continue
            rem = left_no.get(k, size)
            q = min(rem, tq[j] * participation, max_pos - pos)
            if q <= 0:
                continue
            left_no[k] = rem - q
            pos += q
            fills.append((decide_sec[k], "no", float(b), float(q), float(tt[j])))
    return fills
