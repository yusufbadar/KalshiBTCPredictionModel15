"""
Taker backtest over the per-second research panel.

For each market, seconds are visited in time order. When a rule fires the
simulated IOC order is filled at the *next public taker trade* on that side
printed after decision time + latency (``exec_ask`` / ``exec_bid`` in the
panel), only if that price is within the rule's limit. Size is capped by that
trade's size times a participation fraction, the per-order cap and the
per-market position cap. Every contract pays $1 if its outcome wins; the
taker fee 0.07 * p * (1 - p) is charged on every fill.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

FEE = 0.07


@dataclass
class TakerRule:
    name: str
    signal: Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray]]
    # signal(df) -> (want_yes_limit, want_no_limit): limit price per row or NaN
    cooldown_s: int = 5
    max_per_order: float = 50.0
    max_per_market: float = 200.0
    participation: float = 0.5
    one_side_per_market: bool = True


@dataclass
class BTResult:
    trades: pd.DataFrame
    summary: dict = field(default_factory=dict)


def run_taker(panel: pd.DataFrame, rule: TakerRule) -> BTResult:
    df = panel
    sig = rule.signal(df)
    yes_lim, no_lim = sig[0], sig[1]
    size_mult = np.asarray(sig[2], float) if len(sig) > 2 else np.ones(len(df))
    yes_lim = np.asarray(yes_lim, float)
    no_lim = np.asarray(no_lim, float)
    ea = df.exec_ask.values
    eb = df.exec_bid.values
    eaq = df.exec_ask_q.values
    ebq = df.exec_bid_q.values
    tick = df.ticker.values
    sec = df.sec.values
    y = df.y.values
    m = df.m.values

    fire_yes = ~np.isnan(yes_lim) & ~np.isnan(ea) & (ea <= yes_lim + 1e-9)
    fire_no = ~np.isnan(no_lim) & ~np.isnan(eb) & ((1 - eb) <= no_lim + 1e-9)
    idx = np.flatnonzero(fire_yes | fire_no)
    rows = []
    last_t: dict = {}
    pos: dict = {}
    side_of: dict = {}
    for i in idx:
        tk = tick[i]
        if sec[i] - last_t.get(tk, -10**9) < rule.cooldown_s:
            continue
        held = pos.get(tk, 0.0)
        if held >= rule.max_per_market:
            continue
        if fire_yes[i] and (not rule.one_side_per_market or side_of.get(tk, "yes") == "yes"):
            outcome, price, avail = "yes", ea[i], eaq[i]
        elif fire_no[i] and (not rule.one_side_per_market or side_of.get(tk, "no") == "no"):
            outcome, price, avail = "no", 1 - eb[i], ebq[i]
        else:
            continue
        q = min(rule.max_per_order * size_mult[i], rule.participation * avail, rule.max_per_market - held)
        q = np.floor(q * 100) / 100
        if q < 1:
            continue
        win = (y[i] == 1) if outcome == "yes" else (y[i] == 0)
        fee = FEE * price * (1 - price)
        pnl = (1.0 if win else 0.0) - price - fee
        rows.append((tk, sec[i], m[i], outcome, price, q, fee, int(win), pnl, pnl * q))
        last_t[tk] = sec[i]
        pos[tk] = held + q
        side_of[tk] = outcome
    tr = pd.DataFrame(rows, columns=["ticker", "sec", "m", "outcome", "price", "qty", "fee_pc",
                                     "win", "pnl_pc", "pnl"])
    return BTResult(tr, summarize(tr))


def summarize(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {"trades": 0}
    per_mkt = tr.groupby("ticker").pnl.sum()
    cost = (tr.price * tr.qty).sum()
    days = (tr.sec.max() - tr.sec.min()) / 86400 or 1
    daily = tr.assign(day=(tr.sec // 86400)).groupby("day").pnl.sum()
    return {
        "trades": int(len(tr)), "markets": int(tr.ticker.nunique()),
        "contracts": float(tr.qty.sum()),
        "avg_price": float((tr.price * tr.qty).sum() / tr.qty.sum()),
        "win_rate": float((tr.win * tr.qty).sum() / tr.qty.sum()),
        "pnl": float(tr.pnl.sum()),
        "pnl_per_contract_c": float(100 * tr.pnl.sum() / tr.qty.sum()),
        "roi_on_cost": float(tr.pnl.sum() / cost) if cost else 0.0,
        "fees": float((tr.fee_pc * tr.qty).sum()),
        "market_win_rate": float((per_mkt > 0).mean()),
        "worst_market": float(per_mkt.min()), "best_market": float(per_mkt.max()),
        "daily_mean": float(daily.mean()), "daily_std": float(daily.std(ddof=1)) if len(daily) > 1 else 0.0,
        "sharpe_daily": float(daily.mean() / daily.std(ddof=1) * np.sqrt(365)) if len(daily) > 1 and daily.std(ddof=1) > 0 else 0.0,
        "t_stat_markets": float(per_mkt.mean() / (per_mkt.std(ddof=1) / np.sqrt(len(per_mkt)))) if len(per_mkt) > 2 and per_mkt.std(ddof=1) > 0 else 0.0,
        "days": float(days),
    }
