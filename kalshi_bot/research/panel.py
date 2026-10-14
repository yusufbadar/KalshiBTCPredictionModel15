"""
Build a per-second research panel from markets that have trade tapes.

Each row is one (market, second) with only information available at that
second (plus the executable prices of the *next* taker trades for
backtesting, clearly prefixed ``exec_``).
"""
from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from kalshi_bot.model.fair_value import fair_yes_vec, N_AVG
from kalshi_bot.research.market_data import MarketData, load_market, markets_with_trades

LATENCY = 0.25          # seconds from decision to order arrival at the exchange
MAX_WAIT = 2.0          # a fill must be confirmed by a public trade within this window


def ewma_var(inc2_per_s: np.ndarray, halflife: float) -> np.ndarray:
    """EWMA of a per-second variance series (inc2 already divided by horizon)."""
    lam = math.exp(math.log(0.5) / halflife)
    out = np.empty_like(inc2_per_s)
    acc = np.nanmean(inc2_per_s[:600]) if len(inc2_per_s) > 600 else np.nanmean(inc2_per_s)
    for i, v in enumerate(inc2_per_s):
        if not np.isnan(v):
            acc = lam * acc + (1 - lam) * v
        out[i] = acc
    return out


def _last_before(tt: np.ndarray, vals: np.ndarray, grid: np.ndarray):
    """Value and age of the last event with time <= grid point."""
    j = np.searchsorted(tt, grid, side="right") - 1
    ok = j >= 0
    v = np.full(len(grid), np.nan)
    age = np.full(len(grid), np.inf)
    v[ok] = vals[j[ok]]
    age[ok] = grid[ok] - tt[j[ok]]
    return v, age


def _next_after(tt: np.ndarray, vals: np.ndarray, qty: np.ndarray, grid: np.ndarray, wait: float):
    v = np.full(len(grid), np.nan)
    q = np.full(len(grid), np.nan)
    if len(tt) == 0:                    # nobody traded on this side at all
        return v, q
    j = np.searchsorted(tt, grid, side="right")
    ok = j < len(tt)
    jj = np.where(ok, j, 0)
    ok &= tt[jj] <= grid + wait
    v[ok] = vals[jj[ok]]
    q[ok] = qty[jj[ok]]
    return v, q


def _vol_in_window(tt, qq, grid, window):
    cs = np.r_[0.0, np.cumsum(qq)]
    hi = np.searchsorted(tt, grid, side="right")
    lo = np.searchsorted(tt, grid - window, side="right")
    return cs[hi] - cs[lo]


def market_panel(md: MarketData, step: int = 1, window_s: int | None = None) -> pd.DataFrame:
    x = md.x
    # 5-second increments -> per-second variance, aligned to the end second
    inc5 = np.full(len(x), np.nan)
    inc5[5:] = (x[5:] - x[:-5]) ** 2 / 5.0
    inc1 = np.full(len(x), np.nan)
    inc1[1:] = np.diff(x) ** 2
    v_fast = ewma_var(inc5, 60.0)
    v_mid = ewma_var(inc5, 300.0)
    v_slow = ewma_var(inc5, 1200.0)
    v1_mid = ewma_var(inc1, 300.0)

    first = md.open_ts + 1 if window_s is None else max(md.open_ts + 1, md.close_ts - window_s)
    secs = np.arange(first, md.close_ts, step)
    i = secs - md.t0
    m = md.close_ts - secs
    # prints already inside the settlement window [close-60, close-1]
    cs = np.r_[0.0, np.cumsum(x)]
    win_start = md.close_ts - N_AVG - md.t0
    past_n = np.clip(i - win_start + 1, 0, N_AVG)
    past_sum = np.where(past_n > 0, cs[np.minimum(i, win_start + N_AVG - 1) + 1] - cs[max(win_start, 0)], 0.0)

    buy = md.tr_buy
    tb_t, tb_p, tb_q = md.tr_t[buy], md.tr_p[buy], md.tr_q[buy]
    ts_t, ts_p, ts_q = md.tr_t[~buy], md.tr_p[~buy], md.tr_q[~buy]
    g = secs.astype(float)
    ask_prev, ask_age = _last_before(tb_t, tb_p, g)
    bid_prev, bid_age = _last_before(ts_t, ts_p, g)
    last_px, last_age = _last_before(md.tr_t, md.tr_p, g)
    exec_ask, exec_ask_q = _next_after(tb_t, tb_p, tb_q, g + LATENCY, MAX_WAIT)
    exec_bid, exec_bid_q = _next_after(ts_t, ts_p, ts_q, g + LATENCY, MAX_WAIT)
    # slower execution for stress tests (1 s decision-to-exchange)
    exec_ask_slow, exec_ask_slow_q = _next_after(tb_t, tb_p, tb_q, g + 1.0, MAX_WAIT)
    exec_bid_slow, exec_bid_slow_q = _next_after(ts_t, ts_p, ts_q, g + 1.0, MAX_WAIT)
    vb10 = _vol_in_window(tb_t, tb_q, g, 10.0)
    vs10 = _vol_in_window(ts_t, ts_q, g, 10.0)
    vb60 = _vol_in_window(tb_t, tb_q, g, 60.0)
    vs60 = _vol_in_window(ts_t, ts_q, g, 60.0)

    df = pd.DataFrame({
        "ticker": md.ticker, "sec": secs, "m": m, "K": md.strike, "x": x[i],
        "past_sum": past_sum, "past_n": past_n,
        "d5": x[i] - x[i - 5], "d30": x[i] - x[i - 30], "d60": x[i] - x[i - 60],
        "d300": x[i] - x[i - 300],
        "v_fast": v_fast[i], "v_mid": v_mid[i], "v_slow": v_slow[i], "v1_mid": v1_mid[i],
        "ask_prev": ask_prev, "ask_age": ask_age, "bid_prev": bid_prev, "bid_age": bid_age,
        "last_px": last_px, "last_age": last_age,
        "exec_ask": exec_ask, "exec_ask_q": exec_ask_q, "exec_bid": exec_bid, "exec_bid_q": exec_bid_q,
        "exec_ask_slow": exec_ask_slow, "exec_ask_slow_q": exec_ask_slow_q,
        "exec_bid_slow": exec_bid_slow, "exec_bid_slow_q": exec_bid_slow_q,
        "vb10": vb10, "vs10": vs10, "vb60": vb60, "vs60": vs60,
        "y": md.y, "open_ts": md.open_ts,
    })
    for name in ("v_fast", "v_mid", "v_slow"):
        df["p_" + name] = fair_yes_vec(df.x.values, md.strike, np.sqrt(df[name].values),
                                       df.m.values, df.past_sum.values, df.past_n.values, 5.0)
    return df


def series_decimals(ticker: str) -> int:
    """Precision Kalshi rounds the settlement average to for this market's coin."""
    from kalshi_bot.config import ASSETS
    series = ticker.split("-")[0]
    for a in ASSETS.values():
        if series in (a.series15, a.hourly):
            return a.decimals
    return 2


def _one(args):
    row, step, root, window_s = args
    md = load_market(row, root)
    if md is None:
        return None
    try:
        df = market_panel(md, step, window_s)
        st = getattr(row, "strike_type", "greater_or_equal") or "greater_or_equal"
        df["strike_type"] = st
        # settlement compares the *rounded* average with the strike
        half = 0.5 * 10.0 ** (-series_decimals(md.ticker))
        df["Kt"] = df.K + (half if st == "greater" else -half)
        return df
    except Exception:
        return None


def build_panel(step: int = 1, workers: int = 6, limit: int | None = None, root=None,
                window_s: int | None = None, since_ts: int | None = None,
                until_ts: int | None = None) -> pd.DataFrame:
    """``since_ts`` / ``until_ts``: only markets closing after / at or before that epoch second."""
    from types import SimpleNamespace
    from kalshi_bot.research.history import ROOT
    root = root or ROOT
    mk = markets_with_trades(root)
    if since_ts is not None:
        mk = mk[mk.close_ts > since_ts]
    if until_ts is not None:
        mk = mk[mk.close_ts <= until_ts]
    rows = [SimpleNamespace(**r) for r in mk.to_dict("records")]
    if limit:
        rows = rows[:limit]
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for df in ex.map(_one, [(r, step, root, window_s) for r in rows], chunksize=4):
            if df is not None:
                out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
