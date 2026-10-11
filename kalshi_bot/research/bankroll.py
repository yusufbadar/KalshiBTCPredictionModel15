"""
Small-account simulation: what the backtest's trades mean for an account
that starts at $100 and sizes every position as a fraction of itself.

For each traded market the fills are replayed in time order until a fixed
stake has been spent, giving that market's return on stake (a full loss is
-100%). An account that puts the fraction ``f`` of its value into each market
then multiplies by (1 + f * return) market after market.

Two views:
* ``replay``     the actual sequence of markets, once;
* ``bootstrap``  many alternative histories built by drawing whole days with
                 replacement (keeps the within-day clustering of losses).

These are only as good as the backtest's fill assumptions and assume the edge
persists. They are for choosing a position size, not a forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def market_returns(trades: pd.DataFrame, stake: float = 20.0) -> pd.DataFrame:
    """One row per traded market: ticker, first fill second, dollars spent
    (<= stake) and P&L when the fills are taken in order up to ``stake``."""
    rows = []
    for tk, g in trades.sort_values("sec").groupby("ticker", sort=False):
        left, cost, pnl = stake, 0.0, 0.0
        for px, q, fee, win in zip(g.price.values, g.qty.values, g.fee_pc.values, g.win.values):
            unit = px + fee
            n = min(q, np.floor(left / unit * 100) / 100)
            if n < 1:
                break
            cost += n * unit
            pnl += n * (win - unit)
            left -= n * unit
        if cost > 0:
            rows.append((tk, int(g.sec.iloc[0]), cost, pnl))
    out = pd.DataFrame(rows, columns=["ticker", "sec", "cost", "pnl"]).sort_values("sec").reset_index(drop=True)
    out["ret"] = out.pnl / stake              # return on the intended stake (unfilled part earns 0)
    return out


def replay(mr: pd.DataFrame, f: float, start: float = 100.0) -> dict:
    eq = start * np.cumprod(1.0 + f * mr.ret.values)
    path = np.r_[start, eq]
    dd = float((1.0 - path / np.maximum.accumulate(path)).max()) if len(eq) else 0.0
    return {"final": float(path[-1]), "max_dd": dd}


def bootstrap(mr: pd.DataFrame, f: float, total_days: int, days: int = 30, n: int = 3000,
              start: float = 100.0, seed: int = 7) -> dict:
    """Draw ``days`` calendar days with replacement (days without trades included)."""
    rng = np.random.default_rng(seed)
    by_day = [g.ret.values for _, g in mr.groupby(mr.sec // 86400)]
    total_days = max(int(total_days), len(by_day))
    finals, dds = np.empty(n), np.empty(n)
    for i in range(n):
        picks = rng.integers(0, total_days, size=days)
        seq = [by_day[k] for k in picks if k < len(by_day)]
        if seq:
            path = np.r_[1.0, np.cumprod(1.0 + f * np.concatenate(seq))]
            finals[i] = start * path[-1]
            dds[i] = (1.0 - path / np.maximum.accumulate(path)).max()
        else:
            finals[i], dds[i] = start, 0.0
    return {"median": float(np.median(finals)), "p5": float(np.percentile(finals, 5)),
            "p1": float(np.percentile(finals, 1)), "p_loss": float(np.mean(finals < start)),
            "p_dd30": float(np.mean(dds > 0.30)), "p_half": float(np.mean(finals < 0.5 * start))}


def kelly_fraction(mr: pd.DataFrame) -> float:
    """Per-market stake fraction that maximises log growth on these returns."""
    fs = np.linspace(0.01, 0.99, 99)
    g = [np.mean(np.log1p(np.clip(f * mr.ret.values, -0.999999, None))) for f in fs]
    return float(fs[int(np.argmax(g))])


def account_table(trades: pd.DataFrame, total_days: float, fractions=(0.03, 0.06, 0.10, 0.15),
                  stake: float = 20.0) -> dict:
    mr = market_returns(trades, stake)
    if mr.empty:
        return {"rows": [], "markets": 0}
    rows = []
    for f in fractions:
        b = bootstrap(mr, f, int(round(total_days)))
        h = replay(mr, f)
        rows.append({"f": f, "per_day": len(mr) / max(total_days, 1e-9), **b,
                     "hist_dd": h["max_dd"], "hist_final": h["final"]})
    r = mr.ret
    return {"rows": rows, "markets": int(len(mr)), "mean_return": float(r.mean()),
            "loss_rate": float((r < 0).mean()), "kelly": kelly_fraction(mr)}


# ---------------------------------------------------------------------------
# Several series at once, with the live bot's account-relative limits
# ---------------------------------------------------------------------------
def close_returns(trades: pd.DataFrame, f_market: float, f_close: float, equity: float = 100.0) -> pd.DataFrame:
    """Return of a fixed-size account at every settlement moment.

    ``trades`` needs a ``close_ts`` column (and may mix series). Contracts
    that settle at the same moment share the per-close limit, exactly like the
    live risk manager: at most ``f_market`` of the account in one contract and
    ``f_close`` across everything settling together. Whole contracts only."""
    rows = []
    for close_ts, g in trades.sort_values("sec").groupby("close_ts", sort=True):
        spent_c: dict = {}
        spent = pnl = 0.0
        n_mk = set()
        for tk, px, q, fee, win in zip(g.ticker.values, g.price.values, g.qty.values, g.fee_pc.values, g.win.values):
            unit = px + fee
            room = min(f_market * equity - spent_c.get(tk, 0.0), f_close * equity - spent)
            n = np.floor(min(q, max(room, 0.0) / unit) + 1e-9)
            if n < 1:
                continue
            spent_c[tk] = spent_c.get(tk, 0.0) + n * unit
            spent += n * unit
            pnl += n * (win - unit)
            n_mk.add(tk)
        if n_mk:
            rows.append((int(close_ts), len(n_mk), spent, pnl, pnl / equity))
    return pd.DataFrame(rows, columns=["close_ts", "markets", "cost", "pnl", "ret"])


def account_summary(cr: pd.DataFrame, total_days: float, days: int = 30, n: int = 3000,
                    start: float = 100.0, seed: int = 7) -> dict:
    """Replay and day-bootstrap of an account compounding the per-close returns."""
    if cr.empty:
        return {"markets": 0}
    path = start * np.r_[1.0, np.cumprod(1.0 + cr.ret.values)]
    hist_dd = float((1.0 - path / np.maximum.accumulate(path)).max())
    rng = np.random.default_rng(seed)
    by_day = [g.ret.values for _, g in cr.groupby(cr.close_ts // 86400)]
    total = max(int(round(total_days)), len(by_day))
    finals, dds = np.empty(n), np.empty(n)
    for i in range(n):
        seq = [by_day[k] for k in rng.integers(0, total, size=days) if k < len(by_day)]
        if seq:
            pth = np.r_[1.0, np.cumprod(1.0 + np.concatenate(seq))]
            finals[i], dds[i] = start * pth[-1], (1.0 - pth / np.maximum.accumulate(pth)).max()
        else:
            finals[i], dds[i] = start, 0.0
    day_ret = cr.groupby(cr.close_ts // 86400).ret.sum()
    return {"markets": int(cr.markets.sum()), "per_day": float(cr.markets.sum() / max(total_days, 1e-9)),
            "closes": int(len(cr)), "loss_closes": float((cr.ret < 0).mean()),
            "worst_close": float(cr.ret.min()), "mean_day": float(day_ret.sum() / max(total, 1)),
            "losing_days": float((day_ret < 0).sum() / max(total, 1)),
            "median": float(np.median(finals)), "p5": float(np.percentile(finals, 5)),
            "p1": float(np.percentile(finals, 1)), "p_loss": float(np.mean(finals < start)),
            "p_dd30": float(np.mean(dds > 0.30)), "p_half": float(np.mean(finals < 0.5 * start)),
            "hist_dd": hist_dd, "hist_final": float(path[-1])}
