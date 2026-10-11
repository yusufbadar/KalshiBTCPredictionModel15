"""
Walk-forward evaluation of the optional maker add-on: in the final seconds,
rest a zero-fee post-only bid for the model favourite at (p_fav - margin),
only when the expected settlement is >= z_min standard deviations from the
strike. Fills use the conservative trade-through rule in ``maker.py``.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from kalshi_bot.research.lockin import model_prob, model_z
from kalshi_bot.research.maker import simulate_maker_market
from kalshi_bot.research.market_data import load_market, markets_with_trades


def _snap_down(p: np.ndarray) -> np.ndarray:
    step = np.where((p < 0.1) | (p >= 0.9), 0.001, 0.01)
    return np.floor(p / step + 1e-9) * step


def maker_trades(panel: pd.DataFrame, prob: np.ndarray, z: np.ndarray, max_m: int, z_min: float,
                 margin: float, cap: float = 0.97, size: float = 50.0, max_pos: float = 300.0,
                 mk: pd.DataFrame | None = None) -> pd.DataFrame:
    mk = markets_with_trades().set_index("ticker") if mk is None else mk
    d = panel.assign(p=prob, z=z)
    d = d[d.m <= max_m]
    out = []
    for tk, g in d.groupby("ticker", sort=False):
        if not ((g.z.values >= z_min) | (-g.z.values >= z_min)).any():
            continue
        md = load_market(SimpleNamespace(ticker=tk, **mk.loc[tk].to_dict()))
        if md is None:
            continue
        g = g.sort_values("sec")
        p, zz = g.p.values, g.z.values
        want = _snap_down(np.minimum(np.maximum(p, 1 - p) - margin, cap))
        by = np.where(zz >= z_min, want, np.nan)
        bn = np.where(-zz >= z_min, want, np.nan)
        for sec, o, px, q, _ in simulate_maker_market(md, g.sec.values, by, bn, size=size, max_pos=max_pos):
            win = (md.y == 1) if o == "yes" else (md.y == 0)
            out.append((tk, int(sec), int(md.close_ts - sec), o, px, q, 0.0, int(win),
                        float(win) - px, (float(win) - px) * q))
    return pd.DataFrame(out, columns=["ticker", "sec", "m", "outcome", "price", "qty", "fee_pc", "win",
                                      "pnl_pc", "pnl"])


def walk_forward_maker(panel: pd.DataFrame, sigma_scale: float = 1.2, fold_days: float = 7.0,
                       warmup_days: float = 7.0, log=print) -> pd.DataFrame:
    prob = model_prob(panel, sigma_scale, 5.0)
    z = model_z(panel, sigma_scale)
    mk = markets_with_trades().set_index("ticker")
    grid = [(m, zm, mg) for m in (60, 90) for zm in (2.0, 2.5, 3.0) for mg in (0.03, 0.05, 0.08)]
    t0, t1 = panel.sec.min(), panel.sec.max()
    cache: dict = {}
    for g in grid:
        cache[g] = maker_trades(panel, prob, z, *g, mk=mk)
    out = []
    for start in np.arange(t0 + warmup_days * 86400, t1 + 1, fold_days * 86400):
        end = start + fold_days * 86400

        def sc(tr):
            tr = tr[tr.sec < start]
            if tr.empty:
                return -1e9
            pm = tr.groupby("ticker").pnl.sum()
            return pm.sum() + 3 * min(0.0, pm.min())
        best = max(grid, key=lambda g: sc(cache[g]))
        te = cache[best]
        te = te[(te.sec >= start) & (te.sec < end)].assign(fold_params=str(best))
        log(f"maker fold {pd.to_datetime(start, unit='s'):%m-%d}: params {best} -> pnl {te.pnl.sum():.2f} "
            f"({te.ticker.nunique()} mkts)")
        out.append(te)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
