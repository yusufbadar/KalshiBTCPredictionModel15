"""
Load cached KXBTC15M history into aligned numpy arrays for research/backtests.

BRTI reconstruction
-------------------
Kalshi's event live-data gives A(t), the trailing 60-print average of BRTI
sampled every second (A(T) at a market's close is exactly its settlement
value). Raw prints are recovered from

    x(t) - x(t-60) = 60 * (A(t) - A(t-1))

which fixes x up to one unknown constant per residue class (t mod 60). Those
60 constants are chosen to make the path as smooth as possible (least squares
on first differences, a closed form here) and the overall level is pinned by
A itself. Validated against the live CF Benchmarks feed in tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_bot.research.history import ROOT, epoch_seconds


def reconstruct_prints(avg60: np.ndarray) -> np.ndarray:
    """Recover 1-second BRTI prints from a gap-free trailing-60 average series."""
    a = np.asarray(avg60, dtype=np.float64)
    n = len(a)
    if n < 180:
        raise ValueError("need at least 3 minutes of data")
    d = np.zeros(n)
    d[1:] = 60.0 * np.diff(a)            # x(t) - x(t-60)
    # y: cumulative sums inside each residue class, starting at 0
    y = np.zeros(n)
    for r in range(60):
        idx = np.arange(r, n, 60)
        y[idx] = np.cumsum(np.r_[0.0, d[idx[1:]]])
    # choose offsets c_r minimising sum_t (x(t) - x(t-1))^2, x = y + c[t mod 60]
    dy = np.diff(y)                       # dy[t-1] = y(t) - y(t-1)
    t = np.arange(1, n)
    r = t % 60
    delta = np.zeros(60)                  # delta_r = c_r - c_{r-1}
    for k in range(60):
        delta[k] = -dy[r == k].mean()
    delta -= delta.mean()                 # offsets must close the cycle
    c = np.cumsum(delta)                  # c_r up to a common constant (c_0 := delta_0)
    x = y + c[np.arange(n) % 60]
    # pin the level: the mean of x over the last 60 prints must equal a[-1]
    x += a[-1] - x[-60:].mean()
    return x


@dataclass
class Day:
    t0: int              # epoch seconds of first sample
    avg60: np.ndarray    # settlement-style trailing average A(t)
    x: np.ndarray        # reconstructed 1 s BRTI prints

    def idx(self, ts: int) -> int:
        return ts - self.t0


def load_markets() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "markets.parquet")
    df["open_ts"] = epoch_seconds(df.open_time)
    df["close_ts"] = epoch_seconds(df.close_time)
    return df


def load_brti(days: list[str] | None = None) -> tuple[int, np.ndarray, np.ndarray]:
    """Concatenate cached days into one continuous 1 s series.

    Returns (t0 epoch seconds, avg60 array, reconstructed prints array).
    Gaps are forward-filled for avg60 and the reconstruction is done per
    contiguous block so a gap never corrupts the residue offsets.
    """
    files = sorted((ROOT / "brti").glob("*.parquet"))
    if days:
        keep = set(days)
        files = [f for f in files if f.stem in keep]
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames).drop_duplicates("t_ms").sort_values("t_ms")
    ts = (df.t_ms.values // 1000).astype(np.int64)
    t0, t1 = int(ts[0]), int(ts[-1])
    n = t1 - t0 + 1
    a = np.full(n, np.nan)
    a[ts - t0] = df.avg60.values
    x = np.full(n, np.nan)
    ok = ~np.isnan(a)
    # contiguous blocks
    edges = np.flatnonzero(np.diff(ok.astype(np.int8)) != 0) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, n]
    for s, e in zip(starts, ends):
        if ok[s] and e - s >= 180:
            x[s:e] = reconstruct_prints(a[s:e])
    return t0, a, x


def market_prints(t0: int, avg60: np.ndarray, close_ts: int, span: int = 3600) -> tuple[int, np.ndarray] | None:
    """Reconstruct prints over the hour ending at ``close_ts`` (the same window
    Kalshi publishes per event). Reconstructing per market keeps any rare
    glitch in the published averages from leaking across markets.
    Returns (start epoch second, prints) or None if data is missing."""
    s = close_ts - span - t0
    e = close_ts - t0 + 1
    if s < 0 or e > len(avg60):
        return None
    seg = avg60[s:e]
    if np.isnan(seg).any():
        return None
    return close_ts - span, reconstruct_prints(seg)


def load_candles() -> pd.DataFrame:
    files = sorted((ROOT / "candles").glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_trades(ticker: str, close_time: pd.Timestamp) -> pd.DataFrame | None:
    day = (close_time - pd.Timedelta(microseconds=1)).normalize()
    p = ROOT / "trades" / f"{day:%Y-%m-%d}" / f"{ticker}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)
