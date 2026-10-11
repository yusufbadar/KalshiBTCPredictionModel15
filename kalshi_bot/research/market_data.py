"""
Per-market research data: settlement index, trade tape and a 1-second grid.

Execution realism (used by the backtester)
------------------------------------------
There is no public history of Kalshi order books, so executable prices are
taken from the public trade tape:

* a taker BUY of YES decided at time t is filled at the price of the first
  public taker-buy-YES trade that prints after t + latency (within
  ``max_wait`` seconds); if that price is above our limit, or no such trade
  exists, there is no fill. Size is capped at a fraction of that trade's size.
* a taker BUY of NO (= selling YES) uses the first taker-sell-YES trade.

So we only ever "fill" at a price another trader actually received a moment
after our decision - never at a stale or hypothetical quote.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_bot.research.dataset import reconstruct_prints
from kalshi_bot.research.history import ROOT, epoch_seconds


@dataclass
class MarketData:
    ticker: str
    open_ts: int
    close_ts: int
    strike: float
    y: int
    t0: int                   # epoch second of x[0] / a[0]
    a: np.ndarray             # own event series: trailing 60-print average
    x: np.ndarray             # reconstructed 1 s prints
    tr_t: np.ndarray          # trade time, float epoch seconds
    tr_p: np.ndarray          # yes price
    tr_q: np.ndarray          # contracts
    tr_buy: np.ndarray        # True if taker bought YES

    def xi(self, sec: int) -> int:
        return sec - self.t0


def _day_of(close_time: pd.Timestamp) -> str:
    return f"{(close_time - pd.Timedelta(microseconds=1)).normalize():%Y-%m-%d}"


def _event_groups(root: Path, day: str) -> dict:
    p = root / "event_series" / f"{day}.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    return {k: g for k, g in df.groupby("ticker", sort=False)}


_group_cache: dict[tuple, dict] = {}


def load_market(row, root: Path = ROOT, span_s: int = 3600) -> MarketData | None:
    """row: a markets.parquet row (namedtuple/namespace) with open_ts/close_ts.
    Uses the last ``span_s`` seconds of the event's own index series."""
    if row.floor_strike is None or not np.isfinite(float(row.floor_strike)):
        return None                      # a few archived markets lack their strike
    day = _day_of(row.close_time)
    key = (str(root), day)
    if key not in _group_cache:
        if len(_group_cache) > 6:
            _group_cache.clear()
        _group_cache[key] = _event_groups(root, day)
    g = _group_cache[key].get(row.ticker)
    if g is None or len(g) < 3000:
        return None
    ts = (g.t_ms.values // 1000).astype(np.int64)
    order = np.argsort(ts)
    ts, a = ts[order], g.avg60.values[order]
    if ts[-1] != row.close_ts:
        return None
    ts, a = ts[-(span_s + 1):], a[-(span_s + 1):]
    if np.any(np.diff(ts) != 1):
        return None
    x = reconstruct_prints(a)
    tp = root / "trades" / day / f"{row.ticker}.parquet"
    if not tp.exists():
        return None
    tr = pd.read_parquet(tp)
    tr = tr[~tr.is_block]
    return MarketData(
        ticker=row.ticker, open_ts=int(row.open_ts), close_ts=int(row.close_ts),
        strike=float(row.floor_strike), y=int(row.result == "yes"), t0=int(ts[0]),
        a=a, x=x,
        tr_t=tr.ts_us.values / 1e6, tr_p=tr.yes_price.values.astype(float),
        tr_q=tr["count"].values.astype(float), tr_buy=tr.taker_yes.values.astype(bool),
    )


def markets_with_trades(root: Path = ROOT) -> pd.DataFrame:
    mk = pd.read_parquet(root / "markets.parquet")
    mk["open_ts"] = epoch_seconds(mk.open_time)
    mk["close_ts"] = epoch_seconds(mk.close_time)
    have = []
    for r in mk.itertuples():
        day = _day_of(r.close_time)
        have.append((root / "trades" / day / f"{r.ticker}.parquet").exists()
                    and (root / "event_series" / f"{day}.parquet").exists())
    return mk[np.array(have, bool)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Execution against the tape
# ---------------------------------------------------------------------------
def next_taker_trade(md: MarketData, t: float, buy_yes: bool, max_wait: float = 2.0):
    """First public taker trade on the given side strictly after t."""
    idx = np.searchsorted(md.tr_t, t, side="right")
    end = np.searchsorted(md.tr_t, t + max_wait, side="right")
    if idx >= end:
        return None
    side = md.tr_buy[idx:end] == buy_yes
    if not side.any():
        return None
    j = idx + int(np.argmax(side))
    return md.tr_t[j], md.tr_p[j], md.tr_q[j]


def fee_taker(p: float | np.ndarray, count: float | np.ndarray = 1.0):
    return 0.07 * count * p * (1 - p)
