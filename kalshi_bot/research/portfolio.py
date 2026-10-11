"""
Combined backtest: KXBTC15M + the KXBTCD hourly strike ladder, with the
live bot's joint risk cap (dollars at risk across every contract that settles
at the same moment) and per-contract cap, applied in time order.

    python -m kalshi_bot.research.portfolio
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from kalshi_bot.config import STORAGE_DIR
from kalshi_bot.research.lockin import LockinParams, backtest, house_for

REPORTS = STORAGE_DIR / "reports"


def _trades(series: str, p: LockinParams) -> pd.DataFrame:
    panel = pd.read_parquet(REPORTS / f"panel_{series.lower()}.parquet")
    panel = panel[panel.K.notna() & (panel.m <= 120)].reset_index(drop=True)
    tr = backtest(panel, p).trades
    close = panel.groupby("ticker").sec.max() + 1          # grid ends one second before the close
    tr["close_ts"] = tr.ticker.map(close).astype(np.int64)
    tr["series"] = series
    return tr


def combine(tr: pd.DataFrame, per_contract: float, per_close: float) -> pd.DataFrame:
    """Apply per-contract and per-settlement-moment dollar caps in time order,
    scaling down (or dropping) fills that would breach them."""
    tr = tr.sort_values(["sec", "series"]).reset_index(drop=True)
    spent_c: dict = {}
    spent_t: dict = {}
    keep = []
    for r in tr.itertuples():
        cost_pc = r.price + r.fee_pc
        room = min(per_contract - spent_c.get(r.ticker, 0.0), per_close - spent_t.get(r.close_ts, 0.0))
        q = min(r.qty, np.floor(max(room, 0.0) / cost_pc * 100) / 100)
        if q < 1:
            continue
        spent_c[r.ticker] = spent_c.get(r.ticker, 0.0) + q * cost_pc
        spent_t[r.close_ts] = spent_t.get(r.close_ts, 0.0) + q * cost_pc
        keep.append((r.ticker, r.series, r.sec, r.close_ts, r.outcome, r.price, q, r.win, r.pnl_pc * q))
    return pd.DataFrame(keep, columns=["ticker", "series", "sec", "close_ts", "outcome", "price", "qty", "win", "pnl"])


def stats(t: pd.DataFrame, bankroll: float) -> dict:
    per_close = t.groupby("close_ts").pnl.sum().sort_index()
    eq = per_close.cumsum().values
    dd = float(np.max(np.maximum.accumulate(np.r_[0.0, eq])[1:] - eq)) if len(eq) else 0.0
    day = t.assign(d=t.sec // 86400).groupby("d").pnl.sum()
    days = (t.sec.max() - t.sec.min()) / 86400
    return {"pnl": float(t.pnl.sum()), "days": float(days), "per_day": float(t.pnl.sum() / days),
            "contracts": float(t.qty.sum()), "markets": int(t.ticker.nunique()),
            "max_drawdown": dd, "worst_close": float(per_close.min()),
            "neg_days": int((day < 0).sum()), "n_days": int(len(day)),
            "return_pct": 100 * float(t.pnl.sum()) / bankroll}


def main() -> None:
    both = pd.concat([_trades(s, house_for(s)) for s in ("KXBTC15M", "KXBTCD")], ignore_index=True)
    # common period (the hourly ladder was downloaded for the most recent 35 days)
    start = both[both.series == "KXBTCD"].sec.min()
    both = both[both.sec >= start]
    print(f"period {pd.to_datetime(start, unit='s'):%Y-%m-%d} -> {pd.to_datetime(both.sec.max(), unit='s'):%Y-%m-%d}")
    for label, per_c, per_t, bank in [("demo-sized ($100/contract, $150/close)", 100, 150, 200),
                                      ("$300/contract, $450/close", 300, 450, 1000),
                                      ("15-min only, $300/contract", 300, 300, 1000)]:
        src = both if "only" not in label else both[both.series == "KXBTC15M"]
        s = stats(combine(src, per_c, per_t), bank)
        print(f"{label:40s} pnl ${s['pnl']:8.0f}  ${s['per_day']:6.1f}/day  {s['markets']:4d} contracts traded  "
              f"maxDD ${s['max_drawdown']:6.0f}  worst close ${s['worst_close']:6.0f}  "
              f"losing days {s['neg_days']}/{s['n_days']}  return {s['return_pct']:.0f}% on ${bank}")


if __name__ == "__main__":
    main()
