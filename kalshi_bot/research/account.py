"""
What the backtested trades mean for a small account.

Replays every traded market of the chosen series for an account that starts
at $100 and uses the live bot's limits - a share of the account per contract
and a (larger) share across everything that settles at the same moment.

    python -m kalshi_bot.research.account                      # BTC 15-min, and with the hourly strikes
    python -m kalshi_bot.research.account --series KXBTC15M KXBTCD KXETH15M

Writes ``storage/reports/account.json`` (shown on the dashboard's Backtest tab).
The numbers assume the backtest's fills and edge carry over to live trading;
they are for choosing a position size, not a forecast.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from kalshi_bot.config import STORAGE_DIR
from kalshi_bot.research.bankroll import account_summary, close_returns
from kalshi_bot.research.lockin import LockinParams, backtest, house_for
from kalshi_bot.research.panel import series_decimals
from kalshi_bot.research.report import HOUSE

REPORTS = STORAGE_DIR / "reports"


def load_panel(series: str) -> pd.DataFrame | None:
    """The report panel if there is one, else the all-markets panel built by research.holdout --all."""
    for name in (f"panel_{series.lower()}.parquet", f"panel_{series.lower()}_all.parquet"):
        p = REPORTS / name
        if p.exists():
            panel = pd.read_parquet(p)
            panel = panel[panel.K.notna() & (panel.m <= 60)].reset_index(drop=True)
            if "Kt" not in panel.columns or panel.Kt.isna().any():
                half = 0.5 * 10.0 ** (-series_decimals(series + "-"))
                kt = panel.K + np.where(panel.strike_type.values == "greater", half, -half)
                panel["Kt"] = panel["Kt"].fillna(kt) if "Kt" in panel.columns else kt
            return panel
    return None


def series_trades(series: str, rule: LockinParams, since: int | None = None) -> pd.DataFrame:
    panel = load_panel(series)
    if panel is None:
        return pd.DataFrame()
    if since is not None:
        panel = panel[panel.sec >= since].reset_index(drop=True)
    # small-account view: no size-up, and only a quarter of each confirming trade
    r = backtest(panel, replace(rule, edge_size_max=1.0, max_per_order=400.0, max_per_market=4000.0,
                                participation=0.25))
    tr = r.trades
    if tr.empty:
        return tr
    tr["close_ts"] = (tr.sec + tr.m).astype(np.int64)
    tr["series"] = series
    return tr


def run(sets: dict[str, list[str]], rule: LockinParams | None = None, common_period: bool = True,
        limits=((0.03, 0.05), (0.06, 0.10), (0.10, 0.15))) -> dict:
    """``rule``: one rule for every series; by default each series uses the rule the bot ships for it."""
    all_series = sorted({s for v in sets.values() for s in v})
    trades = {s: series_trades(s, rule or house_for(s)) for s in all_series}
    trades = {s: t for s, t in trades.items() if not t.empty}
    out = {"rule": (rule or HOUSE).label(), "sets": []}
    for label, members in sets.items():
        have = [s for s in members if s in trades]
        if not have:
            continue
        tr = pd.concat([trades[s] for s in have], ignore_index=True)
        if common_period and len(have) > 1:
            start = max(trades[s].sec.min() for s in have)     # every series has data from here on
            tr = tr[tr.sec >= start]
        t0, t1 = tr.sec.min(), tr.sec.max()
        days = (t1 - t0) / 86400
        rows = []
        for fm, fc in limits:
            cr = close_returns(tr, fm, fc)
            rows.append({"f_market": fm, "f_close": fc, **account_summary(cr, days)})
        out["sets"].append({"label": label, "series": have, "days": float(days),
                            "range": f"{pd.to_datetime(t0, unit='s'):%b %d} - {pd.to_datetime(t1, unit='s'):%b %d}",
                            "rows": rows})
    return out


# the markets the bot trades by default (other coins showed no reliable edge - see RESEARCH.md)
DEFAULT_SETS = {
    "BTC 15-min only": ["KXBTC15M"],
    "BTC 15-min + hourly BTC strikes": ["KXBTC15M", "KXBTCD"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="*", default=None, help="one custom set instead of the default comparison")
    a = ap.parse_args()
    sets = {" + ".join(a.series): [s.upper() for s in a.series]} if a.series else DEFAULT_SETS
    res = run(sets)
    for st in res["sets"]:
        print(f"\n{st['label']}  ({st['range']}, {st['days']:.1f} days)")
        print(f"  {'limits (contract/close)':24s} {'mkts/day':>8s} {'avg day':>8s} {'losing days':>11s} {'worst close':>11s} "
              f"{'30d median':>10s} {'30d 5th pct':>11s} {'P(<$100)':>8s} {'P(DD>30%)':>9s} {'replay DD':>9s}")
        for r in st["rows"]:
            print(f"  {100 * r['f_market']:.0f}% / {100 * r['f_close']:.0f}%{'':15s} {r['per_day']:8.1f} "
                  f"{100 * r['mean_day']:7.2f}% {100 * r['losing_days']:10.0f}% {100 * r['worst_close']:10.1f}% "
                  f"{r['median']:10.0f} {r['p5']:11.0f} {100 * r['p_loss']:7.1f}% {100 * r['p_dd30']:8.1f}% "
                  f"{100 * r['hist_dd']:8.1f}%")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "account.json").write_text(json.dumps(res, default=float), encoding="utf-8")
    print(f"\nwrote {REPORTS / 'account.json'}")


if __name__ == "__main__":
    main()
