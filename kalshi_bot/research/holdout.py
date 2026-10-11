"""
Unseen-data check.

The strategy and its parameters were frozen on data up to 27 Sep 2026. This
module replays fixed rules on markets that closed *after* that - data that
played no part in building or tuning anything - and reports how they did.

    python -m kalshi_bot.research.history --days 14 --tail-seconds 130   # fetch new markets
    python -m kalshi_bot.research.holdout                                # KXBTC15M
    python -m kalshi_bot.research.holdout --series KXETH15M --all        # a coin with no earlier study
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from kalshi_bot.config import STORAGE_DIR
from kalshi_bot.research.backtest import summarize
from kalshi_bot.research.history import series_root
from kalshi_bot.research.lockin import HOUSE_15M, LockinParams, backtest
from kalshi_bot.research.panel import build_panel

REPORT_DIR = STORAGE_DIR / "reports"
# last moment covered by the original research (per series)
FROZEN = {"KXBTC15M": "2026-09-27 22:45", "KXBTCD": "2026-09-28 00:00"}

# v1: the rule shipped on 28 Sep (frozen before any of the holdout existed)
V1 = LockinParams(max_seconds_left=60, p_min=0.9, z_min=1.5, min_edge=0.02, sigma_scale=1.0, edge_size_max=3.0)
RULES = {
    "v1 as shipped (edge >= 2c)": V1,
    "v2: edge >= 1c + early-window caution": LockinParams(60, 0.9, 1.5, 0.01, 1.0, edge_size_max=3.0,
                                                           sigma_time_k=0.25),
    "edge >= 1c, no extra caution": LockinParams(60, 0.9, 1.5, 0.01, 1.0, edge_size_max=3.0),
    "conservative: sigma x1.2, edge >= 2c": LockinParams(60, 0.9, 1.5, 0.02, 1.2, edge_size_max=3.0),
    "strict: z >= 2, edge >= 2c": LockinParams(60, 0.9, 2.0, 0.02, 1.0, edge_size_max=3.0),
}
CURRENT_15M = "current: v2 in the final 55 s + early lead (chosen 11 Oct)"


def rules_for(series: str) -> dict:
    """Rules to replay for a series. 15-minute markets also get the current
    default, which was picked after this period (see the report's note); for
    the hourly ladder the current default is still v2."""
    rules = dict(RULES)
    if str(series).upper().endswith("15M"):
        rules[CURRENT_15M] = HOUSE_15M
    return rules


def frozen_ts(series: str) -> int:
    return int(pd.Timestamp(FROZEN.get(series, "2026-09-28 00:00"), tz="UTC").timestamp())


def holdout_panel(series: str, workers: int = 6, rebuild: bool = False, everything: bool = False) -> pd.DataFrame:
    """Final two minutes of every market after the freeze (``everything``:
    all cached markets, for series the original research never used)."""
    cache = REPORT_DIR / f"panel_{series.lower()}_{'all' if everything else 'holdout'}.parquet"
    if cache.exists() and not rebuild:
        return pd.read_parquet(cache)
    panel = build_panel(root=series_root(series), window_s=120, workers=workers,
                        since_ts=None if everything else frozen_ts(series))
    if not panel.empty:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cache)
    return panel


def max_drawdown(tr: pd.DataFrame) -> float:
    if tr.empty:
        return 0.0
    eq = tr.groupby("ticker").agg(sec=("sec", "max"), pnl=("pnl", "sum")).sort_values("sec").pnl.cumsum().values
    return float((np.maximum.accumulate(np.r_[0.0, eq]) - np.r_[0.0, eq]).max())


def evaluate(panel: pd.DataFrame, rules: dict | None = None) -> tuple[list[dict], dict]:
    """Rows of results per rule and the trades of each."""
    panel = panel[panel.K.notna() & (panel.m <= 60)].reset_index(drop=True)
    days = max((panel.sec.max() - panel.sec.min()) / 86400, 1e-9)
    rows, trades = [], {}
    for label, p in (rules or RULES).items():
        r = backtest(panel, p)
        s = r.summary if r.summary.get("trades") else summarize(r.trades)
        trades[label] = r.trades
        rows.append({"label": label, "markets": s.get("markets", 0), "per_day": s.get("markets", 0) / days,
                     "contracts": s.get("contracts", 0.0), "pnl": s.get("pnl", 0.0),
                     "pnl_per_contract_c": s.get("pnl_per_contract_c", 0.0),
                     "market_win_rate": s.get("market_win_rate", 0.0),
                     "worst_market": s.get("worst_market", 0.0), "t_stat": s.get("t_stat_markets", 0.0),
                     "max_drawdown": max_drawdown(r.trades)})
    meta = {"markets_in_panel": int(panel.ticker.nunique()), "days": float(days),
            "start": int(panel.sec.min()), "end": int(panel.sec.max())}
    return rows, {"meta": meta, "trades": trades}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="KXBTC15M")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--all", action="store_true", help="use every cached market (series with no earlier study)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    series = a.series.upper()
    panel = holdout_panel(series, a.workers, a.rebuild, a.all)
    if panel.empty:
        raise SystemExit("no markets after the freeze are cached - run research.history first")
    rows, extra = evaluate(panel, rules_for(series))
    m = extra["meta"]
    print(f"\n{series}: {m['markets_in_panel']} markets, {pd.to_datetime(m['start'], unit='s'):%b %d %H:%M} -> "
          f"{pd.to_datetime(m['end'], unit='s'):%b %d %H:%M} UTC ({m['days']:.1f} days)\n")
    print(f"{'rule':40s} {'mkts':>5s} {'/day':>5s} {'contracts':>9s} {'pnl':>9s} {'c/ct':>6s} {'win':>6s} "
          f"{'worst':>8s} {'maxDD':>8s} {'t':>5s}")
    for r in rows:
        print(f"{r['label']:40s} {r['markets']:5d} {r['per_day']:5.1f} {r['contracts']:9.0f} {r['pnl']:9.2f} "
              f"{r['pnl_per_contract_c']:6.2f} {r['market_win_rate']:6.3f} {r['worst_market']:8.2f} "
              f"{r['max_drawdown']:8.2f} {r['t_stat']:5.2f}")
    tr = extra["trades"][next(iter(RULES))]
    if not tr.empty:
        d = tr.assign(day=pd.to_datetime(tr.sec, unit="s").dt.strftime("%b %d")).groupby("day", sort=False).agg(
            markets=("ticker", "nunique"), contracts=("qty", "sum"), pnl=("pnl", "sum"))
        print("\nby day, first rule:\n" + d.round(2).to_string())
    out = REPORT_DIR / f"holdout_{series.lower()}.json"
    out.write_text(json.dumps({"series": series, **m, "rows": rows}, default=float), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
