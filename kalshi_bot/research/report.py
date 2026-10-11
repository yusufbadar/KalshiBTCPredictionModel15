"""
End-to-end backtest report for the dashboard.

    python -m kalshi_bot.research.report            # uses cached data
    python -m kalshi_bot.research.report --download # refresh data first

Steps: build the per-second panel from every cached market with a full trade
tape -> expanding-window walk-forward parameter selection (weekly folds) ->
out-of-sample equity, daily/weekly breakdowns and robustness scenarios ->
``storage/reports/backtest.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from kalshi_bot.config import STORAGE_DIR
from kalshi_bot.research.backtest import summarize
from kalshi_bot.research.bankroll import account_table
from kalshi_bot.research.holdout import FROZEN, frozen_ts
from kalshi_bot.research.lockin import (HOUSE_15M, HOUSE_HOURLY, LockinParams, backtest, house_for, model_prob,
                                        model_z, walk_forward)
from kalshi_bot.research.panel import build_panel

REPORT_DIR = STORAGE_DIR / "reports"
# Defaults shipped in the live bot (config.StrategyParams mirrors these); defined in research/lockin.py
HOUSE = HOUSE_15M
BANKROLL = 1000.0


def _period_table(tr: pd.DataFrame, freq: str) -> list[dict]:
    if tr.empty:
        return []
    t = tr.assign(dt=pd.to_datetime(tr.sec, unit="s"))      # naive UTC
    key = t.dt.dt.to_period(freq)
    rows = []
    for k, g in t.groupby(key):
        pm = g.groupby("ticker").pnl.sum()
        label = (f"{k.start_time:%b %d} - {k.end_time:%b %d}" if freq.startswith("W") else f"{k.start_time:%a %b %d}")
        rows.append({"label": label, "markets": int(pm.size), "win_rate": float((pm > 0).mean()),
                     "pnl": float(g.pnl.sum()), "contracts": float(g.qty.sum())})
    return rows


def _calibration(panel: pd.DataFrame, prob: np.ndarray) -> list[dict]:
    """Model probability of the favourite vs realised, one observation per market
    per bin (final 60 s, where the strategy trades), so each market counts once per row."""
    d = panel[panel.m <= 60].assign(p=prob[panel.m.values <= 60] if len(prob) == len(panel) else prob)
    d = d.assign(pf=np.maximum(d.p, 1 - d.p), fw=np.where(d.p >= 0.5, d.y, 1 - d.y))
    d["bin"] = pd.cut(d.pf, [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1.0])
    first = d.sort_values("sec").groupby(["ticker", "bin"], observed=True).head(1)
    out = []
    for b, g in first.groupby("bin", observed=True):
        out.append({"label": f"{100 * b.left:.1f}-{100 * b.right:.1f}%", "markets": int(len(g)),
                    "model": float(g.pf.mean()), "realised": float(g.fw.mean())})
    return out


def _hour_table(tr: pd.DataFrame) -> list[dict]:
    if tr.empty:
        return []
    t = tr.assign(h=pd.to_datetime(tr.sec, unit="s", utc=True).dt.tz_convert("America/New_York").dt.hour
                  if _has_tz() else pd.to_datetime(tr.sec, unit="s", utc=True).dt.hour)
    rows = []
    for h, g in t.groupby("h"):
        rows.append({"label": f"{int(h):02d}:00", "markets": int(g.ticker.nunique()), "pnl": float(g.pnl.sum())})
    return rows


def _has_tz() -> bool:
    try:
        pd.Timestamp("2026-01-01", tz="UTC").tz_convert("America/New_York")
        return True
    except Exception:  # noqa: BLE001
        return False


def _equity(tr: pd.DataFrame) -> tuple[list, float, float]:
    if tr.empty:
        return [], 0.0, 0.0
    per_mkt = tr.groupby("ticker").agg(sec=("sec", "max"), pnl=("pnl", "sum")).sort_values("sec")
    # settle at the market close (last fill second is a close proxy)
    eq = per_mkt.pnl.cumsum()
    peak = np.maximum.accumulate(np.r_[0.0, eq.values])[1:]
    dd = float(np.max(peak - eq.values)) if len(eq) else 0.0
    dd_pct = float(np.max((peak - eq.values) / (BANKROLL + peak))) if len(eq) else 0.0
    pts = [[int(s), round(float(v), 4)] for s, v in zip(per_mkt.sec.values, eq.values)]
    return pts, dd, dd_pct


def build_report(panel: pd.DataFrame, log=print, fold_days: float = 7.0, warmup_days: float = 7.0,
                 series: str | None = None) -> dict:
    panel = panel[(panel.m <= 300) & panel.K.notna()].reset_index(drop=True)
    n_mk = panel.ticker.nunique()
    t0, t1 = panel.sec.min(), panel.sec.max()
    log(f"panel: {len(panel):,} rows, {n_mk} markets, "
        f"{pd.to_datetime(t0, unit='s'):%Y-%m-%d} -> {pd.to_datetime(t1, unit='s'):%Y-%m-%d}")

    wf = walk_forward(panel, fold_days=fold_days, warmup_days=warmup_days, log=log)
    tr = wf["trades"]
    s = summarize(tr)
    eq, dd, dd_pct = _equity(tr)
    s.update({"max_drawdown": dd, "max_drawdown_pct": dd_pct,
              "return_on_bankroll": s.get("pnl", 0.0) / BANKROLL})

    # robustness: fixed "house" parameters under harsher / different assumptions
    house = house_for(series) if series else HOUSE
    base_prob = model_prob(panel, house.sigma_scale, house.nu, time_k=house.sigma_time_k)
    base_z = model_z(panel, house.sigma_scale, time_k=house.sigma_time_k)
    scen = []
    variants = [
        ("House parameters, whole period", house),
        ("Participation 25% of each confirming trade", replace(house, participation=0.25)),
        ("Max 20 contracts per order", replace(house, max_per_order=20.0)),
        ("Flat sizing (no edge scaling)", replace(house, edge_size_max=1.0)),
        ("2x size caps (100/order, 600/market)", replace(house, max_per_order=100.0, max_per_market=600.0)),
        ("Final 60 s and no early-lead rule (the default until 11 Oct)", replace(house, max_seconds_left=60,
                                                                                early_cushion_bp=0.0)),
        ("Min edge 2c (the first version's threshold)", replace(house, min_edge=0.02)),
        ("No early-window caution", replace(house, sigma_time_k=0.0)),
        ("Conservative: edge >= 3c, z >= 2", replace(house, min_edge=0.03, z_min=2.0)),
        ("Min edge 5c", replace(house, min_edge=0.05)),
        ("z >= 2.5", replace(house, z_min=2.5)),
        ("Last 45 s only", replace(house, max_seconds_left=45)),
        ("Last 90 s (starts before the averaging window)", replace(house, max_seconds_left=90)),
        ("Jump floor 2e-5 x price", replace(house, sigma_floor_rel=2e-5)),
        ("Volatility factor 1.2 (more conservative)", replace(house, sigma_scale=1.2)),
        ("Volatility factor 1.5 (much more conservative)", replace(house, sigma_scale=1.5)),
    ]
    for label, p in variants:
        if (p.sigma_scale, p.nu, p.sigma_time_k, p.sigma_floor_rel) == (
                house.sigma_scale, house.nu, house.sigma_time_k, house.sigma_floor_rel):
            r = backtest(panel, p, base_prob, base_z)
        else:
            r = backtest(panel, p)
        scen.append({"label": label, "pnl": r.summary.get("pnl", 0.0),
                     "pnl_per_contract_c": r.summary.get("pnl_per_contract_c", 0.0),
                     "markets": r.summary.get("markets", 0),
                     "worst_market": r.summary.get("worst_market", 0.0)})
    # execution stress: every fill 2c worse than the tape says
    r = backtest(panel, house, base_prob, base_z)
    t2 = r.trades
    scen.insert(1, {"label": "Every fill 2c worse than the tape", "pnl": float((t2.pnl - 0.02 * t2.qty).sum()),
                    "pnl_per_contract_c": float(100 * (t2.pnl - 0.02 * t2.qty).sum() / t2.qty.sum()),
                    "markets": int(t2.ticker.nunique()), "worst_market": float((t2.pnl - 0.02 * t2.qty).groupby(t2.ticker).sum().min())})
    # latency stress: orders reach the exchange 1 s after the decision instead of 0.25 s
    slow = panel.copy()
    for c in ("exec_ask", "exec_ask_q", "exec_bid", "exec_bid_q"):
        slow[c] = slow[c.replace("exec_ask", "exec_ask_slow").replace("exec_bid", "exec_bid_slow")]
    r = backtest(slow, house, base_prob, base_z)
    scen.insert(1, {"label": "Latency 1.0 s instead of 0.25 s", "pnl": r.summary.get("pnl", 0.0),
                    "pnl_per_contract_c": r.summary.get("pnl_per_contract_c", 0.0),
                    "markets": r.summary.get("markets", 0), "worst_market": r.summary.get("worst_market", 0.0)})
    calib = _calibration(panel, base_prob)
    by_hour = _hour_table(tr)
    # ---- data the strategy had never seen when it was frozen
    holdout = None
    cut = frozen_ts(series) if series in FROZEN else None
    if cut is not None and t1 > cut + 86400:
        from kalshi_bot.research.holdout import evaluate, rules_for
        hp = panel[panel.sec > cut]
        rows, extra = evaluate(hp, rules_for(series))
        m = extra["meta"]
        holdout = {
            "range": f"{pd.to_datetime(m['start'], unit='s'):%b %d} - {pd.to_datetime(m['end'], unit='s'):%b %d, %Y} "
                     f"· {m['markets_in_panel']} markets · {m['days']:.1f} days",
            "note": ("These markets closed <b>after</b> the first version was finished on 27 Sep, so nothing here "
                     "was used to build or tune it. The first row is that version exactly as shipped. Rows two to "
                     "five were written down before this data was examined. The row marked <b>current</b> is what "
                     "the bot runs now; it was chosen on 11 Oct, after the first live day, using history from before "
                     "the bot went live - so for that row this period is not unseen."),
            "rows": rows}
    # ---- what the trades mean for a $100 account whose position size follows it
    full = backtest(panel, replace(house, edge_size_max=1.0, max_per_order=400.0, max_per_market=4000.0,
                                   participation=0.25), base_prob, base_z)
    days_all = (t1 - t0) / 86400
    acct = account_table(full.trades, days_all)
    acct["note"] = (
        f"Every traded market of the whole period ({acct.get('markets', 0)} markets, current default rule) replayed "
        f"for a small account that puts a fixed share of itself into each contract. A losing contract loses its "
        f"whole stake ({100 * acct.get('loss_rate', 0):.1f}% of traded markets here). <b>For choosing limits, not a "
        f"profit forecast:</b> it assumes the backtest's fills and edge carry over to live trading, which is not "
        f"guaranteed, and a 0% means only that it did not happen in the reshuffled backtest history. "
        f"The shipped limit is 6% per contract.")
    folds = [{"label": f"{pd.to_datetime(f['start'], unit='s'):%b %d} - "
                       f"{pd.to_datetime(f['end'] - 1, unit='s'):%b %d}",
              "markets": f.get("markets") or 0, "pnl": f.get("pnl") or 0.0,
              "pnl_per_contract_c": f.get("pnl_per_contract_c") or 0.0,
              "params": f["params"]} for f in wf["folds"]]
    oos_start = pd.to_datetime(wf["folds"][0]["start"], unit="s") if wf["folds"] else pd.to_datetime(t0, unit="s")
    rep = {
        "available": True,
        "generated": time.time(),
        "bankroll": BANKROLL,
        "range": f"out-of-sample {oos_start:%b %d} - {pd.to_datetime(t1, unit='s'):%b %d, %Y} "
                 f"({n_mk} markets in panel)",
        "description": (
            "<b>Settlement lock-in.</b> KXBTC15M settles on the average of the 60 one-second CF Benchmarks "
            "BRTI prints before the close. Once part of that minute has printed, the result is far more "
            "certain than the spot price suggests, and the market keeps pricing off spot. The bot tracks "
            "BRTI in real time, prices each contract with an exact averaging model (Student-t tails, "
            "live volatility), and buys the side the model rates a strong favourite whenever it can be "
            "taken below model value after Kalshi's taker fee."),
        "method": (
            "Data: Kalshi's public trade tape (every trade, microsecond timestamps) and the published "
            "1-second settlement-index series for every market; settlement values reconstructed from it "
            "match Kalshi's official values to the cent. Fills: an order decided at second t is filled only "
            "at the price of the next public taker trade on that side printed after t + 0.25 s (and within "
            "2 s), for at most 50% of that trade's size - never at a hypothetical quote. Fees: 0.07*p*(1-p) "
            "per contract on every fill. Parameters are chosen on past data only (expanding window) and "
            "evaluated on the following unseen week; only those out-of-sample weeks are shown."),
        "summary": s,
        "equity": eq,
        "by_week": _period_table(tr, "W-SUN"),
        "by_day": _period_table(tr, "D"),
        "folds": folds,
        "robustness": scen,
        "calibration": calib,
        "by_hour": by_hour,
        "house_params": asdict(house),
        "holdout": holdout,
        "account": acct,
    }
    return rep


def main() -> None:
    from kalshi_bot.research.history import series_root
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="KXBTC15M")
    ap.add_argument("--rebuild-panel", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fold-days", type=float, default=7.0)
    ap.add_argument("--warmup-days", type=float, default=7.0)
    a = ap.parse_args()
    series = a.series.upper()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cache = REPORT_DIR / f"panel_{series.lower()}.parquet"
    if a.rebuild_panel or not cache.exists():
        print(f"building panel for {series} ...", flush=True)
        # only the final 300 s of each contract are needed for the lock-in study
        panel = build_panel(step=1, workers=a.workers, root=series_root(series), window_s=300)
        panel.to_parquet(cache)
    else:
        panel = pd.read_parquet(cache)
        # markets downloaded since the panel was cached (their tapes hold the final ~2 minutes)
        newer = build_panel(step=1, workers=a.workers, root=series_root(series), window_s=120,
                            since_ts=int(panel.sec.max()) + 1)
        if not newer.empty:
            print(f"added {newer.ticker.nunique()} newer markets to the panel", flush=True)
            panel = pd.concat([panel, newer], ignore_index=True)
            panel.to_parquet(cache)
    if "Kt" not in panel.columns or panel.Kt.isna().any():
        from kalshi_bot.research.panel import series_decimals
        half = 0.5 * 10.0 ** (-series_decimals(series + "-"))
        kt = panel.K + np.where(panel.strike_type.values == "greater", half, -half)
        panel["Kt"] = panel["Kt"].fillna(kt) if "Kt" in panel.columns else kt
    rep = build_report(panel, fold_days=a.fold_days, warmup_days=a.warmup_days, series=series)
    rep["series"] = series
    name = "backtest.json" if series == "KXBTC15M" else f"backtest_{series.lower()}.json"
    out = REPORT_DIR / name
    out.write_text(json.dumps(rep, default=float), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(rep["summary"], indent=2, default=float))


if __name__ == "__main__":
    main()
