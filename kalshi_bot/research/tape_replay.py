"""
Tape replay: the LIVE strategy code run over recorded markets, trade by trade.

The per-second backtest (research/lockin.py, research/report.py) looks at each
market once a second. The live bot reacts inside the second, and about half of
what it does never shows up in that backtest. This replay hands
``strategy/lockin.py`` what the engine hands it:

* the index print it would have had at that moment (prints reach the bot just
  after the second they are stamped with),
* every public trade up to that moment (the ``TradeTape`` the "market moved
  first" checks read),
* an offer wherever another taker actually bought the favourite: a real trade
  at that price proves the offer was there.

and books a fill at that trade's price, sized like the live risk limits (whole
contracts, a share of the account per market and per closing moment) and never
for more than a fraction of what really traded at or under the bot's limit in
the following second.

    python -m kalshi_bot.research.tape_replay                       # shipped rule vs. the checks switched off
    python -m kalshi_bot.research.tape_replay --per-market 0.10 --per-close 0.15

Checked against live trading on 13 Oct 2026: it reproduced 24 of the bot's 25
trades since the v3 rule, including both losing ones (RESEARCH.md section 8).
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kalshi_bot.config import STORAGE_DIR, RiskParams, StrategyParams
from kalshi_bot.core.fees import taker_fee
from kalshi_bot.core.markets import Contract
from kalshi_bot.core.orderbook import OrderBook
from kalshi_bot.core.tape import TradeTape
from kalshi_bot.model.fair_value import fair_yes, fair_yes_vec, settlement_zscore
from kalshi_bot.research.history import series_root
from kalshi_bot.research.lockin import max_price_for_edge as research_max_price
from kalshi_bot.research.market_data import load_market, markets_with_trades
from kalshi_bot.research.panel import market_panel, series_decimals
from kalshi_bot.strategy.base import TickState
from kalshi_bot.strategy.lockin import LockInStrategy, model_sigma

REPORT = STORAGE_DIR / "reports" / "tape_replay.json"
PRINT_DELAY = 0.05        # seconds after its timestamp that an index print is usable
PARTICIPATION = 0.25      # share of what traded at or under the limit that the bot is assumed to get
WARM_UP = 14.0            # seconds of tape fed before the settlement minute (the checks look back 11 s)
GRID = [(0.0, 0.1, 0.001), (0.1, 0.9, 0.01), (0.9, 1.0, 0.001)]
DEEP = 1e9                # the offer is assumed deep; liquidity is capped afterwards from what really traded


def replay_market(md, strike_type: str, variants: dict[str, StrategyParams], equity: float = 100.0,
                  per_market: float = 0.10) -> dict[str, dict]:
    """Replay one market for each named parameter set. Returns, per name, the
    fills ``(time, contracts, price, cost per contract at the limit)`` and what
    the strategy saw at the first one (empty dict = no trade)."""
    df = market_panel(md, 1, 75)
    close, sec0 = md.close_ts, int(df.sec.iloc[0])
    c = Contract(md.ticker, "EV", md.open_ts, close, md.strike, "api", "active", 2, GRID,
                 strike_type=strike_type or "greater_or_equal", decimals=series_decimals(md.ticker))
    thr = c.threshold
    x, psum, pn = df.x.values, df.past_sum.values, df.past_n.values.astype(int)
    sig0 = np.sqrt(df.v_mid.values)
    m = (close - df.sec.values).astype(float)
    T, P, Q, B = md.tr_t, md.tr_p, md.tr_q, md.tr_buy
    cap = per_market * equity
    out = {name: {} for name in variants}
    # which index print the bot has at the time of each trade
    row = np.ceil(T - PRINT_DELAY).astype(np.int64) - 1 - sec0
    inside = (row >= 0) & (row < len(x)) & (T < close)
    ri = np.where(inside, row, 0)
    # Cheap screen: a trade can only matter to a parameter set if, with the most confident reading of
    # that second (the latest moment in it), the favourite clears the gates and the trade's price is at or
    # under the highest price the bot would pay. Everything else is settled by the strategy itself below.
    states = {}
    any_cand = np.zeros(len(T), bool)
    for name, sp in variants.items():
        sl_lo = np.maximum(m - 1.0 - PRINT_DELAY, 0.0)
        sig_lo = np.array([model_sigma(float(s), float(xx), float(sl), sp) for s, xx, sl in zip(sig0, x, sl_lo)])
        p_hi = fair_yes_vec(x, thr, sig_lo, m, psum, pn, sp.tail_nu)
        yes_row = p_hi >= 0.5
        pf_hi = np.where(yes_row, p_hi, 1.0 - p_hi)
        z_hi = np.abs(settlement_zscore(x, thr, sig_lo, m, psum, pn))
        top = np.minimum(research_max_price(np.minimum(pf_hi + 0.002, 0.9999), sp.min_edge), sp.max_price)
        ok_row = (pf_hi >= sp.p_min - 0.002) & (z_hi >= sp.z_min - 0.02) & (m <= 62)
        fav_px = np.where(yes_row[ri], P, 1.0 - P)
        cand = inside & ok_row[ri] & (B == yes_row[ri]) & (fav_px <= top[ri] + 1e-9)
        if cand.any():
            any_cand |= cand
            states[name] = dict(sp=sp, strat=LockInStrategy(sp), cand=cand, yes=0.0, no=0.0, cost=0.0, fills=[],
                                first=None, waits=0, done=False)
    if not states:
        return out
    idx = np.flatnonzero(any_cand)
    tape = TradeTape()
    book = OrderBook(md.ticker)
    book.valid = True
    j = int(np.searchsorted(T, T[idx[0]] - WARM_UP))             # the checks look back 11 s from the first decision
    for jc in idx:
        while j < jc:                                            # everything that traded before this moment
            tape.add(md.ticker, float(T[j]), float(P[j]), float(Q[j]), bool(B[j]))
            j += 1
        t = float(T[jc])
        i = int(row[jc])
        g = sec0 + i
        for name, s in states.items():
            if s["done"] or not s["cand"][jc]:
                continue
            sp = s["sp"]
            sl = close - t
            sigma = model_sigma(float(sig0[i]), float(x[i]), sl, sp)
            p_yes = fair_yes(float(x[i]), thr, sigma, g, close, float(psum[i]), int(pn[i]), sp.tail_nu)
            yes_fav = p_yes >= 0.5
            if bool(B[jc]) != yes_fav:
                continue                                         # this taker bought the other side: no offer shown
            if yes_fav:
                book.asks, book.bids = {int(round(P[jc] * 10000)): DEEP}, {}
            else:
                book.bids, book.asks = {int(round(P[jc] * 10000)): DEEP}, {}
            st = TickState(now=t, contract=c, seconds_left=sl, x=float(x[i]), x_print=float(x[i]), strike=thr,
                           sigma=sigma, past_sum=float(psum[i]), past_n=int(pn[i]), p_model=p_yes, book=book,
                           yes_qty=s["yes"], no_qty=s["no"], tape=tape)
            before = s["strat"].waited.get(md.ticker)
            intents = s["strat"].decide(st)
            if s["strat"].waited.get(md.ticker) is not before:
                s["waits"] += 1
            if not intents:
                continue
            it = intents[0]
            px = float(P[jc]) if yes_fav else 1.0 - float(P[jc])
            unit = it.price + taker_fee(it.price)                # the risk manager budgets at the limit price
            k = int(np.searchsorted(T, t + 1.0, side="right"))
            fav = P[jc:k] if yes_fav else 1.0 - P[jc:k]
            traded = float(Q[jc:k][(B[jc:k] == yes_fav) & (fav <= it.price + 1e-9)].sum())
            n = math.floor(min(it.count, (cap - s["cost"]) / unit, PARTICIPATION * traded) + 1e-9)
            if n < 1:
                continue
            s["fills"].append((t, float(n), px, unit))
            s["cost"] += n * (px + taker_fee(px))
            s["yes" if yes_fav else "no"] += n
            if s["first"] is None:
                s["first"] = dict(t=t, left=sl, yes=bool(yes_fav), px=px, limit=float(it.price),
                                  p=float(p_yes if yes_fav else 1 - p_yes), waits=s["waits"])
            if (cap - s["cost"]) / unit < 1:
                s["done"] = True
        if all(s["done"] for s in states.values()):
            break
    for name, s in states.items():
        if s["fills"]:
            out[name] = dict(ticker=md.ticker, close_ts=int(close), y=int(md.y), fills=s["fills"], **s["first"])
    return out


def _worker(args):
    row, root, variants, equity, per_market = args
    md = load_market(row, Path(root))
    if md is None:
        return None
    try:
        return replay_market(md, getattr(row, "strike_type", None) or "greater_or_equal",
                             {k: StrategyParams(**v) for k, v in variants.items()}, equity, per_market)
    except Exception:  # noqa: BLE001 - a broken market file must not stop the run
        return None


def settle(markets: list[dict], equity: float, per_close: float) -> pd.DataFrame:
    """Apply the cap shared by everything that settles at the same moment
    (earliest fills first) and turn fills into one row per traded market."""
    rows = []
    by_close: dict[int, list[dict]] = {}
    for mk in markets:
        by_close.setdefault(mk["close_ts"], []).append(mk)
    for close, group in by_close.items():
        spent = 0.0
        book = {mk["ticker"]: dict(qty=0.0, cost=0.0, pnl=0.0) for mk in group}
        events = sorted((f[0], mk["ticker"], f, mk) for mk in group for f in mk["fills"])
        for _, tk, (t, n, px, unit), mk in events:
            n = min(n, math.floor((per_close * equity - spent) / unit + 1e-9))
            if n < 1:
                continue
            fee = taker_fee(px)
            win = (mk["y"] == 1) if mk["yes"] else (mk["y"] == 0)
            b = book[tk]
            b["qty"] += n
            b["cost"] += n * (px + fee)
            b["pnl"] += n * ((1.0 if win else 0.0) - px - fee)
            spent += n * (px + fee)
        for mk in group:
            b = book[mk["ticker"]]
            if b["qty"] > 0:
                rows.append({k: v for k, v in mk.items() if k != "fills"} | b)
    R = pd.DataFrame(rows)
    if not R.empty:
        R = R.sort_values(["close_ts", "t"]).reset_index(drop=True)
        R["hourly"] = ~R.ticker.str.split("-").str[0].str.endswith("15M")
    return R


def run(series: list[str], variants: dict[str, StrategyParams], equity: float = 100.0, per_market: float = 0.10,
        per_close: float = 0.15, workers: int = 6, since_ts: int | None = None, roots: dict | None = None,
        limit: int | None = None) -> dict[str, pd.DataFrame]:
    """Replay every downloaded market of the given series. Returns one frame of traded markets per variant."""
    jobs = []
    packed = {k: asdict(v) for k, v in variants.items()}
    for sname in series:
        root = (roots or {}).get(sname) or series_root(sname)
        mk = markets_with_trades(Path(root))
        if since_ts is not None:
            mk = mk[mk.close_ts > since_ts]
        recs = mk.to_dict("records")[:limit] if limit else mk.to_dict("records")
        jobs += [(SimpleNamespace(**r), str(root), packed, equity, per_market) for r in recs]
    found: dict[str, list] = {k: [] for k in variants}
    usable = 0
    if workers <= 1:
        results = map(_worker, jobs)
    else:
        ex = ProcessPoolExecutor(max_workers=workers)
        results = ex.map(_worker, jobs, chunksize=8)
    for res in results:
        if res is None:
            continue
        usable += 1
        for k, v in res.items():
            if v:
                found[k].append(v)
    out = {k: settle(v, equity, per_close) for k, v in found.items()}
    for R in out.values():
        R.attrs["markets"] = usable
    return out


def summary(R: pd.DataFrame, a: int | None = None, b: int | None = None) -> dict:
    if R is None or R.empty:
        return dict(trades=0, per_day=0.0, lost=0, lost_pct=0.0, pnl=0.0, dip=0.0, worst_day=0.0, red_days=0, days=0,
                    avg_win=0.0, avg_loss=0.0)
    if a is not None:
        R = R[R.close_ts > a]
    if b is not None:
        R = R[R.close_ts <= b]
    if R.empty:
        return summary(R.iloc[0:0])
    cum = np.r_[0.0, R.pnl.cumsum().values]
    day = R.groupby(R.close_ts // 86400).pnl.sum()
    span = max((R.close_ts.max() - (a if a is not None else R.close_ts.min())) / 86400, 1e-9)
    return dict(trades=int(len(R)), per_day=float(len(R) / span), lost=int((R.pnl < 0).sum()),
                lost_pct=float(100 * (R.pnl < 0).mean()), pnl=float(R.pnl.sum()),
                dip=float((np.maximum.accumulate(cum) - cum).max()), worst_day=float(day.min()),
                red_days=int((day < 0).sum()), days=int(len(day)),
                avg_win=float(R.pnl[R.pnl > 0].mean()) if (R.pnl > 0).any() else 0.0,
                avg_loss=float(R.pnl[R.pnl < 0].mean()) if (R.pnl < 0).any() else 0.0)


def shipped_variants() -> dict[str, StrategyParams]:
    sp = StrategyParams()
    return {"shipped rule (both checks)": sp,
            "price-fall check only": replace(sp, sell_guard_x=0.0),
            "selling check only": replace(sp, fall_guard=0.0),
            "without the checks": replace(sp, fall_guard=0.0, sell_guard_x=0.0)}


def main() -> None:
    rk = RiskParams()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--series", default="KXBTC15M,KXBTCD")
    ap.add_argument("--equity", type=float, default=100.0)
    ap.add_argument("--per-market", type=float, default=rk.pct_per_market)
    ap.add_argument("--per-close", type=float, default=rk.pct_per_close)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--since", default=None, help="only markets closing after this UTC date, e.g. 2026-09-27")
    a = ap.parse_args()
    since = int(pd.Timestamp(a.since, tz="UTC").timestamp()) if a.since else None
    series = [s.strip().upper() for s in a.series.split(",") if s.strip()]
    res = run(series, shipped_variants(), a.equity, a.per_market, a.per_close, a.workers, since)
    any_r = next(iter(res.values()))
    print(f"\n{any_r.attrs.get('markets', 0):,} markets replayed ({', '.join(series)}); ${a.equity:,.0f} account, "
          f"at most {100 * a.per_market:g}% per market and {100 * a.per_close:g}% per closing moment\n")
    print(f"{'':30s} {'trades':>6s} {'/day':>6s} {'lost':>5s} {'lost %':>7s} {'profit $':>9s} {'worst dip':>9s} {'worst day':>9s} {'red days':>9s}")
    rep = {}
    for name, R in res.items():
        s = summary(R)
        rep[name] = s
        print(f"{name:30s} {s['trades']:6d} {s['per_day']:6.1f} {s['lost']:5d} {s['lost_pct']:6.1f}% {s['pnl']:+9.1f} "
              f"{s['dip']:9.1f} {s['worst_day']:+9.1f} {s['red_days']:4d}/{s['days']}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"series": series, "equity": a.equity, "per_market": a.per_market,
                                  "per_close": a.per_close, "markets": any_r.attrs.get("markets", 0),
                                  "variants": rep}, indent=1), encoding="utf-8")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
