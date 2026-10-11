"""
The "settlement lock-in" strategy, research version.

Idea
----
KXBTC15M settles on the *average* of the last 60 one-second BRTI prints.
Once part of that minute has printed, the outcome is far more certain than
the spot-vs-strike distance suggests - but many traders keep pricing off
spot. When the exact averaging model says one side is a strong favourite
(p_fav >= p_min) and that side can be bought for at least ``min_edge`` below
the model value after the taker fee, buy it with an IOC order.

Everything here is vectorised over the per-second panel from ``panel.py``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from kalshi_bot.model.fair_value import fair_yes_vec, settlement_zscore
from kalshi_bot.research.backtest import TakerRule, run_taker

FEE = 0.07


@dataclass(frozen=True)
class LockinParams:
    max_seconds_left: int = 60       # only act in the last N seconds
    p_min: float = 0.95              # model probability of the side bought
    z_min: float = 2.0               # expected settlement distance from strike (s.d.)
    min_edge: float = 0.05           # p_fav - price - fee, per contract
    sigma_scale: float = 1.2         # conservative inflation of the EWMA vol
    nu: float = 5.0                  # Student-t tails
    max_quote_age: float = 3.0       # seconds since the last trade on that side
    max_price: float = 0.99          # never pay more than this
    cooldown_s: int = 1
    max_per_order: float = 50.0
    max_per_market: float = 300.0
    participation: float = 0.5
    edge_size_max: float = 1.0       # order size x clip(edge / min_edge, 1, edge_size_max)
    sigma_floor_rel: float = 0.0     # jump-risk floor: sigma_eff^2 = (scale*sigma)^2 + (floor*price)^2
    min_cushion_rel: float = 0.0     # |E[settlement] - strike| must be >= this fraction of price
    sigma_time_k: float = 0.0        # extra caution early in the window: sigma x (1 + k (m-30)/30) for m > 30
    early_seconds: int = 45          # while more than this many seconds remain ...
    early_cushion_bp: float = 0.0    # ... |E[settlement] - strike| must be >= this many bp of price (0 = off)

    def label(self) -> str:
        return (f"last {self.max_seconds_left}s, p>={self.p_min}, edge>={100 * self.min_edge:.0f}c, "
                f"sigma x{self.sigma_scale}")


VOL_FLOOR_REL = 6e-6      # live sigma floor, as a fraction of price per sqrt(s) (config.StrategyParams)


def eff_sigma(panel: pd.DataFrame, sigma_scale: float, floor_rel: float = 0.0,
              vol_col: str = "v_mid", time_k: float = 0.0) -> np.ndarray:
    """Volatility handed to the model, mirroring the live engine: the EWMA
    estimate (with the live floor), times the safety factor, times extra
    caution for decisions taken early in the averaging window."""
    s = np.maximum(np.sqrt(panel[vol_col].values), VOL_FLOOR_REL * panel.x.values) * sigma_scale
    if floor_rel > 0:
        s = np.sqrt(s ** 2 + (floor_rel * panel.x.values) ** 2)
    if time_k > 0:
        s = s * (1.0 + time_k * np.clip((panel.m.values - 30.0) / 30.0, 0.0, None))
    return s


def strike_level(panel: pd.DataFrame) -> np.ndarray:
    """Level the unrounded average must reach for YES (strike adjusted for
    settlement rounding) when the panel carries it, else the raw strike."""
    return panel.Kt.values if "Kt" in panel.columns else panel.K.values


def expected_settlement(panel: pd.DataFrame) -> np.ndarray:
    pn = panel.past_n.values
    return np.where(pn > 0, (panel.past_sum.values + (60 - pn) * panel.x.values) / 60, panel.x.values)


def model_prob(panel: pd.DataFrame, sigma_scale: float, nu: float, vol_col: str = "v_mid",
               floor_rel: float = 0.0, time_k: float = 0.0) -> np.ndarray:
    return fair_yes_vec(panel.x.values, strike_level(panel),
                        eff_sigma(panel, sigma_scale, floor_rel, vol_col, time_k),
                        panel.m.values, panel.past_sum.values, panel.past_n.values, nu)


def max_price_for_edge(p_win: np.ndarray, min_edge: float) -> np.ndarray:
    """Largest L with p_win - L - 0.07 L (1 - L) >= min_edge (vectorised).
    Solves 0.07 L^2 - 1.07 L + (p_win - min_edge) = 0 for the smaller root."""
    c = np.asarray(p_win, float) - min_edge
    disc = np.maximum(1.07 ** 2 - 4 * 0.07 * c, 0.0)
    return np.clip((1.07 - np.sqrt(disc)) / (2 * 0.07), 0.0, 0.999)


def make_rule(p: LockinParams, prob: np.ndarray, z: np.ndarray | None = None) -> TakerRule:
    """Mirror of the live strategy: buy the model favourite when the last
    traded price on that side is at or below the break-even-plus-edge limit;
    the simulated IOC carries that limit."""
    def signal(d: pd.DataFrame):
        a = d.ask_prev.values
        b = d.bid_prev.values
        inwin = (d.m.values <= p.max_seconds_left) & (d.m.values >= 1)
        fy = d.ask_age.values < p.max_quote_age
        fn = d.bid_age.values < p.max_quote_age
        zz = np.zeros(len(d)) if z is None else z
        if p.min_cushion_rel > 0:
            cushion_ok = np.abs(expected_settlement(d) - strike_level(d)) >= p.min_cushion_rel * d.x.values
            zz = np.where(cushion_ok, zz, 0.0)
        if p.early_cushion_bp > 0:           # mirrors strategy/lockin.py: a small lead early in the window is not enough
            lead = np.abs(expected_settlement(d) - strike_level(d))
            inwin = inwin & ((d.m.values <= p.early_seconds) | (lead >= p.early_cushion_bp * 1e-4 * d.x.values))
        lim_y = np.minimum(max_price_for_edge(prob, p.min_edge), p.max_price)
        lim_n = np.minimum(max_price_for_edge(1 - prob, p.min_edge), p.max_price)
        yes = np.where(inwin & fy & (prob >= p.p_min) & (zz >= p.z_min) & (a <= lim_y + 1e-9),
                       lim_y, np.nan)
        no = np.where(inwin & fn & ((1 - prob) >= p.p_min) & (-zz >= p.z_min) & ((1 - b) <= lim_n + 1e-9),
                      lim_n, np.nan)
        # size up with the edge available at the last traded price on the favourite side
        edge = np.where(prob >= 0.5, prob - a - FEE * a * (1 - a),
                        (1 - prob) - (1 - b) - FEE * (1 - b) * b)
        mult = np.clip(np.nan_to_num(edge / p.min_edge, nan=1.0), 1.0, p.edge_size_max)
        return yes, no, mult
    return TakerRule("lockin", signal, cooldown_s=p.cooldown_s, max_per_order=p.max_per_order,
                     max_per_market=p.max_per_market, participation=p.participation)


def model_z(panel: pd.DataFrame, sigma_scale: float, vol_col: str = "v_mid",
            floor_rel: float = 0.0, time_k: float = 0.0) -> np.ndarray:
    return settlement_zscore(panel.x.values, strike_level(panel),
                             eff_sigma(panel, sigma_scale, floor_rel, vol_col, time_k),
                             panel.m.values, panel.past_sum.values, panel.past_n.values)


def backtest(panel: pd.DataFrame, p: LockinParams, prob: np.ndarray | None = None,
             z: np.ndarray | None = None):
    if prob is None:
        prob = model_prob(panel, p.sigma_scale, p.nu, floor_rel=p.sigma_floor_rel, time_k=p.sigma_time_k)
    if z is None:
        z = model_z(panel, p.sigma_scale, floor_rel=p.sigma_floor_rel, time_k=p.sigma_time_k)
    return run_taker(panel, make_rule(p, prob, z))


# The rules the live bot ships with (config.StrategyParams mirrors these; tests/test_parity.py checks it).
# 15-minute markets: final 55 s, and a >= 2 bp lead while more than 45 s remain. Chosen on 11 Oct 2026 after
# the first live day, on history from before the bot went live (same profit, about half the drawdown).
HOUSE_15M = LockinParams(max_seconds_left=55, p_min=0.9, z_min=1.5, min_edge=0.01, sigma_scale=1.0,
                         edge_size_max=3.0, sigma_time_k=0.25, early_seconds=45, early_cushion_bp=2.0)
# Hourly strike ladder: unchanged (it loses about 1% of the time; both rules above would only cost profit there).
HOUSE_HOURLY = LockinParams(max_seconds_left=60, p_min=0.9, z_min=1.5, min_edge=0.01, sigma_scale=1.0,
                            edge_size_max=3.0, sigma_time_k=0.25)


def house_for(series: str) -> LockinParams:
    return HOUSE_15M if str(series).upper().endswith("15M") else HOUSE_HOURLY


SIGMA_SCALES = (1.0, 1.1, 1.2, 1.3, 1.5)
FLOORS = (0.0, 1e-5, 2e-5)
VOL_MODELS = tuple((sc, fl) for sc in SIGMA_SCALES for fl in FLOORS)


TIME_KS = (0.0, 0.25)          # early-window caution levels searched
EDGES = (0.01, 0.02, 0.03, 0.05, 0.08)


def param_grid(sigma_scale: float = 1.2, floor_rel: float = 0.0, time_k: float = 0.0) -> list[LockinParams]:
    """Trading thresholds searched once the volatility model is fixed by calibration.
    Only inside the settlement averaging minute, where the edge is structural."""
    out = []
    for m in (30, 45, 60):
        for zmin in (1.5, 2.0, 2.5):
            for e in EDGES:
                out.append(LockinParams(max_seconds_left=m, p_min=0.9, z_min=zmin, min_edge=e,
                                        sigma_scale=sigma_scale, sigma_floor_rel=floor_rel,
                                        edge_size_max=3.0, sigma_time_k=time_k))
    return out


def calibrate_vol_model(panel: pd.DataFrame, probs: dict, mask: np.ndarray, max_m: int = 60) -> tuple:
    """Pick (volatility factor, jump floor) with the best log-loss - a proper
    scoring rule - on the final ``max_m`` seconds of past markets, i.e. the
    region the strategy trades in."""
    sel = mask & (panel.m.values <= max_m) & (panel.m.values % 2 == 0)
    for k in VOL_MODELS:                          # only rows every candidate can score
        sel &= np.isfinite(probs[k])
    y = panel.y.values[sel]
    out = {}
    for k in VOL_MODELS:
        p = np.clip(probs[k][sel], 1e-7, 1 - 1e-7)
        out[k] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    if not all(np.isfinite(v) for v in out.values()):
        raise ValueError(f"non-finite calibration losses: {out}")
    best = min(out, key=out.get)
    return best, out


_K_CACHE: dict = {}


def _model_k(panel: pd.DataFrame, vm: tuple, k: float, nu: float, probs: dict, zs: dict):
    """Model probability / z-score for a volatility model plus an early-window factor."""
    if k == 0.0:
        return probs[vm], zs[vm]
    key = (id(panel), vm, k)
    if key not in _K_CACHE:
        if len(_K_CACHE) > 8:
            _K_CACHE.clear()
        _K_CACHE[key] = (model_prob(panel, vm[0], nu, floor_rel=vm[1], time_k=k),
                         model_z(panel, vm[0], floor_rel=vm[1], time_k=k))
    return _K_CACHE[key]


def score(summary: dict) -> float:
    """Selection objective: total P&L penalised three times over by the worst
    single market, so picks favour consistent edges over tail-risky ones."""
    if not summary.get("trades"):
        return -1e9
    return summary["pnl"] + 3.0 * min(0.0, summary["worst_market"])


def walk_forward(panel: pd.DataFrame, fold_days: float = 7.0, warmup_days: float = 7.0,
                 log=print, nu: float = 5.0) -> dict:
    """Expanding-window walk-forward. Before each fold, using only earlier data:
      1. choose the volatility factor by log-loss (calibration, not P&L);
      2. choose the trading thresholds by risk-penalised P&L.
    The fold is then traded with those choices. Returns out-of-sample trades."""
    t0, t1 = panel.sec.min(), panel.sec.max()
    probs = {(sc, fl): model_prob(panel, sc, nu, floor_rel=fl) for sc, fl in VOL_MODELS}
    zs = {(sc, fl): model_z(panel, sc, floor_rel=fl) for sc, fl in VOL_MODELS}
    edges = np.arange(t0 + warmup_days * 86400, t1 + 1, fold_days * 86400)
    oos, folds = [], []
    for i, start in enumerate(edges):
        end = min(start + fold_days * 86400, t1 + 1)
        train = panel.sec.values < start
        test = (panel.sec.values >= start) & (panel.sec.values < end)
        if test.sum() == 0:
            continue
        vm, losses = calibrate_vol_model(panel, probs, train)
        best, best_s, best_k = None, -1e18, 0.0
        ptr = panel[train]
        for k in TIME_KS:
            pk, zk = _model_k(panel, vm, k, nu, probs, zs)
            ptrain, ztrain = pk[train], zk[train]
            for g in param_grid(*vm, time_k=k):
                r = run_taker(ptr, make_rule(g, ptrain, ztrain))
                s = score(r.summary)
                if s > best_s:
                    best, best_s, best_k = g, s, k
        pk, zk = _model_k(panel, vm, best_k, nu, probs, zs)
        r = run_taker(panel[test], make_rule(best, pk[test], zk[test]))
        r.trades["fold"] = i
        oos.append(r.trades)
        folds.append({"fold": i, "start": int(start), "end": int(end), "params": asdict(best),
                      "vol_logloss": {f"{k[0]}|{k[1]}": v for k, v in losses.items()}, "train_score": best_s,
                      **{k: r.summary.get(k) for k in ("trades", "markets", "pnl", "pnl_per_contract_c",
                                                       "win_rate", "worst_market")}})
        log(f"fold {i}: {pd.to_datetime(start, unit='s'):%m-%d} -> {pd.to_datetime(end, unit='s'):%m-%d} "
            f"params [{best.label()}, floor {best.sigma_floor_rel:g}, z>={best.z_min}, early-k {best.sigma_time_k:g}] "
            f"-> test pnl {r.summary.get('pnl', 0):.2f} "
            f"({r.summary.get('trades', 0)} fills, {r.summary.get('markets', 0)} mkts)")
    trades = pd.concat(oos, ignore_index=True) if oos else pd.DataFrame()
    return {"trades": trades, "folds": folds}
