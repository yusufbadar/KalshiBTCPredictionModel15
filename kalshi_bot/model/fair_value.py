"""
Fair value of a KXBTC15M contract.

Contract: YES pays $1 if  S >= K  where

    S = mean of the 60 one-second BRTI prints stamped close-60 ... close-1
    K = the same average taken at the market's open (floor_strike)

(Verified against 568 official settlements: the published expiration value
equals the mean of the 1 Hz prints stamped T-60..T-1 in 98.6% of markets to
the cent, and the remainder differ by a cent. Kalshi's docs describe the
window as (T-60, T], which is one second later than what settles.)

Model: over a 15-minute horizon BTC is a driftless arithmetic random walk in
dollars, x(s) = x(t) + sigma * W(s - t), with per-second dollar volatility
``sigma`` supplied by the caller. The averaging is handled exactly for 1 s
sampling. Let L = close - 1 (last print in the window) and m = L - t the
number of prints still to come after the current one at t.

* Before the window starts (t < close - 60), with gap d = (close - 60) - t:
      S - x(t) ~ N(0, sigma^2 * (d - 1 + (n+1)(2n+1)/(6n)))    n = 60
  i.e. the final minute contributes ~20.5 s of variance, not 60 s.
* Inside the window, with k prints already observed (sum ``past_sum``):
      S = (past_sum + sum_{j=1..m} x(t+j)) / 60,     m = 60 - k
      E[S] = (past_sum + m x(t)) / 60
      Var[S] = sigma^2 * m (m+1) (2m+1) / (6 * 60^2)

Fat tails: BTC second-level returns are leptokurtic, so a Student-t with the
same variance is used (``nu``; nu = inf gives the Gaussian).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import special, stats

N_AVG = 60
_WINDOW_VAR = (N_AVG + 1) * (2 * N_AVG + 1) / (6.0 * N_AVG)   # 20.508 s


def window_bounds(close_ts: int) -> tuple[int, int]:
    """First and last print second that enter the settlement average."""
    return close_ts - N_AVG, close_ts - 1


def _var_seconds(t_rel_first: np.ndarray | float, remaining: np.ndarray | float, inside: np.ndarray | bool):
    """Variance multiplier (sigma^2 units) of the not-yet-known part of S.

    t_rel_first: seconds from now until the first window print (>0 outside)
    remaining:   prints still to come inside the window (inside case)
    """
    out_var = (np.asarray(t_rel_first, float) - 1.0) + _WINDOW_VAR
    r = np.asarray(remaining, float)
    in_var = r * (r + 1) * (2 * r + 1) / (6.0 * N_AVG ** 2)
    return np.where(inside, in_var, out_var)


def _cdf(z, nu: float):
    if not np.isfinite(nu):
        return special.ndtr(z)
    scale = math.sqrt(nu / (nu - 2.0))
    return stats.t.cdf(np.asarray(z) * scale, df=nu)


def fair_yes(x_now: float, strike: float, sigma: float, now_sec: float, close_ts: int,
             past_sum: float = 0.0, past_n: int = 0, nu: float = 5.0) -> float:
    """P(YES) at time ``now_sec`` given the current print ``x_now``.

    ``past_sum``/``past_n``: sum and count of window prints already observed
    (prints stamped >= close-60 and <= now). Callers keep past_n consistent
    with now_sec.
    """
    return float(fair_yes_vec(x_now, strike, sigma, close_ts - now_sec, past_sum, past_n, nu))


def fair_yes_vec(x_now, strike, sigma, seconds_left, past_sum, past_n, nu: float = 5.0):
    """Vectorised P(YES). ``seconds_left`` = close_ts - now (integer seconds)."""
    x_now = np.asarray(x_now, float)
    strike = np.asarray(strike, float)
    sigma = np.maximum(np.asarray(sigma, float), 1e-9)
    sl = np.asarray(seconds_left, float)
    past_sum = np.asarray(past_sum, float)
    past_n = np.asarray(past_n, float)
    inside = past_n > 0
    remaining = N_AVG - past_n                       # prints still to come (inside)
    until_first = sl - N_AVG                         # seconds until first window print (outside)
    mu = np.where(inside, (past_sum + remaining * x_now) / N_AVG, x_now)
    var = _var_seconds(np.maximum(until_first, 1.0), remaining, inside)
    sd = sigma * np.sqrt(np.maximum(var, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (mu - strike) / sd
    p = _cdf(z, nu)
    done = inside & (remaining <= 0)
    p = np.where(done, (mu >= strike).astype(float), p)
    return p


def settlement_zscore(x_now, strike, sigma, seconds_left, past_sum, past_n):
    """Signed distance of the expected settlement from the strike in s.d. units."""
    x_now = np.asarray(x_now, float)
    past_n = np.asarray(past_n, float)
    inside = past_n > 0
    remaining = N_AVG - past_n
    mu = np.where(inside, (np.asarray(past_sum, float) + remaining * x_now) / N_AVG, x_now)
    var = _var_seconds(np.maximum(np.asarray(seconds_left, float) - N_AVG, 1.0), remaining, inside)
    return (mu - np.asarray(strike, float)) / (np.asarray(sigma, float) * np.sqrt(np.maximum(var, 1e-12)))
