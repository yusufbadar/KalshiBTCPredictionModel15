"""
Backtest of the original v1 strategy on real Kalshi execution, for comparison.

v1 rule (legacy/kalshi_bot/strategy/trading_logic.py): at minute 10 of each
15-minute window look at the two completed 5-minute candles:
  both green and price above the 20-period EMA of 5-minute closes -> buy YES
  both red and price below the EMA                                 -> buy NO
  otherwise                         -> side of the candle with the bigger body
then hold to settlement.

Here candles are built from the BRTI index itself (the settlement source)
instead of Binance BTCUSDT, and the entry is filled at the next public taker
trade on that side after minute 10 + 0.25 s, with the 0.07*p*(1-p) taker fee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kalshi_bot.research.dataset import load_brti
from kalshi_bot.research.market_data import load_market, markets_with_trades, next_taker_trade


def ema(values: np.ndarray, period: int = 20) -> float:
    if len(values) < period:
        return float(np.mean(values))
    k = 2.0 / (period + 1)
    v = float(np.mean(values[:period]))
    for x in values[period:]:
        v = (x - v) * k + v
    return v


def run(max_markets: int | None = None) -> pd.DataFrame:
    from types import SimpleNamespace
    t0, a, _ = load_brti()           # trailing-60 s average series: robust price proxy for 5 m closes
    mk = markets_with_trades()
    rows = []
    for r in mk.to_dict("records")[:max_markets]:
        row = SimpleNamespace(**r)
        o = int(row.open_ts)
        i0 = o - t0
        if i0 - 110 * 60 < 0 or i0 + 600 >= len(a):
            continue
        # 5-minute closes for the EMA: the 22 candles ending at minute 10
        closes = a[i0 + 600 - 300 * np.arange(22)[::-1]]
        opens1, close1 = a[i0], a[i0 + 300]
        opens2, close2 = a[i0 + 300], a[i0 + 600]
        if np.isnan(closes).any() or np.isnan([opens1, close1, close2]).any():
            continue
        g1, g2 = close1 > opens1, close2 > opens2
        above = close2 > ema(closes)
        if g1 and g2 and above:
            side = "yes"
        elif (not g1) and (not g2) and (not above):
            side = "no"
        else:
            side = ("yes" if g1 else "no") if abs(close1 - opens1) >= abs(close2 - opens2) else ("yes" if g2 else "no")
        md = load_market(row)
        if md is None:
            continue
        nt = next_taker_trade(md, o + 600 + 0.25, buy_yes=(side == "yes"), max_wait=5.0)
        if nt is None:
            continue
        _, yes_px, _ = nt
        price = yes_px if side == "yes" else 1 - yes_px
        win = (md.y == 1) if side == "yes" else (md.y == 0)
        fee = 0.07 * price * (1 - price)
        rows.append((row.ticker, o, side, price, fee, int(win), float(win) - price - fee))
    return pd.DataFrame(rows, columns=["ticker", "open_ts", "side", "price", "fee", "win", "pnl_pc"])


if __name__ == "__main__":
    df = run()
    n = len(df)
    print(f"markets {n}  win rate {df.win.mean():.3f}  avg price {df.price.mean():.3f}  "
          f"avg fee {100 * df.fee.mean():.2f}c  P&L per contract {100 * df.pnl_pc.mean():+.2f}c  "
          f"t={df.pnl_pc.mean() / (df.pnl_pc.std(ddof=1) / np.sqrt(n)):.2f}")
