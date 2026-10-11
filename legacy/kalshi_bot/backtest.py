"""
Walk-forward backtest for the 5m-candle rule-based BTC 15-minute strategy.

Strategy:
  At minute 10 of each 15-min window (after two 5m candles complete):
    - Both GREEN + price above 20 EMA  ->  buy YES
    - Both RED   + price below 20 EMA  ->  buy NO
    - Mixed / ambiguous                ->  trade the bigger-body candle's direction
  Hold to settlement (~5 min).

No lookahead bias:
  - Candle data at entry is only from completed candles (minutes 0-10).
  - The third 5m candle (minutes 10-15) is never seen before the trade.
  - Kalshi mid-price is estimated from where BTC sits at minute 10 relative
    to the window open, using a normal-CDF model calibrated on training data.
  - The remaining 5-minute volatility is estimated from out-of-sample data.

Usage:
    python -m kalshi_bot.backtest
    python -m kalshi_bot.backtest --days 30 --spread 0.04
"""
from __future__ import annotations

import argparse
import asyncio
import math
import time as _time
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np

BINANCE_ENDPOINTS = [
    "https://api.binance.us/api/v3",
    "https://api.binance.com/api/v3",
]


# ---- data fetching ---------------------------------------------------------

async def fetch_klines_paginated(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    days: int = 30,
) -> list[dict]:
    """Fetch historical klines from Binance, paginating in 1000-candle chunks."""
    end_ms = int(_time.time() * 1000)
    start_ms = end_ms - (days + 2) * 24 * 60 * 60 * 1000

    all_candles: list[dict] = []
    base_url = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        for ep in BINANCE_ENDPOINTS:
            try:
                r = await client.get(f"{ep}/ping")
                if r.status_code == 200:
                    base_url = ep
                    break
            except Exception:
                continue

        if base_url is None:
            print("ERROR: Could not connect to any Binance endpoint.")
            return []

        cursor = start_ms
        while cursor < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            }
            try:
                r = await client.get(f"{base_url}/klines", params=params)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print(f"  Klines fetch error: {e}")
                break

            if not data:
                break

            for k in data:
                all_candles.append({
                    "ts": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_ts": datetime.fromtimestamp(k[6] / 1000, tz=timezone.utc),
                })

            cursor = data[-1][0] + 1
            if len(data) < 1000:
                break

    print(f"Fetched {len(all_candles)} candles ({interval}) over ~{days} days")
    return all_candles


# ---- helpers ----------------------------------------------------------------

def ema_20(closes: list[float]) -> float:
    """20-period exponential moving average."""
    period = 20
    if len(closes) < period:
        return sum(closes) / len(closes) if closes else 0.0
    mult = 2.0 / (period + 1)
    val = sum(closes[:period]) / period
    for price in closes[period:]:
        val = (price - val) * mult + val
    return val


def norm_cdf(x: float) -> float:
    """Standard normal CDF approximation via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def estimate_fee_cents(price_cents: int) -> float:
    p = min(price_cents, 100 - price_cents)
    return max(1.0, math.ceil(p * p / 10000.0))


# ---- 15-min window grouping ------------------------------------------------

def group_into_windows(candles_5m: list[dict]) -> list[dict]:
    """Group 5m candles into 15-minute windows (3 candles each).

    Each window dict:
      candle1, candle2, candle3  -- the three 5m candles
      window_open, window_close  -- first open, last close
      went_up                    -- settlement outcome
    """
    by_window: dict[datetime, list[dict]] = {}
    for c in candles_5m:
        ts = c["ts"]
        ws = ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)
        by_window.setdefault(ws, []).append(c)

    windows = []
    for ws in sorted(by_window):
        group = sorted(by_window[ws], key=lambda c: c["ts"])
        if len(group) < 3:
            continue
        c1, c2, c3 = group[0], group[1], group[2]
        windows.append({
            "ws": ws,
            "candle1": c1,
            "candle2": c2,
            "candle3": c3,
            "window_open": c1["open"],
            "window_close": c3["close"],
            "went_up": c3["close"] > c1["open"],
        })
    return windows


# ---- strategy logic ---------------------------------------------------------

def decide_side(
    candle1: dict,
    candle2: dict,
    trailing_closes: list[float],
) -> tuple[str, str]:
    """Apply the three rules. Returns (side, rule_name)."""
    c1_green = candle1["close"] > candle1["open"]
    c2_green = candle2["close"] > candle2["open"]
    above_ema = candle2["close"] > ema_20(trailing_closes)

    both_green = c1_green and c2_green
    both_red = (not c1_green) and (not c2_green)

    if both_green and above_ema:
        return "yes", "2xGREEN+aboveEMA"
    elif both_red and (not above_ema):
        return "no", "2xRED+belowEMA"
    else:
        body1 = abs(candle1["close"] - candle1["open"])
        body2 = abs(candle2["close"] - candle2["open"])
        if body1 >= body2:
            return ("yes" if c1_green else "no"), "bigger_body"
        else:
            return ("yes" if c2_green else "no"), "bigger_body"


# ---- backtest core ----------------------------------------------------------

class BacktestTrade:
    __slots__ = (
        "window_start", "side", "rule", "price_cents", "fee_cents",
        "kalshi_mid", "went_up", "won", "pnl_cents",
        "btc_open", "btc_at_entry", "btc_close",
        "c1_green", "c2_green", "above_ema",
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def run_backtest(
    candles_5m: list[dict],
    *,
    train_days: int = 15,
    spread: float = 0.04,
) -> list[BacktestTrade]:
    """Walk-forward backtest of the 5m-candle strategy."""
    windows = group_into_windows(candles_5m)
    if len(windows) < 100:
        print(f"Only {len(windows)} windows -- need more data.")
        return []

    first_ts = windows[0]["ws"]
    train_cutoff = first_ts + timedelta(days=train_days)

    train_windows = [w for w in windows if w["ws"] < train_cutoff]
    test_windows = [w for w in windows if w["ws"] >= train_cutoff]

    if len(train_windows) < 50 or len(test_windows) < 50:
        print("Not enough windows for train/test split.")
        return []

    print(f"Train windows: {len(train_windows)}  |  Test windows: {len(test_windows)}")
    print(f"Train period: {train_windows[0]['ws'].date()} to "
          f"{train_windows[-1]['ws'].date()}")
    print(f"Test  period: {test_windows[0]['ws'].date()} to "
          f"{test_windows[-1]['ws'].date()}")

    # Estimate 5-minute volatility from training data
    all_closes = [c["close"] for c in candles_5m if c["close_ts"] <= train_cutoff]
    log_rets = [
        math.log(all_closes[i] / all_closes[i - 1])
        for i in range(1, len(all_closes))
        if all_closes[i - 1] > 0
    ]
    vol_5m = float(np.std(log_rets)) if log_rets else 0.001
    print(f"Estimated 5m vol: {vol_5m:.6f} ({vol_5m*100:.4f}%)")

    # Build a lookup for trailing closes
    sorted_candles = sorted(candles_5m, key=lambda c: c["ts"])

    trades: list[BacktestTrade] = []

    for w in test_windows:
        c1 = w["candle1"]
        c2 = w["candle2"]

        # Trailing closes for EMA: all 5m candles completed by candle2's close
        trailing = [
            c["close"] for c in sorted_candles
            if c["close_ts"] <= c2["close_ts"]
        ][-30:]

        if len(trailing) < 20:
            continue

        side, rule = decide_side(c1, c2, trailing)
        direction = "up" if side == "yes" else "down"

        # Realistic Kalshi mid: P(close > open | price at minute 10)
        btc_at_entry = c2["close"]
        btc_open = w["window_open"]
        pct_move = (btc_at_entry - btc_open) / btc_open if btc_open > 0 else 0.0
        d = pct_move / vol_5m if vol_5m > 0 else 0.0
        kalshi_yes_mid = norm_cdf(d)
        kalshi_yes_mid = max(0.05, min(0.95, kalshi_yes_mid))

        half_spread = spread / 2.0
        if side == "yes":
            ask = kalshi_yes_mid + half_spread
        else:
            ask = (1.0 - kalshi_yes_mid) + half_spread

        ask = max(0.02, min(0.98, ask))
        price_cents = int(round(ask * 100))
        fee_c = estimate_fee_cents(price_cents)

        went_up = w["went_up"]
        won = (direction == "up" and went_up) or (direction == "down" and not went_up)

        if won:
            pnl_cents = (100 - price_cents) - fee_c
        else:
            pnl_cents = -price_cents - fee_c

        c1_green = c1["close"] > c1["open"]
        c2_green = c2["close"] > c2["open"]
        above_ema = btc_at_entry > ema_20(trailing)

        trades.append(BacktestTrade(
            window_start=w["ws"],
            side=side,
            rule=rule,
            price_cents=price_cents,
            fee_cents=fee_c,
            kalshi_mid=kalshi_yes_mid,
            went_up=went_up,
            won=won,
            pnl_cents=pnl_cents,
            btc_open=btc_open,
            btc_at_entry=btc_at_entry,
            btc_close=w["window_close"],
            c1_green=c1_green,
            c2_green=c2_green,
            above_ema=above_ema,
        ))

    return trades


# ---- reporting --------------------------------------------------------------

def print_report(trades: list[BacktestTrade]):
    if not trades:
        print("No trades to report.")
        return

    n = len(trades)
    wins = sum(1 for t in trades if t.won)
    losses = n - wins
    win_rate = wins / n

    pnl_list = [t.pnl_cents for t in trades]
    total_pnl = sum(pnl_list)
    avg_pnl = total_pnl / n
    total_fees = sum(t.fee_cents for t in trades)

    cum = np.cumsum(pnl_list)
    peak = np.maximum.accumulate(cum)
    drawdown = peak - cum
    max_dd = float(np.max(drawdown)) if len(drawdown) else 0.0

    arr = np.array(pnl_list, dtype=float)
    std = float(np.std(arr))
    sharpe = (float(np.mean(arr)) / std * (96 ** 0.5)) if std > 0 else 0.0

    # Break down by rule
    rules = {}
    for t in trades:
        rules.setdefault(t.rule, {"n": 0, "wins": 0, "pnl": 0.0, "avg_entry": 0.0})
        rules[t.rule]["n"] += 1
        rules[t.rule]["wins"] += int(t.won)
        rules[t.rule]["pnl"] += t.pnl_cents
        rules[t.rule]["avg_entry"] += t.price_cents

    # Break down by side
    yes_trades = [t for t in trades if t.side == "yes"]
    no_trades = [t for t in trades if t.side == "no"]
    yes_wr = sum(1 for t in yes_trades if t.won) / max(1, len(yes_trades))
    no_wr = sum(1 for t in no_trades if t.won) / max(1, len(no_trades))
    yes_pnl = sum(t.pnl_cents for t in yes_trades)
    no_pnl = sum(t.pnl_cents for t in no_trades)

    avg_entry_yes = np.mean([t.price_cents for t in yes_trades]) if yes_trades else 0
    avg_entry_no = np.mean([t.price_cents for t in no_trades]) if no_trades else 0

    actual_up_pct = sum(1 for t in trades if t.went_up) / n

    # Streaks
    max_w_streak = max_l_streak = cur = 0
    last = None
    for t in trades:
        if t.won == last:
            cur += 1
        else:
            cur = 1
            last = t.won
        if t.won:
            max_w_streak = max(max_w_streak, cur)
        else:
            max_l_streak = max(max_l_streak, cur)

    print("\n" + "=" * 70)
    print("  BACKTEST RESULTS -- 5m Candle Rule-Based Strategy")
    print("=" * 70)
    print(f"  Period:        {trades[0].window_start.date()} to "
          f"{trades[-1].window_start.date()}")
    print(f"  Total trades:  {n}")
    print(f"  Actual UP %:   {actual_up_pct:.1%}")
    print("-" * 70)
    print(f"  Win rate:      {win_rate:.1%}  ({wins}W / {losses}L)")
    print(f"  YES trades:    {len(yes_trades)} (WR {yes_wr:.1%}, "
          f"avg entry {avg_entry_yes:.0f}c, PnL {yes_pnl:+.0f}c)")
    print(f"  NO  trades:    {len(no_trades)} (WR {no_wr:.1%}, "
          f"avg entry {avg_entry_no:.0f}c, PnL {no_pnl:+.0f}c)")
    print("-" * 70)
    print(f"  Total PnL:     {total_pnl:+.0f}c  (${total_pnl/100:+.2f} per contract)")
    print(f"  Avg PnL/trade: {avg_pnl:+.2f}c")
    print(f"  Total fees:    {total_fees:.0f}c  (${total_fees/100:.2f})")
    print(f"  Max drawdown:  {max_dd:.0f}c")
    print(f"  Sharpe (ann.): {sharpe:.2f}")
    print(f"  Best streak:   {max_w_streak}W  |  Worst: {max_l_streak}L")
    print("-" * 70)

    print("\n  Breakdown by rule:")
    print(f"  {'Rule':<22} {'Count':>6} {'WR':>7} {'PnL':>10} {'Avg Entry':>10}")
    for rule_name in sorted(rules):
        r = rules[rule_name]
        wr = r["wins"] / r["n"] if r["n"] else 0
        ae = r["avg_entry"] / r["n"] if r["n"] else 0
        print(f"  {rule_name:<22} {r['n']:>6} {wr:>7.1%} {r['pnl']:>+10.0f}c {ae:>9.0f}c")

    print("\n  Equity curve (sampled):")
    cum_pnl = np.cumsum(pnl_list)
    pts = list(range(0, n, max(1, n // 20))) + [n - 1]
    for idx in pts:
        t = trades[idx]
        bar_width = int(max(0, min(40, (cum_pnl[idx] + 2000) / 100)))
        bar = "#" * bar_width
        print(f"  {t.window_start.strftime('%m-%d %H:%M')}  {cum_pnl[idx]:>+8.0f}c  {bar}")

    print("\n" + "=" * 70)
    final = cum_pnl[-1]
    if final > 0:
        print(f"  RESULT: +{final:.0f}c net profit per contract over {n} trades")
    else:
        print(f"  RESULT: {final:.0f}c net loss per contract over {n} trades")
    print(f"  At $5/trade:  ${final * 5 / 100:+.2f} total")
    print(f"  At $10/trade: ${final * 10 / 100:+.2f} total")
    print("=" * 70)


# ---- entry point ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="5m candle strategy backtest")
    p.add_argument("--days", type=int, default=30, help="Days of data (default: 30)")
    p.add_argument("--train-days", type=int, default=15, help="Days for vol calibration")
    p.add_argument("--spread", type=float, default=0.04, help="Kalshi spread (default: 0.04)")
    return p.parse_args()


async def async_main():
    args = parse_args()

    print("=" * 70)
    print("  5m Candle Rule-Based Strategy -- Walk-Forward Backtest")
    print("=" * 70)
    print(f"  Data:     last {args.days} days of BTC 5-min candles from Binance")
    print(f"  Spread:   {int(args.spread*100)}c")
    print(f"  Pricing:  Kalshi mid estimated via normal CDF (realistic)")
    print(f"  Rules:")
    print(f"    - 2x GREEN + above 20 EMA -> YES")
    print(f"    - 2x RED   + below 20 EMA -> NO")
    print(f"    - Mixed -> bigger body direction")
    print(f"  Entry:    minute 10 of each window")
    print(f"  Exit:     hold to settlement (minute 15)")
    print()

    print("Fetching 5m candles from Binance...")
    candles = await fetch_klines_paginated(
        symbol="BTCUSDT", interval="5m", days=args.days,
    )
    if not candles:
        print("Failed to fetch data.")
        return

    trades = run_backtest(
        candles,
        train_days=args.train_days,
        spread=args.spread,
    )

    print_report(trades)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
