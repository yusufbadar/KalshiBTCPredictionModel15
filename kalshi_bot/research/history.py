"""
Historical data downloader for KXBTC15M research and backtests.

Everything here uses Kalshi's *public* production market-data endpoints, so no
API key is needed and nothing is traded. Data is cached under ``data/kxbtc15m``:

    markets.parquet             one row per settled 15-minute market
    brti/<YYYY-MM-DD>.parquet   1-second series of the settlement index
                                (trailing 60 s average of CF Benchmarks BRTI)
    candles/<YYYY-MM-DD>.parquet  1-minute bid/ask/trade candles per market
    trades/<YYYY-MM-DD>/<ticker>.parquet  full public trade tape per market

Usage:
    python -m kalshi_bot.research.history --days 35            # everything
    python -m kalshi_bot.research.history --days 35 --no-trades
"""
from __future__ import annotations

import argparse
import asyncio
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

PUBLIC_BASE = "https://external-api.kalshi.com/trade-api/v2"
SERIES = "KXBTC15M"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def series_root(series: str) -> Path:
    return DATA_DIR / series.lower()


def epoch_seconds(s: pd.Series) -> pd.Series:
    """Whole seconds since 1970 for a datetime column, whatever resolution it
    is stored in (pandas 2 used nanoseconds everywhere; pandas 3 keeps the
    resolution of the source, usually microseconds)."""
    return s.dt.as_unit("ns").astype("int64") // 10**9


def epoch_micros(s: pd.Series) -> pd.Series:
    """Microseconds since 1970 for a datetime column, independent of its stored resolution."""
    return s.dt.as_unit("us").astype("int64")


ROOT = series_root(SERIES)


# ---------------------------------------------------------------------------
# HTTP with a shared token bucket + backoff (public endpoints are rate limited)
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, rate: float, burst: float):
        self.rate, self.capacity = rate, burst
        self.tokens, self.last = burst, time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                await asyncio.sleep((1 - self.tokens) / self.rate)


class PublicClient:
    def __init__(self, rate: float = 18.0):
        self.http = httpx.AsyncClient(base_url=PUBLIC_BASE, timeout=30.0,
                                      limits=httpx.Limits(max_connections=16))
        self.limiter = RateLimiter(rate, burst=rate)
        self.requests = 0

    async def get(self, path: str, params: dict | None = None) -> dict:
        delay = 0.5
        for attempt in range(40):
            await self.limiter.acquire()
            try:
                r = await self.http.get(path, params=params)
            except (httpx.TransportError, httpx.TimeoutException):
                await asyncio.sleep(delay + random.random() * delay)
                delay = min(delay * 2, 30)
                continue
            self.requests += 1
            if r.status_code == 429 or r.status_code >= 500:
                await asyncio.sleep(delay + random.random() * delay)
                delay = min(delay * 2, 30)
                continue
            if r.status_code == 404:
                return {}
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"GET {path} failed after retries")

    async def close(self) -> None:
        await self.http.aclose()


def _ts(s: str | None) -> pd.Timestamp | None:
    return pd.Timestamp(s) if s else None


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------
MARKET_COLS = ["ticker", "event_ticker", "open_time", "close_time", "floor_strike",
               "expiration_value", "result", "volume", "status", "strike_type"]


def _market_row(m: dict) -> dict:
    return {
        "ticker": m["ticker"],
        "event_ticker": m.get("event_ticker", ""),
        "open_time": _ts(m.get("open_time")),
        "close_time": _ts(m.get("close_time")),
        "floor_strike": _f(m.get("floor_strike")),
        "expiration_value": _f(m.get("expiration_value")),
        "result": m.get("result", ""),
        "volume": _f(m.get("volume_fp", m.get("volume"))),
        "status": m.get("status", ""),
        "strike_type": m.get("strike_type", ""),
    }


async def fetch_markets(cl: PublicClient, start: datetime, end: datetime,
                        series: str = SERIES) -> pd.DataFrame:
    """All settled markets of ``series`` with close_time in [start, end)."""
    rows: list[dict] = []
    cutoff = await cl.get("/historical/cutoff")
    cut = pd.Timestamp(cutoff["market_settled_ts"])
    # live tier (settled after the cutoff)
    cursor = None
    while True:
        p = {"series_ticker": series, "status": "settled", "limit": 1000,
             "min_settled_ts": int(max(start, cut.to_pydatetime()).timestamp()) - 3600,
             "max_settled_ts": int(end.timestamp()) + 3600}
        if cursor:
            p["cursor"] = cursor
        d = await cl.get("/markets", p)
        rows += [_market_row(m) for m in d.get("markets", [])]
        cursor = d.get("cursor")
        if not cursor or not d.get("markets"):
            break
    # historical tier (only if the window reaches before the cutoff)
    if start < cut.to_pydatetime():
        cursor = None
        while True:
            p = {"series_ticker": series, "limit": 1000}
            if cursor:
                p["cursor"] = cursor
            d = await cl.get("/historical/markets", p)
            ms = d.get("markets", [])
            rows += [_market_row(m) for m in ms]
            cursor = d.get("cursor")
            oldest = min((_ts(m.get("close_time")) for m in ms), default=None)
            if not cursor or not ms or (oldest is not None and oldest < pd.Timestamp(start)):
                break
    df = pd.DataFrame(rows, columns=MARKET_COLS).drop_duplicates("ticker")
    df = df[(df.close_time >= pd.Timestamp(start)) & (df.close_time < pd.Timestamp(end))]
    df = df[df.result.isin(["yes", "no"])]
    return df.sort_values("close_time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Settlement index (trailing 60 s BRTI average at 1 s resolution)
# ---------------------------------------------------------------------------
async def fetch_brti_day(cl: PublicClient, markets: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Each event's live-data payload covers the hour before its close, so the
    events closing on the hour tile the day without gaps."""
    day_m = markets[(markets.close_time > day) & (markets.close_time <= day + pd.Timedelta(days=1))]
    hourly = day_m[day_m.close_time.dt.minute == 0]
    frames = []

    async def one(ev: str):
        d = await cl.get(f"/live_data/events/{ev}")
        ts = ((d.get("live_data") or {}).get("details") or {}).get("timeseries") or []
        if ts:
            frames.append(pd.DataFrame({"t_ms": [p["t"] for p in ts], "avg60": [p["v"] for p in ts]}))

    await asyncio.gather(*(one(ev) for ev in hourly.event_ticker))
    if not frames:
        return pd.DataFrame(columns=["t_ms", "avg60"])
    df = pd.concat(frames).drop_duplicates("t_ms").sort_values("t_ms")
    lo, hi = int(day.timestamp() * 1000), int((day + pd.Timedelta(days=1)).timestamp() * 1000)
    return df[(df.t_ms >= lo) & (df.t_ms <= hi)].reset_index(drop=True)


async def fetch_event_series_day(cl: PublicClient, day_m: pd.DataFrame) -> pd.DataFrame:
    """Every market's *own* hour-long index series. Overlapping payloads from
    neighbouring events can differ by cents at the same timestamp; each
    payload's final point is that market's exact settlement value, so the
    per-market series is the faithful view for near-expiry modelling."""
    frames = []
    by_event: dict[str, list[str]] = defaultdict(list)
    for r in day_m.itertuples():
        by_event[r.event_ticker].append(r.ticker)

    async def one(ev: str, tickers: list[str]):
        d = await cl.get(f"/live_data/events/{ev}")
        ts = ((d.get("live_data") or {}).get("details") or {}).get("timeseries") or []
        if ts:
            t_ms = np.array([p["t"] for p in ts], dtype=np.int64)
            v = np.array([p["v"] for p in ts], dtype=np.float64)
            for tk in tickers:          # strikes of one event share its index series
                frames.append(pd.DataFrame({"ticker": tk, "t_ms": t_ms, "avg60": v}))

    await asyncio.gather(*(one(ev, tks) for ev, tks in by_event.items()))
    if not frames:
        return pd.DataFrame(columns=["ticker", "t_ms", "avg60"])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 1-minute candles
# ---------------------------------------------------------------------------
def _candle_rows(ticker: str, cs: list[dict]) -> list[dict]:
    out = []
    for c in cs:
        def g(block: str, key: str):
            b = c.get(block) or {}
            return _f(b.get(f"{key}_dollars", b.get(key)))
        out.append({
            "ticker": ticker, "end_ts": int(c["end_period_ts"]),
            "bid_o": g("yes_bid", "open"), "bid_h": g("yes_bid", "high"),
            "bid_l": g("yes_bid", "low"), "bid_c": g("yes_bid", "close"),
            "ask_o": g("yes_ask", "open"), "ask_h": g("yes_ask", "high"),
            "ask_l": g("yes_ask", "low"), "ask_c": g("yes_ask", "close"),
            "px_o": g("price", "open"), "px_h": g("price", "high"),
            "px_l": g("price", "low"), "px_c": g("price", "close"),
            "px_mean": g("price", "mean"),
            "volume": _f(c.get("volume_fp", c.get("volume"))),
            "oi": _f(c.get("open_interest_fp", c.get("open_interest"))),
        })
    return out


async def fetch_candles_day(cl: PublicClient, day_m: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict] = []
    live = day_m[day_m.close_time >= cut]
    hist = day_m[day_m.close_time < cut]
    # The batch endpoint rejects requests where markets x minutes-in-range
    # exceeds its 10k-candle budget, so walk consecutive markets in groups.
    live = live.sort_values("open_time")
    for i in range(0, len(live), 20):
        chunk = live.iloc[i:i + 20]
        p = {"market_tickers": ",".join(chunk.ticker),
             "start_ts": int(chunk.open_time.min().timestamp()),
             "end_ts": int(chunk.close_time.max().timestamp()),
             "period_interval": 1}
        d = await cl.get("/markets/candlesticks", p)
        for m in d.get("markets", []):
            rows += _candle_rows(m.get("market_ticker") or m.get("ticker"), m.get("candlesticks", []))

    async def one(r):
        d = await cl.get(f"/historical/markets/{r.ticker}/candlesticks",
                         {"start_ts": int(r.open_time.timestamp()),
                          "end_ts": int(r.close_time.timestamp()), "period_interval": 1})
        rows.extend(_candle_rows(r.ticker, d.get("candlesticks", [])))

    await asyncio.gather(*(one(r) for r in hist.itertuples()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full trade tape
# ---------------------------------------------------------------------------
async def fetch_trades(cl: PublicClient, ticker: str, historical: bool,
                       min_ts: int | None = None) -> pd.DataFrame:
    """Public trade tape of one market. ``min_ts`` keeps only trades at or
    after that epoch second (the strategy only needs the final minutes)."""
    path = "/historical/trades" if historical else "/markets/trades"
    ts, px, cnt, taker_yes, block = [], [], [], [], []
    cursor = None
    while True:
        p = {"ticker": ticker, "limit": 1000}
        if min_ts is not None:
            p["min_ts"] = int(min_ts)
        if cursor:
            p["cursor"] = cursor
        d = await cl.get(path, p)
        tr = d.get("trades") or []
        for t in tr:
            ts.append(t["created_time"])
            px.append(t["yes_price_dollars"])
            cnt.append(t["count_fp"])
            side = t.get("taker_outcome_side") or t.get("taker_side")
            taker_yes.append(side == "yes")
            block.append(bool(t.get("is_block_trade", False)))
        cursor = d.get("cursor")
        if not cursor or not tr:
            break
    df = pd.DataFrame({
        "ts_us": epoch_micros(pd.to_datetime(pd.Series(ts, dtype="string"), utc=True, format="ISO8601"))
        if ts else pd.Series([], dtype="int64"),
        "yes_price": pd.Series(px, dtype="float64"),
        "count": pd.Series(cnt, dtype="float64"),
        "taker_yes": pd.Series(taker_yes, dtype="bool"),
        "is_block": pd.Series(block, dtype="bool"),
    })
    return df.sort_values("ts_us", kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)

async def download(days: int, end: datetime | None, trades: bool, rate: float,
                   workers: int, series: str = SERIES, min_volume: float = 0.0,
                   tail_seconds: int = 0) -> None:
    ROOT = series_root(series)
    ROOT.mkdir(parents=True, exist_ok=True)
    end = end or datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = (end - timedelta(days=days)).replace(hour=0)
    cl = PublicClient(rate)
    t0 = time.time()
    try:
        cutoff = await cl.get("/historical/cutoff")
        cut_m = pd.Timestamp(cutoff["market_settled_ts"])
        cut_t = pd.Timestamp(cutoff["trades_created_ts"])
        print(f"[markets] {start:%Y-%m-%d} -> {end:%Y-%m-%d %H:%M}  (cutoff {cut_m})", flush=True)
        mk = await fetch_markets(cl, start, end, series)
        mpath = ROOT / "markets.parquet"
        if mpath.exists():
            old = pd.read_parquet(mpath)
            mk = pd.concat([old, mk]).drop_duplicates("ticker", keep="last").sort_values("close_time")
        mk.to_parquet(mpath, index=False)
        print(f"[markets] {len(mk)} settled markets cached", flush=True)

        sel = mk[(mk.close_time > pd.Timestamp(start)) & (mk.close_time <= pd.Timestamp(end))]
        if min_volume > 0:
            sel = sel[sel.volume >= min_volume]
        day_list = sorted({c.normalize() for c in sel.close_time - pd.Timedelta(microseconds=1)})
        for day in day_list:
            day_m = sel[(sel.close_time > day) & (sel.close_time <= day + pd.Timedelta(days=1))]
            tag = f"{day:%Y-%m-%d}"
            bpath = ROOT / "brti" / f"{tag}.parquet"
            if series.endswith("15M") and not tail_seconds and (not bpath.exists() or day + pd.Timedelta(days=1) > pd.Timestamp(end)):
                bpath.parent.mkdir(parents=True, exist_ok=True)
                b = await fetch_brti_day(cl, mk, day)
                _atomic_parquet(b, bpath)
            cpath = ROOT / "candles" / f"{tag}.parquet"
            if series.endswith("15M") and not tail_seconds and (not cpath.exists() or day + pd.Timedelta(days=1) > pd.Timestamp(end)):
                cpath.parent.mkdir(parents=True, exist_ok=True)
                c = await fetch_candles_day(cl, day_m, cut_m)
                _atomic_parquet(c, cpath)
            epath = ROOT / "event_series" / f"{tag}.parquet"
            # a day file written while that day was still running lacks its later
            # markets: fetch whatever is missing instead of trusting the file's existence
            old_es = pd.read_parquet(epath) if epath.exists() else None
            have = set(old_es.ticker.unique()) if old_es is not None else set()
            missing = day_m[~day_m.ticker.isin(have)]
            if len(missing):
                epath.parent.mkdir(parents=True, exist_ok=True)
                es = await fetch_event_series_day(cl, missing)
                if old_es is not None and len(old_es):
                    es = pd.concat([old_es, es], ignore_index=True)
                _atomic_parquet(es, epath)
            print(f"[{tag}] brti+candles ok  ({cl.requests} req, {time.time()-t0:.0f}s)", flush=True)

        if not trades:
            return
        todo = []
        for r in sel.itertuples():
            day = (r.close_time - pd.Timedelta(microseconds=1)).normalize()
            p = ROOT / "trades" / f"{day:%Y-%m-%d}" / f"{r.ticker}.parquet"
            if not p.exists():
                todo.append((r, p))
        print(f"[trades] {len(todo)} markets to fetch", flush=True)
        q: asyncio.Queue = asyncio.Queue()
        for item in todo:
            q.put_nowait(item)
        done = 0
        counts = defaultdict(int)

        async def worker():
            nonlocal done
            while not q.empty():
                r, p = q.get_nowait()
                min_ts = int(r.close_time.timestamp()) - tail_seconds if tail_seconds else None
                try:
                    df = await fetch_trades(cl, r.ticker, historical=r.close_time <= cut_t, min_ts=min_ts)
                except Exception as e:  # noqa: BLE001 - one bad market must not stop the run
                    counts["failed"] += 1
                    print(f"[trades] {r.ticker} failed ({e}); re-run to retry", flush=True)
                    await asyncio.sleep(5)
                    continue
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".tmp")
                df.to_parquet(tmp, index=False)
                tmp.replace(p)
                done += 1
                counts["trades"] += len(df)
                if done % 25 == 0 or done == len(todo):
                    el = time.time() - t0
                    print(f"[trades] {done}/{len(todo)} markets, {counts['trades']:,} trades, "
                          f"{cl.requests} req, {el/60:.1f} min", flush=True)

        await asyncio.gather(*(worker() for _ in range(workers)))
    finally:
        await cl.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Download KXBTC15M history (public data)")
    ap.add_argument("--days", type=int, default=35)
    ap.add_argument("--end", type=str, default=None, help="UTC end, e.g. 2026-09-27T00:00")
    ap.add_argument("--no-trades", action="store_true")
    ap.add_argument("--rate", type=float, default=18.0, help="requests per second")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--series", default=SERIES, help="e.g. KXBTC15M, KXETH15M, KXBTCD")
    ap.add_argument("--min-volume", type=float, default=0.0,
                    help="skip markets that traded fewer contracts (useful for hourly strike ladders)")
    ap.add_argument("--tail-seconds", type=int, default=0,
                    help="only fetch each market's trades from the last N seconds before its close "
                         "(the backtest panel uses the final 300 s); 0 = the whole tape")
    a = ap.parse_args()
    end = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc) if a.end else None
    asyncio.run(download(a.days, end, not a.no_trades, a.rate, a.workers, a.series.upper(), a.min_volume,
                         a.tail_seconds))


if __name__ == "__main__":
    main()
