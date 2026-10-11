"""
SQLite ledger: every order, fill, settlement and equity sample, per environment.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    client_order_id TEXT PRIMARY KEY, order_id TEXT, env TEXT, ts REAL, ticker TEXT,
    outcome TEXT, book_side TEXT, price REAL, count REAL, tif TEXT, post_only INTEGER,
    status TEXT, filled REAL DEFAULT 0, reason TEXT
);
CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY, env TEXT, ts REAL, ticker TEXT, order_id TEXT,
    outcome TEXT, price REAL, count REAL, fee REAL, is_taker INTEGER
);
CREATE TABLE IF NOT EXISTS settlements (
    ticker TEXT, env TEXT, ts REAL, result TEXT, yes_qty REAL, no_qty REAL,
    cost REAL, fees REAL, payout REAL, pnl REAL, PRIMARY KEY (ticker, env)
);
CREATE TABLE IF NOT EXISTS equity (env TEXT, ts REAL, balance REAL, pnl_day REAL);
CREATE TABLE IF NOT EXISTS events (env TEXT, ts REAL, level TEXT, msg TEXT);
CREATE INDEX IF NOT EXISTS fills_env_ts ON fills(env, ts);
CREATE INDEX IF NOT EXISTS settle_env_ts ON settlements(env, ts);
CREATE INDEX IF NOT EXISTS events_env_ts ON events(env, ts);
"""


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def _exec(self, sql: str, args: tuple = ()) -> None:
        with self._lock:
            self.db.execute(sql, args)

    def _query(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self._lock:
            return self.db.execute(sql, args).fetchall()

    # ------------------------------------------------------------- writes
    def order(self, env: str, o: dict) -> None:
        self._exec("""INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            o["client_order_id"], o.get("order_id"), env, o.get("ts", time.time()), o["ticker"],
            o["outcome"], o["book_side"], o["price"], o["count"], o.get("tif"), int(o.get("post_only", 0)),
            o.get("status", "new"), o.get("filled", 0.0), o.get("reason", "")))

    def order_status(self, client_order_id: str, status: str, filled: float | None = None,
                     order_id: str | None = None) -> None:
        if filled is None:
            self._exec("UPDATE orders SET status=?, order_id=COALESCE(?, order_id) WHERE client_order_id=?",
                       (status, order_id, client_order_id))
        else:
            self._exec("UPDATE orders SET status=?, filled=?, order_id=COALESCE(?, order_id) "
                       "WHERE client_order_id=?", (status, filled, order_id, client_order_id))

    def fill(self, env: str, f: dict) -> bool:
        with self._lock:
            cur = self.db.execute("INSERT OR IGNORE INTO fills VALUES (?,?,?,?,?,?,?,?,?,?)", (
                f["fill_id"], env, f["ts"], f["ticker"], f.get("order_id"), f["outcome"],
                f["price"], f["count"], f["fee"], int(f["is_taker"])))
            return cur.rowcount > 0

    def settlement(self, env: str, s: dict) -> None:
        self._exec("INSERT OR REPLACE INTO settlements VALUES (?,?,?,?,?,?,?,?,?,?)", (
            s["ticker"], env, s["ts"], s["result"], s["yes_qty"], s["no_qty"],
            s["cost"], s["fees"], s["payout"], s["pnl"]))

    def equity(self, env: str, balance: float, pnl_day: float) -> None:
        self._exec("INSERT INTO equity VALUES (?,?,?,?)", (env, time.time(), balance, pnl_day))

    def event(self, env: str, level: str, msg: str) -> None:
        self._exec("INSERT INTO events VALUES (?,?,?,?)", (env, time.time(), level, msg))

    def reset(self, env: str) -> None:
        for t in ("orders", "fills", "settlements", "equity", "events"):
            self._exec(f"DELETE FROM {t} WHERE env=?", (env,))

    # -------------------------------------------------------------- reads
    def fills_for(self, env: str, ticker: str) -> list[dict]:
        rows = self._query("SELECT ts, outcome, price, count, fee, is_taker FROM fills "
                           "WHERE env=? AND ticker=? ORDER BY ts", (env, ticker))
        return [dict(ts=r[0], outcome=r[1], price=r[2], count=r[3], fee=r[4], is_taker=r[5]) for r in rows]

    def fill_ids(self, env: str) -> list[str]:
        return [r[0] for r in self._query("SELECT fill_id FROM fills WHERE env=?", (env,))]

    def unsettled_tickers(self, env: str) -> list[str]:
        rows = self._query("SELECT DISTINCT f.ticker FROM fills f LEFT JOIN settlements s "
                           "ON f.ticker=s.ticker AND f.env=s.env WHERE f.env=? AND s.ticker IS NULL", (env,))
        return [r[0] for r in rows]

    def settlements_list(self, env: str, limit: int = 500) -> list[dict]:
        rows = self._query("SELECT ticker, ts, result, yes_qty, no_qty, cost, fees, payout, pnl "
                           "FROM settlements WHERE env=? ORDER BY ts DESC LIMIT ?", (env, limit))
        keys = ["ticker", "ts", "result", "yes_qty", "no_qty", "cost", "fees", "payout", "pnl"]
        return [dict(zip(keys, r)) for r in rows]

    def recent_fills(self, env: str, limit: int = 100) -> list[dict]:
        rows = self._query("SELECT ts, ticker, outcome, price, count, fee, is_taker FROM fills "
                           "WHERE env=? ORDER BY ts DESC LIMIT ?", (env, limit))
        keys = ["ts", "ticker", "outcome", "price", "count", "fee", "is_taker"]
        return [dict(zip(keys, r)) for r in rows]

    def events_list(self, env: str, limit: int = 200) -> list[dict]:
        rows = self._query("SELECT ts, level, msg FROM events WHERE env=? ORDER BY ts DESC LIMIT ?",
                           (env, limit))
        return [dict(ts=r[0], level=r[1], msg=r[2]) for r in rows]

    def pnl_since(self, env: str, ts: float) -> float:
        r = self._query("SELECT COALESCE(SUM(pnl),0) FROM settlements WHERE env=? AND ts>=?", (env, ts))
        return float(r[0][0])

    def realised_drawdown(self, env: str, since_ts: float = 0.0) -> float:
        """Dollars the cumulative realised P&L sits below its own peak, counting
        settlements after ``since_ts`` (the peak starts at zero then). Based on
        settled trades only, so deposits and withdrawals never move it."""
        rows = self._query("SELECT pnl FROM settlements WHERE env=? AND ts>=? ORDER BY ts", (env, since_ts))
        eq = peak = 0.0
        for (pnl,) in rows:
            eq += pnl
            peak = max(peak, eq)
        return peak - eq

    def stats(self, env: str) -> dict:
        rows = self._query("SELECT pnl, fees, cost, yes_qty+no_qty FROM settlements WHERE env=? ORDER BY ts",
                           (env,))
        pnl = [r[0] for r in rows]
        n = len(pnl)
        wins = sum(1 for p in pnl if p > 0)
        eq, peak, dd = 0.0, 0.0, 0.0
        curve = []
        for p in pnl:
            eq += p
            peak = max(peak, eq)
            dd = max(dd, peak - eq)
            curve.append(round(eq, 4))
        return {
            "markets": n, "wins": wins, "losses": sum(1 for p in pnl if p < 0),
            "win_rate": wins / n if n else 0.0, "pnl": round(sum(pnl), 4),
            "fees": round(sum(r[1] for r in rows), 4), "volume": round(sum(r[3] for r in rows), 2),
            "max_drawdown": round(dd, 4), "curve": curve[-2000:],
        }
