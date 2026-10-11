"""
Real-time tracker for the CF Benchmarks BRTI settlement index.

Kalshi settles KXBTC15M on the mean of the 60 one-second BRTI prints
stamped T-60 .. T-1 (verified against published settlement values). This tracker keeps the 1-second prints (from the 1 Hz
``cfbenchmarks_value`` channel, falling back to the whole-second tick of the
5 Hz channel), the latest sub-second value, and exposes the quantities the
pricing model needs.
"""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np


class IndexTracker:
    def __init__(self, history_s: int = 4 * 3600, decimals: int = 2):
        self.history_s = history_s
        self.decimals = decimals                    # precision of the published settlement value
        self.prints: dict[int, float] = {}          # epoch second -> print
        self.order: deque[int] = deque()
        self.last_value: float | None = None        # freshest (5 Hz) value
        self.last_value_ts: float = 0.0             # source time of last_value (s)
        self.last_rx: float = 0.0                   # local receive time
        self.latency_ms: float = 0.0
        # local clock minus Kalshi's clock (includes the small one-way network
        # delay, which makes exchange-time estimates slightly conservative)
        self._offsets: deque[float] = deque(maxlen=120)
        self.clock_offset: float = 0.0

    # ------------------------------------------------------------- ingest
    def on_tick(self, source_ms: int, value: float, received_ms: int | None = None) -> None:
        ts = source_ms / 1000.0
        if ts >= self.last_value_ts:
            self.last_value, self.last_value_ts = value, ts
        self.last_rx = time.time()
        if received_ms:
            self.latency_ms = received_ms - source_ms
            self._offsets.append(self.last_rx - received_ms / 1000.0)
            if len(self._offsets) >= 5:
                self.clock_offset = float(np.median(self._offsets))
        if source_ms % 1000 == 0:
            sec = source_ms // 1000
            if sec not in self.prints:
                self.order.append(sec)
            self.prints[sec] = value
            while self.order and self.order[0] < sec - self.history_s:
                self.prints.pop(self.order.popleft(), None)

    # ------------------------------------------------------------ queries
    def exchange_time(self) -> float:
        """Current time on Kalshi's clock (local clock corrected by the
        measured offset)."""
        return time.time() - self.clock_offset

    def latest_print_sec(self) -> int | None:
        return self.order[-1] if self.order else None

    def stale(self, max_age_s: float) -> bool:
        return self.last_value is None or time.time() - self.last_rx > max_age_s

    def print_at(self, sec: int) -> float | None:
        return self.prints.get(sec)

    def window_sum(self, start_sec: int, end_sec: int) -> tuple[float, int]:
        """Sum and count of prints with start_sec <= t <= end_sec (missing
        seconds are filled with the previous print, as a 1 Hz sampler would)."""
        total, n, prev = 0.0, 0, None
        for s in range(start_sec, end_sec + 1):
            v = self.prints.get(s)
            if v is None:
                v = prev if prev is not None else self._before(s)
            if v is None:
                continue
            total += v
            n += 1
            prev = v
        return total, n

    def _before(self, sec: int) -> float | None:
        for s in range(sec - 1, sec - 10, -1):
            if s in self.prints:
                return self.prints[s]
        return None

    def series(self, end_sec: int, n: int) -> np.ndarray:
        """Last n one-second prints ending at end_sec (forward filled)."""
        out = np.empty(n)
        prev = self._before(end_sec - n + 1)
        for i, s in enumerate(range(end_sec - n + 1, end_sec + 1)):
            v = self.prints.get(s, prev)
            out[i] = np.nan if v is None else v
            prev = v if v is not None else prev
        return out

    def missing_in(self, start_sec: int, end_sec: int) -> int:
        """Number of seconds in [start_sec, end_sec] with no print received."""
        return sum(1 for s in range(start_sec, end_sec + 1) if s not in self.prints)

    def settlement_avg(self, close_sec: int) -> float | None:
        """Kalshi settlement value for a contract closing at ``close_sec``:
        mean of the prints stamped close-60 .. close-1, rounded to the
        index's published precision (cents for BTC).
        Requires all 60 prints to have been received."""
        vals = [self.prints.get(s) for s in range(close_sec - 60, close_sec)]
        if any(v is None for v in vals):
            return None
        return round(sum(vals) / 60.0, self.decimals)

    def avg60(self, sec: int) -> float | None:
        s, n = self.window_sum(sec - 59, sec)
        return s / n if n else None
