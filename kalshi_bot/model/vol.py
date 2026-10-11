"""
Real-time volatility estimate for BRTI (dollars per sqrt(second)).

EWMA of squared 5-second print changes scaled to one second. Five-second
increments sit past the ~2 s negative autocorrelation of the index's
one-second prints (variance ratio ~0.8 from 5 s out to 15 min), so they
estimate the variance that actually accumulates over a contract's life.
"""
from __future__ import annotations

import math


class EwmaVol:
    def __init__(self, halflife_s: float = 300.0, horizon_s: int = 5, floor: float = 1.0,
                 init_sigma: float = 6.0, init_rel: float | None = None):
        """``init_rel``: start the estimate at this fraction of the first price
        seen instead of ``init_sigma`` dollars (for indexes of any price level).
        Both starting points are deliberately high - an unseeded estimate
        makes the model less confident, never more."""
        self.lam = math.exp(math.log(0.5) / halflife_s)
        self.h = horizon_s
        self.floor = floor
        self.init_rel = init_rel
        self.var = init_sigma ** 2
        self._anchored = init_rel is None
        self.n = 0
        self._last_sec: int | None = None

    def update(self, sec: int, prints: dict[int, float]) -> None:
        """Fold in every whole second since the last update."""
        start = sec if self._last_sec is None else self._last_sec + 1
        for s in range(max(start, sec - 3600), sec + 1):
            a, b = prints.get(s), prints.get(s - self.h)
            if a is None or b is None:
                continue
            if not self._anchored:
                self.var = (self.init_rel * a) ** 2
                self._anchored = True
            v = (a - b) ** 2 / self.h
            self.var = self.lam * self.var + (1 - self.lam) * v
            self.n += 1
        self._last_sec = sec

    def warm_start(self, prints: dict[int, float], end_sec: int, span_s: int = 1800) -> None:
        self._last_sec = end_sec - span_s
        self.update(end_sec, prints)

    @property
    def sigma(self) -> float:
        return max(self.floor, math.sqrt(max(self.var, 0.0)))

    @property
    def ready(self) -> bool:
        """Enough real observations (about one EWMA half-life) to trust."""
        return self.n >= 300
