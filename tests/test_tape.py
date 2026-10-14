"""The trade tape (core/tape.py) answers three questions about the last moments
of a market. They must mean exactly what the research measured (RESEARCH.md
section 8), so they are checked here against a plain, slow reference written
straight from those definitions."""
from __future__ import annotations

import numpy as np
import pytest

from kalshi_bot.core.tape import TradeTape


def _reference(T, P, Q, B, now, yes_side):
    """The research definitions, on arrays of everything that has traded so far."""
    fav = P if yes_side else 1.0 - P
    in1 = T >= now - 1.0
    in05 = T >= now - 0.5
    against = B != yes_side                         # the taker bought the other outcome = sold this one
    base = (T >= now - 11.0) & (T < now - 1.0)
    return (float(fav[in1].max()) if in1.any() else None,
            float(Q[in05 & against].sum()),
            float(Q[base].sum()) / 10.0,
            float(Q[T >= now - 10.0].sum()))


@pytest.mark.parametrize("seed", range(12))
def test_tape_matches_the_research_definitions(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(50, 1500))
    T = np.sort(rng.uniform(1000.0, 1040.0, n))
    T[rng.integers(0, n, n // 10)] = np.round(T[rng.integers(0, n, n // 10)], 1)     # bursts sharing a timestamp
    T = np.sort(T)
    P = np.round(rng.uniform(0.02, 0.98, n), 2)
    Q = np.round(rng.exponential(60.0, n) + 0.01, 2)
    B = rng.random(n) < 0.5
    tape = TradeTape(keep_s=60.0)
    fed = 0
    for now in np.sort(rng.uniform(1012.0, 1041.0, 60)):
        while fed < n and T[fed] < now:                # live: only what has already happened is known
            tape.add("M", T[fed], P[fed], Q[fed], bool(B[fed]))
            fed += 1
        for yes_side in (True, False):
            hi, sold, normal, vol10 = _reference(T[:fed], P[:fed], Q[:fed], B[:fed], now, yes_side)
            got = tape.high("M", yes_side, now, 1.0)
            assert (got is None) == (hi is None)
            if hi is not None:
                assert got == pytest.approx(hi)
            assert tape.sold("M", yes_side, now, 0.5) == pytest.approx(sold)
            assert tape.per_second("M", now, 11.0, 1.0) == pytest.approx(normal)
            assert tape.volume("M", now, 10.0) == pytest.approx(vol10)


def test_selling_means_takers_buying_the_other_side():
    tape = TradeTape()
    tape.add("M", 100.0, 0.90, 40, taker_yes=False)      # someone hit the YES bid: YES was sold
    tape.add("M", 100.1, 0.91, 25, taker_yes=True)       # someone lifted the YES offer: NO was "sold"
    assert tape.sold("M", True, 100.2, 0.5) == 40
    assert tape.sold("M", False, 100.2, 0.5) == 25
    assert tape.high("M", True, 100.2, 1.0) == pytest.approx(0.91)
    assert tape.high("M", False, 100.2, 1.0) == pytest.approx(0.10)       # NO traded at 10c and 9c


def test_markets_are_separate_and_old_trades_are_dropped():
    tape = TradeTape(keep_s=20.0)
    tape.add("A", 100.0, 0.5, 10, True)
    tape.add("B", 100.0, 0.6, 99, False)
    assert tape.sold("A", True, 100.1, 0.5) == 0 and tape.sold("B", True, 100.1, 0.5) == 99
    assert tape.high("C", True, 100.1, 1.0) is None and tape.sold("C", True, 100.1, 0.5) == 0
    for i in range(300):
        tape.add("A", 101.0 + i * 0.2, 0.5, 1, True)
    assert tape.trades("A") <= 102                       # 20 s at five trades a second, not all 301
    tape.keep_only(["A"])
    assert tape.trades("B") == 0 and tape.trades("A") > 0


def test_bad_prints_are_ignored_and_small_disorder_is_tolerated():
    tape = TradeTape()
    tape.add("M", 100.0, 0.0, 10, True)                  # impossible price
    tape.add("M", 100.0, 0.5, 0, True)                   # no size
    assert tape.trades("M") == 0 and tape.received == 0
    tape.add("M", 100.30, 0.80, 10, False)
    tape.add("M", 100.10, 0.95, 7, False)                # arrived late, stamped earlier
    tape.add("M", 99.00, 0.99, 5, False)                 # older than the half-second look-back
    assert tape.sold("M", True, 100.4, 0.5) == 17
    assert tape.high("M", True, 100.4, 1.0) == pytest.approx(0.95)
    assert tape.high("M", True, 100.4, 2.0) == pytest.approx(0.99)
    # a trade stamped slightly ahead of the engine's estimate of the exchange clock still counts
    tape.add("M", 100.45, 0.70, 3, False)
    assert tape.sold("M", True, 100.4, 0.5) == 20
