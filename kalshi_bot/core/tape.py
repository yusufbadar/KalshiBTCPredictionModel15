"""
Recent public trades, per market: what just traded, at what price, and which
side was the aggressor.

Kalshi's own traders react to a jump in the underlying price up to a second
before the index value this bot reads shows it. The trade tape is how the bot
notices: the two "the market moved first" checks in strategy/lockin.py ask it
whether the favourite's price has just dropped and whether a rush of selling
has just hit it.

Times are on Kalshi's clock (the timestamp of each trade), the same clock the
engine's ``now`` is on. A query looks back from ``now``; trades stamped a few
milliseconds "ahead" of the engine's estimate of that clock still count.

The engine also notes the best bid and offer it sees on every pass
(``quote``). A bid is a price somebody was prepared to pay, so "this outcome
was bid at 97c a moment ago and is offered at 80c now" is the same warning as
a trade at 97c, and it does not depend on a trade message having arrived yet.
"""
from __future__ import annotations

import time
from collections import deque

# a trade may arrive this many seconds out of order and still be found by a look-back
_DISORDER_S = 1.0


class TradeTape:
    def __init__(self, keep_s: float = 20.0):
        self.keep_s = keep_s
        self._t: dict[str, deque] = {}            # ticker -> (ts, yes_price, contracts, taker_bought_yes)
        self._q: dict[str, deque] = {}            # ticker -> (ts, best yes bid, best yes ask) as the engine saw them
        self.last_rx = 0.0                        # local time the last trade arrived (any market)
        self.received = 0                         # trades taken in since start

    # ------------------------------------------------------------- ingest
    def add(self, ticker: str, ts: float, yes_price: float, count: float, taker_yes: bool) -> None:
        if count <= 0 or not (0.0 < yes_price < 1.0):
            return
        d = self._t.get(ticker)
        if d is None:
            d = self._t[ticker] = deque()
        d.append((float(ts), float(yes_price), float(count), bool(taker_yes)))
        self.last_rx = time.time()
        self.received += 1
        newest = d[-1][0]
        while d and d[0][0] < newest - self.keep_s:
            d.popleft()

    def quote(self, ticker: str, ts: float, yes_bid: float | None, yes_ask: float | None) -> None:
        """Best bid and offer of the YES side as the engine sees them now."""
        d = self._q.get(ticker)
        if d is None:
            d = self._q[ticker] = deque()
        d.append((float(ts), yes_bid, yes_ask))
        while d and d[0][0] < ts - 5.0:
            d.popleft()

    def keep_only(self, tickers) -> None:
        """Forget markets that are no longer watched."""
        keep = set(tickers)
        for book in (self._t, self._q):
            for tk in [tk for tk in book if tk not in keep]:
                del book[tk]

    # ------------------------------------------------------------ queries
    def _recent(self, ticker: str, since: float):
        """Trades stamped at or after ``since``, newest first."""
        d = self._t.get(ticker)
        if not d:
            return
        for tr in reversed(d):
            if tr[0] >= since:
                yield tr
            elif tr[0] < since - _DISORDER_S:
                return

    def high(self, ticker: str, yes_side: bool, now: float, window: float) -> float | None:
        """Highest price this outcome traded at, or was bid at, in the last
        ``window`` seconds (every trade counts, whoever was the aggressor).
        None when there is nothing to go on."""
        best = None
        for _, p, _, _ in self._recent(ticker, now - window):
            px = p if yes_side else 1.0 - p
            if best is None or px > best:
                best = px
        q = self._q.get(ticker)
        if q:
            since = now - window
            for ts, bid, ask in reversed(q):
                if ts < since:
                    break
                px = bid if yes_side else (None if ask is None else 1.0 - ask)   # a NO bid is a YES offer
                if px is not None and (best is None or px > best):
                    best = px
        return best

    def sold(self, ticker: str, yes_side: bool, now: float, window: float) -> float:
        """Contracts of this outcome that were *sold* by aggressors (takers who
        bought the other outcome) in the last ``window`` seconds."""
        tot = 0.0
        for _, _, q, taker_yes in self._recent(ticker, now - window):
            if taker_yes != yes_side:
                tot += q
        return tot

    def per_second(self, ticker: str, now: float, far: float = 11.0, near: float = 1.0) -> float:
        """Average contracts traded per second (both sides) between ``far``
        and ``near`` seconds ago: what a normal second in this market looks like."""
        if far <= near:
            return 0.0
        hi = now - near
        tot = 0.0
        for ts, _, q, _ in self._recent(ticker, now - far):
            if ts < hi:
                tot += q
        return tot / (far - near)

    def volume(self, ticker: str, now: float, window: float) -> float:
        """Contracts traded (both sides) in the last ``window`` seconds."""
        return sum(q for _, _, q, _ in self._recent(ticker, now - window))

    def trades(self, ticker: str) -> int:
        d = self._t.get(ticker)
        return len(d) if d else 0
