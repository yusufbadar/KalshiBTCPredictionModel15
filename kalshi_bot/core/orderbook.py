"""
Local order book for one binary market, kept in YES-price terms.

Kalshi's book has two bid ladders: YES bids and NO bids. A NO bid at price q
is an offer to sell YES at 1 - q, so in YES terms:

    bids = YES bids                (buy YES)
    asks = 1 - NO bids             (sell YES)

WebSocket subscriptions use ``use_yes_price: true`` so NO-side levels arrive
already converted to YES prices. REST snapshots arrive in NO-leg prices and
are converted here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

PRICE_Q = 10000  # prices stored as integer ten-thousandths of a dollar


def q(p: float | str) -> int:
    return int(round(float(p) * PRICE_Q))


@dataclass
class OrderBook:
    ticker: str
    bids: dict[int, float] = field(default_factory=dict)   # yes price (1e-4 $) -> size
    asks: dict[int, float] = field(default_factory=dict)
    seq: int = 0
    updated: float = 0.0
    valid: bool = False

    # ------------------------------------------------------------ mutation
    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.valid = False

    def apply_snapshot_ws(self, msg: dict) -> None:
        """WS snapshot with use_yes_price=true (no-side already in YES terms)."""
        self.bids = {q(p): float(s) for p, s in msg.get("yes_dollars_fp", []) if float(s) > 0}
        self.asks = {q(p): float(s) for p, s in msg.get("no_dollars_fp", []) if float(s) > 0}
        self.valid = True
        self.updated = time.time()

    def apply_snapshot_rest(self, ob: dict) -> None:
        """REST snapshot: ``yes_dollars`` YES bids, ``no_dollars`` NO bids (NO-leg prices)."""
        self.bids = {q(p): float(s) for p, s in (ob.get("yes_dollars") or []) if float(s) > 0}
        self.asks = {PRICE_Q - q(p): float(s) for p, s in (ob.get("no_dollars") or []) if float(s) > 0}
        self.valid = True
        self.updated = time.time()

    def apply_delta_ws(self, msg: dict) -> None:
        side = msg.get("side")
        book = self.bids if side == "yes" else self.asks
        k = q(msg["price_dollars"])
        v = book.get(k, 0.0) + float(msg["delta_fp"])
        if v <= 1e-9:
            book.pop(k, None)
        else:
            book[k] = v
        self.updated = time.time()

    # ------------------------------------------------------------- queries
    @property
    def best_bid(self) -> float | None:
        return max(self.bids) / PRICE_Q if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self.asks) / PRICE_Q if self.asks else None

    def bid_size(self, price: float | None = None) -> float:
        if not self.bids:
            return 0.0
        k = max(self.bids) if price is None else q(price)
        return self.bids.get(k, 0.0)

    def ask_size(self, price: float | None = None) -> float:
        if not self.asks:
            return 0.0
        k = min(self.asks) if price is None else q(price)
        return self.asks.get(k, 0.0)

    @property
    def mid(self) -> float | None:
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return None
        return (b + a) / 2

    def crossed(self) -> bool:
        b, a = self.best_bid, self.best_ask
        return b is not None and a is not None and b >= a

    def cost_to_buy_yes(self, count: float) -> tuple[float, float, float]:
        """Walk the asks. Returns (filled, total_cost, worst_price)."""
        left, cost, worst = count, 0.0, 0.0
        for k in sorted(self.asks):
            take = min(left, self.asks[k])
            cost += take * k / PRICE_Q
            worst = k / PRICE_Q
            left -= take
            if left <= 1e-9:
                break
        return count - left, cost, worst

    def cost_to_buy_no(self, count: float) -> tuple[float, float, float]:
        """Buying NO = selling YES into the bids. Cost per contract = 1 - bid."""
        left, cost, worst = count, 0.0, 0.0
        for k in sorted(self.bids, reverse=True):
            take = min(left, self.bids[k])
            cost += take * (1 - k / PRICE_Q)
            worst = 1 - k / PRICE_Q
            left -= take
            if left <= 1e-9:
                break
        return count - left, cost, worst

    def levels(self, n: int = 10) -> dict:
        return {
            "bids": [(k / PRICE_Q, self.bids[k]) for k in sorted(self.bids, reverse=True)[:n]],
            "asks": [(k / PRICE_Q, self.asks[k]) for k in sorted(self.asks)[:n]],
        }
