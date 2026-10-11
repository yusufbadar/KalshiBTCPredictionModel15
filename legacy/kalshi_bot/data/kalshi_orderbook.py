"""
Kalshi orderbook data feed for KXBTC15M markets.

Pulls the YES / NO bid arrays and computes imbalance, spread, and depth
features that feed into the ML model.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from kalshi_bot.kalshi.client import KalshiClient


def _clamp_cents(c: int) -> int:
    return max(1, min(99, c))


def _implied_yes_ask_cents(no_bid_dollars: float) -> int:
    return _clamp_cents(int(round((1.0 - float(no_bid_dollars)) * 100)))


def _implied_no_ask_cents(yes_bid_dollars: float) -> int:
    return _clamp_cents(int(round((1.0 - float(yes_bid_dollars)) * 100)))


def sweep_yes_buy_limit(no_bids: list, want: int) -> tuple[int, int] | None:
    """Walk NO bid ladder (implied YES asks), best first.

    Returns (yes_limit_cents, fillable_count) with fillable_count <= want,
    or None if no size.
    """
    if want < 1 or not no_bids:
        return None
    need = want
    worst = 0
    for i in range(len(no_bids) - 1, -1, -1):
        row = no_bids[i]
        nb = float(row[0])
        qty = int(float(row[1]))
        if qty < 1:
            continue
        yac = _implied_yes_ask_cents(nb)
        take = min(need, qty)
        if take > 0:
            worst = max(worst, yac)
        need -= take
        if need <= 0:
            lim = min(99, worst + 1)
            return lim, want
    got = want - need
    if got < 1:
        return None
    lim = min(99, worst + 1)
    return lim, got


def sweep_no_buy_limit(yes_bids: list, want: int) -> tuple[int, int] | None:
    """Walk YES bid ladder (implied NO asks), best first."""
    if want < 1 or not yes_bids:
        return None
    need = want
    worst = 0
    for i in range(len(yes_bids) - 1, -1, -1):
        row = yes_bids[i]
        yb = float(row[0])
        qty = int(float(row[1]))
        if qty < 1:
            continue
        nac = _implied_no_ask_cents(yb)
        take = min(need, qty)
        if take > 0:
            worst = max(worst, nac)
        need -= take
        if need <= 0:
            lim = min(99, worst + 1)
            return lim, want
    got = want - need
    if got < 1:
        return None
    lim = min(99, worst + 1)
    return lim, got


def _size_yes_buy_for_budget(no_bids: list, budget: float) -> tuple[int, int] | None:
    """Max YES contracts spend ≤ ``budget`` (walk book at implied YES ask dollars).

    Returns ``(count, limit_cents)`` for FOK, or ``None``.
    """
    if budget < 0.01 or not no_bids:
        return None
    remaining = float(budget)
    got = 0
    worst_cents = 0
    for i in range(len(no_bids) - 1, -1, -1):
        row = no_bids[i]
        nb = float(row[0])
        qty = int(float(row[1]))
        if qty < 1:
            continue
        price_d = 1.0 - nb
        if price_d <= 0:
            continue
        yac = _implied_yes_ask_cents(nb)
        can_afford = int(remaining / price_d + 1e-12)
        take = min(qty, can_afford)
        if take < 1:
            continue
        got += take
        worst_cents = max(worst_cents, yac)
        remaining -= take * price_d
    if got < 1:
        return None
    lim = min(99, worst_cents + 1)
    cap = int(budget / (lim / 100.0))
    if cap < 1:
        return None
    got = min(got, cap)
    sw = sweep_yes_buy_limit(no_bids, got)
    if not sw:
        return None
    lim_f, book_cap = sw
    got = min(got, book_cap)
    cap2 = int(budget / (lim_f / 100.0))
    got = min(got, cap2)
    if got < 1:
        return None
    sw2 = sweep_yes_buy_limit(no_bids, got)
    if not sw2:
        return None
    return sw2[1], sw2[0]


def _size_no_buy_for_budget(yes_bids: list, budget: float) -> tuple[int, int] | None:
    """Max NO contracts spend ≤ ``budget`` (walk book at implied NO ask dollars)."""
    if budget < 0.01 or not yes_bids:
        return None
    remaining = float(budget)
    got = 0
    worst_cents = 0
    for i in range(len(yes_bids) - 1, -1, -1):
        row = yes_bids[i]
        yb = float(row[0])
        qty = int(float(row[1]))
        if qty < 1:
            continue
        price_d = 1.0 - yb
        if price_d <= 0:
            continue
        nac = _implied_no_ask_cents(yb)
        can_afford = int(remaining / price_d + 1e-12)
        take = min(qty, can_afford)
        if take < 1:
            continue
        got += take
        worst_cents = max(worst_cents, nac)
        remaining -= take * price_d
    if got < 1:
        return None
    lim = min(99, worst_cents + 1)
    cap = int(budget / (lim / 100.0))
    if cap < 1:
        return None
    got = min(got, cap)
    sw = sweep_no_buy_limit(yes_bids, got)
    if not sw:
        return None
    lim_f, book_cap = sw
    got = min(got, book_cap)
    cap2 = int(budget / (lim_f / 100.0))
    got = min(got, cap2)
    if got < 1:
        return None
    sw2 = sweep_no_buy_limit(yes_bids, got)
    if not sw2:
        return None
    return sw2[1], sw2[0]


class KalshiOrderbookFeed:
    def __init__(self, client: KalshiClient):
        self.client = client

    def fetch(self, ticker: str) -> Optional[dict]:
        """Fetch orderbook and compute derived features.

        Returns dict with:
            yes_bids, no_bids       – raw arrays
            best_yes_bid, best_no_bid
            best_yes_ask (implied), best_no_ask (implied)
            spread, mid_price
            yes_depth, no_depth     – total contracts on each side
            imbalance               – (yes_depth - no_depth) / total
        """
        try:
            raw = self.client.get_market_orderbook(ticker)
            ob = raw.get("orderbook_fp") or raw.get("orderbook", {})

            yes_bids = ob.get("yes_dollars", ob.get("yes", []))
            no_bids = ob.get("no_dollars", ob.get("no", []))

            best_yes_bid = float(yes_bids[-1][0]) if yes_bids else 0.0
            best_no_bid = float(no_bids[-1][0]) if no_bids else 0.0

            best_yes_ask = 1.0 - best_no_bid if best_no_bid else 1.0
            best_no_ask = 1.0 - best_yes_bid if best_yes_bid else 1.0

            spread = best_yes_ask - best_yes_bid if best_yes_bid else 1.0
            mid_price = (best_yes_bid + best_yes_ask) / 2 if best_yes_bid else 0.5

            yes_depth = sum(float(b[1]) for b in yes_bids) if yes_bids else 0.0
            no_depth = sum(float(b[1]) for b in no_bids) if no_bids else 0.0
            total_depth = yes_depth + no_depth
            imbalance = (yes_depth - no_depth) / total_depth if total_depth > 0 else 0.0

            return {
                "best_yes_bid": best_yes_bid,
                "best_no_bid": best_no_bid,
                "best_yes_ask": best_yes_ask,
                "best_no_ask": best_no_ask,
                "spread": round(spread, 4),
                "mid_price": round(mid_price, 4),
                "yes_depth": yes_depth,
                "no_depth": no_depth,
                "imbalance": round(imbalance, 4),
                "yes_levels": len(yes_bids),
                "no_levels": len(no_bids),
            }

        except Exception as e:
            logger.warning("Kalshi orderbook fetch failed for {}: {}", ticker, e)
            return None

    def sweep_buy(self, ticker: str, side: str, want: int) -> Optional[tuple[int, int]]:
        """Fresh orderbook sweep for an immediate buy: (limit_cents, max_contracts).

        ``side`` is ``yes`` or ``no``. Limit is set to cover the worst level
        needed for up to ``want`` contracts (plus 1¢ headroom), capped at 99.
        """
        if want < 1:
            return None
        try:
            raw = self.client.get_market_orderbook(ticker)
            ob = raw.get("orderbook_fp") or raw.get("orderbook", {})
            yes_bids = ob.get("yes_dollars", ob.get("yes", []))
            no_bids = ob.get("no_dollars", ob.get("no", []))
            if side == "yes":
                return sweep_yes_buy_limit(no_bids, want)
            return sweep_no_buy_limit(yes_bids, want)
        except Exception as e:
            logger.warning("Kalshi sweep_buy failed for {} {}: {}", ticker, side, e)
            return None

    def size_buy_for_budget(
        self, ticker: str, side: str, budget_dollars: float
    ) -> Optional[tuple[int, int]]:
        """Contracts and FOK limit to spend up to ``budget_dollars`` at book prices.

        Returns ``(count, limit_cents)`` or ``None`` if the book cannot support
        at least one contract within budget.
        """
        if budget_dollars < 0.01:
            return None
        try:
            raw = self.client.get_market_orderbook(ticker)
            ob = raw.get("orderbook_fp") or raw.get("orderbook", {})
            yes_bids = ob.get("yes_dollars", ob.get("yes", []))
            no_bids = ob.get("no_dollars", ob.get("no", []))
            if side == "yes":
                return _size_yes_buy_for_budget(no_bids, budget_dollars)
            return _size_no_buy_for_budget(yes_bids, budget_dollars)
        except Exception as e:
            logger.warning(
                "Kalshi size_buy_for_budget failed for {} {}: {}", ticker, side, e
            )
            return None
