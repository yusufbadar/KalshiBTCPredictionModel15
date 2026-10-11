"""
Settlement lock-in strategy (live version of research/lockin.py).

KXBTC15M settles on the mean of the 60 BRTI prints stamped close-60..close-1.
Inside that final minute part of the average is already fixed, so the
outcome is much more certain than spot-vs-strike suggests. When the exact
averaging model rates one side a strong favourite and that side is offered
below model value (after the taker fee) by at least ``min_edge``, buy it with
an immediate-or-cancel order capped at the break-even-plus-edge price.

15-minute markets skip the first seconds of the window and, while more than
``early_seconds`` remain, need the expected settlement to be a minimum
distance from the strike: live trading (Oct 2026) and the backtest agree
that small early leads are where single-second index jumps do the damage.
The hourly strike ladder is not affected by either rule.
"""
from __future__ import annotations

import math

from kalshi_bot.config import StrategyParams
from kalshi_bot.core.fees import taker_fee
from kalshi_bot.model.fair_value import settlement_zscore
from kalshi_bot.strategy.base import Intent, TickState


def expected_settlement(st: TickState) -> float:
    """Expected value of the 60-print settlement average: prints already in
    the window as they are, the rest at the current index level."""
    if st.past_n > 0:
        return (st.past_sum + (60 - st.past_n) * st.x) / 60.0
    return st.x


def max_price_for_edge(p_win: float, min_edge: float) -> float:
    """Largest price L with  p_win - L - 0.07 L (1-L) >= min_edge."""
    lo, hi = 0.0, min(0.999, p_win)
    for _ in range(40):
        mid = (lo + hi) / 2
        if p_win - mid - taker_fee(mid) >= min_edge:
            lo = mid
        else:
            hi = mid
    return lo


class LockInStrategy:
    def __init__(self, p: StrategyParams):
        self.p = p
        self.note = ""
        self._last_order: dict[str, float] = {}

    def decide(self, st: TickState) -> list[Intent]:
        p = self.p
        b = st.book
        sl = st.seconds_left
        quarter = st.contract.is_quarter_hour
        window = p.max_seconds_left if quarter else p.hourly_max_seconds_left
        if sl > window:
            self.note = f"waiting for the final {window:.0f}s ({sl:.0f}s left)"
            return []
        if sl < p.min_seconds_left:
            self.note = "too close to the close"
            return []
        if b is None or not b.valid:
            self.note = "no order book"
            return []
        yes_fav = st.p_model >= 0.5
        p_fav = st.p_model if yes_fav else 1.0 - st.p_model
        outcome = "yes" if yes_fav else "no"
        z = float(settlement_zscore(st.x, st.strike, st.sigma, st.contract.close_ts - int(st.now),
                                    st.past_sum, st.past_n))
        z_fav = z if yes_fav else -z
        if p_fav < p.p_min or z_fav < p.z_min:
            self.note = f"model {100 * p_fav:.1f}% on {outcome.upper()} (z {z_fav:.1f}) — not confident enough"
            return []
        # early in a 15-minute market's window a small lead is one jump away from flipping
        if quarter and p.early_cushion_bp > 0 and sl > p.early_seconds:
            lead = abs(expected_settlement(st) - st.strike)
            if lead < p.early_cushion_bp * 1e-4 * st.x:
                self.note = (f"model {100 * p_fav:.1f}% on {outcome.upper()} but the lead is only "
                             f"{1e4 * lead / st.x:.1f} bp with {sl:.0f}s left — waiting")
                return []
        # never add to the other side
        if (yes_fav and st.no_qty > 0) or (not yes_fav and st.yes_qty > 0):
            self.note = "holding the other side"
            return []
        held = st.yes_qty if yes_fav else st.no_qty
        room = p.max_position_per_market - held
        if room < 1:
            self.note = "position limit reached"
            return []
        limit = max_price_for_edge(p_fav, p.min_edge)
        limit = min(limit, p.max_price)
        if yes_fav:
            best = b.best_ask
            # size only what is offered at or below our limit
            avail = sum(sz for k, sz in b.asks.items() if k / 10000 <= limit + 1e-9)
        else:
            best = None if b.best_bid is None else round(1 - b.best_bid, 4)
            avail = sum(sz for k, sz in b.bids.items() if 1 - k / 10000 <= limit + 1e-9)
        if best is None:
            self.note = f"model {100 * p_fav:.1f}% {outcome.upper()} — no offers"
            return []
        edge_now = p_fav - best - taker_fee(best)
        if best > limit + 1e-9 or avail < 1:
            self.note = (f"model {100 * p_fav:.1f}% {outcome.upper()}, best {100 * best:.1f}c "
                         f"(edge {100 * edge_now:.1f}c < {100 * p.min_edge:.0f}c)")
            return []
        tk = st.contract.ticker
        if st.now - self._last_order.get(tk, 0.0) < p.cooldown_s:
            self.note = "cooldown"
            return []
        # size up on bigger mispricings: base size x clip(edge / min_edge, 1, edge_size_max)
        mult = min(max(edge_now / p.min_edge, 1.0), max(1.0, p.edge_size_max))
        count = math.floor(min(p.max_contracts_per_order * mult, room, avail) * 100) / 100
        if count < 1:
            return []
        self._last_order[tk] = st.now
        self.note = (f"BUY {outcome.upper()} ≤{100 * limit:.1f}c: model {100 * p_fav:.2f}%, "
                     f"best {100 * best:.1f}c, edge {100 * edge_now:.1f}c")
        return [Intent("buy", outcome, limit, count, "immediate_or_cancel", False,
                       f"lock-in p={p_fav:.4f} z={z_fav:.1f} edge={100 * edge_now:.1f}c",
                       edge=edge_now)]
