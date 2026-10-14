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

Two "the market moved first" checks apply to every market. Kalshi's traders
react to a jump in the underlying price up to a second before the index value
this bot reads shows it, so right after a jump the model still calls the old
favourite and its collapsed price looks like a bargain. The bot does not buy
while (a) the favourite traded, or was bid, much higher a moment ago or (b) a
rush of selling has just hit it, nor for one more second after the warning
has passed; it looks again on every pass, and by then the index has caught up
(see RESEARCH.md section 8).
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


def model_sigma(sigma: float, x: float, seconds_left: float, p: StrategyParams) -> float:
    """Volatility handed to the model: the live estimate (floored), times the
    safety factor, times extra caution early in the averaging window."""
    sig = max(sigma, p.vol_floor_rel * (x or 0.0)) * p.sigma_scale
    if p.sigma_time_k > 0:
        sig *= 1.0 + p.sigma_time_k * max(0.0, (seconds_left - 30.0) / 30.0)
    return sig


def market_moved_first(p: StrategyParams, tape, ticker: str, yes_fav: bool, best: float, now: float) -> str:
    """The two checks. Returns '' when it is fine to buy, otherwise the reason
    to wait. ``best`` is the price the favourite can be bought at right now."""
    if tape is None:
        return ""
    if p.fall_guard > 0:
        hi = tape.high(ticker, yes_fav, now, p.fall_window_s)
        if hi is not None and hi - best > p.fall_guard + 1e-9:
            return (f"its price just fell {100 * (hi - best):.0f}c (it traded at {100 * hi:.0f}c "
                    f"within the last {p.fall_window_s:g}s)")
    if p.sell_guard_x > 0:
        sold = tape.sold(ticker, yes_fav, now, p.sell_window_s)
        if sold > 0:
            normal = max(tape.per_second(ticker, now, p.sell_base_s + 1.0, 1.0), p.sell_base_floor)
            if sold > p.sell_guard_x * normal + 1e-9:
                return (f"a rush of selling just hit it ({sold:,.0f} contracts in {p.sell_window_s:g}s, "
                        f"{sold / normal:.1f}x a normal second)")
    return ""


class LockInStrategy:
    def __init__(self, p: StrategyParams):
        self.p = p
        self.note = ""
        self._last_order: dict[str, float] = {}
        self.waited: dict[str, tuple[float, str]] = {}      # ticker -> (time, reason) of the last "market moved first" wait

    def not_sent(self, ticker: str) -> None:
        """The engine dropped the buy this strategy just asked for (missing
        data, not a rejection by the exchange): the order cooldown should not
        run, so the buy can go out as soon as the reason is gone."""
        self._last_order.pop(ticker, None)

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
        # the market moved first: a price that has just dropped, or a rush of selling, means other
        # traders are reacting to something the index has not shown yet - do not buy, look again
        why = market_moved_first(p, st.tape, tk, yes_fav, best, st.now)
        if why:
            self.waited[tk] = (st.now, why)
            self.note = f"model {100 * p_fav:.1f}% {outcome.upper()} at {100 * best:.0f}c, but {why} — waiting"
            return []
        # ... and for a moment after the warning has passed: the index needs about a second to show
        # what the market reacted to, and prices often bounce before it does
        last = self.waited.get(tk)
        if last is not None and 0.0 <= st.now - last[0] < p.guard_hold_s:
            self.note = (f"model {100 * p_fav:.1f}% {outcome.upper()} at {100 * best:.0f}c — letting the index catch up "
                         f"for {p.guard_hold_s - (st.now - last[0]):.1f}s more ({last[1]})")
            return []
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
