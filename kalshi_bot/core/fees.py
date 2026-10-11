"""
Kalshi trading fees for KXBTC15M (fee_type ``quadratic``, multiplier 1).

    taker fee = 0.07   * C * P * (1 - P)
    maker fee = 0.0175 * C * P * (1 - P) * maker_multiplier   (0 on this series)

P is the execution price of the contract bought (YES price for YES, NO price
for NO - the formula is symmetric). Fees are computed to $0.000001 and then
rounded against the member's balance precision, with an accumulator that
refunds over-collection across an order's fills, so for sizing purposes the
unrounded value is accurate to well under a cent per order.
"""
from __future__ import annotations

TAKER_RATE = 0.07
MAKER_RATE = 0.0175
MAKER_MULTIPLIER = 0.0   # KXBTC15M carries no maker fee (verified on fills)


def taker_fee(price: float, count: float = 1.0) -> float:
    return TAKER_RATE * count * price * (1.0 - price)


def maker_fee(price: float, count: float = 1.0) -> float:
    return MAKER_RATE * MAKER_MULTIPLIER * count * price * (1.0 - price)


def all_in_cost(price: float, taker: bool = True) -> float:
    """Cost per contract including fees."""
    return price + (taker_fee(price) if taker else maker_fee(price))
