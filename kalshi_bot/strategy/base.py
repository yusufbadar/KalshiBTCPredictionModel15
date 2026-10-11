"""Strategy interface: a pure function of the current tick state -> intents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kalshi_bot.core.markets import Contract
from kalshi_bot.core.orderbook import OrderBook


@dataclass
class TickState:
    now: float
    contract: Contract
    seconds_left: float
    x: float                    # latest BRTI value (5 Hz)
    x_print: float              # last whole-second print
    strike: float
    sigma: float                # $/sqrt(s)
    past_sum: float             # settlement-window prints observed so far
    past_n: int
    p_model: float              # model P(YES)
    book: OrderBook | None
    yes_qty: float
    no_qty: float
    working: list = field(default_factory=list)
    balance: float = 0.0


@dataclass
class Intent:
    kind: str                   # 'buy' | 'cancel' | 'cancel_all'
    outcome: str = ""
    price: float = 0.0          # limit price of the outcome bought
    count: float = 0.0
    tif: str = "immediate_or_cancel"
    post_only: bool = False
    reason: str = ""
    order: Any = None
    edge: float = 0.0           # model value - price - fee at the best offer, per contract
