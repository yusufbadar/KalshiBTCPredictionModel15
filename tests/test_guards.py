"""The two "the market moved first" checks (added 13 Oct 2026, RESEARCH.md section 8).

Kalshi's traders react to a jump in Bitcoin up to a second before the index
value this bot reads shows it. Right after such a jump the model still calls
the old favourite, and its collapsed price looks like a bargain. Both losing
trades after the v3 update were bought exactly there. The bot now waits while

* the favourite traded more than 10c higher within the last second, or
* more of it was sold in the last half second than normally trades in a second.

``tests/data/real_losses.json`` holds the public tape and the bot's own inputs
at those two real buys; the tests replay them through the live strategy."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

import kalshi_bot.config as config
from kalshi_bot.api.ws import KalshiWS
from kalshi_bot.config import RiskParams, Settings, StrategyParams
from kalshi_bot.core.markets import Contract
from kalshi_bot.core.orderbook import OrderBook
from kalshi_bot.core.orders import Broker, OrderManager
from kalshi_bot.core.tape import TradeTape
from kalshi_bot.engine import Engine
from kalshi_bot.model.fair_value import fair_yes
from kalshi_bot.strategy.base import TickState
from kalshi_bot.strategy.lockin import LockInStrategy, market_moved_first

GRID = [(0.0, 0.1, 0.001), (0.1, 0.9, 0.01), (0.9, 1.0, 0.001)]
STRIKE = 80_000.0
NOW = 1_800_000_000.0
REAL = json.loads((Path(__file__).parent / "data" / "real_losses.json").read_text(encoding="utf-8"))


def _state(tape, *, yes_fav=True, price=0.93, seconds_left=30.0, hourly=False, now=NOW, size=500.0):
    """A confident favourite (well clear of the strike, half the window already
    in) that can be bought at ``price``."""
    close = int(now + seconds_left)
    c = Contract("KXBTCD-T" if hourly else "KXBTC15M-T", "EV", close - (3600 if hourly else 900), close, STRIKE,
                 "api", "active", 2, GRID, strike_type="greater" if hourly else "greater_or_equal")
    x = STRIKE + (30.0 if yes_fav else -30.0)
    pn = 60 - int(seconds_left)
    sigma = 2.0
    p = fair_yes(x, c.threshold, sigma, int(now), close, pn * x, pn, 5.0)
    b = OrderBook(c.ticker)
    if yes_fav:
        b.asks, b.bids = {int(round(price * 10000)): size}, {int(round((price - 0.02) * 10000)): size}
    else:                                   # NO at `price` = a YES bid at 1 - price
        b.bids, b.asks = {int(round((1 - price) * 10000)): size}, {int(round((1 - price + 0.02) * 10000)): size}
    b.valid, b.updated = True, time.time()
    return TickState(now=now, contract=c, seconds_left=seconds_left, x=x, x_print=x, strike=c.threshold, sigma=sigma,
                     past_sum=pn * x, past_n=pn, p_model=p, book=b, yes_qty=0.0, no_qty=0.0, tape=tape)


def _decide(tape, params: StrategyParams | None = None, **kw):
    s = LockInStrategy(params or StrategyParams(cooldown_s=0.0))
    out = s.decide(_state(tape, **kw))
    return out, s.note


def _quiet(tape, tk, per_second=100.0, now=NOW):
    """Ten ordinary seconds of trading, ending a second ago."""
    for i in range(10):
        tape.add(tk, now - 10.5 + i, 0.93, per_second, taker_yes=bool(i % 2))


# ------------------------------------------------------------------ defaults
def test_shipped_settings_are_the_tested_ones():
    sp = StrategyParams()
    assert (sp.fall_guard, sp.fall_window_s) == (0.10, 1.0)
    assert (sp.sell_guard_x, sp.sell_window_s, sp.sell_base_s, sp.sell_base_floor) == (1.0, 0.5, 10.0, 50.0)
    assert sp.guard_hold_s == 1.0


def test_calm_market_buys_as_before():
    for tape in (TradeTape(), None):                     # no trades at all, or no tape object
        out, note = _decide(tape)
        assert len(out) == 1 and out[0].outcome == "yes", note
    tape = TradeTape()
    _quiet(tape, "KXBTC15M-T")
    tape.add("KXBTC15M-T", NOW - 0.3, 0.93, 40, taker_yes=False)      # a little selling, same price
    out, note = _decide(tape)
    assert len(out) == 1, note


# ------------------------------------------------------- price just fell
def test_price_that_just_fell_is_not_bought():
    tape = TradeTape()
    tape.add("KXBTC15M-T", NOW - 0.6, 0.98, 5, taker_yes=True)        # it was trading at 98c ...
    out, note = _decide(tape, price=0.85)                             # ... and is offered at 85c now
    assert out == [] and "price just fell 13c" in note and "waiting" in note
    # once that trade is more than a second old the bot may buy (the index has had time to catch up)
    out, note = _decide(tape, price=0.85, now=NOW + 0.5)
    assert len(out) == 1, note


def test_price_fall_threshold_is_ten_cents():
    for was, expect_buy in ((0.95, True), (0.96, False)):             # 10c is allowed, 11c is not
        tape = TradeTape()
        tape.add("KXBTC15M-T", NOW - 0.2, was, 5, taker_yes=False)
        out, note = _decide(tape, price=0.85)
        assert (len(out) == 1) == expect_buy, note
    tape = TradeTape()
    tape.add("KXBTC15M-T", NOW - 0.2, 0.70, 5, taker_yes=True)        # the price ROSE to here: fine
    out, note = _decide(tape, price=0.85)
    assert len(out) == 1, note


def test_a_recent_higher_bid_counts_as_a_fall_even_before_any_trade_is_reported():
    """The engine notes the best bid on every pass. A side that was BID at 97c a
    moment ago and is offered at 85c now has fallen, whether or not the trade
    messages for that move have arrived yet."""
    tk = "KXBTC15M-T"
    tape = TradeTape()
    tape.quote(tk, NOW - 0.3, yes_bid=0.97, yes_ask=0.98)
    out, note = _decide(tape, price=0.85)
    assert out == [] and "price just fell 12c" in note
    out, note = _decide(tape, price=0.85, now=NOW + 0.8)              # that quote is now more than a second old
    assert len(out) == 1, note
    # an OFFER that was higher a moment ago is no warning (thin books gap upwards all the time)
    tape = TradeTape()
    tape.quote(tk, NOW - 0.3, yes_bid=0.84, yes_ask=0.99)
    out, note = _decide(tape, price=0.85)
    assert len(out) == 1, note
    # NO side: a NO bid is a YES offer
    tape = TradeTape()
    tape.quote(tk, NOW - 0.3, yes_bid=0.02, yes_ask=0.03)             # NO was bid at 97c
    out, note = _decide(tape, yes_fav=False, price=0.85)
    assert out == [] and "price just fell 12c" in note
    tape = TradeTape()
    tape.quote(tk, NOW - 0.3, yes_bid=None, yes_ask=None)             # empty book a moment ago: nothing to go on
    out, note = _decide(tape, price=0.85)
    assert len(out) == 1, note


# ------------------------------------------------------ rush of selling
def test_rush_of_selling_is_not_bought_into():
    tk = "KXBTC15M-T"
    for sold, expect_buy in ((90.0, True), (100.0, True), (101.0, False), (150.0, False)):
        tape = TradeTape()
        _quiet(tape, tk, per_second=100.0)                            # a normal second here is 100 contracts
        tape.add(tk, NOW - 0.2, 0.93, sold, taker_yes=False)          # takers selling YES
        out, note = _decide(tape)
        assert (len(out) == 1) == expect_buy, (sold, note)
        if not expect_buy:
            assert "rush of selling" in note and "waiting" in note
    # the same amount BOUGHT is no warning, and selling older than half a second no longer counts
    tape = TradeTape()
    _quiet(tape, tk)
    tape.add(tk, NOW - 0.2, 0.93, 5000, taker_yes=True)
    tape.add(tk, NOW - 0.7, 0.93, 5000, taker_yes=False)
    out, note = _decide(tape)
    assert len(out) == 1, note


def test_thin_markets_use_a_floor_for_the_normal_second():
    for sold, expect_buy in ((40.0, True), (60.0, False)):            # nothing traded before: 50 a second is assumed
        tape = TradeTape()
        tape.add("KXBTCD-T", NOW - 0.1, 0.93, sold, taker_yes=False)
        out, note = _decide(tape, hourly=True)
        assert (len(out) == 1) == expect_buy, note


def test_checks_mirror_for_the_no_side():
    tk = "KXBTC15M-T"
    # favourite is NO at 93c. NO "sold" = takers buying YES; NO traded high = YES traded low
    tape = TradeTape()
    _quiet(tape, tk)
    tape.add(tk, NOW - 0.2, 0.07, 500, taker_yes=True)
    out, note = _decide(tape, yes_fav=False)
    assert out == [] and "rush of selling" in note
    tape = TradeTape()
    _quiet(tape, tk)
    tape.add(tk, NOW - 0.2, 0.07, 500, taker_yes=False)               # takers buying NO: not selling
    out, note = _decide(tape, yes_fav=False)
    assert len(out) == 1 and out[0].outcome == "no", note
    tape = TradeTape()
    tape.add(tk, NOW - 0.4, 0.02, 1, taker_yes=False)                 # NO traded at 98c a moment ago
    out, note = _decide(tape, yes_fav=False, price=0.85)
    assert out == [] and "price just fell 13c" in note


def test_each_check_can_be_switched_off():
    tk = "KXBTC15M-T"
    tape = TradeTape()
    _quiet(tape, tk)
    tape.add(tk, NOW - 0.2, 0.98, 5000, taker_yes=False)              # both checks would fire
    assert _decide(tape, price=0.85)[0] == []
    assert _decide(tape, StrategyParams(cooldown_s=0.0, fall_guard=0.0), price=0.85)[0] == []      # selling still blocks
    assert _decide(tape, StrategyParams(cooldown_s=0.0, sell_guard_x=0.0), price=0.85)[0] == []    # the fall still blocks
    out, note = _decide(tape, StrategyParams(cooldown_s=0.0, fall_guard=0.0, sell_guard_x=0.0), price=0.85)
    assert len(out) == 1, note


def test_after_a_warning_it_stays_out_one_more_second():
    """A check stops firing a fraction of a second before the index has caught
    up (the selling check after half a second). Replaying the live code over
    the whole history showed the checks do half the good without this pause."""
    tk = "KXBTC15M-T"
    tape = TradeTape()
    _quiet(tape, tk)
    tape.add(tk, NOW - 0.3, 0.93, 400, taker_yes=False)               # a rush of selling, 0.3 s ago
    s = LockInStrategy(StrategyParams())
    assert s.decide(_state(tape)) == [] and "rush of selling" in s.note
    assert tk in s.waited
    # 0.4 s later the selling is more than half a second old: the check itself is quiet again ...
    assert market_moved_first(s.p, tape, tk, True, 0.93, NOW + 0.4) == ""
    # ... but the bot still stays out, and says why
    assert s.decide(_state(tape, now=NOW + 0.4)) == []
    assert "letting the index catch up" in s.note and "rush of selling" in s.note
    assert s.decide(_state(tape, now=NOW + 0.95)) == []
    out = s.decide(_state(tape, now=NOW + 1.05))                      # one second after the last warning
    assert len(out) == 1, s.note
    # the pause is per market, and it is a setting
    s2 = LockInStrategy(StrategyParams(guard_hold_s=0.0))
    assert s2.decide(_state(tape)) == []
    assert len(s2.decide(_state(tape, now=NOW + 0.4))) == 1


def test_a_wait_does_not_start_the_order_cooldown():
    """Waiting must not delay the buy that follows once the market has settled."""
    tk = "KXBTC15M-T"
    tape = TradeTape()
    tape.add(tk, NOW - 0.9, 0.98, 5, taker_yes=True)
    s = LockInStrategy(StrategyParams(guard_hold_s=0.2, cooldown_s=0.5))
    assert s.decide(_state(tape, price=0.85)) == []
    assert tk in s.waited
    assert len(s.decide(_state(tape, price=0.85, now=NOW + 0.25))) == 1   # inside what would be the cooldown


# ------------------------------------------------- the two real losing buys
@pytest.mark.parametrize("case", REAL, ids=[c["ticker"] for c in REAL])
def test_the_real_losing_buys_are_now_waited_out(case):
    c = Contract(case["ticker"], "EV", case["open_ts"], case["close_ts"], case["strike"], "api", "active", 2, GRID,
                 strike_type=case["strike_type"])
    now = case["order_ts"]
    tape = TradeTape()
    for dt_ms, yes_price, count, taker_yes in case["trades"]:
        tape.add(c.ticker, now + dt_ms / 1000.0, yes_price, count, taker_yes)
    yes = case["bought"] == "yes"
    b = OrderBook(c.ticker)                               # what the bot could buy at: the price it really paid
    if yes:
        b.asks = {int(round(case["paid"] * 10000)): 500.0}
    else:
        b.bids = {int(round((1 - case["paid"]) * 10000)): 500.0}
    b.valid, b.updated = True, time.time()
    st = TickState(now=now, contract=c, seconds_left=case["close_ts"] - now, x=case["x"], x_print=case["x"],
                   strike=c.threshold, sigma=case["sigma"], past_sum=case["past_sum"], past_n=case["past_n"],
                   p_model=case["p_model"], book=b, yes_qty=0.0, no_qty=0.0, tape=tape)
    # without the checks the strategy does what the bot really did
    off = LockInStrategy(StrategyParams(fall_guard=0.0, sell_guard_x=0.0))
    out = off.decide(st)
    assert len(out) == 1 and out[0].outcome == case["bought"], off.note
    assert case["settled"] != case["bought"]              # ... and that trade lost
    # with them it waits, and each check on its own is enough
    on = LockInStrategy(StrategyParams())
    assert on.decide(st) == [] and "waiting" in on.note, on.note
    sp = StrategyParams()
    assert "price just fell" in market_moved_first(StrategyParams(sell_guard_x=0.0), tape, c.ticker, yes, case["paid"], now)
    assert "rush of selling" in market_moved_first(StrategyParams(fall_guard=0.0), tape, c.ticker, yes, case["paid"], now)
    # the numbers behind it, as measured in the research
    sold = tape.sold(c.ticker, yes, now, sp.sell_window_s)
    normal = max(tape.per_second(c.ticker, now, sp.sell_base_s + 1.0, 1.0), sp.sell_base_floor)
    assert sold > 2 * normal
    assert tape.high(c.ticker, yes, now, sp.fall_window_s) - case["paid"] > 0.15


# ------------------------------------------------------------- the engine
class FakeBroker(Broker):
    def __init__(self):
        self.placed = []

    async def place(self, o):
        self.placed.append(o)
        return {"order_id": f"o{len(self.placed)}", "fill_count": f"{o.count:.2f}", "remaining_count": "0"}

    async def cancel(self, o):
        pass

    async def cancel_all(self):
        pass


class FakeClock:
    def __init__(self, c):
        self.current = c

    async def refresh(self, index=None):
        pass


class FakeFeed:
    def __init__(self):
        self.trade_listeners = []

    def set_tickers(self, tks):
        self.tickers = tks


class FakeWS:
    def __init__(self, confirmed=True):
        self.connected = asyncio.Event()
        self.connected.set()
        self.last_msg_ts = time.time()
        self.confirmed = confirmed
        self.subs, self.unsubs = {}, []

    def subscribed(self, name):
        return self.confirmed

    async def update(self, *a, **k):
        pass

    async def unsubscribe(self, name):
        self.unsubs.append(name)
        self.subs.pop(name, None)

    async def subscribe(self, name, channels, **params):
        self.subs[name] = (channels, params)


def _engine(tmp_path, monkeypatch, *, env="paper", seconds_left=30, lead=100.0):
    """A 15-minute market `seconds_left` before its close: index flat at
    strike + lead (YES is the favourite), YES offered at 80c."""
    import kalshi_bot.engine as eng
    monkeypatch.setattr(eng, "DB_PATH", tmp_path / "e.sqlite3")
    now = int(time.time())
    close = now + seconds_left
    sp = StrategyParams(trade_hourly=False)
    s = Settings(env="paper", api_key_id="x", private_key_path="x", strategy=sp,
                 risk=RiskParams(pct_per_market=0.10, pct_per_close=0.15), paper_start=100.0)
    e = Engine(s)
    e.feed = FakeFeed()
    broker = FakeBroker()
    e.om = OrderManager(broker, e.ledger, "paper")
    a = e.assets["BTC"]
    c = Contract("KXBTC15M-TEST", "EV", close - 900, close, STRIKE, "api", "active", 2, GRID)
    a.clock = FakeClock(c)
    b = OrderBook(c.ticker)
    b.asks, b.bids = {8000: 500.0}, {7900: 500.0}
    b.valid, b.updated = True, time.time()
    e.books[c.ticker] = b
    level = STRIKE + lead
    for sec in range(now - 400, now + 1):
        a.index.on_tick(sec * 1000, level, received_ms=int(time.time() * 1000))
    a.index.clock_offset = 0.0
    a.vol.var = (2e-5 * level) ** 2
    a.vol._anchored = True
    a.vol.n = 1000
    a.vol._last_sec = now
    e._book_tickers = (c.ticker,)
    e.running = True
    if env != "paper":
        e.env = env
        e.feed = None
        e.ws = FakeWS()
        e._apply_balance({"balance": 10000, "balance_breakdown": [{"exchange_index": 2, "balance": "100.00"}]},
                         asked_at=time.time())
    return e, broker, c


WS_TRADE = {"type": "trade", "sid": 1, "seq": 2, "msg": {            # as received from Kalshi on 13 Oct 2026
    "trade_id": "0725ed59-8cb1-bc5c-bd20-ba6ac385a092", "market_ticker": "KXBTC15M-TEST",
    "yes_price_dollars": "0.2600", "no_price_dollars": "0.7400", "count_fp": "2.00", "taker_side": "yes",
    "taker_outcome_side": "yes", "taker_book_side": "bid", "is_block_trade": False,
    "ts": 1791922874, "ts_ms": 1791922874840}, "sending_ts_ms": 1791922874841}


def test_engine_reads_kalshi_trade_messages(tmp_path, monkeypatch):
    e, _, c = _engine(tmp_path, monkeypatch)
    e._on_trade(WS_TRADE)
    assert e.tape.trades(c.ticker) == 1
    assert e.tape.high(c.ticker, True, 1791922875.0, 1.0) == pytest.approx(0.26)
    assert e.tape.sold(c.ticker, False, 1791922875.0, 0.5) == 2 and e.tape.sold(c.ticker, True, 1791922875.0, 0.5) == 0
    block = {"msg": {**WS_TRADE["msg"], "is_block_trade": True}}
    cents = {"msg": {"market_ticker": c.ticker, "yes_price": 31, "count": 4, "taker_side": "no", "ts_ms": 1791922874900}}
    for bad in (block, {"msg": {}}, {}, {"msg": {**WS_TRADE["msg"], "taker_side": None, "taker_outcome_side": None}},
                {"msg": {**WS_TRADE["msg"], "count_fp": "abc"}}):
        e._on_trade(bad)
    assert e.tape.trades(c.ticker) == 1
    e._on_trade(cents)                                    # older message shape: whole cents and contracts
    assert e.tape.trades(c.ticker) == 2 and e.tape.sold(c.ticker, True, 1791922875.0, 0.5) == 4


def test_engine_waits_out_a_rush_of_selling_then_buys(tmp_path, monkeypatch):
    e, broker, c = _engine(tmp_path, monkeypatch)
    now = e.index.exchange_time()
    e._on_feed_trades(c.ticker, [{"ts": now - 0.2, "yes_price": 0.80, "count": 400.0, "taker_yes": False}])
    asyncio.run(e._step())
    assert broker.placed == []
    assert "rush of selling" in e.last_decision["note"] and "waiting" in e.last_decision["note"]
    e.tape.keep_only([])                                  # the selling is over ...
    asyncio.run(e._step())
    assert broker.placed == [] and "letting the index catch up" in e.last_decision["note"]
    t, why = e.strategy.waited[c.ticker]                  # ... and a second has passed since the warning
    e.strategy.waited[c.ticker] = (t - 1.5, why)
    asyncio.run(e._step())
    assert len(broker.placed) == 1 and broker.placed[0].outcome == "yes"


def test_engine_logs_the_first_wait_in_each_market_once(tmp_path, monkeypatch):
    e, broker, c = _engine(tmp_path, monkeypatch)
    assert not [l for l in e.logs if l["msg"].startswith("WAIT ")]
    e._on_feed_trades(c.ticker, [{"ts": e.index.exchange_time() - 0.2, "yes_price": 0.80, "count": 400.0, "taker_yes": False}])
    for _ in range(3):                                    # the check fires on every pass ...
        asyncio.run(e._step())
    lines = [l["msg"] for l in e.logs if l["msg"].startswith("WAIT ")]
    assert len(lines) == 1 and broker.placed == []        # ... and is written down once
    assert c.ticker in lines[0] and "rush of selling" in lines[0] and "400 contracts" in lines[0]
    e._wait_logged.add("KXBTC15M-OLD")
    e._book_tickers = ()                                  # markets roll over: only watched ones are remembered
    asyncio.run(e._sync_books())
    assert e._wait_logged == {c.ticker}


def test_a_calm_buy_logs_no_wait(tmp_path, monkeypatch):
    e, broker, c = _engine(tmp_path, monkeypatch)
    asyncio.run(e._step())
    assert len(broker.placed) == 1 and not [l for l in e.logs if l["msg"].startswith("WAIT ")]


def test_engine_waits_out_a_price_fall(tmp_path, monkeypatch):
    e, broker, c = _engine(tmp_path, monkeypatch)
    e._on_feed_trades(c.ticker, [{"ts": e.index.exchange_time() - 0.3, "yes_price": 0.97, "count": 1.0, "taker_yes": True}])
    asyncio.run(e._step())
    assert broker.placed == [] and "price just fell 17c" in e.last_decision["note"]


def test_engine_remembers_the_bid_it_saw_a_moment_ago(tmp_path, monkeypatch):
    """Pass 1: YES is bid 96 / offered 99 - too dear. Pass 2, a moment later: the
    bid is gone and YES is offered at 80. No trade has been reported, but the
    engine saw the 96c bid itself: it waits."""
    e, broker, c = _engine(tmp_path, monkeypatch)
    book = e.books[c.ticker]
    book.asks, book.bids = {9950: 500.0}, {9600: 500.0}
    asyncio.run(e._step())
    assert broker.placed == [] and "edge" in e.last_decision["note"]
    book.asks, book.bids = {8000: 500.0}, {7900: 500.0}
    asyncio.run(e._step())
    assert broker.placed == [] and "price just fell 16c" in e.last_decision["note"]
    assert e.tape.trades(c.ticker) == 0                   # nothing came from the trade stream


def test_engine_does_not_buy_without_a_confirmed_trade_feed(tmp_path, monkeypatch):
    e, broker, c = _engine(tmp_path, monkeypatch, env="demo")
    e.ws.confirmed = False
    asyncio.run(e._step())
    assert broker.placed == [] and "trade feed" in e.last_decision["note"]
    assert e.state()["tape"]["ok"] is False
    assert any("trade feed" in l["msg"] for l in e.logs)
    e.ws.confirmed = True
    asyncio.run(e._step())
    assert len(broker.placed) == 1
    # with both checks switched off the trade feed is not needed
    e2, broker2, _ = _engine(tmp_path / "b", monkeypatch, env="demo")
    e2.ws.confirmed = False
    e2.s.strategy.fall_guard = e2.s.strategy.sell_guard_x = 0.0
    asyncio.run(e2._step())
    assert len(broker2.placed) == 1


def test_engine_subscribes_to_trades_for_every_watched_market(tmp_path, monkeypatch):
    e, _, c = _engine(tmp_path, monkeypatch, env="demo")
    e.tape.add("KXBTC15M-OLD", 1.0, 0.5, 1, True)
    e._book_tickers = ()                                  # force a resubscribe
    asyncio.run(e._sync_books())
    assert e.ws.subs["trades"] == (["trade"], {"market_tickers": [c.ticker]})
    assert e.ws.subs["book"][1]["market_tickers"] == [c.ticker]
    assert "trades" in e.ws.unsubs
    assert e.tape.trades("KXBTC15M-OLD") == 0             # tapes of markets no longer watched are dropped


def test_websocket_client_knows_which_subscriptions_are_confirmed():
    ws = KalshiWS("wss://example.invalid", signer=None)
    ws._subs["trades"] = {"channels": ["trade"]}
    ws._name_by_cmd[7] = "trades"
    assert not ws.subscribed("trades")
    asyncio.run(ws._dispatch({"type": "subscribed", "id": 7, "msg": {"channel": "trade", "sid": 3}}))
    assert not ws.subscribed("trades")                    # acknowledged, but the socket is not marked connected
    ws.connected.set()
    assert ws.subscribed("trades") and not ws.subscribed("book")
    asyncio.run(ws.unsubscribe("trades"))
    assert not ws.subscribed("trades")


# ----------------------------------------------------------- saved settings
def test_settings_saved_before_the_checks_existed_get_them_switched_on(tmp_path, monkeypatch):
    v3 = {"version": 3, "env": "live", "live_unlocked": True, "run_wanted": True,
          "strategy": {"max_seconds_left": 55.0, "hourly_max_seconds_left": 60.0, "early_cushion_bp": 2.0,
                       "min_edge": 0.01, "z_min": 1.5, "trade_hourly": True},
          "risk": {"pct_per_market": 0.10, "pct_per_close": 0.15, "daily_loss_pct": 0.30, "max_drawdown_pct": 0.40}}
    s = Settings._from_dict(v3)
    assert (s.strategy.fall_guard, s.strategy.sell_guard_x) == (0.10, 1.0)
    assert (s.strategy.max_seconds_left, s.strategy.hourly_max_seconds_left, s.strategy.early_cushion_bp) == (55.0, 60.0, 2.0)
    assert (s.risk.pct_per_market, s.risk.pct_per_close, s.risk.daily_loss_pct, s.risk.max_drawdown_pct) == (0.10, 0.15, 0.30, 0.40)
    assert s.env == "live" and s.live_unlocked and s.run_wanted and s.version == config.SETTINGS_VERSION == 4
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    s.save()
    again = Settings.load()
    assert (again.strategy.fall_guard, again.strategy.sell_guard_x, again.version) == (0.10, 1.0, 4)
    # a choice made on this version (a check switched off) is kept
    off = Settings._from_dict({**v3, "version": 4, "strategy": {**v3["strategy"], "fall_guard": 0.0}})
    assert off.strategy.fall_guard == 0.0 and off.strategy.sell_guard_x == 1.0
