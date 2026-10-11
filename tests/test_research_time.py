"""Research timestamps must not depend on how pandas stores datetimes
(nanoseconds in pandas 2, the source's resolution - usually microseconds - in pandas 3)."""
from __future__ import annotations

import pandas as pd

from kalshi_bot.research.history import epoch_micros, epoch_seconds

CLOSE = "2026-10-11T15:45:00Z"
CLOSE_TS = 1791733500


def test_epoch_seconds_is_the_same_for_every_storage_resolution():
    base = pd.Series([pd.Timestamp(CLOSE), pd.Timestamp("2026-10-11T16:00:00Z")])
    for unit in ("s", "ms", "us", "ns"):
        got = epoch_seconds(base.dt.as_unit(unit))
        assert got.tolist() == [CLOSE_TS, CLOSE_TS + 900], unit
        assert str(got.dtype) == "int64"
    assert epoch_seconds(base).tolist() == [CLOSE_TS, CLOSE_TS + 900]          # whatever this pandas defaults to


def test_trade_times_keep_microseconds():
    raw = pd.Series(["2026-10-11T15:44:59.982675Z", "2026-10-11T15:44:59Z"], dtype="string")
    t = pd.to_datetime(raw, utc=True, format="ISO8601")
    assert epoch_micros(t).tolist() == [(CLOSE_TS - 1) * 1_000_000 + 982_675, (CLOSE_TS - 1) * 1_000_000]
    for unit in ("us", "ns"):
        assert epoch_micros(t.dt.as_unit(unit)).tolist() == epoch_micros(t).tolist()


def test_market_index_uses_real_epoch_seconds(tmp_path):
    """markets_with_trades() feeds open/close times to the whole backtest."""
    from kalshi_bot.research.market_data import markets_with_trades
    root = tmp_path / "kxbtc15m"
    day = "2026-10-11"
    (root / "trades" / day).mkdir(parents=True)
    (root / "event_series").mkdir()
    tk = "KXBTC15M-26OCT111145-45"
    mk = pd.DataFrame({"ticker": [tk], "open_time": [pd.Timestamp("2026-10-11T15:30:00Z")],
                       "close_time": [pd.Timestamp(CLOSE)], "floor_strike": [83000.0], "result": ["yes"]})
    mk.to_parquet(root / "markets.parquet", index=False)
    pd.DataFrame({"ts_us": [1], "yes_price": [0.5], "count": [1.0], "taker_yes": [True], "is_block": [False]}).to_parquet(
        root / "trades" / day / f"{tk}.parquet", index=False)
    pd.DataFrame({"ticker": [tk], "t_ms": [CLOSE_TS * 1000], "avg60": [83000.0]}).to_parquet(
        root / "event_series" / f"{day}.parquet", index=False)
    got = markets_with_trades(root)
    assert got.close_ts.tolist() == [CLOSE_TS] and got.open_ts.tolist() == [CLOSE_TS - 900]
