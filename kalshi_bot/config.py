"""
Central configuration.

Secrets (API key id + private key) come from ``.env`` / environment or from
the dashboard's Settings panel; everything else lives in
``storage/settings.json`` so the UI can change it at runtime.

Environments
------------
demo   real orders on Kalshi's demo exchange (fake money)       <- default
paper  production market data, simulated fills, no orders sent
live   real orders on the production exchange (real money);
       locked unless explicitly unlocked in Settings
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env")

STORAGE_DIR = Path(os.getenv("KALSHI_STORAGE_DIR", PROJECT_DIR / "storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_PATH = STORAGE_DIR / "settings.json"
DB_PATH = STORAGE_DIR / "bot.sqlite3"
LOG_DIR = STORAGE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SERIES_TICKER = "KXBTC15M"
INDEX_ID = "BRTI"


@dataclass(frozen=True)
class AssetSpec:
    """One coin: its CF Benchmarks real-time index on Kalshi's feed, the
    15-minute up/down series that settles on it and (optionally) the hourly
    above/below strike ladder. ``decimals`` is the precision Kalshi rounds
    the 60-second settlement average to."""
    symbol: str
    index_id: str
    series15: str
    hourly: str | None
    decimals: int


ASSETS: dict[str, AssetSpec] = {
    "BTC": AssetSpec("BTC", "BRTI", "KXBTC15M", "KXBTCD", 2),
    "ETH": AssetSpec("ETH", "ETHUSD_RTI", "KXETH15M", None, 2),
    "SOL": AssetSpec("SOL", "SOLUSD_RTI", "KXSOL15M", None, 4),
    "XRP": AssetSpec("XRP", "XRPUSD_RTI", "KXXRP15M", None, 4),
    "DOGE": AssetSpec("DOGE", "DOGEUSD_RTI", "KXDOGE15M", None, 6),
}


def parse_assets(text: str) -> list[AssetSpec]:
    """'BTC,ETH' -> specs, in order, unknown names dropped; never empty."""
    out, seen = [], set()
    for tok in str(text or "").replace(";", ",").split(","):
        k = tok.strip().upper()
        if k in ASSETS and k not in seen:
            seen.add(k)
            out.append(ASSETS[k])
    return out or [ASSETS["BTC"]]

ENDPOINTS = {
    "demo": {
        "rest": "https://external-api.demo.kalshi.co/trade-api/v2",
        "ws": "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2",
    },
    "live": {
        "rest": "https://external-api.kalshi.com/trade-api/v2",
        "ws": "wss://external-api-ws.kalshi.com/trade-api/ws/v2",
    },
}
# Public production market data (no auth) used by paper mode and research.
PUBLIC_REST = ENDPOINTS["live"]["rest"]
WS_PATH = "/trade-api/ws/v2"


SETTINGS_VERSION = 3
# strategy fields whose validated values changed in a settings version: settings
# saved by an older version adopt the new value of exactly these fields
_STRATEGY_RESETS = {2: ("min_edge", "sigma_time_k"),
                    3: ("max_seconds_left",)}          # 15-minute markets: final 60 s -> final 55 s


@dataclass
class StrategyParams:
    """Settlement lock-in strategy (strategy/lockin.py). Defaults mirror
    research/lockin.py HOUSE_15M / HOUSE_HOURLY (see RESEARCH.md);
    tests/test_parity.py fails if the two drift apart."""
    enabled: bool = True
    # when to act: only inside the settlement averaging minute. 15-minute
    # markets skip its first seconds, where entries lose most often
    max_seconds_left: float = 55.0       # 15-minute up/down markets
    hourly_max_seconds_left: float = 60.0   # hourly strike ladder
    min_seconds_left: float = 1.0
    # 15-minute markets, while more than `early_seconds` remain: the expected
    # settlement must be at least this far from the strike (basis points of
    # price; 2 bp is about $17 on BTC) - a small lead that early is one jump
    # away from flipping. 0 switches it off.
    early_seconds: float = 45.0
    early_cushion_bp: float = 2.0
    # model
    vol_halflife_s: float = 300.0        # EWMA half-life of the BRTI variance
    vol_floor_rel: float = 6e-6          # sigma floor as a fraction of price per sqrt(s) (BTC: ~$0.50)
    sigma_scale: float = 1.0             # chosen by out-of-sample calibration (log-loss)
    sigma_time_k: float = 0.25           # extra caution early in the window: sigma x (1 + k (s_left-30)/30) beyond 30 s
    tail_nu: float = 5.0                 # Student-t degrees of freedom
    # entry filters
    p_min: float = 0.90                  # model probability of the side bought
    z_min: float = 1.5                   # expected settlement distance from strike (s.d.)
    min_edge: float = 0.01               # model prob - price - fee, per contract
    max_price: float = 0.99
    cooldown_s: float = 0.5
    # sizing
    max_contracts_per_order: float = 50.0
    edge_size_max: float = 3.0           # scale orders up to 3x on edges >= 3x min_edge
    max_position_per_market: float = 300.0
    # also trade the hourly BTC above/below strike ladder (same settlement index)
    trade_hourly: bool = True
    hourly_band: float = 400.0           # $ distance from BRTI for strikes to watch
    # coins to trade (15-minute up/down markets): any of BTC, ETH, SOL, XRP, DOGE
    assets: str = "BTC"


@dataclass
class RiskParams:
    """Each limit is the tighter of its dollar amount and its fraction of the
    account (cash + money in open positions); 0 switches that half off. The
    fractions make position size follow the account up and down."""
    max_daily_loss: float = 50.0         # dollars; bot halts for the UTC day
    daily_loss_pct: float = 0.15         # ... or this fraction of the day's starting equity
    max_notional_per_market: float = 100.0
    pct_per_market: float = 0.06         # at most this fraction of the account in one contract
    max_notional_per_close: float = 150.0   # all contracts settling at the same moment
    pct_per_close: float = 0.10
    max_drawdown_pct: float = 0.30       # equity this far below its peak -> kill switch
    stale_index_s: float = 3.0           # no index tick for this long -> flatten quotes
    stale_book_s: float = 5.0
    min_balance: float = 5.0


@dataclass
class Settings:
    env: str = "demo"                    # demo | paper | live
    live_unlocked: bool = False
    api_key_id: str = ""                 # demo exchange key
    private_key_path: str = ""
    live_api_key_id: str = ""            # production key (real money)
    live_private_key_path: str = ""
    dd_reset_ts: dict = field(default_factory=dict)   # env -> time the drawdown stop was last released
    paper_start: float = 100.0           # simulated account size in paper mode
    kill_state: dict = field(default_factory=dict)    # env -> reason while a manual kill switch is engaged
    run_wanted: bool = False             # the bot was running when the program last stopped (see --autostart)
    version: int = SETTINGS_VERSION      # bumped when shipped defaults change (see _from_dict)
    strategy: StrategyParams = field(default_factory=StrategyParams)
    risk: RiskParams = field(default_factory=RiskParams)
    shard_topup: bool = True             # move collateral to the crypto shard when needed

    # -- persistence -------------------------------------------------------
    @classmethod
    def load(cls) -> "Settings":
        s = cls()
        saved = SETTINGS_PATH.exists()
        if saved:
            try:
                raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                s = cls._from_dict(raw)
            except Exception:
                saved = False
        # .env supplies credentials when the dashboard has none saved, and the
        # starting environment on first run; after that the dashboard is in charge
        if not s.api_key_id:
            s.api_key_id = os.getenv("KALSHI_API_KEY_ID", "")
        if not s.private_key_path:
            s.private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
        if not saved and os.getenv("KALSHI_ENV") in ("demo", "paper"):
            s.env = os.environ["KALSHI_ENV"]
        if s.env == "live" and not s.live_unlocked:
            s.env = "demo"
        if not s.live_api_key_id:
            s.live_api_key_id = os.getenv("KALSHI_LIVE_API_KEY_ID", "")
        if not s.live_private_key_path:
            s.live_private_key_path = os.getenv("KALSHI_LIVE_PRIVATE_KEY_PATH", "")
        for attr in ("private_key_path", "live_private_key_path"):
            v = getattr(s, attr)
            if v and not Path(v).expanduser().is_absolute():
                setattr(s, attr, str((PROJECT_DIR / v).resolve()))
        return s

    @classmethod
    def _from_dict(cls, raw: dict) -> "Settings":
        sp = StrategyParams(**{k: v for k, v in raw.get("strategy", {}).items()
                               if k in StrategyParams.__dataclass_fields__})
        rp = RiskParams(**{k: v for k, v in raw.get("risk", {}).items()
                           if k in RiskParams.__dataclass_fields__})
        top = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__
               and k not in ("strategy", "risk")}
        saved = int(raw.get("version", 1) or 1)
        fresh = StrategyParams()
        for ver, fields in _STRATEGY_RESETS.items():
            if saved < ver:
                # saved by an older version: adopt the re-validated value of the
                # fields that version changed (new fields arrive as defaults,
                # everything else the user set is kept)
                for k in fields:
                    setattr(sp, k, getattr(fresh, k))
        top["version"] = SETTINGS_VERSION
        return cls(strategy=sp, risk=rp, **top)

    def save(self) -> None:
        data = asdict(self)
        tmp = SETTINGS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(SETTINGS_PATH)

    # -- helpers -----------------------------------------------------------
    def has_creds(self, exchange: str) -> bool:
        kid, path = self.creds(exchange)
        return bool(kid and path and Path(path).expanduser().exists())

    def creds(self, exchange: str | None = None) -> tuple[str, str]:
        """(key id, private key path) for 'demo' or 'live'. Each exchange has
        its own key pair - a demo key is rejected by production and vice versa."""
        if (exchange or self.trading_env) == "live":
            return self.live_api_key_id, self.live_private_key_path
        return self.api_key_id, self.private_key_path

    @property
    def trading_env(self) -> str:
        """Exchange that receives auth (and, outside paper mode, orders).
        Paper never sends orders; it only needs a login for the index feed, so
        it uses the demo key when there is one and the live key otherwise."""
        if self.env == "live":
            return "live"
        if self.env == "paper" and not self.has_creds("demo") and self.has_creds("live"):
            return "live"
        return "demo"

    def rest_url(self) -> str:
        return ENDPOINTS[self.trading_env]["rest"]

    def ws_url(self) -> str:
        return ENDPOINTS[self.trading_env]["ws"]


_lock = threading.Lock()
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    with _lock:
        if _settings is None:
            _settings = Settings.load()
        return _settings


def set_settings(s: Settings) -> None:
    global _settings
    with _lock:
        _settings = s
        s.save()
