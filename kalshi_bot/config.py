"""
KalshiBTCPredictionModel15 – central configuration.
All tuneable parameters live here; secrets come from environment variables.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "storage"
MODEL_DIR = DATA_DIR / "models"
LOG_DIR = DATA_DIR / "logs"
TRADE_LOG_PATH = DATA_DIR / "trades.jsonl"

for d in (DATA_DIR, MODEL_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Kalshi API
# ---------------------------------------------------------------------------
KALSHI_API_KEY_ID: str = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH: str = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")

KALSHI_MODE: str = os.getenv("KALSHI_MODE", "demo")  # "demo" or "live"

KALSHI_BASE_URLS = {
    "demo": "https://demo-api.kalshi.co/trade-api/v2",
    "live": "https://api.elections.kalshi.com/trade-api/v2",
}

KALSHI_WS_URLS = {
    "demo": "wss://demo-api.kalshi.co/trade-api/ws/v2",
    "live": "wss://api.elections.kalshi.com/trade-api/ws/v2",
}

KALSHI_SERIES_TICKER = "KXBTC15M"


def get_kalshi_base_url() -> str:
    return KALSHI_BASE_URLS[KALSHI_MODE]


def get_kalshi_ws_url() -> str:
    return KALSHI_WS_URLS[KALSHI_MODE]


# ---------------------------------------------------------------------------
# Trading parameters
# ---------------------------------------------------------------------------
@dataclass
class TradingConfig:
    bet_fraction: float = 0.09               # bet 9% of balance each trade
    bankroll_floor: float = 5.0
    max_open_positions: int = 1

    # Timing
    entry_delay_seconds: int = 45
    min_time_remaining: int = 120


TRADING = TradingConfig()


# ---------------------------------------------------------------------------
# ML / Feature engineering
# ---------------------------------------------------------------------------
@dataclass
class MLConfig:
    model_type: str = "logistic"             # "logistic" or "xgboost"
    retrain_every_n_trades: int = 96         # ~1 day of 15-min windows
    retrain_every_hours: int = 24
    min_training_samples: int = 200
    lookback_candles: int = 100
    model_version_keep: int = 5
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 2,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.2,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    })


ML = MLConfig()


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
COINBASE_PRODUCT = "BTC-USD"
BINANCE_SYMBOL = "btcusdt"
FEAR_GREED_URL = "https://api.alternative.me/fng/"
DERIBIT_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
