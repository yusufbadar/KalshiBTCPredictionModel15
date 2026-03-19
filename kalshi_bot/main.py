"""
KalshiBTCPredictionModel15 – entry point.

Usage:
    python -m kalshi_bot.main              # demo mode (default)
    python -m kalshi_bot.main --live       # production mode
    python -m kalshi_bot.main --bootstrap  # collect historical data & train, then exit

The bot runs 24/7 in a loop:
  1. On first run: bootstrap an XGBoost model from historical Binance 15m data.
  2. Every ~15 minutes: discover market -> predict -> risk-check -> trade -> settle.
  3. Periodically retrain the model with accumulated live outcomes.
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
import traceback

from loguru import logger

# ── configure logging ──────────────────────────────────────────────
from kalshi_bot.config import LOG_DIR

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
logger.add(LOG_DIR / "bot_{time}.log", rotation="50 MB", retention="7 days", level="DEBUG")


def parse_args():
    p = argparse.ArgumentParser(description="KalshiBTCPredictionModel15")
    p.add_argument("--live", action="store_true", help="Use production Kalshi API (real money)")
    p.add_argument("--bootstrap", action="store_true", help="Only bootstrap model, then exit")
    return p.parse_args()


async def bootstrap_model(predictor, model_store, retrainer):
    """Collect historical data and train the initial model."""
    from kalshi_bot.ml.historical_collector import collect_historical_features

    logger.info("=== BOOTSTRAPPING MODEL FROM HISTORICAL DATA ===")
    X, y, names = await collect_historical_features(n_candles_15m=1000)
    if len(y) == 0:
        logger.error("No historical data collected – cannot bootstrap")
        return False

    retrainer.set_historical_data(X, y)
    metrics = predictor.train(X, y, feature_names=names)
    if "error" in metrics:
        logger.error("Bootstrap training failed: {}", metrics)
        return False

    model_store.save_version(predictor, metrics)
    logger.info("Bootstrap complete: {} samples, val_acc={:.2%}", len(y), metrics["val_acc"])
    return True


async def run_bot(live: bool = False):
    """Main bot loop."""
    import kalshi_bot.config as cfg

    if live:
        cfg.KALSHI_MODE = "live"
        logger.warning(">>> LIVE MODE – REAL MONEY <<<")
    else:
        cfg.KALSHI_MODE = "demo"
        logger.info("Demo mode")

    # ── build all components ─────────────────────────────────────
    from kalshi_bot.kalshi.auth import KalshiAuth
    from kalshi_bot.kalshi.client import KalshiClient
    from kalshi_bot.kalshi.market_discovery import MarketDiscovery
    from kalshi_bot.data.coinbase_feed import CoinbaseFeed
    from kalshi_bot.data.binance_feed import BinanceFeed
    from kalshi_bot.data.fear_greed import FearGreedFeed
    from kalshi_bot.data.deribit_feed import DeribitFeed
    from kalshi_bot.data.kalshi_orderbook import KalshiOrderbookFeed
    from kalshi_bot.data.data_aggregator import DataAggregator
    from kalshi_bot.ml.feature_engineer import FeatureEngineer
    from kalshi_bot.ml.predictor import Predictor
    from kalshi_bot.ml.model_store import ModelStore
    from kalshi_bot.strategy.risk_manager import RiskManager
    from kalshi_bot.strategy.trading_logic import TradingEngine
    from kalshi_bot.learning.trade_logger import TradeLogger
    from kalshi_bot.learning.retrainer import Retrainer
    from kalshi_bot.learning.performance_analyzer import PerformanceAnalyzer
    from kalshi_bot.monitoring.dashboard import print_cycle_summary, print_startup_banner
    from kalshi_bot.monitoring.alerts import alert_trade, alert_settlement, alert_error

    if not cfg.KALSHI_API_KEY_ID or not cfg.KALSHI_PRIVATE_KEY_PATH:
        logger.error(
            "Missing KALSHI_API_KEY_ID or KALSHI_PRIVATE_KEY_PATH in .env – "
            "see .env.example for the template."
        )
        return

    auth = KalshiAuth(cfg.KALSHI_API_KEY_ID, cfg.KALSHI_PRIVATE_KEY_PATH)
    client = KalshiClient(auth)
    discovery = MarketDiscovery(client)

    coinbase = CoinbaseFeed()
    binance = BinanceFeed()
    fear_greed = FearGreedFeed()
    deribit = DeribitFeed()
    kalshi_ob = KalshiOrderbookFeed(client)

    # Ascetic0x signal feeds
    from kalshi_bot.data.coinglass_feed import FundingLiquidationFeed
    from kalshi_bot.data.exchange_orderbook import ExchangeOrderbookFeed
    from kalshi_bot.data.news_feed import NewsFeed
    from kalshi_bot.strategy.signal_analyzer import SignalAnalyzer

    funding_liq = FundingLiquidationFeed()
    exchange_ob = ExchangeOrderbookFeed()
    news_feed = NewsFeed()

    aggregator = DataAggregator(
        coinbase, binance, fear_greed, deribit, kalshi_ob,
        funding_liq=funding_liq,
        exchange_ob=exchange_ob,
        news=news_feed,
    )

    feature_eng = FeatureEngineer()
    predictor = Predictor()
    model_store = ModelStore()
    trade_logger = TradeLogger()
    analyzer = PerformanceAnalyzer(trade_logger, predictor)
    risk_mgr = RiskManager(analyzer)
    retrainer = Retrainer(predictor, model_store, trade_logger)
    signal_analyzer = SignalAnalyzer(min_alignment=3)

    engine = TradingEngine(
        client, discovery, aggregator, feature_eng,
        predictor, risk_mgr, trade_logger, analyzer,
        signal_analyzer=signal_analyzer,
    )

    # ── connect data feeds ───────────────────────────────────────
    if not await aggregator.connect_all():
        logger.error("Failed to connect data feeds")
        return

    # ── load or bootstrap model ──────────────────────────────────
    if not model_store.load_latest(predictor):
        logger.info("No saved model found – bootstrapping…")
        ok = await bootstrap_model(predictor, model_store, retrainer)
        if not ok:
            logger.error("Bootstrap failed – exiting")
            return

    # ── startup banner ───────────────────────────────────────────
    balance = client.get_balance_dollars()
    print_startup_banner(balance, predictor.train_accuracy, predictor.n_train_samples)

    # ── main loop ────────────────────────────────────────────────
    shutdown = False

    def handle_signal(*_):
        nonlocal shutdown
        shutdown = True
        logger.info("Shutdown requested…")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    last_traded_market = None
    backoff = 10  # seconds between error retries

    while not shutdown:
        try:
            # Run one trade cycle
            cycle = await engine.run_cycle()

            # Dashboard
            balance = client.get_balance_dollars()
            print_cycle_summary(cycle, balance, analyzer, trade_logger)

            # Alert on trade
            if cycle.get("action") == "traded":
                alert_trade(
                    cycle.get("direction", "?"), cycle.get("market", "?"),
                    cfg.TRADING.bet_size_dollars, cycle.get("price_cents", 0),
                    alignment_score=cycle.get("alignment_score"),
                )
                last_traded_market = cycle.get("market")

            # Check settlement of previous market
            if last_traded_market:
                for _ in range(60):  # poll for up to 5 min
                    await asyncio.sleep(5)
                    settlement = await engine.check_settlement(last_traded_market)
                    if settlement:
                        if "won" in settlement:
                            alert_settlement(
                                last_traded_market, settlement["won"],
                                settlement["pnl"], analyzer.win_rate_all,
                            )
                        last_traded_market = None
                        break

            # Retrain check
            if retrainer.should_retrain():
                logger.info("=== RETRAINING MODEL ===")
                metrics = retrainer.retrain()
                logger.info("Retrain result: {}", metrics)

            # Wait for next cycle (~15 min minus processing time)
            market = discovery.get_current_market()
            if market and market.seconds_until_close > 0:
                sleep_time = max(10, market.seconds_until_close - cfg.TRADING.trade_entry_minutes_before_close * 60)
                sleep_time = min(sleep_time, 900)
            else:
                sleep_time = 30

            logger.info("Sleeping {:.0f}s until next cycle…", sleep_time)
            for _ in range(int(sleep_time)):
                if shutdown:
                    break
                await asyncio.sleep(1)

            backoff = 10  # reset on success

        except Exception as e:
            logger.error("Cycle error: {}\n{}", e, traceback.format_exc())
            alert_error(str(e))
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 300)

    # ── cleanup ──────────────────────────────────────────────────
    logger.info("Shutting down…")
    await aggregator.close_all()
    client.close()
    predictor.save("latest")
    logger.info("Goodbye.")


def main():
    args = parse_args()

    if args.bootstrap:
        from kalshi_bot.ml.predictor import Predictor
        from kalshi_bot.ml.model_store import ModelStore
        from kalshi_bot.learning.retrainer import Retrainer
        from kalshi_bot.learning.trade_logger import TradeLogger

        predictor = Predictor()
        model_store = ModelStore()
        trade_logger = TradeLogger()
        from kalshi_bot.learning.performance_analyzer import PerformanceAnalyzer
        analyzer = PerformanceAnalyzer(trade_logger, predictor)
        retrainer = Retrainer(predictor, model_store, trade_logger)
        asyncio.run(bootstrap_model(predictor, model_store, retrainer))
        return

    asyncio.run(run_bot(live=args.live))


if __name__ == "__main__":
    main()
