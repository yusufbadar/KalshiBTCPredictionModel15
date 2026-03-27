"""
KalshiBTCPredictionModel15 -- rule-based 5m candle trading bot.

Usage:
    python -m kalshi_bot.main              # demo mode (default)
    python -m kalshi_bot.main --live       # production mode

The bot runs a continuous loop:
  1. Find current KXBTC15M market
  2. Wait 10 minutes for two 5m candles to form
  3. Apply rules: 2x green+above EMA -> YES, 2x red+below EMA -> NO, mixed -> bigger body
  4. Buy the chosen side at the Kalshi ask
  5. Hold to settlement (~5 minutes)
  6. Record outcome, display on live Rich dashboard
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
import traceback

from loguru import logger

from kalshi_bot.config import LOG_DIR

logger.remove()
logger.add(sys.stderr, level="WARNING", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
logger.add(LOG_DIR / "bot_{time}.log", rotation="50 MB", retention="7 days", level="DEBUG")


def parse_args():
    p = argparse.ArgumentParser(description="KalshiBTCPredictionModel15 Candle Bot")
    p.add_argument("--live", action="store_true", help="Use production Kalshi API (real money)")
    return p.parse_args()


async def run_bot(live: bool = False):
    import kalshi_bot.config as cfg
    from rich.live import Live
    from rich.console import Console

    if live:
        cfg.KALSHI_MODE = "live"
    else:
        cfg.KALSHI_MODE = "demo"

    from kalshi_bot.kalshi.auth import KalshiAuth
    from kalshi_bot.kalshi.client import KalshiClient
    from kalshi_bot.kalshi.market_discovery import MarketDiscovery
    from kalshi_bot.data.binance_feed import BinanceFeed
    from kalshi_bot.data.kalshi_orderbook import KalshiOrderbookFeed
    from kalshi_bot.strategy.risk_manager import RiskManager
    from kalshi_bot.strategy.trading_logic import TradingEngine
    from kalshi_bot.learning.trade_logger import TradeLogger
    from kalshi_bot.learning.performance_analyzer import PerformanceAnalyzer
    from kalshi_bot.monitoring.dashboard import build_dashboard, print_startup_banner
    from kalshi_bot.monitoring.alerts import alert_trade, alert_settlement, alert_error

    console = Console()

    if not cfg.KALSHI_API_KEY_ID or not cfg.KALSHI_PRIVATE_KEY_PATH:
        console.print("[red]Missing KALSHI_API_KEY_ID or KALSHI_PRIVATE_KEY_PATH in .env[/red]")
        return

    auth = KalshiAuth(cfg.KALSHI_API_KEY_ID, cfg.KALSHI_PRIVATE_KEY_PATH)
    client = KalshiClient(auth)
    discovery = MarketDiscovery(client)

    binance = BinanceFeed()
    kalshi_ob = KalshiOrderbookFeed(client)

    trade_logger = TradeLogger()
    analyzer = PerformanceAnalyzer(trade_logger)
    risk_mgr = RiskManager()

    engine = TradingEngine(
        client, discovery, binance, kalshi_ob,
        risk_mgr, trade_logger, analyzer,
    )

    # Connect Binance feed
    console.print("[yellow]Connecting to Binance...[/yellow]")
    if not await binance.connect():
        console.print("[red]Failed to connect to Binance[/red]")
        return

    balance = client.get_balance_dollars()
    print_startup_banner(balance)

    # Signal handling
    shutdown = False
    def handle_signal(*_):
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Main loop
    last_traded_market = None
    backoff = 5

    with Live(console=console, refresh_per_second=1, screen=False) as live_display:

        while not shutdown:
            try:
                balance = client.get_balance_dollars()
                market = discovery.get_current_market()
                market_ticker = market.ticker if market else ""
                seconds_to_close = market.seconds_until_close if market else 0

                position_info = None
                if engine.current_position:
                    pos = engine.current_position
                    position_info = {
                        "direction": pos["direction"],
                        "side": pos["side"],
                        "entry_price": pos["entry_price"],
                        "count": pos["count"],
                        "bet_dollars": pos["bet_dollars"],
                        "hold_seconds": time.time() - pos["entry_time"],
                    }

                dashboard = build_dashboard(
                    balance=balance,
                    analyzer=analyzer,
                    trade_logger=trade_logger,
                    cycle_result=engine.last_cycle,
                    position_info=position_info,
                    activity_log=engine.activity_log,
                    market_ticker=market_ticker,
                    seconds_to_close=seconds_to_close,
                )
                live_display.update(dashboard)

                if market is None:
                    engine._log_activity("Scanning for markets...")
                    await asyncio.sleep(8)
                    continue

                # Resolve previous market before entering a new one
                if last_traded_market and market.ticker != last_traded_market:
                    for _ in range(6):
                        settlement = await engine.check_settlement(last_traded_market)
                        if settlement:
                            if "won" in settlement:
                                alert_settlement(
                                    last_traded_market, settlement["won"],
                                    settlement["pnl"], analyzer.win_rate_all,
                                )
                            last_traded_market = None
                            break
                        await asyncio.sleep(5)
                    else:
                        engine._log_activity(
                            f"Previous market {last_traded_market} not settled -- marking stale"
                        )
                        last_traded_market = None
                        engine.current_position = None

                # Enter new market
                if market.ticker != last_traded_market:
                    if market.seconds_until_close >= 300:
                        cycle = await engine.run_cycle()

                        if cycle.get("action") == "traded":
                            last_traded_market = market.ticker
                            alert_trade(
                                cycle.get("direction", "?"),
                                cycle.get("market", "?"),
                                cycle.get("bet_dollars", 0),
                                cycle.get("price_cents", 0),
                            )

                        balance = client.get_balance_dollars()
                        dashboard = build_dashboard(
                            balance=balance,
                            analyzer=analyzer,
                            trade_logger=trade_logger,
                            cycle_result=engine.last_cycle,
                            position_info=position_info,
                            activity_log=engine.activity_log,
                            market_ticker=market.ticker,
                            seconds_to_close=market.seconds_until_close,
                        )
                        live_display.update(dashboard)

                # Check settlement for current market
                if last_traded_market and market.ticker == last_traded_market:
                    settlement = await engine.check_settlement(last_traded_market)
                    if settlement:
                        if "won" in settlement:
                            alert_settlement(
                                last_traded_market, settlement["won"],
                                settlement["pnl"], analyzer.win_rate_all,
                            )
                        last_traded_market = None

                if engine.current_position:
                    await asyncio.sleep(10)
                else:
                    await asyncio.sleep(8)

                backoff = 5

            except Exception as e:
                logger.error("Loop error: {}\n{}", e, traceback.format_exc())
                engine._log_activity(f"ERROR: {e}")
                alert_error(str(e))
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    # Cleanup
    console.print("[yellow]Shutting down...[/yellow]")
    await binance.close()
    client.close()
    console.print("[green]Goodbye.[/green]")


def main():
    args = parse_args()
    asyncio.run(run_bot(live=args.live))


if __name__ == "__main__":
    main()
