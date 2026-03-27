"""
Data aggregator – collects data from all feeds into a single snapshot
that is consumed by the feature engineering pipeline.

Includes the ascetic0x signal feeds:
  - Funding rates + liquidation data (Coinglass / Binance Futures)
  - Exchange order book wall detection (Binance spot depth)
  - Breaking crypto news (CryptoPanic)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from kalshi_bot.data.coinbase_feed import CoinbaseFeed
from kalshi_bot.data.binance_feed import BinanceFeed
from kalshi_bot.data.fear_greed import FearGreedFeed
from kalshi_bot.data.deribit_feed import DeribitFeed
from kalshi_bot.data.kalshi_orderbook import KalshiOrderbookFeed
from kalshi_bot.data.coinglass_feed import FundingLiquidationFeed
from kalshi_bot.data.exchange_orderbook import ExchangeOrderbookFeed
from kalshi_bot.data.news_feed import NewsFeed


class DataSnapshot:
    """Immutable snapshot of all data at a point in time."""

    def __init__(
        self,
        ts: datetime,
        btc_price: Optional[float],
        candles_1m: list[dict],
        candles_5m: list[dict],
        candles_15m: list[dict],
        stats_24h: Optional[dict],
        binance_ticker: Optional[dict],
        fear_greed: Optional[dict],
        deribit_pcr: Optional[dict],
        kalshi_ob: Optional[dict],
        kalshi_mid_price: Optional[float],
        # ── ascetic0x signals ──
        funding_liq: Optional[dict] = None,
        exchange_ob: Optional[dict] = None,
        news: Optional[dict] = None,
    ):
        self.ts = ts
        self.btc_price = btc_price
        self.candles_1m = candles_1m
        self.candles_5m = candles_5m
        self.candles_15m = candles_15m
        self.stats_24h = stats_24h
        self.binance_ticker = binance_ticker
        self.fear_greed = fear_greed
        self.deribit_pcr = deribit_pcr
        self.kalshi_ob = kalshi_ob
        self.kalshi_mid_price = kalshi_mid_price
        self.funding_liq = funding_liq
        self.exchange_ob = exchange_ob
        self.news = news

    @property
    def is_valid(self) -> bool:
        return self.btc_price is not None or len(self.candles_1m) >= 1


class DataAggregator:
    """Orchestrates all feeds and produces a ``DataSnapshot``."""

    def __init__(
        self,
        coinbase: CoinbaseFeed,
        binance: BinanceFeed,
        fear_greed: FearGreedFeed,
        deribit: DeribitFeed,
        kalshi_ob: KalshiOrderbookFeed,
        funding_liq: Optional[FundingLiquidationFeed] = None,
        exchange_ob: Optional[ExchangeOrderbookFeed] = None,
        news: Optional[NewsFeed] = None,
    ):
        self.coinbase = coinbase
        self.binance = binance
        self.fear_greed = fear_greed
        self.deribit = deribit
        self.kalshi_ob = kalshi_ob
        self.funding_liq = funding_liq
        self.exchange_ob = exchange_ob
        self.news = news

    async def connect_all(self) -> bool:
        tasks = [
            self.coinbase.connect(),
            self.binance.connect(),
            self.fear_greed.connect(),
        ]
        if self.funding_liq:
            tasks.append(self.funding_liq.connect())
        if self.exchange_ob:
            tasks.append(self.exchange_ob.connect())
        if self.news:
            tasks.append(self.news.connect())

        results = await asyncio.gather(*tasks, return_exceptions=True)
        ok = sum(1 for r in results if r is True)
        logger.info("Data feeds connected: {}/{}", ok, len(results))
        return ok >= 2  # need at least coinbase + binance

    async def close_all(self):
        tasks = [
            self.coinbase.close(),
            self.binance.close(),
            self.fear_greed.close(),
        ]
        if self.funding_liq:
            tasks.append(self.funding_liq.close())
        if self.exchange_ob:
            tasks.append(self.exchange_ob.close())
        if self.news:
            tasks.append(self.news.close())
        await asyncio.gather(*tasks, return_exceptions=True)

    async def snapshot(self, market_ticker: Optional[str] = None) -> DataSnapshot:
        """Collect data from all sources in parallel."""
        loop = asyncio.get_event_loop()

        price_task = self.coinbase.get_price()
        candles_1m_task = self.binance.get_klines("1m", 100)
        candles_5m_task = self.binance.get_klines("5m", 50)
        candles_15m_task = self.binance.get_klines("15m", 30)
        stats_task = self.coinbase.get_24h_stats()
        ticker_task = self.binance.get_24h_ticker()
        fg_task = self.fear_greed.fetch()
        deribit_task = loop.run_in_executor(None, self.deribit.fetch_pcr)

        core_tasks = [
            price_task, candles_1m_task, candles_5m_task, candles_15m_task,
            stats_task, ticker_task, fg_task, deribit_task,
        ]

        # Ascetic0x signal tasks
        extra_tasks = []
        extra_keys = []
        if self.funding_liq:
            extra_tasks.append(self.funding_liq.fetch())
            extra_keys.append("funding_liq")
        if self.exchange_ob:
            extra_tasks.append(self.exchange_ob.fetch())
            extra_keys.append("exchange_ob")
        if self.news:
            extra_tasks.append(self.news.fetch())
            extra_keys.append("news")

        all_results = await asyncio.gather(
            *core_tasks, *extra_tasks, return_exceptions=True,
        )

        def safe(idx):
            v = all_results[idx]
            return None if isinstance(v, Exception) else v

        extra_data = {}
        for i, key in enumerate(extra_keys):
            extra_data[key] = safe(len(core_tasks) + i)

        kalshi_ob_data = None
        kalshi_mid = None
        if market_ticker:
            try:
                kalshi_ob_data = self.kalshi_ob.fetch(market_ticker)
                if kalshi_ob_data:
                    kalshi_mid = kalshi_ob_data.get("mid_price")
            except Exception as e:
                logger.warning("Kalshi OB snapshot failed: {}", e)

        btc_price_val = safe(0)
        btc_price_float = float(btc_price_val) if btc_price_val is not None else None

        return DataSnapshot(
            ts=datetime.now(timezone.utc),
            btc_price=btc_price_float,
            candles_1m=safe(1) or [],
            candles_5m=safe(2) or [],
            candles_15m=safe(3) or [],
            stats_24h=safe(4),
            binance_ticker=safe(5),
            fear_greed=safe(6),
            deribit_pcr=safe(7),
            kalshi_ob=kalshi_ob_data,
            kalshi_mid_price=kalshi_mid,
            funding_liq=extra_data.get("funding_liq"),
            exchange_ob=extra_data.get("exchange_ob"),
            news=extra_data.get("news"),
        )
