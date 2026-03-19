# KalshiBTCPredictionModel15

A self-learning AI trading bot that predicts Bitcoin price movements in 15-minute windows on [Kalshi](https://kalshi.com) (KXBTC15M markets).

The bot uses an XGBoost machine learning model combined with a multi-signal alignment system to decide whether BTC will go up or down in each 15-minute period, then automatically places trades on Kalshi.

## How It Works

### Dual-Layer Decision System

**Layer 1 – Machine Learning (XGBoost)**
- Trained on 55 features: price returns, technical indicators (RSI, MACD, Bollinger Bands, EMA crossovers), volatility, orderbook data, sentiment, and temporal patterns.
- Bootstraps from historical Binance data, then continuously retrains on live outcomes.

**Layer 2 – Signal Alignment (5-Point Checklist)**

Before every trade, the bot checks five independent signals:

| Signal | Bullish | Bearish |
|--------|---------|---------|
| Price Action | Stalling after a drop | Stalling after a rise |
| Funding Rate | Shorts overcrowded (bottom 20%) | Longs overcrowded (top 20%) |
| Liquidations | Longs flushed → selling done | Shorts squeezed → fuel gone |
| Order Book Walls | Bid wall present, no ask wall | Ask wall present, no bid wall |
| Breaking News | Bullish headline (ETF, adoption) | Bearish headline (hack, war) |

The bot **only trades when 3+ of 5 signals agree** on direction. When both layers agree, confidence is boosted. When they disagree, the bot either skips or reduces position size.

### Data Sources

- **Coinbase** – BTC-USD spot price and 24h stats
- **Binance** – 1m/5m/15m candles, 24h ticker, order book depth (wall detection), funding rates, long/short ratio
- **Coinglass** – Aggregated liquidation data
- **Deribit** – Options put/call ratio
- **CryptoPanic** – Breaking crypto news headlines
- **Alternative.me** – Fear & Greed Index
- **Kalshi** – Market orderbook and pricing

### Risk Management

- Fixed bet size ($10 default on $100 account)
- Reduced bet size when balance drops below threshold
- Hard bankroll floor (stops trading below $20)
- Daily loss limit ($30)
- Consecutive loss circuit breaker (pauses after 4 losses in a row)
- Confidence threshold gate (minimum 60%)

### Self-Learning

- Logs every trade with full feature snapshots
- Retrains the XGBoost model every 50 trades or 24 hours
- Keeps versioned model snapshots; rolls back if new model degrades
- Tracks live performance metrics (win rate, Sharpe ratio, drawdown)

## Setup

### Prerequisites

- Python 3.10+
- A Kalshi account with API access

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example env file and fill in your Kalshi API credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
KALSHI_API_KEY_ID=your-api-key-id
KALSHI_PRIVATE_KEY_PATH=path/to/your/kalshi-private-key.key
KALSHI_MODE=demo
```

**Getting API keys:**
- **Demo:** Go to https://demo.kalshi.co → Account & Security → API Keys
- **Live:** Go to https://kalshi.com → Account & Security → API Keys

When generating a key, Kalshi gives you a private key file (`.key`). Save it somewhere safe and point `KALSHI_PRIVATE_KEY_PATH` to it.

### 3. (Optional) Discord Alerts

Add a Discord webhook URL to `.env` to receive trade alerts:

```env
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your/webhook
```

## Running the Bot

### Demo Mode (Paper Trading)

Start with demo mode to test without risking real money:

```bash
python -m kalshi_bot.main
```

On first run, the bot will automatically:
1. Collect historical BTC data from Binance
2. Train the initial XGBoost model
3. Start the 24/7 trading loop

### Live Mode (Real Money)

Once you are confident in the bot's performance on demo, switch to live:

```bash
python -m kalshi_bot.main --live
```

Make sure your `.env` has:
```env
KALSHI_MODE=live
```

And that your Kalshi account is funded and API keys are from the production site (https://kalshi.com).

### Bootstrap Only

To just train/retrain the model without starting the trading loop:

```bash
python -m kalshi_bot.main --bootstrap
```

### Stopping the Bot

Press `Ctrl+C` for a graceful shutdown. The bot will save the current model and close all connections.

## Project Structure

```
kalshi_bot/
├── __init__.py
├── __main__.py              # python -m kalshi_bot.main
├── main.py                  # Entry point and 24/7 scheduler
├── config.py                # All tuneable parameters
├── kalshi/
│   ├── auth.py              # RSA-PSS API authentication
│   ├── client.py            # Kalshi REST API client
│   └── market_discovery.py  # KXBTC15M market finder
├── data/
│   ├── data_aggregator.py   # Orchestrates all feeds
│   ├── coinbase_feed.py     # BTC spot price
│   ├── binance_feed.py      # Candles, ticker
│   ├── coinglass_feed.py    # Funding rates + liquidations
│   ├── exchange_orderbook.py # Bid/ask wall detection
│   ├── news_feed.py         # Breaking crypto news
│   ├── fear_greed.py        # Fear & Greed Index
│   ├── deribit_feed.py      # Options put/call ratio
│   └── kalshi_orderbook.py  # Kalshi market orderbook
├── ml/
│   ├── feature_engineer.py  # 55-feature pipeline
│   ├── predictor.py         # XGBoost classifier
│   ├── model_store.py       # Versioned model storage
│   └── historical_collector.py
├── strategy/
│   ├── signal_analyzer.py   # 5-point signal checklist
│   ├── trading_logic.py     # Dual-layer trade decisions
│   └── risk_manager.py      # Bankroll protection
├── learning/
│   ├── trade_logger.py      # Trade history (JSONL)
│   ├── retrainer.py         # Periodic model retraining
│   └── performance_analyzer.py
├── monitoring/
│   ├── dashboard.py         # Rich console dashboard
│   └── alerts.py            # Discord webhook alerts
└── storage/                 # Auto-created at runtime
    ├── models/              # Saved model versions
    ├── logs/                # Bot logs
    └── trades.jsonl         # Trade history
```

## Configuration

All parameters are in `kalshi_bot/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bet_size_dollars` | 10.0 | Standard bet size |
| `reduced_bet_size_dollars` | 5.0 | Bet size when balance is low |
| `bankroll_floor` | 20.0 | Stop trading below this balance |
| `daily_loss_limit` | 30.0 | Max daily loss before pausing |
| `confidence_threshold` | 0.60 | Minimum model confidence to trade |
| `consecutive_loss_pause` | 4 | Pause after N consecutive losses |
| `trade_entry_minutes_before_close` | 2 | Enter position ~2 min before market close |

## Disclaimer

This software is provided for educational and research purposes. Trading on prediction markets involves real financial risk. Past performance does not guarantee future results. Only trade with money you can afford to lose.
