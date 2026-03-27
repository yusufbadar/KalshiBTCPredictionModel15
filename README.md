# BTC-15M Candle Bot

A rule-based trading bot for [Kalshi](https://kalshi.com) KXBTC15M markets that predicts whether Bitcoin will close above or below the opening price in each 15-minute window.

The bot watches two 5-minute candles inside each window, compares them against the 20-period EMA, and places a single trade — then holds to settlement.

## Strategy

The bot waits until 10 minutes into each 15-minute Kalshi window so that two complete 5-minute candles are available, then applies three simple rules:

| Condition | Action |
|-----------|--------|
| Both candles green + price above 20 EMA | Buy **YES** (predict close above) |
| Both candles red + price below 20 EMA | Buy **NO** (predict close below) |
| Mixed candles | Trade the direction of whichever candle has the bigger body |

After entering, the bot holds the position until the market settles. No scalping, no mid-window exits.

## Web Dashboard

The bot includes a dark-themed web dashboard built with FastAPI and vanilla JavaScript.

- **BTC price chart** — live 5-minute candles from Binance
- **PnL curve** — cumulative profit/loss over time
- **Stats** — balance, win rate, total trades, fees, max drawdown
- **Live position** — current side, entry price, contracts, time to settlement
- **Strategy panel** — candle colors, EMA position, rule triggered, decision
- **Trade history** — scrollable table of all trades with outcomes
- **Engine log** — real-time activity feed
- **Settings** — switch between demo/live mode, enter API credentials
- **Reset** — clear all trade history and stats

### Running the Dashboard

```bash
python -m kalshi_bot.web
```

Open [http://localhost:8000](http://localhost:8000) in your browser. Start and stop the bot from the UI.

## Setup

### Prerequisites

- Python 3.10+
- A Kalshi account with API access

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example env file and add your Kalshi credentials:

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
- **Demo:** [demo.kalshi.co](https://demo.kalshi.co) → Account & Security → API Keys
- **Live:** [kalshi.com](https://kalshi.com) → Account & Security → API Keys

When generating a key, Kalshi provides a private key file (`.key`). Save it somewhere safe and point `KALSHI_PRIVATE_KEY_PATH` to it.

### 3. (Optional) Discord Alerts

Add a Discord webhook URL to `.env` for trade notifications:

```env
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your/webhook
```

## Running

### Terminal Bot (Demo)

```bash
python -m kalshi_bot.main
```

### Terminal Bot (Live)

```bash
python -m kalshi_bot.main --live
```

### Web Dashboard

```bash
python -m kalshi_bot.web
```

### Backtesting

Run the strategy against historical 5-minute candle data:

```bash
python -m kalshi_bot.backtest --days 30 --spread 0.04
```

## Project Structure

```
kalshi_bot/
├── main.py                  # Terminal entry point with Rich dashboard
├── backtest.py              # Walk-forward backtesting engine
├── config.py                # All tuneable parameters
├── kalshi/
│   ├── auth.py              # RSA-PSS API authentication
│   ├── client.py            # Kalshi REST API client
│   └── market_discovery.py  # KXBTC15M market finder
├── data/
│   ├── binance_feed.py      # 5-minute candles and WebSocket feed
│   └── kalshi_orderbook.py  # Kalshi market orderbook
├── strategy/
│   ├── trading_logic.py     # 5m candle rules + trade execution
│   └── risk_manager.py      # Position sizing and bankroll protection
├── learning/
│   ├── trade_logger.py      # Trade history (JSONL)
│   └── performance_analyzer.py  # Win rate, Sharpe, drawdown
├── monitoring/
│   ├── dashboard.py         # Rich terminal dashboard
│   └── alerts.py            # Discord webhook alerts
├── web/
│   ├── server.py            # FastAPI backend + WebSocket
│   └── static/
│       └── index.html       # Single-page dashboard frontend
└── storage/                 # Auto-created at runtime
    ├── logs/
    └── trades.jsonl
```

## Disclaimer

This software is for educational and research purposes. Trading on prediction markets involves real financial risk. Past performance does not guarantee future results. Only trade with money you can afford to lose.
