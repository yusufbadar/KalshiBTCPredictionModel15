"""
FastAPI backend for the BTC-15M Candle Bot web dashboard.

Run:
    python -m kalshi_bot.web.server
    # or: uvicorn kalshi_bot.web.server:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from kalshi_bot.config import TRADE_LOG_PATH, DATA_DIR

app = FastAPI(title="BTC-15M Candle Bot")

STATIC_DIR = Path(__file__).parent / "static"

# ── state ───────────────────────────────────────────────────────────────

bot_state = {
    "running": False,
    "mode": os.getenv("KALSHI_MODE", "demo"),
    "started_at": None,
    "api_key_id": os.getenv("KALSHI_API_KEY_ID", ""),
    "private_key": "",
    "activity_log": [],
    "current_position": None,
    "last_cycle": {},
    "balance": 0.0,
    "market_ticker": "",
    "seconds_to_close": 0,
}

_bot_task: Optional[asyncio.Task] = None
_ws_clients: list[WebSocket] = []

SETTINGS_PATH = DATA_DIR / "web_settings.json"


def _load_settings():
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text())
            bot_state["mode"] = data.get("mode", bot_state["mode"])
            bot_state["api_key_id"] = data.get("api_key_id", bot_state["api_key_id"])
            bot_state["private_key"] = data.get(
                "private_key", bot_state["private_key"]
            )
        except Exception:
            pass

    pk_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
    if pk_path and not bot_state["private_key"]:
        try:
            bot_state["private_key"] = Path(pk_path).read_text().strip()
        except Exception:
            pass


def _save_settings():
    SETTINGS_PATH.write_text(json.dumps({
        "mode": bot_state["mode"],
        "api_key_id": bot_state["api_key_id"],
        "private_key": bot_state["private_key"],
    }, indent=2))


_load_settings()


# ── helpers ─────────────────────────────────────────────────────────────


def _read_trades() -> list[dict]:
    if not TRADE_LOG_PATH.exists():
        return []
    out = []
    for line in TRADE_LOG_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _compute_stats(trades: list[dict]) -> dict:
    traded = [t for t in trades if t.get("direction") not in ("skip", "order_error")]
    settled = [t for t in traded if t.get("outcome") is not None]
    wins = [t for t in settled if t.get("outcome") is True]

    total_pnl = sum(t.get("pnl_dollars") or 0 for t in settled)
    total_fees = sum(t.get("fees_cents") or 0 for t in traded) / 100.0

    pnl_list = [t.get("pnl_dollars") or 0 for t in settled]
    cum = []
    s = 0
    for p in pnl_list:
        s += p
        cum.append(round(s, 2))

    max_dd = 0.0
    if cum:
        peak = cum[0]
        for v in cum:
            peak = max(peak, v)
            max_dd = max(max_dd, peak - v)

    return {
        "total_trades": len(traded),
        "settled": len(settled),
        "pending": len(traded) - len(settled),
        "wins": len(wins),
        "losses": len(settled) - len(wins),
        "win_rate": len(wins) / len(settled) if settled else 0,
        "total_pnl": round(total_pnl, 2),
        "total_fees": round(total_fees, 2),
        "max_drawdown": round(max_dd, 2),
        "pnl_curve": cum,
        "pnl_timestamps": [t.get("ts", 0) for t in settled],
    }


async def _fetch_btc_prices() -> list[dict]:
    endpoints = [
        "https://api.binance.us/api/v3",
        "https://api.binance.com/api/v3",
    ]
    async with httpx.AsyncClient(timeout=15.0) as client:
        for base in endpoints:
            try:
                r = await client.get(
                    f"{base}/klines",
                    params={"symbol": "BTCUSDT", "interval": "5m", "limit": 288},
                )
                r.raise_for_status()
                return [
                    {
                        "ts": k[0],
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    }
                    for k in r.json()
                ]
            except Exception:
                continue
    return []


# ── bot lifecycle ───────────────────────────────────────────────────────


async def _run_bot_loop():
    """Run the trading bot in the background."""
    import kalshi_bot.config as cfg

    cfg.KALSHI_MODE = bot_state["mode"]
    cfg.KALSHI_API_KEY_ID = bot_state["api_key_id"]
    private_key_pem = bot_state["private_key"]

    if not cfg.KALSHI_API_KEY_ID or not private_key_pem:
        bot_state["activity_log"].append(
            f"{_ts()} Missing API credentials -- configure in Settings"
        )
        bot_state["running"] = False
        return

    from kalshi_bot.kalshi.auth import KalshiAuth
    from kalshi_bot.kalshi.client import KalshiClient
    from kalshi_bot.kalshi.market_discovery import MarketDiscovery
    from kalshi_bot.data.binance_feed import BinanceFeed
    from kalshi_bot.data.kalshi_orderbook import KalshiOrderbookFeed
    from kalshi_bot.strategy.risk_manager import RiskManager
    from kalshi_bot.strategy.trading_logic import TradingEngine
    from kalshi_bot.learning.trade_logger import TradeLogger
    from kalshi_bot.learning.performance_analyzer import PerformanceAnalyzer
    from kalshi_bot.monitoring.alerts import alert_trade, alert_settlement

    try:
        auth = KalshiAuth(cfg.KALSHI_API_KEY_ID, private_key_pem)
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

        bot_state["activity_log"].append(f"{_ts()} Connecting to Binance...")
        if not await binance.connect():
            bot_state["activity_log"].append(f"{_ts()} Binance connection failed")
            bot_state["running"] = False
            return

        bot_state["balance"] = client.get_balance_dollars()
        bot_state["activity_log"].append(
            f"{_ts()} Bot started in {cfg.KALSHI_MODE.upper()} mode "
            f"(balance: ${bot_state['balance']:.2f})"
        )

        last_traded_market = None

        while bot_state["running"]:
            try:
                bot_state["balance"] = client.get_balance_dollars()
                market = discovery.get_current_market()
                bot_state["market_ticker"] = market.ticker if market else ""
                bot_state["seconds_to_close"] = (
                    market.seconds_until_close if market else 0
                )

                if engine.current_position:
                    pos = engine.current_position
                    bot_state["current_position"] = {
                        "direction": pos["direction"],
                        "side": pos["side"],
                        "entry_price": pos["entry_price"],
                        "count": pos["count"],
                        "bet_dollars": pos["bet_dollars"],
                        "hold_seconds": time.time() - pos["entry_time"],
                    }
                else:
                    bot_state["current_position"] = None

                bot_state["last_cycle"] = engine.last_cycle
                bot_state["activity_log"] = engine.activity_log[-30:]

                await _broadcast_state()

                if market is None:
                    await asyncio.sleep(8)
                    continue

                if last_traded_market and market.ticker != last_traded_market:
                    for _ in range(6):
                        settlement = await engine.check_settlement(
                            last_traded_market
                        )
                        if settlement:
                            if "won" in settlement:
                                alert_settlement(
                                    last_traded_market,
                                    settlement["won"],
                                    settlement["pnl"],
                                    analyzer.win_rate_all,
                                )
                            last_traded_market = None
                            break
                        await asyncio.sleep(5)
                    else:
                        last_traded_market = None
                        engine.current_position = None

                if market.ticker != last_traded_market:
                    if market.seconds_until_close >= 300:
                        cycle = await engine.run_cycle()
                        if cycle.get("action") == "traded":
                            last_traded_market = market.ticker

                if last_traded_market and market.ticker == last_traded_market:
                    settlement = await engine.check_settlement(last_traded_market)
                    if settlement:
                        last_traded_market = None

                await asyncio.sleep(10 if engine.current_position else 8)

            except asyncio.CancelledError:
                break
            except Exception as e:
                bot_state["activity_log"].append(f"{_ts()} ERROR: {e}")
                await asyncio.sleep(10)

        await binance.close()
        client.close()

    except Exception as e:
        bot_state["activity_log"].append(f"{_ts()} FATAL: {e}")
        logger.error(traceback.format_exc())
    finally:
        bot_state["running"] = False
        bot_state["current_position"] = None


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


async def _broadcast_state():
    if not _ws_clients:
        return
    payload = json.dumps(_build_ws_payload())
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


def _build_ws_payload() -> dict:
    trades = _read_trades()
    stats = _compute_stats(trades)
    recent = [
        t for t in trades
        if t.get("direction") not in ("skip", "order_error")
    ][-20:]

    return {
        "type": "update",
        "running": bot_state["running"],
        "mode": bot_state["mode"],
        "balance": bot_state["balance"],
        "uptime": (
            time.time() - bot_state["started_at"]
            if bot_state["started_at"]
            else 0
        ),
        "market_ticker": bot_state["market_ticker"],
        "seconds_to_close": bot_state["seconds_to_close"],
        "position": bot_state["current_position"],
        "last_cycle": bot_state["last_cycle"],
        "activity_log": bot_state["activity_log"][-20:],
        "stats": stats,
        "recent_trades": recent,
    }


# ── routes ──────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def api_status():
    return {
        "running": bot_state["running"],
        "mode": bot_state["mode"],
        "balance": bot_state["balance"],
        "uptime": (
            time.time() - bot_state["started_at"]
            if bot_state["started_at"]
            else 0
        ),
        "market_ticker": bot_state["market_ticker"],
        "seconds_to_close": bot_state["seconds_to_close"],
        "position": bot_state["current_position"],
        "last_cycle": bot_state["last_cycle"],
    }


@app.get("/api/stats")
async def api_stats():
    return _compute_stats(_read_trades())


@app.get("/api/trades")
async def api_trades():
    trades = _read_trades()
    traded = [
        t for t in trades if t.get("direction") not in ("skip", "order_error")
    ]
    return traded[-50:]


@app.get("/api/activity")
async def api_activity():
    return bot_state["activity_log"][-30:]


@app.get("/api/btc-prices")
async def api_btc_prices():
    return await _fetch_btc_prices()


@app.get("/api/settings")
async def api_settings():
    pk = bot_state["private_key"]
    has_key = bool(pk and pk.strip().startswith("-----BEGIN"))
    return {
        "mode": bot_state["mode"],
        "api_key_id": bot_state["api_key_id"],
        "has_private_key": has_key,
    }


class SettingsUpdate(BaseModel):
    mode: str = "demo"
    api_key_id: str = ""
    private_key: str = ""


@app.post("/api/settings")
async def api_settings_update(settings: SettingsUpdate):
    if bot_state["running"]:
        return {"error": "Stop the bot before changing settings"}
    bot_state["mode"] = settings.mode
    bot_state["api_key_id"] = settings.api_key_id
    if settings.private_key.strip():
        bot_state["private_key"] = settings.private_key.strip()
    os.environ["KALSHI_MODE"] = settings.mode
    os.environ["KALSHI_API_KEY_ID"] = settings.api_key_id
    _save_settings()
    return {"ok": True}


@app.post("/api/reset")
async def api_reset():
    if bot_state["running"]:
        return {"error": "Stop the bot before resetting"}
    if TRADE_LOG_PATH.exists():
        TRADE_LOG_PATH.unlink()
    bot_state["activity_log"] = []
    bot_state["last_cycle"] = {}
    bot_state["current_position"] = None
    bot_state["balance"] = 0.0
    return {"ok": True}


@app.post("/api/bot/start")
async def api_bot_start():
    global _bot_task
    if bot_state["running"]:
        return {"error": "Bot is already running"}
    bot_state["running"] = True
    bot_state["started_at"] = time.time()
    _bot_task = asyncio.create_task(_run_bot_loop())
    return {"ok": True}


@app.post("/api/bot/stop")
async def api_bot_stop():
    global _bot_task
    if not bot_state["running"]:
        return {"error": "Bot is not running"}
    bot_state["running"] = False
    if _bot_task and not _bot_task.done():
        _bot_task.cancel()
    bot_state["activity_log"].append(f"{_ts()} Bot stopped by user")
    bot_state["current_position"] = None
    return {"ok": True}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        await ws.send_text(json.dumps(_build_ws_payload()))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


# ── main ────────────────────────────────────────────────────────────────

def main():
    import uvicorn

    print("\n  BTC-15M Candle Bot -- Web Dashboard")
    print("  http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
