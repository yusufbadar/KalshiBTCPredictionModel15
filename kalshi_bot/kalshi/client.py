"""
Kalshi REST API client.

Wraps authenticated GET / POST / DELETE calls and exposes typed helpers
for the endpoints the bot needs: balance, markets, orders, fills, settlements.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

import httpx
from loguru import logger

from kalshi_bot.kalshi.auth import KalshiAuth
from kalshi_bot.config import get_kalshi_base_url


class KalshiClient:
    """Thin, synchronous wrapper around the Kalshi Trade-API v2."""

    def __init__(self, auth: KalshiAuth, base_url: str | None = None):
        self.auth = auth
        self.base_url = (base_url or get_kalshi_base_url()).rstrip("/")
        self._http = httpx.Client(timeout=15.0)
        logger.info("KalshiClient ready  base_url={}", self.base_url)

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _sign_path(self, path: str) -> str:
        """Full path from root for signing (includes /trade-api/v2)."""
        from urllib.parse import urlparse
        return urlparse(self._url(path)).path

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------
    def get(self, path: str, params: dict | None = None) -> dict:
        sign_path = self._sign_path(path)
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            if qs:
                sign_path_with_qs = f"{sign_path}?{qs}"
            else:
                sign_path_with_qs = sign_path
        else:
            sign_path_with_qs = sign_path

        headers = self.auth.sign("GET", sign_path)
        resp = self._http.get(self._url(path), params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def post(self, path: str, data: dict | None = None) -> dict:
        sign_path = self._sign_path(path)
        headers = self.auth.sign("POST", sign_path)
        headers["Content-Type"] = "application/json"
        resp = self._http.post(self._url(path), json=data or {}, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def delete(self, path: str) -> dict | None:
        sign_path = self._sign_path(path)
        headers = self.auth.sign("DELETE", sign_path)
        resp = self._http.delete(self._url(path), headers=headers)
        resp.raise_for_status()
        if resp.content:
            return resp.json()
        return None

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------
    def get_balance(self) -> dict:
        """Return balance in cents and portfolio value."""
        return self.get("/portfolio/balance")

    def get_balance_dollars(self) -> float:
        data = self.get_balance()
        return data.get("balance", 0) / 100.0

    def get_positions(self, **kwargs) -> dict:
        return self.get("/portfolio/positions", params=kwargs or None)

    def get_fills(self, **kwargs) -> dict:
        return self.get("/portfolio/fills", params=kwargs or None)

    def get_settlements(self, **kwargs) -> dict:
        return self.get("/portfolio/settlements", params=kwargs or None)

    # ------------------------------------------------------------------
    # Markets (public – but still needs auth header for the connection)
    # ------------------------------------------------------------------
    def get_markets(self, **kwargs) -> dict:
        return self.get("/markets", params=kwargs or None)

    def get_market(self, ticker: str) -> dict:
        return self.get(f"/markets/{ticker}")

    def get_market_orderbook(self, ticker: str) -> dict:
        return self.get(f"/markets/{ticker}/orderbook")

    def get_market_candlesticks(self, ticker: str, period_interval: int = 1) -> dict:
        return self.get(
            f"/markets/{ticker}/candlesticks",
            params={"period_interval": period_interval},
        )

    def get_trades(self, **kwargs) -> dict:
        return self.get("/markets/trades", params=kwargs or None)

    def get_event(self, event_ticker: str) -> dict:
        return self.get(f"/events/{event_ticker}")

    def get_series(self, series_ticker: str) -> dict:
        return self.get(f"/series/{series_ticker}")

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def create_order(
        self,
        ticker: str,
        side: str,
        action: str = "buy",
        count: int = 1,
        order_type: str = "limit",
        yes_price: int | None = None,
        no_price: int | None = None,
        time_in_force: str | None = None,
        client_order_id: str | None = None,
    ) -> dict:
        """Place a new order.

        Args:
            ticker: Market ticker, e.g. ``KXBTC15M-26MAR1915``
            side: ``"yes"`` or ``"no"``
            action: ``"buy"`` or ``"sell"``
            count: Number of contracts (each contract pays $1 if correct)
            order_type: ``"limit"`` (only type supported for retail)
            yes_price: Limit price in cents (1–99) for the YES side
            no_price: Limit price in cents (1–99) for the NO side
            time_in_force: ``"fill_or_kill"``, ``"good_till_canceled"``,
                           ``"immediate_or_cancel"``
            client_order_id: Idempotency key; auto-generated if omitted
        """
        payload: dict[str, Any] = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": order_type,
            "client_order_id": client_order_id or str(uuid.uuid4()),
        }
        if yes_price is not None:
            payload["yes_price"] = yes_price
        if no_price is not None:
            payload["no_price"] = no_price
        if time_in_force:
            payload["time_in_force"] = time_in_force

        logger.info(
            "Creating order: {} {} {} x{} @yes={}c @no={}c",
            action, side, ticker, count, yes_price, no_price,
        )
        return self.post("/portfolio/orders", data=payload)

    def cancel_order(self, order_id: str) -> dict | None:
        return self.delete(f"/portfolio/orders/{order_id}")

    def get_order(self, order_id: str) -> dict:
        return self.get(f"/portfolio/orders/{order_id}")

    def get_orders(self, **kwargs) -> dict:
        return self.get("/portfolio/orders", params=kwargs or None)

    # ------------------------------------------------------------------
    # Exchange status
    # ------------------------------------------------------------------
    def get_exchange_status(self) -> dict:
        return self.get("/exchange/status")

    def close(self):
        self._http.close()
