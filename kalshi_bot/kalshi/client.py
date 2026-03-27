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


def _fp_count(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _fp_dollars(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def order_filled_count(order: dict[str, Any]) -> int:
    """Best-effort filled contracts from a Kalshi order object."""
    if not order:
        return 0
    for key in ("count_filled", "filled_count", "fill_count"):
        raw = order.get(key)
        if raw is not None:
            try:
                return max(0, int(raw))
            except (TypeError, ValueError):
                pass
    leaves = order.get("leaves") or order.get("remaining_count") or order.get("open_count")
    total = order.get("count") or order.get("initial_count") or order.get("order_count")
    if leaves is not None and total is not None:
        try:
            return max(0, int(total) - int(leaves))
        except (TypeError, ValueError):
            pass
    return 0


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

    def aggregate_buy_fills_for_order(self, order_id: str, side: str) -> Optional[dict[str, Any]]:
        """Sum buy fills for ``order_id`` on ``side`` (``yes`` / ``no``).

        Returns ``count``, ``vwap_dollars`` (price for that side), ``fees_dollars``,
        or ``None`` if no matching fills.
        """
        data = self.get_fills(order_id=order_id, limit=200)
        fills = data.get("fills") or []
        if not fills:
            return None
        total = 0
        fee_d = 0.0
        cost_weighted = 0.0
        for f in fills:
            if (f.get("action") or "").lower() != "buy":
                continue
            if (f.get("side") or "").lower() != side:
                continue
            c = _fp_count(f.get("count_fp"))
            if c <= 0:
                c = int(f.get("count") or 0)
            if c <= 0:
                continue
            if side == "yes":
                p = _fp_dollars(f.get("yes_price_dollars"))
            else:
                p = _fp_dollars(f.get("no_price_dollars"))
            total += c
            cost_weighted += p * c
            fee_d += _fp_dollars(f.get("fee_cost"))
        if total <= 0:
            return None
        return {
            "count": total,
            "vwap_dollars": cost_weighted / total,
            "fees_dollars": fee_d,
        }

    def get_market_position_row(self, ticker: str) -> Optional[dict[str, Any]]:
        """Return the ``market_positions`` row for ``ticker``, if any."""
        data = self.get_positions(ticker=ticker, limit=100)
        for row in data.get("market_positions") or []:
            if row.get("ticker") == ticker:
                return row
        return None

    def position_contracts_for_side(self, row: dict[str, Any], side: str) -> int:
        """Long YES -> positive ``position_fp``; long NO -> negative."""
        fp = row.get("position_fp") or row.get("position")
        try:
            v = float(str(fp).strip()) if fp is not None else 0.0
        except (TypeError, ValueError):
            v = 0.0
        if side == "yes" and v > 0:
            return int(v)
        if side == "no" and v < 0:
            return int(abs(v))
        return 0

    # ------------------------------------------------------------------
    # Exchange status
    # ------------------------------------------------------------------
    def get_exchange_status(self) -> dict:
        return self.get("/exchange/status")

    def close(self):
        self._http.close()
