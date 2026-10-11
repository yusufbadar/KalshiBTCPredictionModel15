"""
Async Kalshi Trade API v2 client.

* Ed25519/RSA request signing (``auth.Signer``)
* separate read/write token buckets matching Kalshi's tiered limits
* retries with jittered backoff on 429/5xx for idempotent calls
* V2 event-order endpoints (``/portfolio/events/orders``): prices are YES-leg
  fixed-point dollars, ``side`` is ``bid`` (buy YES) or ``ask`` (sell YES /
  buy NO). Orders are always routed with the market ticker so the exchange
  picks the right shard.
"""
from __future__ import annotations

import asyncio
import random
import time
import uuid
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any
from urllib.parse import urlparse

import httpx
from loguru import logger

from kalshi_bot.api.auth import Signer


class KalshiError(Exception):
    def __init__(self, status: int, code: str, message: str, body: Any = None):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message, self.body = status, code, message, body


class TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self.rate, self.capacity = rate, capacity
        self.tokens = capacity
        self.t = time.monotonic()
        self.lock = asyncio.Lock()

    async def take(self, cost: float) -> None:
        async with self.lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= cost:
                    self.tokens -= cost
                    return
                await asyncio.sleep((cost - self.tokens) / self.rate)


def fmt_price(p: float) -> str:
    return str(Decimal(str(p)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN))


def fmt_count(c: float) -> str:
    return str(Decimal(str(c)).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))


def fnum(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


class KalshiREST:
    def __init__(self, base_url: str, signer: Signer | None = None,
                 read_rate: float = 180.0, write_rate: float = 90.0):
        self.base_url = base_url.rstrip("/")
        self.root_path = urlparse(self.base_url).path          # /trade-api/v2
        self.signer = signer
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0),
                                      limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                                      http2=False)
        # tokens/s; default request cost is 10 tokens (Basic: 200 read / 100 write)
        self.read_bucket = TokenBucket(read_rate, read_rate * 2)
        self.write_bucket = TokenBucket(write_rate, write_rate)
        self.last_latency_ms: float = 0.0

    async def close(self) -> None:
        await self.http.aclose()

    # ------------------------------------------------------------------ core
    async def request(self, method: str, path: str, *, params: dict | None = None,
                      json: dict | None = None, auth: bool = True, write: bool = False,
                      cost: float = 10.0, retries: int = 4) -> Any:
        url = self.base_url + path
        full_path = self.root_path + path
        delay = 0.25
        for attempt in range(retries + 1):
            await (self.write_bucket if write else self.read_bucket).take(cost)
            headers = {"Accept": "application/json"}
            if auth and self.signer is not None:
                headers.update(self.signer.headers(method, full_path))
            elif auth and (write or method != "GET"):
                raise KalshiError(0, "no_credentials", "API key not configured")
            t0 = time.perf_counter()
            try:
                r = await self.http.request(method, url, params=params, json=json, headers=headers)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                if attempt >= retries or (write and method == "POST"):
                    # never blindly resend an order: caller reconciles by client_order_id
                    raise KalshiError(0, "transport", str(e))
                await asyncio.sleep(delay + random.random() * delay)
                delay *= 2
                continue
            self.last_latency_ms = (time.perf_counter() - t0) * 1000
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries and not (write and method == "POST"):
                await asyncio.sleep(delay + random.random() * delay)
                delay *= 2
                continue
            if r.status_code >= 400:
                code, msg, body = str(r.status_code), r.text[:300], None
                try:
                    body = r.json()
                    err = body.get("error", body) if isinstance(body, dict) else {}
                    if isinstance(err, dict):
                        code = err.get("code", code)
                        msg = err.get("message", msg) or msg
                except Exception:
                    pass
                raise KalshiError(r.status_code, code, msg, body)
            if not r.content:
                return {}
            return r.json()
        raise KalshiError(429, "rate_limited", "retries exhausted")

    async def get(self, path: str, params: dict | None = None, auth: bool = True) -> Any:
        return await self.request("GET", path, params=params, auth=auth)

    # ------------------------------------------------------------ market data
    async def get_markets(self, **params) -> dict:
        return await self.get("/markets", params)

    async def get_market(self, ticker: str) -> dict:
        d = await self.get(f"/markets/{ticker}")
        return d.get("market", d)

    async def get_orderbook(self, ticker: str, depth: int = 0) -> dict:
        params = {"depth": depth} if depth else None
        d = await self.get(f"/markets/{ticker}/orderbook", params)
        return d.get("orderbook_fp") or d.get("orderbook") or {}

    async def get_trades(self, **params) -> dict:
        return await self.get("/markets/trades", params)

    async def exchange_status(self) -> dict:
        return await self.get("/exchange/status", auth=False)

    # --------------------------------------------------------------- account
    async def balance(self) -> dict:
        return await self.get("/portfolio/balance")

    async def positions(self, **params) -> dict:
        return await self.get("/portfolio/positions", params)

    async def fills(self, **params) -> dict:
        return await self.get("/portfolio/fills", params)

    async def settlements(self, **params) -> dict:
        return await self.get("/portfolio/settlements", params)

    async def orders(self, **params) -> dict:
        return await self.get("/portfolio/orders", params)

    async def account_limits(self) -> dict:
        return await self.get("/account/limits")

    async def shard_transfer(self, amount_dollars: float, src: int, dst: int) -> dict:
        """Move collateral between exchange shards of the same account."""
        centicents = int(round(amount_dollars * 10000))
        return await self.request("POST", "/portfolio/intra_exchange_instance_transfer", json={
            "source": "event_contract", "destination": "event_contract",
            "amount": centicents, "source_exchange_shard": src, "destination_exchange_shard": dst,
        }, write=True)

    # ----------------------------------------------------------------- orders
    async def create_order(self, ticker: str, side: str, price: float, count: float, *,
                           tif: str = "good_till_canceled", post_only: bool = False,
                           reduce_only: bool = False, expiration_ts: int | None = None,
                           client_order_id: str | None = None,
                           cancel_on_pause: bool = True) -> dict:
        """side: 'bid' = buy YES at ``price``; 'ask' = sell YES at ``price``
        (equivalent to buying NO at 1 - price)."""
        assert side in ("bid", "ask")
        body = {
            "ticker": ticker,
            "client_order_id": client_order_id or str(uuid.uuid4()),
            "side": side,
            "count": fmt_count(count),
            "price": fmt_price(price),
            "time_in_force": tif,
            "self_trade_prevention_type": "maker",
            "post_only": post_only,
            "cancel_order_on_pause": cancel_on_pause,
        }
        if reduce_only:
            body["reduce_only"] = True
        if expiration_ts:
            body["expiration_time"] = int(expiration_ts)
        return await self.request("POST", "/portfolio/events/orders", json=body, write=True)

    async def cancel_order(self, order_id: str, ticker: str) -> dict:
        return await self.request("DELETE", f"/portfolio/events/orders/{order_id}",
                                  params={"market_ticker": ticker}, write=True, cost=2.0)

    async def amend_order(self, order_id: str, ticker: str, side: str, price: float,
                          total_count: float) -> dict:
        return await self.request("POST", f"/portfolio/events/orders/{order_id}/amend", json={
            "ticker": ticker, "side": side, "price": fmt_price(price), "count": fmt_count(total_count),
        }, write=True)

    async def cancel_all(self) -> dict:
        return await self.request("DELETE", "/portfolio/events/orders", write=True)

    async def get_order(self, order_id: str, ticker: str | None = None) -> dict:
        # orders live on an exchange shard; the ticker routes the lookup
        params = {"market_ticker": ticker} if ticker else None
        d = await self.get(f"/portfolio/orders/{order_id}", params)
        return d.get("order", d)
