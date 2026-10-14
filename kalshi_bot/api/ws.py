"""
Authenticated Kalshi WebSocket client with automatic reconnect/resubscribe.

Subscriptions are registered by a local *name*; after a reconnect every
registered subscription is re-sent and ``on_reconnect`` handlers fire so
stateful consumers (order books) can reset. Messages are dispatched to
handlers by ``type``.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from typing import Any, Awaitable, Callable

import websockets
from loguru import logger

from kalshi_bot.api.auth import Signer
from kalshi_bot.config import WS_PATH

Handler = Callable[[dict], Awaitable[None] | None]


class KalshiWS:
    def __init__(self, url: str, signer: Signer):
        self.url = url
        self.signer = signer
        self._ws = None
        self._next_id = 1
        self._subs: dict[str, dict] = {}          # name -> params
        self._sid_by_name: dict[str, int] = {}
        self._name_by_cmd: dict[int, str] = {}
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._reconnect_handlers: list[Callable[[], Awaitable[None] | None]] = []
        self._stop = asyncio.Event()
        self.connected = asyncio.Event()
        self.last_msg_ts = 0.0
        self.reconnects = 0

    # ------------------------------------------------------------ public api
    def on(self, msg_type: str, handler: Handler) -> None:
        self._handlers[msg_type].append(handler)

    def on_reconnect(self, handler) -> None:
        self._reconnect_handlers.append(handler)

    async def subscribe(self, name: str, channels: list[str], **params) -> None:
        self._subs[name] = {"channels": channels, **params}
        if self._ws is not None and self.connected.is_set():
            await self._send_sub(name)

    def subscribed(self, name: str) -> bool:
        """True once the exchange has acknowledged this subscription on the
        current connection."""
        return self.connected.is_set() and name in self._sid_by_name

    async def unsubscribe(self, name: str) -> None:
        self._subs.pop(name, None)
        sid = self._sid_by_name.pop(name, None)
        if sid is not None and self._ws is not None:
            await self._send({"id": self._cmd_id(), "cmd": "unsubscribe", "params": {"sids": [sid]}})

    async def update(self, name: str, action: str, **params) -> None:
        sid = self._sid_by_name.get(name)
        if sid is None or self._ws is None:
            return
        await self._send({"id": self._cmd_id(), "cmd": "update_subscription",
                          "params": {"sid": sid, "action": action, **params}})

    async def run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                headers = self.signer.headers("GET", WS_PATH)
                async with websockets.connect(self.url, additional_headers=headers,
                                              max_size=2 ** 24, ping_interval=15,
                                              ping_timeout=20, close_timeout=3) as ws:
                    self._ws = ws
                    self._sid_by_name.clear()
                    self.connected.set()
                    backoff = 1.0
                    for name in list(self._subs):
                        await self._send_sub(name)
                    if self.reconnects:
                        for h in self._reconnect_handlers:
                            r = h()
                            if asyncio.iscoroutine(r):
                                await r
                    async for raw in ws:
                        self.last_msg_ts = time.time()
                        try:
                            msg = json.loads(raw)
                        except ValueError:
                            continue
                        await self._dispatch(msg)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - reconnect on anything
                logger.warning("WS disconnected: {}", e)
            finally:
                self._ws = None
                self.connected.clear()
            if self._stop.is_set():
                break
            self.reconnects += 1
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    async def close(self) -> None:
        self._stop.set()
        if self._ws is not None:
            await self._ws.close()

    # -------------------------------------------------------------- internals
    def _cmd_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    async def _send(self, obj: dict) -> None:
        if self._ws is not None:
            await self._ws.send(json.dumps(obj))

    async def _send_sub(self, name: str) -> None:
        cid = self._cmd_id()
        self._name_by_cmd[cid] = name
        await self._send({"id": cid, "cmd": "subscribe", "params": self._subs[name]})

    async def _dispatch(self, msg: dict) -> None:
        t = msg.get("type")
        if t == "subscribed":
            name = self._name_by_cmd.get(msg.get("id"))
            if name:
                self._sid_by_name[name] = msg["msg"]["sid"]
        elif t == "error":
            logger.warning("WS error: {}", msg)
        for h in self._handlers.get(t, ()):
            try:
                r = h(msg)
                if asyncio.iscoroutine(r):
                    await r
            except Exception as e:  # noqa: BLE001
                logger.exception("WS handler error for {}: {}", t, e)
