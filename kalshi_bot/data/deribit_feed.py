"""
Deribit BTC options Put/Call Ratio feed (free public API, no auth).

Fetches all BTC option book summaries and computes:
  - Overall PCR  (put OI / call OI)
  - Short-dated PCR  (options expiring within 2 days)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from loguru import logger

from kalshi_bot.config import DERIBIT_URL

CACHE_TTL = 300  # seconds


class DeribitFeed:
    def __init__(self, max_dte: int = 2, min_oi: float = 100.0):
        self.max_dte = max_dte
        self.min_oi = min_oi
        self._cache: Optional[dict] = None
        self._cache_ts: float = 0

    def fetch_pcr(self, force: bool = False) -> Optional[dict]:
        """Synchronous fetch (safe to call from asyncio via run_in_executor)."""
        if not force and self._cache and (time.time() - self._cache_ts < CACHE_TTL):
            return self._cache

        try:
            with httpx.Client(timeout=10.0) as http:
                r = http.get(DERIBIT_URL, params={"currency": "BTC", "kind": "option"})
                r.raise_for_status()
                items = r.json().get("result", [])

            put_oi = call_oi = short_put = short_call = 0.0
            for item in items:
                name = item.get("instrument_name", "")
                oi = float(item.get("open_interest", 0))
                if oi < self.min_oi:
                    continue
                is_put = name.endswith("-P")
                is_call = name.endswith("-C")
                if is_put:
                    put_oi += oi
                elif is_call:
                    call_oi += oi
                dte = self._parse_dte(name)
                if dte is not None and dte <= self.max_dte:
                    if is_put:
                        short_put += oi
                    elif is_call:
                        short_call += oi

            overall = put_oi / call_oi if call_oi else 1.0
            short = short_put / short_call if short_call else overall

            result = {
                "overall_pcr": round(overall, 4),
                "short_pcr": round(short, 4),
                "put_oi": round(put_oi, 2),
                "call_oi": round(call_oi, 2),
                "short_put_oi": round(short_put, 2),
                "short_call_oi": round(short_call, 2),
            }
            self._cache = result
            self._cache_ts = time.time()
            logger.debug("Deribit PCR: overall={:.3f} short={:.3f}", overall, short)
            return result

        except Exception as e:
            logger.warning("Deribit PCR fetch failed: {}", e)
            return self._cache

    @staticmethod
    def _parse_dte(instrument_name: str) -> Optional[int]:
        try:
            parts = instrument_name.split("-")
            if len(parts) < 3:
                return None
            expiry_dt = datetime.strptime(parts[1], "%d%b%y").replace(tzinfo=timezone.utc)
            return max(0, (expiry_dt - datetime.now(timezone.utc)).days)
        except Exception:
            return None
