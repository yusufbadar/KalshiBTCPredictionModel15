"""
Headless runner (no dashboard):

    python -m kalshi_bot.cli                 # uses the saved environment (demo by default)
    python -m kalshi_bot.cli --env paper
    python -m kalshi_bot.cli --env demo --dry-run   # full pipeline, never sends orders

Ctrl+C stops the bot and cancels any resting orders.
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time

from loguru import logger

from kalshi_bot.config import LOG_DIR, get_settings
from kalshi_bot.engine import Engine


def main() -> None:
    ap = argparse.ArgumentParser(description="Settlement lock-in bot (headless)")
    ap.add_argument("--env", choices=["demo", "paper", "live"], default=None)
    ap.add_argument("--dry-run", action="store_true", help="run everything but never place orders")
    ap.add_argument("--assets", default=None, help="coins for this run, e.g. BTC,ETH (not saved)")
    a = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<dim>{time:HH:mm:ss}</dim> {message}")
    logger.add(LOG_DIR / "bot_{time:YYYY-MM-DD}.log", rotation="1 day", retention="14 days", level="DEBUG")

    s = get_settings()
    if a.env:
        if a.env == "live" and not s.live_unlocked:
            sys.exit("Live trading is locked. Unlock it once from the dashboard's Settings tab.")
        s.env = a.env
    if a.dry_run:
        s.strategy.enabled = False
    if a.assets:
        from kalshi_bot.config import parse_assets
        s.strategy.assets = ",".join(x.symbol for x in parse_assets(a.assets))

    async def run() -> None:
        loop = asyncio.get_running_loop()

        def quiet(lp, context):         # Windows logs peer resets noisily; the bot reconnects itself
            if isinstance(context.get("exception"), (ConnectionResetError, ConnectionAbortedError)):
                return
            lp.default_exception_handler(context)

        loop.set_exception_handler(quiet)
        eng = Engine(s)
        try:
            await eng.start()
        except RuntimeError as e:            # missing key, rejected login, another copy already running
            raise SystemExit(f"Could not start: {e}") from None
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:          # Windows
                signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))
        last = 0.0
        while not stop.is_set():
            await asyncio.sleep(1.0)
            if time.time() - last > 60:
                last = time.time()
                st = eng.state()
                m = st.get("market") or {}
                logger.info("{} | {:.0f}s left | account ${:,.2f} | today {:+.2f} | {}",
                            st["env"].upper(), m.get("seconds_left") or 0,
                            st.get("equity") or 0.0, st.get("day_pnl") or 0.0,
                            " | ".join(f"{x['symbol']}: {(x.get('decision') or {}).get('note', '')}"
                                       for x in st.get("assets", [])))
        await eng.stop("interrupted")

    asyncio.run(run())


if __name__ == "__main__":
    main()
