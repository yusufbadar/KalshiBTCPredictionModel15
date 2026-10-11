"""
Go / no-go checklist for the selected environment.

Read-only: it logs in, reads the balance, lists the markets it would trade and
listens to the settlement-index feed for a few seconds. It never places,
amends or cancels an order and never moves money.

    python -m kalshi_bot.preflight            # prints the checklist
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import websockets

from kalshi_bot.api.auth import Signer
from kalshi_bot.api.rest import KalshiError, KalshiREST, fnum
from kalshi_bot.config import ENDPOINTS, WS_PATH, Settings, parse_assets
from kalshi_bot.core.risk import RiskManager

CRYPTO_SHARD = 2


def _check(name: str, level: str, detail: str) -> dict:
    return {"name": name, "level": level, "ok": level != "fail", "detail": detail}


async def _feed_check(url: str, signer: Signer, index_ids: list[str], wait_s: float = 7.0) -> dict:
    """Connect to the WebSocket and wait for one tick per index. Returns
    {index_id: (value, local_minus_exchange_seconds)}."""
    got: dict[str, tuple[float, float]] = {}
    async with websockets.connect(url, additional_headers=signer.headers("GET", WS_PATH),
                                  open_timeout=10, close_timeout=2) as ws:
        await ws.send(json.dumps({"id": 1, "cmd": "subscribe",
                                  "params": {"channels": ["cfbenchmarks_value"], "index_ids": index_ids}}))
        t0 = time.time()
        while time.time() - t0 < wait_s and len(got) < len(index_ids):
            try:
                raw = await asyncio.wait_for(ws.recv(), 2.0)
            except asyncio.TimeoutError:
                continue
            m = json.loads(raw)
            if m.get("type") != "cfbenchmarks_value":
                continue
            mm = m.get("msg", {})
            try:
                d = json.loads(mm.get("data", "{}"))
                iid = mm.get("index_id") or d.get("id")
                rx = mm.get("received_at")
                got[iid] = (float(d["value"]), time.time() - rx / 1000.0 if rx else 0.0)
            except (KeyError, TypeError, ValueError):
                continue
    return got


async def run_preflight(s: Settings, running: bool = False) -> dict:
    exch = s.trading_env
    specs = parse_assets(s.strategy.assets)
    checks: list[dict] = []
    add = lambda *a: checks.append(_check(*a))  # noqa: E731
    label = {"demo": "Demo (fake money)", "paper": "Paper (simulated fills, real prices)",
             "live": "LIVE (real money)"}[s.env]
    add("Mode", "ok" if s.env != "live" or s.live_unlocked else "fail",
        label + ("" if s.env != "paper" else f" - price feed login via the {exch} exchange"))

    kid, path = s.creds(exch)
    if not kid or not path or not Path(path).expanduser().exists():
        add("API key", "fail", f"No {exch} API key saved. Add the key id and private key in Settings.")
        return {"env": s.env, "exchange": exch, "ok": False, "checks": checks}
    try:
        signer = Signer(kid, path)
        add("API key", "ok", f"{exch} key loaded")
    except Exception as e:  # noqa: BLE001
        add("API key", "fail", f"Private key could not be read: {e}")
        return {"env": s.env, "exchange": exch, "ok": False, "checks": checks}

    rest = KalshiREST(ENDPOINTS[exch]["rest"], signer)
    equity = None
    try:
        try:
            b = await rest.balance()
            total = fnum(b.get("balance_dollars"), fnum(b.get("balance")) / 100)
            shards = {int(x.get("exchange_index", 0)): fnum(x.get("balance")) for x in b.get("balance_breakdown", [])}
            cash = sum(shards.values()) if shards else total
            equity = s.paper_start if s.env == "paper" else cash
            if s.env == "paper":
                add("Login", "ok", f"Logged in to {exch}. Paper account size ${s.paper_start:,.2f} (simulated).")
            else:
                crypto = shards.get(CRYPTO_SHARD)
                extra = "" if crypto is None else (f" (${crypto:,.2f} already on the crypto sub-account; the bot "
                                                   f"moves what it needs)" if s.shard_topup else
                                                   f" (${crypto:,.2f} on the crypto sub-account)")
                add("Login and balance", "ok" if cash >= s.risk.min_balance + 2 else "fail",
                    f"Logged in. Cash ${cash:,.2f}{extra}" if cash >= s.risk.min_balance + 2 else
                    f"Cash ${cash:,.2f} is below the ${s.risk.min_balance:.0f} minimum plus one trade - deposit first.")
        except KalshiError as e:
            add("Login and balance", "fail", f"Kalshi rejected the key ({e.status}): {e.message}")
            return {"env": s.env, "exchange": exch, "ok": False, "checks": checks}

        try:
            st = await rest.get("/exchange/status", auth=False)
            active = bool(st.get("trading_active", True)) and bool(st.get("exchange_active", True))
            add("Exchange", "ok" if active else "warn",
                "Trading is open" if active else "Exchange trading is paused right now (maintenance?) - "
                "the bot waits and resumes by itself")
        except Exception as e:  # noqa: BLE001
            add("Exchange", "warn", f"Could not read exchange status: {e}")

        # markets the bot would trade
        market_rest = KalshiREST(ENDPOINTS["live"]["rest"], None) if s.env == "paper" else rest
        try:
            now = time.time()
            for sp in specs:
                try:
                    d = await market_rest.get("/markets", {"series_ticker": sp.series15, "limit": 10,
                                                           "min_close_ts": int(now), "max_close_ts": int(now) + 1800},
                                              auth=s.env != "paper")
                    ms = [m for m in d.get("markets", []) if m.get("close_time")]
                    live = []
                    for m in ms:
                        o = datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp()
                        c = datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp()
                        if o <= now < c:
                            live.append((c, m))
                    if live:
                        c, m = min(live, key=lambda t: t[0])
                        add(f"{sp.symbol} market", "ok", f"{m['ticker']} - strike {m.get('floor_strike')}, "
                                                         f"closes in {c - now:.0f}s")
                    else:
                        add(f"{sp.symbol} market", "warn", f"No open {sp.series15} market right now")
                except Exception as e:  # noqa: BLE001
                    add(f"{sp.symbol} market", "fail", f"Could not list {sp.series15}: {e}")
        finally:
            if market_rest is not rest:
                await market_rest.close()

        # settlement-index feed + clock
        try:
            ids = [sp.index_id for sp in specs]
            got = await _feed_check(ENDPOINTS[exch]["ws"], signer, ids)
            missing = [i for i in ids if i not in got]
            if missing:
                add("Price feed", "fail", f"No ticks received for {', '.join(missing)} in 7 s")
            else:
                add("Price feed", "ok", "Live settlement index for " + ", ".join(
                    f"{sp.symbol} {got[sp.index_id][0]:,.6g}" for sp in specs))
                off = sorted(v[1] for v in got.values())[len(got) // 2]
                lvl = "ok" if abs(off) < 2.0 else "warn"
                add("Clock", lvl, f"This PC is {abs(off):.2f}s {'ahead of' if off > 0 else 'behind'} Kalshi "
                    f"(the bot corrects for it)" + ("" if lvl == "ok" else
                                                    " - large gap: turn on automatic time sync in Windows"))
        except Exception as e:  # noqa: BLE001
            add("Price feed", "fail", f"Could not connect to the live feed: {e}")
    finally:
        await rest.close()

    # limits with this account size
    if equity is not None:
        r = RiskManager(s.risk)
        mc, cc = r.market_cap(equity), r.close_cap(equity)
        dl, ddl = r.daily_loss_limit(equity, 0.0), r.drawdown_limit(equity, 0.0)
        fmt = lambda v: "off" if v == float("inf") else f"${v:,.2f}"  # noqa: E731
        lvl = "ok"
        note = ""
        if mc < 1.0:
            lvl, note = "fail", " - the per-contract limit is below the price of one contract"
        elif mc == float("inf") or (equity > 0 and mc > 0.25 * equity):
            lvl, note = "warn", " - more than 25% of the account can ride on one contract"
        add("Risk limits", lvl, f"Per contract {fmt(mc)}, per settlement moment {fmt(cc)}, daily loss stop "
            f"{fmt(dl)}, drawdown stop {fmt(ddl)}{note}")
    sp_ = s.strategy
    add("Strategy", "ok" if sp_.enabled else "warn",
        (f"Trading ON - {', '.join(x.symbol for x in specs)}; final {sp_.max_seconds_left:.0f}s"
         f"{f' (lead >= {sp_.early_cushion_bp:g} bp while > {sp_.early_seconds:.0f}s left)' if sp_.early_cushion_bp > 0 else ''}, "
         f"model >= {100 * sp_.p_min:.0f}%, z >= {sp_.z_min:g}, edge >= {100 * sp_.min_edge:g}c"
         + (", hourly BTC strikes" if sp_.trade_hourly else ""))
        if sp_.enabled else "Trading is switched OFF in Settings (the bot will only watch)")
    if running:
        add("Bot", "ok", "The bot is running")
    ok = all(c["ok"] for c in checks)
    return {"env": s.env, "exchange": exch, "ok": ok, "checks": checks}


def main() -> None:
    from kalshi_bot.config import get_settings
    res = asyncio.run(run_preflight(get_settings()))
    mark = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}
    print(f"\nPreflight - {res['env'].upper()} (login: {res['exchange']})\n")
    for c in res["checks"]:
        print(f"  {mark[c['level']]} {c['name']}: {c['detail']}")
    print("\n  READY" if res["ok"] else "\n  NOT READY - fix the [FAIL] lines first")


if __name__ == "__main__":
    main()
