"""
Dashboard backend: FastAPI + a WebSocket that pushes engine state twice a second.

    python -m kalshi_bot            -> http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from loguru import logger
from pydantic import BaseModel

from kalshi_bot.config import (DB_PATH, PROJECT_DIR, STORAGE_DIR, RiskParams, Settings, StrategyParams,
                               get_settings, set_settings)
from kalshi_bot.core.ledger import Ledger
from kalshi_bot.engine import Engine
from kalshi_bot.web.auth import COOKIE, LOOPBACK_HOSTS, SESSION_HOURS, Auth, host_only, is_loopback, same_origin

STATIC = Path(__file__).parent / "static"
REPORTS = STORAGE_DIR / "reports"
SECRETS = PROJECT_DIR / "secrets"

app = FastAPI(title="Lock-In bot", docs_url=None, redoc_url=None, openapi_url=None)
engine: Engine | None = None
_engine_lock = asyncio.Lock()
auth = Auth()
AUTOSTART = os.getenv("KALSHI_AUTOSTART", "").strip().lower() in ("1", "true", "yes", "on")
# extra host names this dashboard may be opened under when no password is set
# (with a password any name works - the login is what protects it)
EXTRA_HOSTS = {h.strip().lower() for h in os.getenv("KALSHI_ALLOWED_HOSTS", "").split(",") if h.strip()}
OPEN_PATHS = {"/login", "/api/login", "/favicon.svg"}
SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": ("default-src 'self'; script-src 'self' 'unsafe-inline'; "
                                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                                "connect-src 'self' ws: wss:; frame-ancestors 'none'; base-uri 'none'; "
                                "form-action 'self'"),
}


def _host_allowed(host_header: str) -> bool:
    """Without a password the dashboard answers only to this machine's own
    names, so a hostile web page cannot reach it through a made-up domain."""
    return auth.enabled or host_only(host_header) in LOOPBACK_HOSTS | EXTRA_HOSTS


@app.middleware("http")
async def guard(request: Request, call_next):
    auth.maybe_reload()
    path = request.url.path
    host = request.headers.get("host", "")
    if not _host_allowed(host):
        return JSONResponse({"detail": "Set a dashboard password before opening it from another address."},
                            status_code=403)
    if request.method not in ("GET", "HEAD", "OPTIONS") and not same_origin(request.headers.get("origin"), host):
        return JSONResponse({"detail": "Cross-site request refused."}, status_code=403)
    if auth.enabled and path not in OPEN_PATHS and not auth.valid_session(request.cookies.get(COOKIE)):
        if path.startswith("/api/"):
            return JSONResponse({"detail": "Not signed in."}, status_code=401)
        return RedirectResponse("/login", status_code=303)
    resp = await call_next(request)
    for k, v in SECURITY_HEADERS.items():
        resp.headers.setdefault(k, v)
    if path.startswith("/api/") or path in ("/", "/login"):
        resp.headers.setdefault("Cache-Control", "no-store")
    return resp


def _json(obj) -> str:
    return json.dumps(obj, default=lambda o: None, allow_nan=False, separators=(",", ":"))


def _clean(o):
    """Replace NaN/inf with None so the payload is valid JSON."""
    import math
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _idle_state() -> dict:
    s = get_settings()
    led = Ledger(DB_PATH)
    return {"env": s.env, "status": "stopped", "running": False, "error": "",
            "stats": led.stats(s.env), "settlements": led.settlements_list(s.env, 50),
            "fills": led.recent_fills(s.env, 50),
            "logs": [dict(e, ts=e["ts"]) for e in reversed(led.events_list(s.env, 80))],
            "history": [], "orders": [], "market": None, "index": {}, "decision": {},
            "assets": [], "equity": None, "limits": None, "killed": False, "kill_reason": "",
            "trading_enabled": s.strategy.enabled, "day_pnl": 0.0, "balance": None}


def current_state() -> dict:
    if engine is None:
        return _idle_state()
    st = engine.state()
    return st


# ----------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/favicon.svg")
async def favicon():
    return FileResponse(STATIC / "favicon.svg", media_type="image/svg+xml")


# ------------------------------------------------------------------ login
class LoginIn(BaseModel):
    password: str = ""


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if not auth.enabled or auth.valid_session(request.cookies.get(COOKIE)):
        return RedirectResponse("/", status_code=303)
    return HTMLResponse((STATIC / "login.html").read_text(encoding="utf-8"))


@app.post("/api/login")
async def api_login(body: LoginIn, request: Request):
    if not auth.enabled:
        return {"ok": True}
    try:
        # scrypt is deliberately slow: keep it off the trading loop
        ok = await asyncio.to_thread(auth.check_password, body.password)
    except PermissionError as e:
        raise HTTPException(429, str(e))
    if not ok:
        logger.warning("dashboard login failed from {}", request.client.host if request.client else "?")
        raise HTTPException(401, "Wrong password.")
    resp = JSONResponse({"ok": True})
    https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "") == "https"
    resp.set_cookie(COOKIE, auth.new_session(), max_age=int(SESSION_HOURS * 3600), httponly=True,
                    samesite="strict", secure=https, path="/")
    return resp


@app.post("/api/logout")
async def api_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE, path="/")
    return resp


@app.get("/api/session")
async def api_session():
    return {"auth": auth.enabled, "autostart": AUTOSTART}


# ------------------------------------------------------------- engine api
def _remember_running(wanted: bool) -> None:
    s = get_settings()
    if s.run_wanted != wanted:
        s.run_wanted = wanted
        set_settings(s)


async def _start_engine() -> None:
    global engine
    async with _engine_lock:
        if engine and engine.running:
            return
        engine = Engine(get_settings())
        try:
            await engine.start()
        except Exception as e:  # noqa: BLE001
            engine.status, engine.error = "stopped", str(e)
            raise


@app.post("/api/start")
async def api_start():
    try:
        await _start_engine()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, str(e))
    _remember_running(True)
    return {"ok": True}


@app.post("/api/stop")
async def api_stop():
    _remember_running(False)          # a bot you stopped stays stopped across restarts
    async with _engine_lock:
        if engine and engine.running:
            await engine.stop()
    return {"ok": True}


@app.post("/api/kill")
async def api_kill():
    """Kill switch: cancel every resting order and block new ones."""
    if engine and engine.running:
        engine.engage_kill("manual kill switch")
        await engine.om.cancel_everything()
        engine.log("KILL SWITCH engaged - all orders cancelled, trading blocked", "warn")
    return {"ok": True}


@app.post("/api/unkill")
async def api_unkill():
    if engine:
        engine.release_kill()
        engine.log("Kill switch released")
    return {"ok": True}


@app.post("/api/cancel_all")
async def api_cancel_all():
    if engine and engine.running:
        await engine.om.cancel_everything()
        engine.log("All resting orders cancelled")
    return {"ok": True}


@app.post("/api/reset_ledger")
async def api_reset_ledger():
    if engine and engine.running:
        raise HTTPException(400, "Stop the bot before resetting history")
    s = get_settings()
    Ledger(DB_PATH).reset(s.env)
    return {"ok": True}


# ---------------------------------------------------------------- settings
class SettingsIn(BaseModel):
    env: str
    api_key_id: str = ""               # key id for the exchange `env` logs in to
    private_key_pem: str = ""          # optional: paste a new key for that exchange
    private_key_path: str = ""
    unlock_live: bool = False
    unlock_phrase: str = ""
    strategy: dict = {}
    risk: dict = {}
    shard_topup: bool = True
    paper_start: float | None = None


def _key_info(s: Settings, exchange: str) -> dict:
    kid, path = s.creds(exchange)
    out = {"key_id": kid, "has_key": s.has_creds(exchange), "key_type": None}
    if out["has_key"]:
        try:
            from kalshi_bot.api.auth import load_private_key
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            k = load_private_key(path)
            out["key_type"] = "Ed25519" if isinstance(k, Ed25519PrivateKey) else "RSA"
        except Exception:  # noqa: BLE001
            out["key_type"] = "invalid"
    return out


@app.get("/api/settings")
async def get_settings_api():
    s = get_settings()
    d = asdict(s)
    for k in ("private_key_path", "live_private_key_path"):      # paths stay on this machine
        d.pop(k, None)
    d["keys"] = {"demo": _key_info(s, "demo"), "live": _key_info(s, "live")}
    d["paper_login"] = s.trading_env if s.env == "paper" else ("demo" if s.has_creds("demo") else "live")
    cur = d["keys"]["live" if s.env == "live" else d["paper_login"] if s.env == "paper" else "demo"]
    d["has_key"], d["key_type"] = cur["has_key"], cur["key_type"]
    from kalshi_bot.config import ASSETS
    d["asset_choices"] = list(ASSETS)
    return _clean(d)


@app.post("/api/settings")
async def post_settings(body: SettingsIn):
    if engine and engine.running:
        raise HTTPException(400, "Stop the bot before changing settings")
    s = get_settings()
    if body.env not in ("demo", "paper", "live"):
        raise HTTPException(400, "env must be demo, paper or live")
    if body.env == "live" and not (s.live_unlocked or (body.unlock_live and
                                                       body.unlock_phrase.strip().upper() == "TRADE REAL MONEY")):
        raise HTTPException(400, "Live trading is locked. Type TRADE REAL MONEY to unlock it.")
    if body.env == "live":
        s.live_unlocked = True
    # which exchange's key the form is editing: live in live mode, otherwise demo
    # (paper only needs a login for the price feed and falls back to the live key)
    exch = "live" if body.env == "live" else "demo"
    if body.env == "paper" and not s.has_creds("demo") and s.has_creds("live") and not body.private_key_pem.strip():
        exch = "live"
    kid_attr, path_attr = (("live_api_key_id", "live_private_key_path") if exch == "live"
                           else ("api_key_id", "private_key_path"))
    s.env = body.env
    if body.api_key_id.strip():
        setattr(s, kid_attr, body.api_key_id.strip())
    if body.private_key_pem.strip():
        pem = body.private_key_pem.strip()
        from kalshi_bot.api.auth import load_private_key
        try:
            load_private_key(pem)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"That private key could not be parsed: {e}")
        SECRETS.mkdir(exist_ok=True)
        path = SECRETS / f"kalshi_{exch}.key"
        path.write_text(pem + chr(10), encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        setattr(s, path_attr, str(path))
    elif body.private_key_path.strip():
        setattr(s, path_attr, body.private_key_path.strip())
    sp = asdict(s.strategy)
    sp.update({k: v for k, v in body.strategy.items() if k in sp})
    from kalshi_bot.config import parse_assets
    sp["assets"] = ",".join(a.symbol for a in parse_assets(sp.get("assets", "BTC")))
    s.strategy = StrategyParams(**sp)
    rp = asdict(s.risk)
    rp.update({k: v for k, v in body.risk.items() if k in rp})
    s.risk = RiskParams(**rp)
    s.shard_topup = body.shard_topup
    if body.paper_start is not None and body.paper_start > 0:
        s.paper_start = float(body.paper_start)
    set_settings(s)
    return {"ok": True}


@app.post("/api/test_connection")
async def test_connection(exchange: str | None = None):
    s = get_settings()
    exch = exchange if exchange in ("demo", "live") else s.trading_env
    from kalshi_bot.api.auth import Signer
    from kalshi_bot.api.rest import KalshiREST, KalshiError, fnum
    from kalshi_bot.config import ENDPOINTS
    kid, path = s.creds(exch)
    if not kid or not path:
        raise HTTPException(400, f"No {exch} API key saved yet.")
    try:
        rest = KalshiREST(ENDPOINTS[exch]["rest"], Signer(kid, path))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Key problem: {e}")
    try:
        b = await rest.balance()
        lim = await rest.account_limits()
    except KalshiError as e:
        raise HTTPException(400, f"Kalshi said {e.status}: {e.message}")
    finally:
        await rest.close()
    return {"ok": True, "env": exch, "balance": fnum(b.get("balance_dollars"), fnum(b.get("balance")) / 100),
            "shards": b.get("balance_breakdown", []), "tier": lim.get("usage_tier")}


@app.get("/api/preflight")
async def preflight():
    """Go/no-go checklist for the selected environment. Read-only: logs in,
    reads the balance and listens to the price feed; never places an order."""
    from kalshi_bot.preflight import run_preflight
    return _clean(await run_preflight(get_settings(), running=bool(engine and engine.running)))


# ---------------------------------------------------------------- reports
@app.get("/api/backtests")
async def backtest_list():
    out = []
    for p in sorted(REPORTS.glob("backtest*.json")):
        series = "KXBTC15M" if p.stem == "backtest" else p.stem.split("_", 1)[1].upper()
        if series.startswith("KX"):
            out.append(series)
    return {"series": out}


@app.get("/api/backtest")
async def backtest_report(series: str = "KXBTC15M"):
    series = series.upper()
    name = "backtest.json" if series == "KXBTC15M" else f"backtest_{series.lower()}.json"
    p = REPORTS / name
    if not p.exists() or not p.resolve().is_relative_to(REPORTS.resolve()):
        return {"available": False}
    rep = json.loads(p.read_text(encoding="utf-8"))
    acct = REPORTS / "account.json"
    if series == "KXBTC15M" and acct.exists():          # whole-book small-account simulation
        try:
            rep["account_sets"] = json.loads(acct.read_text(encoding="utf-8")).get("sets", [])
        except ValueError:
            pass
    return JSONResponse(rep)


@app.get("/api/stats")
async def stats():
    return current_state().get("stats", {})


# -------------------------------------------------------------- websocket
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    auth.maybe_reload()
    host = ws.headers.get("host", "")
    if (not _host_allowed(host) or not same_origin(ws.headers.get("origin"), host)
            or (auth.enabled and not auth.valid_session(ws.cookies.get(COOKIE)))):
        await ws.close(code=4401)
        return
    await ws.accept()
    try:
        while True:
            try:
                payload = _clean(current_state())
                await ws.send_text(_json(payload))
            except WebSocketDisconnect:
                break
            except Exception as e:  # noqa: BLE001
                logger.warning("ws push failed: {}", e)
            if auth.enabled and not auth.valid_session(ws.cookies.get(COOKIE)):
                await ws.close(code=4401)          # session expired or password changed
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


@app.on_event("startup")
async def _quiet_windows_resets():
    """Windows' proactor loop logs a traceback whenever a peer drops a socket
    (e.g. after the PC wakes from sleep). The bot reconnects on its own, so
    route those to the debug log instead of alarming the console."""
    loop = asyncio.get_running_loop()
    default = loop.get_exception_handler()

    def handler(lp, context):
        exc = context.get("exception")
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
            logger.debug("socket reset: {}", exc)
            return
        if default is not None:
            default(lp, context)
        else:
            lp.default_exception_handler(context)

    loop.set_exception_handler(handler)


@app.on_event("startup")
async def _resume_after_restart():
    """With --autostart, a bot that was running when the program stopped
    (reboot, crash, update) starts again by itself. A bot you stopped by hand
    stays stopped, and an engaged kill switch stays engaged."""
    if not (AUTOSTART and get_settings().run_wanted):
        return

    async def resume():
        delay = 5.0
        while get_settings().run_wanted and not (engine and engine.running):
            try:
                await _start_engine()
                logger.info("autostart: bot resumed in {} mode", get_settings().env.upper())
                return
            except Exception as e:  # noqa: BLE001 - network not up yet, Kalshi maintenance, ...
                logger.warning("autostart failed ({}); retrying in {:.0f}s", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 300.0)

    asyncio.create_task(resume(), name="autostart")


@app.on_event("shutdown")
async def _shutdown():
    if engine and engine.running:
        await engine.stop("server shutdown")


def main(host: str = "127.0.0.1", port: int = 8000, autostart: bool = False) -> None:
    import sys
    import uvicorn
    from kalshi_bot.config import LOG_DIR
    global AUTOSTART
    AUTOSTART = AUTOSTART or autostart
    if not is_loopback(host) and not auth.enabled:
        sys.exit(f"Refusing to listen on {host}: the dashboard has no password, and anyone who can reach that "
                 f"address could control the bot.\nSet one first:  python -m kalshi_bot --set-password\n"
                 f"(or keep the default 127.0.0.1 and reach it through an SSH tunnel)")
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<dim>{time:HH:mm:ss}</dim> {message}")
    logger.add(LOG_DIR / "bot_{time:YYYY-MM-DD}.log", rotation="1 day", retention="14 days", level="DEBUG")
    lock = "password required" if auth.enabled else "no password - this machine only"
    print(f"\n  Lock-In dashboard  ->  http://{host}:{port}   ({lock}"
          f"{', resumes the bot after restarts' if AUTOSTART else ''})\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
