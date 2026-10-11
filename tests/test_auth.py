"""Dashboard login, cross-site protection, and the state that must survive a
restart on an unattended server (kill switch, whether the bot was running)."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import kalshi_bot.config as config
from kalshi_bot.config import Settings
from kalshi_bot.web.auth import COOKIE, Auth, same_origin

GOOD = "correct horse battery"          # test-only value, never used outside this file
LOCAL = "http://127.0.0.1:8000"


# ------------------------------------------------------------------ Auth
def test_password_is_hashed_and_sessions_are_signed(tmp_path):
    a = Auth(tmp_path / "auth.json")
    assert not a.enabled and not a.valid_session("anything")
    with pytest.raises(ValueError):
        a.set_password("short")
    a.set_password(GOOD)
    assert a.enabled
    assert GOOD not in (tmp_path / "auth.json").read_text(encoding="utf-8")       # only a salted hash on disk
    assert a.check_password(GOOD) and not a.check_password(GOOD + "x")
    tok = a.new_session()
    assert a.valid_session(tok)
    exp, nonce, sig = tok.split(".")
    assert not a.valid_session(f"{int(exp) + 9999}.{nonce}.{sig}")               # tampered expiry
    assert not a.valid_session(a.new_session(hours=-1))                          # expired
    assert Auth(tmp_path / "auth.json").valid_session(tok)                       # survives a restart
    a.set_password(GOOD + "!")                                                   # new password = new signing key
    assert not a.valid_session(tok)


def test_wrong_passwords_lock_the_login(tmp_path):
    a = Auth(tmp_path / "auth.json")
    a.set_password(GOOD)
    for _ in range(5):
        assert not a.check_password("nope-nope-nope")
    with pytest.raises(PermissionError):
        a.check_password(GOOD)                    # even the right password waits out the lock
    assert 0 < a.locked_for() <= 60
    a._locked_until = time.time() - 1
    assert a.check_password(GOOD)


def test_origin_check():
    assert same_origin(None, "127.0.0.1:8000")                                   # curl / same-origin GET
    assert same_origin("http://localhost:8000", "localhost:8000")
    assert same_origin("https://bot.example.com", "bot.example.com")
    assert not same_origin("http://evil.example", "localhost:8000")
    assert not same_origin("http://localhost:9999", "localhost:8000")
    assert not same_origin("null", "localhost:8000")


# ---------------------------------------------------------------- server
@pytest.fixture
def app(tmp_path, monkeypatch):
    import kalshi_bot.web.server as server
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "_settings", Settings(env="demo"))
    monkeypatch.setattr(server, "engine", None)
    a = Auth(tmp_path / "auth.json")
    monkeypatch.setattr(server, "auth", a)
    return server, a


def test_without_a_password_only_this_machine_is_served(app):
    server, _ = app
    assert TestClient(server.app, base_url=LOCAL).get("/api/settings").status_code == 200
    assert TestClient(server.app, base_url="http://localhost:8000").get("/").status_code == 200
    # a made-up domain pointed at 127.0.0.1 (DNS rebinding) or the server's public address
    assert TestClient(server.app, base_url="http://evil.example:8000").get("/api/settings").status_code == 403
    assert TestClient(server.app, base_url="http://203.0.113.7:8000").get("/").status_code == 403
    c = TestClient(server.app, base_url=LOCAL)
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/ws", headers={"host": "evil.example:8000"}):
            pass
    with c.websocket_connect("/ws", headers={"host": "127.0.0.1:8000", "origin": LOCAL}) as ws:
        assert ws.receive_json()["status"] == "stopped"


def test_other_sites_cannot_press_buttons(app):
    server, _ = app
    c = TestClient(server.app, base_url=LOCAL)
    r = c.post("/api/stop", headers={"Origin": "http://evil.example"})
    assert r.status_code == 403 and "Cross-site" in r.json()["detail"]
    assert c.post("/api/stop", headers={"Origin": LOCAL}).status_code == 200
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/ws", headers={"host": "127.0.0.1:8000", "origin": "http://evil.example"}):
            pass


def test_login_flow_protects_pages_api_and_websocket(app):
    server, a = app
    a.set_password(GOOD)
    c = TestClient(server.app, base_url="http://203.0.113.7:8000", follow_redirects=False)
    r = c.get("/")
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert c.get("/api/settings").status_code == 401
    assert c.post("/api/start").status_code == 401
    assert c.get("/api/preflight").status_code == 401
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/ws"):
            pass
    page = c.get("/login")
    assert page.status_code == 200 and "password" in page.text.lower()
    assert page.headers["x-frame-options"] == "DENY" and "frame-ancestors 'none'" in page.headers["content-security-policy"]

    assert c.post("/api/login", json={"password": "wrong-wrong-wrong"}).status_code == 401
    r = c.post("/api/login", json={"password": GOOD})
    assert r.status_code == 200
    cookie = r.headers["set-cookie"].lower()
    assert COOKIE in cookie and "httponly" in cookie and "samesite=strict" in cookie
    assert c.get("/").status_code == 200
    assert c.get("/api/settings").status_code == 200
    assert c.get("/api/session").json()["auth"] is True
    session = {"cookie": f"{COOKIE}={c.cookies.get(COOKIE)}", "host": "203.0.113.7:8000"}
    with c.websocket_connect("/ws", headers={**session, "origin": "http://203.0.113.7:8000"}) as ws:
        assert ws.receive_json()["status"] == "stopped"
    with pytest.raises(WebSocketDisconnect):                     # a valid session used from another site
        with c.websocket_connect("/ws", headers={**session, "origin": "http://evil.example"}):
            pass
    with pytest.raises(WebSocketDisconnect):                     # a forged cookie
        with c.websocket_connect("/ws", headers={"cookie": f"{COOKIE}=1999999999.abcd.AAAA", "host": "203.0.113.7:8000"}):
            pass
    # still no cross-site writes, even with a valid session
    assert c.post("/api/stop", headers={"Origin": "http://evil.example"}).status_code == 403

    assert c.post("/api/logout").status_code == 200
    assert c.get("/api/settings").status_code == 401


def test_login_cookie_is_secure_behind_https(app):
    server, a = app
    a.set_password(GOOD)
    c = TestClient(server.app, base_url="https://bot.example.com")
    r = c.post("/api/login", json={"password": GOOD})
    assert r.status_code == 200 and "secure" in r.headers["set-cookie"].lower()


def test_login_locks_after_repeated_failures(app):
    server, a = app
    a.set_password(GOOD)
    c = TestClient(server.app, base_url=LOCAL)
    for _ in range(5):
        assert c.post("/api/login", json={"password": "wrong-wrong-wrong"}).status_code == 401
    r = c.post("/api/login", json={"password": GOOD})
    assert r.status_code == 429 and "Too many" in r.json()["detail"]


def test_refuses_to_listen_on_the_network_without_a_password(app):
    server, a = app
    with pytest.raises(SystemExit) as e:
        server.main("0.0.0.0", 8000)
    assert "password" in str(e.value)


def test_stop_is_remembered_and_start_failure_is_not(app):
    server, _ = app
    c = TestClient(server.app, base_url=LOCAL)
    config.get_settings().run_wanted = True
    assert c.post("/api/stop").status_code == 200
    assert config.get_settings().run_wanted is False            # a bot stopped by hand stays stopped after a reboot
    r = c.post("/api/start")                                     # no API key configured -> cannot start
    assert r.status_code == 400 and config.get_settings().run_wanted is False


def test_manual_kill_survives_a_restart(tmp_path, monkeypatch):
    import kalshi_bot.engine as eng
    monkeypatch.setattr(eng, "DB_PATH", tmp_path / "e.sqlite3")
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    s = Settings(env="paper", api_key_id="x", private_key_path="x")
    monkeypatch.setattr(config, "_settings", s)
    e = eng.Engine(s)
    e.engage_kill("manual kill switch")
    assert e.risk.killed
    saved = Settings.load()                                      # what a restarted program reads
    assert saved.kill_state == {"paper": "manual kill switch"}
    e.release_kill()
    assert not e.risk.killed and Settings.load().kill_state == {}
