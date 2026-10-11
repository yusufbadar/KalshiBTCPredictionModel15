"""Settings API: the live lock, per-exchange key storage and the settings migration.
No network: only the settings endpoints are exercised, against temporary paths."""
from __future__ import annotations

import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

import kalshi_bot.config as config
from kalshi_bot.config import Settings


def _pem() -> str:
    k = Ed25519PrivateKey.generate()
    return k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                           serialization.NoEncryption()).decode()


@pytest.fixture
def client(tmp_path, monkeypatch):
    import kalshi_bot.web.server as server
    demo_key = tmp_path / "demo.key"
    demo_key.write_text(_pem(), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "_settings", Settings(env="demo", api_key_id="demo-id",
                                                       private_key_path=str(demo_key)))
    monkeypatch.setattr(server, "SECRETS", tmp_path / "secrets")
    monkeypatch.setattr(server, "engine", None)
    from kalshi_bot.web.auth import Auth
    monkeypatch.setattr(server, "auth", Auth(tmp_path / "auth.json"))          # no password set
    return TestClient(server.app, base_url="http://127.0.0.1:8000"), tmp_path, demo_key


def test_live_needs_the_phrase_and_keeps_keys_apart(client):
    c, tmp, demo_key = client
    body = {"env": "live", "api_key_id": "prod-id", "private_key_pem": _pem(), "strategy": {}, "risk": {}}
    r = c.post("/api/settings", json=body)
    assert r.status_code == 400 and "locked" in r.json()["detail"].lower()
    assert config.get_settings().env == "demo" and config.get_settings().live_api_key_id == ""

    r = c.post("/api/settings", json={**body, "unlock_live": True, "unlock_phrase": "trade real money"})
    assert r.status_code == 200
    s = config.get_settings()
    assert s.env == "live" and s.live_unlocked
    assert s.creds("live")[0] == "prod-id" and s.creds("live")[1].endswith("kalshi_live.key")
    assert s.creds("demo") == ("demo-id", str(demo_key))          # the demo key is untouched
    assert (tmp / "secrets" / "kalshi_live.key").exists() and not (tmp / "secrets" / "kalshi_demo.key").exists()

    g = c.get("/api/settings").json()
    assert g["keys"]["live"]["has_key"] and g["keys"]["live"]["key_type"] == "Ed25519"
    assert g["keys"]["demo"]["key_id"] == "demo-id"
    assert "private_key_path" not in g and "live_private_key_path" not in g
    assert "PRIVATE KEY" not in json.dumps(g)

    # back to demo: the key-id field now refers to the demo key again
    r = c.post("/api/settings", json={"env": "demo", "api_key_id": "demo-id", "strategy": {}, "risk": {}})
    assert r.status_code == 200
    s = config.get_settings()
    assert s.env == "demo" and s.creds() == ("demo-id", str(demo_key)) and s.creds("live")[0] == "prod-id"


def test_settings_reject_bad_input_and_save_limits(client):
    c, tmp, _ = client
    assert c.post("/api/settings", json={"env": "moon", "strategy": {}, "risk": {}}).status_code == 400
    bad = c.post("/api/settings", json={"env": "demo", "private_key_pem": "not a key", "strategy": {}, "risk": {}})
    assert bad.status_code == 400
    ok = c.post("/api/settings", json={"env": "paper", "paper_start": 250,
                                       "strategy": {"assets": "eth,btc,xyz", "min_edge": 0.02, "bogus": 1},
                                       "risk": {"pct_per_market": 0.03, "max_drawdown_pct": 0.2}})
    assert ok.status_code == 200
    s = config.get_settings()
    assert s.env == "paper" and s.paper_start == 250 and s.trading_env == "demo"
    assert s.strategy.assets == "ETH,BTC" and s.strategy.min_edge == 0.02
    assert s.risk.pct_per_market == 0.03 and s.risk.max_drawdown_pct == 0.2
    saved = json.loads((tmp / "settings.json").read_text(encoding="utf-8"))
    assert saved["version"] == config.SETTINGS_VERSION and saved["risk"]["pct_per_market"] == 0.03


def test_first_version_settings_adopt_the_revalidated_thresholds():
    old = {"env": "demo", "strategy": {"min_edge": 0.02, "z_min": 1.5, "max_price": 0.98},
           "risk": {"max_notional_per_market": 100.0, "max_daily_loss": 100.0}}
    s = Settings._from_dict(old)                                   # saved before versions existed
    assert s.strategy.min_edge == 0.01 and s.strategy.sigma_time_k == 0.25
    assert s.strategy.max_seconds_left == 55.0                     # ...and every later re-validated field too
    assert s.strategy.max_price == 0.98                            # everything else the user set is kept
    assert s.risk.max_notional_per_market == 100.0 and s.risk.pct_per_market == 0.06
    assert s.version == config.SETTINGS_VERSION
    mine = Settings._from_dict({**old, "version": config.SETTINGS_VERSION})
    assert mine.strategy.min_edge == 0.02                          # a choice made on this version is respected
