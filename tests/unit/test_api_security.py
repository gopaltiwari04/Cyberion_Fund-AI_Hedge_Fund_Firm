import json
import time

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.ratelimit import LIMITS, limiter
from api.schemas import BacktestRequest, HoldingsUpdate, ProfileUpdate
from api.security import User, require_user, sign
from common import config

SECRET = "test-secret-value"


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_SECRET", SECRET)
    config.get_settings.cache_clear()
    application = FastAPI()

    @application.post("/echo")
    def echo(payload: dict, user: User = Depends(require_user)):
        return {"email": user.email, "payload": payload}

    yield application
    config.get_settings.cache_clear()


def signed_headers(body: bytes, path="/echo", email="analyst@example.com", ts=None, secret=SECRET):
    ts = str(int(ts if ts is not None else time.time()))
    return {"content-type": "application/json", "x-user-email": email, "x-auth-timestamp": ts,
            "x-auth-signature": sign(secret, ts, "POST", path, email, body)}


def test_valid_signature_accepted(app):
    body = json.dumps({"a": 1}).encode()
    response = TestClient(app).post("/echo", content=body, headers=signed_headers(body))
    assert response.status_code == 200 and response.json()["email"] == "analyst@example.com"


@pytest.mark.parametrize("mutate", ["body", "email", "secret", "expired", "missing"])
def test_tampered_requests_rejected(app, mutate):
    body = json.dumps({"a": 1}).encode()
    headers = signed_headers(body)
    if mutate == "body":
        body = json.dumps({"a": 2}).encode()
    elif mutate == "email":
        headers["x-user-email"] = "attacker@example.com"
    elif mutate == "secret":
        headers = signed_headers(body, secret="wrong")
    elif mutate == "expired":
        headers = signed_headers(body, ts=time.time() - 3600)
    elif mutate == "missing":
        headers.pop("x-auth-signature")
    assert TestClient(app).post("/echo", content=body, headers=headers).status_code == 401


def test_production_without_secret_disables_user_endpoints(monkeypatch):
    monkeypatch.delenv("INTERNAL_API_SECRET", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    config.get_settings.cache_clear()
    application = FastAPI()

    @application.get("/me")
    def me(user: User = Depends(require_user)):
        return {"email": user.email}

    try:
        assert TestClient(application).get("/me").status_code == 503
    finally:
        config.get_settings.cache_clear()


def test_rate_limiter(monkeypatch):
    monkeypatch.setitem(LIMITS, "test", (2, 60))
    application = FastAPI()

    @application.get("/limited", dependencies=[Depends(limiter("test"))])
    def limited():
        return {"ok": True}

    client = TestClient(application)
    assert [client.get("/limited").status_code for _ in range(3)] == [200, 200, 429]


def test_schema_validation():
    with pytest.raises(ValidationError):
        HoldingsUpdate(holdings=[{"ticker": "FAKE", "quantity": 1}])
    with pytest.raises(ValidationError):
        HoldingsUpdate(holdings=[{"ticker": "AAPL", "quantity": 1}, {"ticker": "aapl", "quantity": 2}])
    with pytest.raises(ValidationError):
        HoldingsUpdate(holdings=[{"ticker": "AAPL", "quantity": -1}])
    with pytest.raises(ValidationError):
        ProfileUpdate(cvar_limit=0.5)
    with pytest.raises(ValidationError):
        ProfileUpdate(unexpected=True)
    with pytest.raises(ValidationError):
        BacktestRequest(start="2024-01-01", end="2024-02-01")
    assert HoldingsUpdate(holdings=[{"ticker": "aapl", "quantity": 3}]).holdings[0].ticker == "AAPL"
