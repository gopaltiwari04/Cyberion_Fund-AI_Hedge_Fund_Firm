"""
End-to-end integration test on a disposable PostgreSQL database.

migrations (from empty) -> synthetic prices (test fixture, seeded) -> feature pipeline
-> walk-forward training + registry -> inference + SHAP -> CVaR house portfolio
-> API (signed + validation) -> recommendation -> backtest (no look-ahead) -> publish

Requires TEST_DATABASE_URL pointing at a database whose name contains "test"
(it is wiped). Skipped otherwise.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

TEST_URL = os.getenv("TEST_DATABASE_URL")
TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "SPY", "TLT"]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not TEST_URL, reason="TEST_DATABASE_URL not set"),
]


def _reset(url: str) -> None:
    name = url.rsplit("/", 1)[-1]
    assert "test" in name, "refusing to wipe a database whose name does not contain 'test'"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    engine.dispose()


def _synthetic_prices(days: int = 1100, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=days)
    market = rng.normal(0.0004, 0.01, days)
    rows = []
    for i, ticker in enumerate(TICKERS):
        beta = 0.3 if ticker == "TLT" else 1.0 + 0.1 * i
        returns = beta * market + rng.normal(0.0001 * i, 0.012, days)
        close = 100 * np.cumprod(1 + returns)
        open_ = close / (1 + rng.normal(0, 0.003, days))
        high = np.maximum(open_, close) * 1.004
        low = np.minimum(open_, close) * 0.996
        dividend = np.where((np.arange(days) % 63 == 30) & (ticker in ("JPM", "XOM", "SPY", "TLT")), 0.4, 0.0)
        rows.append(pd.DataFrame({"ticker": ticker, "date": dates.date, "open": open_, "high": high, "low": low,
                                  "close": close, "adj_close": close, "volume": rng.integers(1e6, 5e6, days),
                                  "dividend": dividend, "split_ratio": 1.0, "source": "synthetic-test"}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def environment(tmp_path_factory):
    prod_url = TEST_URL.rsplit("/", 1)[0] + "/" + TEST_URL.rsplit("/", 1)[1] + "_publish"
    admin = create_engine(TEST_URL.rsplit("/", 1)[0] + "/postgres", isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        if not conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": prod_url.rsplit("/", 1)[1]}).scalar():
            conn.execute(text(f'CREATE DATABASE "{prod_url.rsplit("/", 1)[1]}"'))
    admin.dispose()
    _reset(TEST_URL)
    _reset(prod_url)

    saved = dict(os.environ)
    os.environ.update({
        "DATABASE_URL": TEST_URL, "UNIVERSE_TICKERS": ",".join(TICKERS), "STORAGE_BACKEND": "local",
        "LOCAL_LAKE_DIR": str(tmp_path_factory.mktemp("lake")), "ENVIRONMENT": "development",
        "INTERNAL_API_SECRET": "integration-secret", "GEMINI_API_KEY": "", "RAG_EMBEDDINGS": "off",
        "FORECAST_MIN_TRAIN_DAYS": "400", "FORECAST_REFIT_EVERY": "126",
    })
    from common import config, storage

    config.get_settings.cache_clear()
    storage.get_object_store.cache_clear()
    yield {"prod_url": prod_url}
    os.environ.clear()
    os.environ.update(saved)
    config.get_settings.cache_clear()
    storage.get_object_store.cache_clear()


def test_full_platform_flow(environment, monkeypatch):
    from api.main import run_migrations
    from common.db import get_engine

    # 1. migrations from an empty database
    run_migrations()
    engine = get_engine()
    with engine.connect() as conn:
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == "5f0c1d2e3a4b"

    # 2. synthetic market data (fixture) through the real loader
    from data_pipeline.clean_and_load import upsert_asset_metadata

    prices = _synthetic_prices()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO market_data (ticker, date, open, high, low, close, adj_close, volume, dividend, split_ratio, source) "
                "VALUES (:ticker, :date, :open, :high, :low, :close, :adj_close, :volume, :dividend, :split_ratio, :source)"
            ),
            prices.astype(object).to_dict(orient="records"),
        )
    upsert_asset_metadata()

    # 3. feature pipeline (technical + external signals + graph)
    from ml_core.feature_store import engineer_features

    monkeypatch.setattr(engineer_features, "MARKET_TICKERS", TICKERS)
    engineer_features.run_pipeline()
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM feature_store")).scalar() == len(prices)
        assert conn.execute(text("SELECT COUNT(corr_eigenvector_60d) FROM feature_store")).scalar() > 0

    # 4. walk-forward training + registry (small model for speed)
    from ml_core.models import forecasting

    monkeypatch.setattr(forecasting, "ROUNDS", {"mean": 30, "quantile": 30, "vol": 30})
    monkeypatch.setattr(forecasting, "MIN_TRAIN_DAYS", 400)
    monkeypatch.setattr(forecasting, "REFIT_EVERY", 126)
    frame = forecasting.load_frame(TICKERS)
    summary = forecasting.train_horizon(frame, 5, "integration")
    assert summary["observations"] > 1000
    if not summary["active"]:
        from ml_core.models import registry

        registry.promote(summary["model_id"])  # synthetic data may miss coverage gates; flow must still work

    with engine.connect() as conn:
        violations = conn.execute(
            text(
                "SELECT COUNT(*) FROM model_predictions WHERE kind = 'walk_forward' "
                "AND train_end + 10 > as_of_date"  # train_end must precede as_of by > horizon + embargo days
            )
        ).scalar()
    assert violations == 0

    # 5. inference with SHAP explanations
    from ml_core.models.inference import run_inference

    result = run_inference(frame)
    assert result["horizons"][5]["predictions"] == len(TICKERS)
    with engine.connect() as conn:
        explanation = conn.execute(text("SELECT explanation FROM model_predictions WHERE kind = 'live' LIMIT 1")).scalar()
    assert explanation["top_contributions"] and "base_value" in explanation

    # 6. house portfolio with CVaR constraint
    from ml_core.portfolio.portfolio_optimizer import run_optimizer

    run_optimizer()
    with engine.connect() as conn:
        risk = conn.execute(text("SELECT feasible, cvar_95, cvar_limit FROM portfolio_risk_metrics ORDER BY id DESC LIMIT 1")).one()
    assert risk.feasible and risk.cvar_95 <= risk.cvar_limit * 1.05  # empirical CVaR vs RU-constrained limit

    # 7. API: auth, validation, recommendation, research
    from fastapi.testclient import TestClient

    from api.main import app
    from api.security import sign

    client = TestClient(app)

    def call(method: str, path: str, body: dict | None = None):
        payload = json.dumps(body).encode() if body is not None else b""
        ts = str(int(time.time()))
        headers = {"X-User-Email": "tester@example.com", "X-Auth-Timestamp": ts, "Content-Type": "application/json",
                   "X-Auth-Signature": sign("integration-secret", ts, method, path, "tester@example.com", payload)}
        return client.request(method, path, content=payload or None, headers=headers)

    assert client.get("/health").json()["status"] == "healthy"
    assert client.get("/markets").status_code == 401  # unsigned
    assert call("GET", "/markets").status_code == 200
    assert call("GET", "/portfolio").json()["strategy"] == "xgb_cvar_v2"
    assert call("GET", "/research/model?horizon=5").status_code == 200
    assert call("GET", "/markets/AAPL/forecast").json()["forecasts"][0]["explanation"]["top_contributions"]
    assert call("PUT", "/me/holdings", {"holdings": [{"ticker": "AAPL", "quantity": 10}], "cash_balance": 5000}).status_code == 200
    assert call("PUT", "/me/holdings", {"holdings": [{"ticker": "ZZZ", "quantity": 1}]}).status_code == 422
    rec = call("POST", "/me/recommendations", {"horizon_days": 5}).json()
    assert rec["status"] in ("optimal", "infeasible")
    if rec["feasible"]:
        assert rec["proposal"]["metrics"]["cvar_95_1d_ru"] <= rec["constraints"]["cvar_limit"] * 1.001 + 1e-6
        assert rec["scenarios"]["proposed"]["stress_tests"]
    status = call("GET", "/system/status").json()
    assert {d["dataset"] for d in status["datasets"]} >= {"market_prices", "forecasts"}

    # 8. backtest: event-driven, out-of-sample only
    from ml_core.backtest.engine import BacktestConfig, run_backtest

    with engine.connect() as conn:
        first_pred = conn.execute(text("SELECT MIN(as_of_date) FROM model_predictions WHERE kind='walk_forward'")).scalar()
    backtest = run_backtest(BacktestConfig(start=first_pred, end=prices["date"].max(), tickers=TICKERS,
                                           scenario_window=252, rebalance_every=10))
    metrics = backtest["metrics"]["ml_cvar"]
    assert metrics["trades"] > 0 and np.isfinite(metrics["sharpe"])
    assert set(backtest["metrics"]) == {"ml_cvar", "spy_buy_and_hold", "equal_weight"}
    assert run_backtest(BacktestConfig(start=first_pred, end=prices["date"].max(), tickers=TICKERS,
                                       scenario_window=252, rebalance_every=10))["metrics"]["ml_cvar"] == metrics  # reproducible

    # 9. publish serving tables to a second database
    from data_pipeline.publish import publish

    published = publish(environment["prod_url"])
    assert published["market_data"] == len(prices)
    prod = create_engine(environment["prod_url"])
    with prod.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM model_predictions WHERE kind = 'live'")).scalar() == len(TICKERS)
        assert conn.execute(text("SELECT COUNT(*) FROM model_registry WHERE artifact IS NOT NULL")).scalar() == 0
    again = publish(environment["prod_url"])  # incremental: nothing new to copy
    assert again["market_data"] == 0
    prod.dispose()
    assert date.today()  # sanity
