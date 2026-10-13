"""Health checks, data freshness, pipeline runs, model registry and monitoring."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.deps import get_db, jsonable
from api.security import optional_user
from common.market_time import business_days_between, last_completed_session

router = APIRouter(tags=["system"])

FRESHNESS_RULES = [
    # name, SQL returning a date/timestamp, max business-day lag, description
    ("market_prices", "SELECT MAX(date) FROM market_data", 2, "Daily OHLCV + corporate actions (yfinance)"),
    ("features", "SELECT MAX(date) FROM feature_store", 2, "Point-in-time feature store"),
    ("macro", "SELECT MAX(date) FROM macro_data WHERE series_id = 'DFF'", 5, "FRED macro series"),
    ("news", "SELECT MAX(published_at) FROM financial_news", 2, "Marketaux news + FinBERT sentiment"),
    ("sec_filings", "SELECT MAX(filing_date) FROM sec_filings", 10, "SEC EDGAR 10-K / 10-Q / 8-K"),
    ("forecasts", "SELECT MAX(as_of_date) FROM model_predictions WHERE kind = 'live'", 2, "Live model forecasts"),
    ("house_portfolio", "SELECT MAX(date) FROM portfolio_risk_metrics WHERE strategy_name = 'xgb_cvar_v2'", 2,
     "CVaR-constrained model portfolio"),
    ("model_training", "SELECT MAX(trained_at) FROM model_registry WHERE status = 'active'", 10, "Active model training date"),
]


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Liveness + database connectivity (public; used by Render health checks)."""
    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Database unavailable") from None
    return {"status": "healthy", "database": "connected"}


def _as_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date() if value.tzinfo else value.date()
    return value


def freshness(db: Session) -> list[dict]:
    session = last_completed_session()
    results = []
    for name, sql, max_lag, description in FRESHNESS_RULES:
        try:
            value = db.execute(text(sql)).scalar()
        except SQLAlchemyError:
            db.rollback()
            value = None
        latest = _as_date(value)
        lag = business_days_between(latest, session) if latest else None
        status = "missing" if latest is None else "fresh" if lag <= max_lag else "stale"
        results.append({"dataset": name, "description": description, "latest": jsonable(value),
                        "business_days_behind": lag, "max_allowed_lag": max_lag, "status": status})
    return results


@router.get("/health/ready")
def readiness(db: Session = Depends(get_db)):
    """Readiness: DB reachable, schema migrated, core datasets present."""
    try:
        version = db.execute(text("SELECT version_num FROM alembic_version")).scalar()
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Database unavailable or not migrated") from None
    checks = {item["dataset"]: item["status"] for item in freshness(db)}
    ready = checks.get("market_prices") != "missing"
    return {"status": "ready" if ready else "degraded", "schema_version": version, "datasets": checks}


@router.get("/system/status", dependencies=[Depends(optional_user)])
def system_status(db: Session = Depends(get_db)):
    data = freshness(db)
    runs = db.execute(
        text(
            """
            SELECT DISTINCT ON (pipeline, task) pipeline, task, status, started_at, finished_at, rows_processed,
                   error, details->'warnings' AS warnings, details->>'duration_seconds' AS duration
            FROM pipeline_runs ORDER BY pipeline, task, started_at DESC
            """
        )
    ).mappings().all()
    failures = db.execute(
        text("SELECT COUNT(*) FROM pipeline_runs WHERE status = 'failed' AND started_at > now() - interval '24 hours'")
    ).scalar()
    filings = db.execute(text("SELECT status, COUNT(*) FROM sec_filings GROUP BY status")).all()
    chunks = db.execute(
        text("SELECT COUNT(*), COUNT(*) FILTER (WHERE embedding_model IS NOT NULL) FROM sec_filing_chunks")
    ).one()
    overall = "healthy"
    if any(item["status"] != "fresh" for item in data) or failures:
        overall = "degraded"
    if any(item["status"] == "missing" for item in data if item["dataset"] in ("market_prices", "forecasts")):
        overall = "unhealthy"
    return {
        "status": overall,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "last_completed_session": str(last_completed_session()),
        "datasets": data,
        "pipelines": [jsonable(dict(row)) for row in runs],
        "failed_runs_24h": failures,
        "sec_index": {"filings_by_status": {status: count for status, count in filings},
                      "chunks": chunks[0], "embedded_chunks": chunks[1]},
    }


@router.get("/system/pipelines", dependencies=[Depends(optional_user)])
def pipeline_history(db: Session = Depends(get_db), limit: int = 100):
    limit = max(1, min(limit, 500))
    rows = db.execute(
        text(
            """
            SELECT id, pipeline, task, run_id, status, started_at, finished_at, rows_processed, error,
                   details->'warnings' AS warnings
            FROM pipeline_runs ORDER BY started_at DESC LIMIT :limit
            """
        ),
        {"limit": limit},
    ).mappings().all()
    return {"runs": [jsonable(dict(row)) for row in rows]}


@router.get("/system/models", dependencies=[Depends(optional_user)])
def model_registry(db: Session = Depends(get_db)):
    rows = db.execute(
        text(
            """
            SELECT id, model_name, version, algorithm, horizon_days, target, status, trained_at, train_start,
                   train_end, data_end, metrics, calibration, feature_importance, data_hash, code_version, notes,
                   jsonb_array_length(feature_columns) AS feature_count
            FROM model_registry ORDER BY trained_at DESC LIMIT 50
            """
        )
    ).mappings().all()
    return {"models": [jsonable(dict(row)) for row in rows]}


@router.get("/system/monitoring", dependencies=[Depends(optional_user)])
def monitoring(db: Session = Depends(get_db), days: int = 120):
    since = date.today() - timedelta(days=max(7, min(days, 730)))
    rows = db.execute(
        text(
            """
            SELECT m.as_of_date, m.metric, m.value, m.details, r.horizon_days, r.version
            FROM model_monitoring m JOIN model_registry r ON r.id = m.model_id
            WHERE m.as_of_date >= :since ORDER BY m.as_of_date
            """
        ),
        {"since": since},
    ).mappings().all()
    return {"since": str(since), "metrics": [jsonable(dict(row)) for row in rows]}
