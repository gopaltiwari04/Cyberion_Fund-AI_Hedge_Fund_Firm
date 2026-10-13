"""Backtest jobs: submitted by users, executed in a bounded background worker pool."""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.deps import get_db, jsonable
from api.ratelimit import limiter
from api.schemas import BacktestRequest
from api.security import User, require_user
from common.db import get_engine
from common.logs import get_logger
from ml_core.backtest.engine import BacktestConfig, LookAheadError, run_backtest
from ml_core.models.registry import code_version

logger = get_logger(__name__)
router = APIRouter(prefix="/backtests", tags=["backtests"])

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="backtest")
_lock = threading.Lock()


def _execute(run_id: uuid.UUID, config: BacktestConfig) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("UPDATE backtest_runs SET status = 'running', started_at = now() WHERE id = :id"), {"id": run_id})
    try:
        result = jsonable(run_backtest(config))
        status, payload, error = "completed", json.dumps(result), None
    except (ValueError, LookAheadError) as exc:
        status, payload, error = "failed", None, str(exc)
    except Exception:  # noqa: BLE001
        logger.exception("Backtest %s crashed", run_id)
        status, payload, error = "failed", None, "Internal error while running the backtest"
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE backtest_runs SET status = :s, result = CAST(:r AS jsonb), error = :e, finished_at = now() "
                "WHERE id = :id"
            ),
            {"s": status, "r": payload, "e": error, "id": run_id},
        )


def recover_stale_jobs() -> None:
    """Jobs interrupted by a restart are marked failed so users can resubmit."""
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE backtest_runs SET status = 'failed', error = 'Interrupted by service restart', finished_at = now() "
                "WHERE status IN ('queued', 'running')"
            )
        )


@router.post("", status_code=202, dependencies=[])
def submit(payload: BacktestRequest, user: User = Depends(require_user), _: None = Depends(limiter("backtest")),
           db: Session = Depends(get_db)):
    config = BacktestConfig(
        start=payload.start, end=payload.end, strategy=payload.strategy, initial_capital=payload.initial_capital,
        horizon_days=payload.horizon_days, rebalance_every=payload.rebalance_every, profile=payload.risk_profile,
        cvar_limit=payload.cvar_limit, max_weight=payload.max_weight, max_turnover=payload.max_turnover,
        commission_bps=payload.commission_bps, slippage_bps=payload.slippage_bps, tickers=payload.tickers or [],
    )
    try:
        config.validate()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
    with _lock:
        active = db.execute(
            text("SELECT COUNT(*) FROM backtest_runs WHERE email = :e AND status IN ('queued', 'running')"),
            {"e": user.email},
        ).scalar()
        if active:
            raise HTTPException(status_code=409, detail="You already have a backtest running; wait for it to finish.")
        run_id = uuid.uuid4()
        db.execute(
            text(
                "INSERT INTO backtest_runs (id, email, status, config, config_hash, code_version) "
                "VALUES (:id, :e, 'queued', CAST(:c AS jsonb), :h, :v)"
            ),
            {"id": run_id, "e": user.email, "c": json.dumps(config.to_dict()), "h": config.config_hash(), "v": code_version()},
        )
        db.commit()
    _executor.submit(_execute, run_id, config)
    return {"id": str(run_id), "status": "queued", "config_hash": config.config_hash()}


@router.get("")
def list_runs(user: User = Depends(require_user), db: Session = Depends(get_db)):
    rows = db.execute(
        text(
            """
            SELECT id, status, config, config_hash, error, created_at, finished_at,
                   result->'metrics'->(config->>'strategy') AS summary
            FROM backtest_runs WHERE email = :e ORDER BY created_at DESC LIMIT 25
            """
        ),
        {"e": user.email},
    ).mappings().all()
    return {"runs": [jsonable(dict(row)) for row in rows]}


@router.get("/{run_id}")
def get_run(run_id: uuid.UUID, user: User = Depends(require_user), db: Session = Depends(get_db)):
    row = db.execute(
        text(
            "SELECT id, status, config, config_hash, code_version, result, error, created_at, started_at, finished_at "
            "FROM backtest_runs WHERE id = :id AND email = :e"
        ),
        {"id": run_id, "e": user.email},
    ).mappings().one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return jsonable(dict(row))
