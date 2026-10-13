"""
Cyberion quant platform API.

Entry point for uvicorn (``uvicorn api.main:app``), unchanged from the
original deployment so the Render start command keeps working.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from api.routers import analyst, backtests, markets, portfolio, research, system, user
from common.config import PROJECT_ROOT, get_settings
from common.db import get_engine
from common.logs import configure_logging, get_logger

configure_logging()
logger = get_logger("api")
MIGRATION_LOCK_ID = 727_001


def run_migrations() -> None:
    """Apply Alembic migrations once across instances (Postgres advisory lock)."""
    from alembic import command
    from alembic.config import Config

    settings = get_settings()
    with get_engine().connect() as conn:
        conn.execute(text("SELECT pg_advisory_lock(:id)"), {"id": MIGRATION_LOCK_ID})
        try:
            config = Config(str(PROJECT_ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(PROJECT_ROOT / "db" / "migrations"))
            config.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))
            config.attributes["skip_env_url"] = True
            command.upgrade(config, "head")
            logger.info("Database migrations applied")
        finally:
            conn.execute(text("SELECT pg_advisory_unlock(:id)"), {"id": MIGRATION_LOCK_ID})


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    if settings.run_migrations_on_startup:
        try:
            run_migrations()
        except Exception:  # noqa: BLE001 - keep serving read endpoints; readiness reports the problem
            logger.exception("Startup migration failed")
    try:
        backtests.recover_stale_jobs()
    except SQLAlchemyError:
        logger.warning("Could not recover stale backtest jobs (schema not migrated yet?)")
    if not settings.internal_api_secret:
        logger.warning(
            "INTERNAL_API_SECRET is not set: user endpoints are %s",
            "disabled (production)" if settings.is_production else "using a local developer identity",
        )
    yield


app = FastAPI(
    title="Cyberion Quant Platform API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url=None if get_settings().is_production else "/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().frontend_origins),
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Content-Type", "X-User-Email", "X-User-Name", "X-Auth-Timestamp", "X-Auth-Signature"],
    max_age=600,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    if request.url.path not in ("/health",):
        user = getattr(request.state, "user", None)
        logger.info(
            "request",
            extra={"request_id": request_id, "method": request.method, "path": request.url.path,
                   "status": response.status_code, "ms": round(elapsed, 1), "user": getattr(user, "email", None)},
        )
    return response


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": getattr(request.state, "request_id", None)},
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    errors = [{"field": ".".join(str(p) for p in error["loc"][1:]), "message": error["msg"]} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": "Invalid request", "errors": errors,
                                                  "request_id": getattr(request.state, "request_id", None)})


@app.exception_handler(SQLAlchemyError)
async def database_error(request: Request, exc: SQLAlchemyError):
    logger.error("Database error", extra={"request_id": getattr(request.state, "request_id", None), "error": str(exc)[:500]})
    return JSONResponse(status_code=503, content={"detail": "Database temporarily unavailable",
                                                  "request_id": getattr(request.state, "request_id", None)})


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    logger.exception("Unhandled error", extra={"request_id": getattr(request.state, "request_id", None)})
    return JSONResponse(status_code=500, content={"detail": "Internal server error",
                                                  "request_id": getattr(request.state, "request_id", None)})


for module in (system, markets, portfolio, user, research, analyst, backtests):
    app.include_router(module.router)
