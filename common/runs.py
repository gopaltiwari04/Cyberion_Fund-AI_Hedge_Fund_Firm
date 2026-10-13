"""
Pipeline observability.

``track_run`` records every pipeline task execution in ``pipeline_runs``
(start, finish, status, row counts, details, error) and sends an alert to
``ALERT_WEBHOOK_URL`` (Slack-compatible JSON) when a task fails. The API's
system endpoints read this table to show pipeline health and data freshness.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import requests
from sqlalchemy import text

from common.config import get_settings
from common.db import get_engine
from common.logs import get_logger

logger = get_logger(__name__)


@dataclass
class RunContext:
    pipeline: str
    task: str
    rows_processed: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def add_rows(self, count: int) -> None:
        self.rows_processed += int(count)

    def warn(self, message: str) -> None:
        logger.warning(message, extra={"pipeline": self.pipeline, "task": self.task})
        self.warnings.append(message)


def send_alert(title: str, message: str) -> None:
    url = get_settings().alert_webhook_url
    logger.error("ALERT: %s | %s", title, message)
    if not url:
        return
    try:
        requests.post(url, json={"text": f"*{title}*\n{message[:3000]}"}, timeout=10)
    except requests.RequestException as error:  # alerting must never mask the original failure
        logger.warning("Alert webhook failed: %s", error)


def _insert_run(pipeline: str, task: str, run_id: str | None) -> int | None:
    try:
        with get_engine().begin() as conn:
            # Runs killed without cleanup (container restart, OOM) would otherwise stay "running" forever.
            conn.execute(
                text(
                    """
                    UPDATE pipeline_runs SET status = 'abandoned', finished_at = now(),
                        error = 'Process ended without reporting completion'
                    WHERE pipeline = :pipeline AND task = :task AND status = 'running'
                    """
                ),
                {"pipeline": pipeline, "task": task},
            )
            return conn.execute(
                text(
                    """
                    INSERT INTO pipeline_runs (pipeline, task, run_id, status, started_at)
                    VALUES (:pipeline, :task, :run_id, 'running', now())
                    RETURNING id
                    """
                ),
                {"pipeline": pipeline, "task": task, "run_id": run_id},
            ).scalar()
    except Exception as error:  # noqa: BLE001 - tracking table may not exist before migrations
        logger.warning("Could not record pipeline run start: %s", error)
        return None


def _finish_run(run_pk: int | None, status: str, ctx: RunContext, error: str | None, duration: float) -> None:
    if run_pk is None:
        return
    details = dict(ctx.details)
    details["duration_seconds"] = round(duration, 2)
    if ctx.warnings:
        details["warnings"] = ctx.warnings[:50]
    try:
        with get_engine().begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE pipeline_runs
                    SET status = :status, finished_at = now(), rows_processed = :rows,
                        details = CAST(:details AS jsonb), error = :error
                    WHERE id = :id
                    """
                ),
                {
                    "id": run_pk,
                    "status": status,
                    "rows": ctx.rows_processed,
                    "details": json.dumps(details, default=str),
                    "error": error[:4000] if error else None,
                },
            )
    except Exception as db_error:  # noqa: BLE001
        logger.warning("Could not record pipeline run finish: %s", db_error)


@contextmanager
def track_run(pipeline: str, task: str) -> Iterator[RunContext]:
    run_id = os.getenv("AIRFLOW_CTX_DAG_RUN_ID")
    ctx = RunContext(pipeline=pipeline, task=task)
    started = time.monotonic()
    run_pk = _insert_run(pipeline, task, run_id)
    logger.info("Pipeline task started", extra={"pipeline": pipeline, "task": task, "run_id": run_id})
    try:
        yield ctx
    except Exception as error:
        duration = time.monotonic() - started
        message = f"{type(error).__name__}: {error}"
        _finish_run(run_pk, "failed", ctx, message + "\n" + traceback.format_exc(limit=5), duration)
        send_alert(f"Pipeline failure: {pipeline}.{task}", message)
        raise
    duration = time.monotonic() - started
    status = "success_with_warnings" if ctx.warnings else "success"
    _finish_run(run_pk, status, ctx, None, duration)
    logger.info(
        "Pipeline task finished",
        extra={"pipeline": pipeline, "task": task, "status": status, "rows": ctx.rows_processed, "seconds": round(duration, 2)},
    )
