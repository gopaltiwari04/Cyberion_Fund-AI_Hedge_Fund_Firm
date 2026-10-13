"""Shared Airflow helpers: retry policy, failure alerts and pipeline task factory."""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import timedelta

from airflow.operators.bash import BashOperator

PROJECT_DIR = os.getenv("PROJECT_DIR", "/opt/airflow/dags/project")
PIPELINE_PYTHON = os.getenv("PIPELINE_PYTHON", "python")


def alert_on_failure(context) -> None:
    """Final-failure alert (after retries). Pipelines also record failures in pipeline_runs."""
    task = context.get("task_instance")
    message = (
        f"Airflow task failed: {task.dag_id}.{task.task_id} "
        f"(run {context.get('run_id')}, try {task.try_number})\n{context.get('exception')}"
    )
    print(message)
    url = os.getenv("ALERT_WEBHOOK_URL")
    if not url:
        return
    request = urllib.request.Request(url, data=json.dumps({"text": message[:3000]}).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(request, timeout=10)  # noqa: S310 - operator-configured webhook
    except OSError as error:
        print(f"Alert webhook failed: {error}")


DEFAULT_ARGS = {
    "owner": "cyberion",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(hours=1),
    "on_failure_callback": alert_on_failure,
}


def pipeline_task(task_id: str, module: str, args: str = "", timeout_minutes: int = 60, **kwargs) -> BashOperator:
    """Run ``python -m <module>`` inside the isolated pipeline virtualenv."""
    return BashOperator(
        task_id=task_id,
        bash_command=f"set -euo pipefail; cd {PROJECT_DIR} && {PIPELINE_PYTHON} -m {module} {args}",
        env={
            "RUN_DATE": "{{ data_interval_end | ds }}",
            "PYTHONPATH": PROJECT_DIR,
            "PYTHONUNBUFFERED": "1",
        },
        append_env=True,
        execution_timeout=timedelta(minutes=timeout_minutes),
        **kwargs,
    )
