"""
Model registry backed by PostgreSQL (model_registry) with MinIO archival.

Artifacts are stored in the database row (bytea) so any service with DB
access - including the Render API - can load the exact active version, and
mirrored to object storage for long-term archival. Every entry records the
training window, data hash, code version, parameters and evaluation metrics
so a prediction can always be traced back to how it was produced.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from datetime import date, datetime, timezone
from typing import Any

import joblib
import pandas as pd
from sqlalchemy import text

from common.config import PROJECT_ROOT
from common.db import get_engine
from common.logs import get_logger

logger = get_logger(__name__)


def code_version() -> str:
    if os.getenv("GIT_SHA"):
        return os.environ["GIT_SHA"][:12]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL) != 0
        return sha.decode().strip()[:12] + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def frame_hash(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame.reset_index(drop=True), index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


def serialize(artifact: dict) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer, compress=3)
    return buffer.getvalue()


def deserialize(blob: bytes) -> dict:
    return joblib.load(io.BytesIO(blob))


def _json(value: Any) -> str:
    return json.dumps(value, default=lambda v: v.isoformat() if isinstance(v, (date, datetime)) else str(v))


def register_model(
    *,
    model_name: str,
    algorithm: str,
    horizon_days: int,
    target: str,
    artifact: dict | None,
    feature_columns: list[str],
    params: dict,
    metrics: dict,
    calibration: dict,
    feature_importance: dict,
    train_start: date | None,
    train_end: date | None,
    data_end: date | None,
    data_hash: str,
    status: str = "candidate",
    notes: str | None = None,
) -> int:
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{data_hash[:8]}"
    blob = serialize(artifact) if artifact is not None else None
    sha = hashlib.sha256(blob).hexdigest() if blob else None
    artifact_key = None
    if blob:
        try:
            from common.storage import get_object_store

            artifact_key = f"models/{model_name}/{version}/artifact.joblib"
            get_object_store().put_bytes(artifact_key, blob)
        except Exception as error:  # noqa: BLE001 - archival copy is best effort
            logger.warning("Model archival to object storage failed: %s", error)
            artifact_key = None

    with get_engine().begin() as conn:
        model_id = conn.execute(
            text(
                """
                INSERT INTO model_registry (model_name, version, algorithm, horizon_days, target, status,
                    train_start, train_end, data_end, feature_columns, params, metrics, calibration,
                    feature_importance, data_hash, code_version, artifact_key, artifact, artifact_sha256, notes)
                VALUES (:model_name, :version, :algorithm, :horizon, :target, 'candidate',
                    :train_start, :train_end, :data_end, CAST(:features AS jsonb), CAST(:params AS jsonb),
                    CAST(:metrics AS jsonb), CAST(:calibration AS jsonb), CAST(:importance AS jsonb),
                    :data_hash, :code_version, :artifact_key, :artifact, :sha, :notes)
                RETURNING id
                """
            ),
            {
                "model_name": model_name,
                "version": version,
                "algorithm": algorithm,
                "horizon": horizon_days,
                "target": target,
                "train_start": train_start,
                "train_end": train_end,
                "data_end": data_end,
                "features": _json(feature_columns),
                "params": _json(params),
                "metrics": _json(metrics),
                "calibration": _json(calibration),
                "importance": _json(feature_importance),
                "data_hash": data_hash,
                "code_version": code_version(),
                "artifact_key": artifact_key,
                "artifact": blob,
                "sha": sha,
                "notes": notes,
            },
        ).scalar_one()
    if status == "active":
        promote(model_id)
    return model_id


def promote(model_id: int) -> None:
    """Make one model the active version for its (name, horizon); the previous one is archived."""
    with get_engine().begin() as conn:
        row = conn.execute(
            text("SELECT model_name, horizon_days FROM model_registry WHERE id = :id"), {"id": model_id}
        ).one()
        conn.execute(
            text(
                "UPDATE model_registry SET status = 'archived' "
                "WHERE model_name = :name AND horizon_days = :h AND status = 'active' AND id <> :id"
            ),
            {"name": row.model_name, "h": row.horizon_days, "id": model_id},
        )
        conn.execute(text("UPDATE model_registry SET status = 'active' WHERE id = :id"), {"id": model_id})


def load_active(model_name: str, horizon_days: int) -> tuple[int, dict, dict] | None:
    with get_engine().connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT id, artifact, artifact_sha256, metrics FROM model_registry
                WHERE model_name = :name AND horizon_days = :h AND status = 'active'
                ORDER BY trained_at DESC LIMIT 1
                """
            ),
            {"name": model_name, "h": horizon_days},
        ).one_or_none()
    if row is None or row.artifact is None:
        return None
    blob = bytes(row.artifact)
    if hashlib.sha256(blob).hexdigest() != row.artifact_sha256:
        raise RuntimeError(f"Artifact checksum mismatch for model {row.id}")
    return row.id, deserialize(blob), row.metrics
