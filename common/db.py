from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from common.config import get_settings, normalize_database_url


@lru_cache(maxsize=4)
def _engine_for(url: str) -> Engine:
    # Serverless Postgres (Neon) suspends idle computes and drops connections: recycle
    # quickly, ping before use, and keep TCP keepalives so stale sockets are detected.
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "5")),
        pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "300")),
        connect_args={"connect_timeout": 15, "keepalives": 1, "keepalives_idle": 30, "keepalives_interval": 10,
                      "keepalives_count": 3, "application_name": os.getenv("DB_APPLICATION_NAME", "cyberion")},
    )


def get_engine(url: str | None = None) -> Engine:
    return _engine_for(normalize_database_url(url or get_settings().database_url))
