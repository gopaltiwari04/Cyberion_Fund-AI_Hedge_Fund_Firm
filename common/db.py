from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from common.config import get_settings, normalize_database_url


@lru_cache(maxsize=4)
def _engine_for(url: str) -> Engine:
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        pool_recycle=1800,
    )


def get_engine(url: str | None = None) -> Engine:
    return _engine_for(normalize_database_url(url or get_settings().database_url))
