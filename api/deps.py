from __future__ import annotations

import math
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterator

from sqlalchemy.orm import Session, sessionmaker

from common.db import get_engine

_SessionLocal: sessionmaker | None = None


def get_db() -> Iterator[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def jsonable(value: Any) -> Any:
    """Convert DB values to JSON-safe primitives (NaN/inf -> None)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):  # numpy scalars
        return jsonable(value.item())
    return value
