"""Convert numpy / pandas / date values to JSON- and DB-driver-safe builtins."""

from __future__ import annotations

import math
from datetime import date, datetime
from decimal import Decimal
from typing import Any


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if hasattr(value, "item") and not isinstance(value, (datetime, date)):  # numpy scalar
        value = value.item()
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
