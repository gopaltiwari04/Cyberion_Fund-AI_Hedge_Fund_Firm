"""
Point-in-time helpers.

All daily features are "as of the US market close" of their date. Any event
(news article, SEC filing acceptance) that happens after 16:00 New York time
is only usable from the *next* trading day. These helpers enforce that rule
so text signals can never leak into an earlier feature row.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NEW_YORK = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)


def to_utc(values: pd.Series) -> pd.Series:
    series = pd.to_datetime(values, errors="coerce")
    if getattr(series.dt, "tz", None) is None:
        return series.dt.tz_localize(timezone.utc)
    return series.dt.tz_convert(timezone.utc)


def effective_trading_dates(timestamps: pd.Series, trading_dates: Sequence[date] | Iterable[date]) -> pd.Series:
    """
    Map event timestamps to the first trading date whose close is at or after
    the event. Events after the last known close map to NaT (not yet usable).
    """
    sorted_dates = np.array(sorted(set(pd.to_datetime(list(trading_dates)).date)), dtype="datetime64[D]")
    local = to_utc(timestamps).dt.tz_convert(NEW_YORK)
    after_close = local.dt.time >= MARKET_CLOSE
    candidate = local.dt.normalize().dt.tz_localize(None) + pd.to_timedelta(after_close.astype(int), unit="D")
    candidate_days = candidate.to_numpy(dtype="datetime64[D]")

    result = pd.Series(pd.NaT, index=timestamps.index, dtype="datetime64[ns]")
    if len(sorted_dates) == 0:
        return result

    valid = ~pd.isna(candidate)
    positions = np.searchsorted(sorted_dates, candidate_days[valid.to_numpy()], side="left")
    mapped = np.full(positions.shape, np.datetime64("NaT"), dtype="datetime64[D]")
    in_range = positions < len(sorted_dates)
    mapped[in_range] = sorted_dates[positions[in_range]]
    result.loc[valid[valid].index] = pd.to_datetime(mapped)
    return result.dt.date.where(result.notna(), None)


def last_completed_session(now: datetime | None = None) -> date:
    """Most recent weekday whose 16:00 New York close has passed (holidays ignored)."""
    now = (now or datetime.now(timezone.utc)).astimezone(NEW_YORK)
    day = now.date()
    if now.time() < MARKET_CLOSE:
        day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def business_days_between(start: date, end: date) -> int:
    if end <= start:
        return 0
    return int(np.busday_count(start, end))
