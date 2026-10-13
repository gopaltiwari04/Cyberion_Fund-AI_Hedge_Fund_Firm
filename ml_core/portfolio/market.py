"""Market data access shared by the optimiser, backtester and API."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text

from common.config import get_settings
from common.db import get_engine


def load_prices(tickers: list[str], start: date | None = None, end: date | None = None) -> pd.DataFrame:
    query = "SELECT ticker, date, open, close, dividend FROM market_data WHERE ticker = ANY(:t)"
    params: dict = {"t": tickers}
    if start:
        query += " AND date >= :start"
        params["start"] = start
    if end:
        query += " AND date <= :end"
        params["end"] = end
    with get_engine().connect() as conn:
        frame = pd.read_sql(text(query + " ORDER BY date"), conn, params=params)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def total_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Date x ticker matrix of daily total returns (close-to-close incl. dividends)."""
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    dividend = prices.pivot(index="date", columns="ticker", values="dividend").reindex(close.index).fillna(0.0)
    return (close + dividend) / close.shift(1) - 1.0


def return_scenarios(tickers: list[str], end: date | None = None, window: int = 756) -> pd.DataFrame:
    prices = load_prices(tickers, end=end)
    returns = total_returns(prices)[tickers].dropna()
    if len(returns) < 60:
        raise ValueError("Not enough synchronized return history for risk scenarios")
    return returns.tail(window)


def latest_prices(tickers: list[str]) -> dict[str, dict]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT ON (ticker) ticker, date, close FROM market_data
                WHERE ticker = ANY(:t) ORDER BY ticker, date DESC
                """
            ),
            {"t": tickers},
        ).mappings().all()
    return {row["ticker"]: {"date": row["date"], "close": float(row["close"])} for row in rows}


def risk_free_daily(as_of: date | None = None) -> tuple[float, str]:
    """Daily risk-free rate from the effective fed funds rate (FRED DFF) known at ``as_of``."""
    query = "SELECT date, value FROM macro_data WHERE series_id = 'DFF' AND value IS NOT NULL"
    params: dict = {}
    if as_of:
        query += " AND date < :as_of"
        params["as_of"] = as_of
    with get_engine().connect() as conn:
        row = conn.execute(text(query + " ORDER BY date DESC LIMIT 1"), params).one_or_none()
    if row is None:
        return get_settings().risk_free_fallback / 252, "fallback"
    return float(row.value) / 100 / 252, f"FRED DFF {row.date}"


def alpha_confidence(metrics: dict | None) -> float:
    """
    Scale model expected returns by demonstrated out-of-sample skill:
    0 when the overlap-adjusted IC t-stat <= 0, rising linearly to 1 at t >= 3.
    Prefers the untouched post-2020 holdout statistics when available.
    """
    if not metrics:
        return 0.0
    holdout = metrics.get("holdout_since_2020") or {}
    t_stat = holdout.get("ic_tstat_overlap_adjusted", metrics.get("ic_tstat_overlap_adjusted"))
    if t_stat is None or not math.isfinite(t_stat):
        return 0.0
    return float(np.clip(t_stat / 3.0, 0.0, 1.0))
