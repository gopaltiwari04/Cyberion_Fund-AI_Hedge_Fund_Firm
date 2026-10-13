"""
Market data extraction: yfinance -> MinIO raw landing zone.

* Incremental: fetches from (last stored date - overlap) instead of the full
  history on every run.
* Corporate actions: Yahoo back-adjusts historical prices when a split
  happens. If the fetched window contains a split, or the overlap disagrees
  with stored closes, the ticker's full history is re-fetched so the stored
  series stays internally consistent.
* Idempotent: raw objects are written to a deterministic key per run date
  (market/yfinance/<ticker>/run_date=<YYYY-MM-DD>/prices.parquet).
* Retries with exponential backoff and jitter; a task only fails when too
  many tickers fail, so one bad symbol does not block the whole universe.

Run: python -m data_pipeline.market_ingestion
"""

from __future__ import annotations

import os
import random
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import text

from common.db import get_engine
from common.logs import get_logger
from common.runs import RunContext, track_run
from common.storage import get_object_store
from common.universe import tickers

logger = get_logger(__name__)

HISTORY_START = date(2015, 1, 1)
OVERLAP_DAYS = 20
MAX_ATTEMPTS = 4
MAX_FAILURE_RATIO = 0.3
ADJUSTMENT_TOLERANCE = 0.005


def run_date() -> date:
    value = os.getenv("RUN_DATE")
    return date.fromisoformat(value) if value else datetime.now(timezone.utc).date()


def raw_key(ticker: str, partition: date) -> str:
    return f"market/yfinance/{ticker}/run_date={partition.isoformat()}/prices.parquet"


def stored_closes(ticker: str, since: date) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(
            text("SELECT date, close FROM market_data WHERE ticker = :ticker AND date >= :since ORDER BY date"),
            conn,
            params={"ticker": ticker, "since": since},
        )


def last_stored_date(ticker: str) -> date | None:
    with get_engine().connect() as conn:
        return conn.execute(text("SELECT MAX(date) FROM market_data WHERE ticker = :ticker"), {"ticker": ticker}).scalar()


def has_legacy_rows(ticker: str) -> bool:
    """Rows loaded before corporate-action capture (no lineage) need one full backfill."""
    with get_engine().connect() as conn:
        return bool(
            conn.execute(
                text("SELECT EXISTS (SELECT 1 FROM market_data WHERE ticker = :ticker AND source IS NULL)"),
                {"ticker": ticker},
            ).scalar()
        )


def download(ticker: str, start: date) -> pd.DataFrame:
    import yfinance as yf

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            frame = yf.Ticker(ticker).history(
                start=start.isoformat(),
                interval="1d",
                auto_adjust=False,
                actions=True,
            )
            if frame is not None and not frame.empty:
                frame = frame.reset_index()
                frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None).dt.normalize()
                return frame
            last_error = RuntimeError("empty response")
        except Exception as error:  # noqa: BLE001 - yfinance raises many types
            last_error = error
        delay = min(120, 5 * 2 ** (attempt - 1)) + random.uniform(0, 3)
        logger.warning("Download failed for %s (attempt %s/%s): %s", ticker, attempt, MAX_ATTEMPTS, last_error)
        if attempt < MAX_ATTEMPTS:
            time.sleep(delay)
    raise RuntimeError(f"Could not download {ticker}: {last_error}")


def needs_full_refresh(fetched: pd.DataFrame, existing: pd.DataFrame, last_date: date | None) -> tuple[bool, str | None]:
    splits = fetched.get("Stock Splits")
    if splits is not None and last_date is not None:
        new_split = (splits.fillna(0) != 0) & (fetched["Date"].dt.date > last_date)
        if new_split.any():
            return True, f"split detected on {fetched.loc[new_split, 'Date'].dt.date.tolist()}"
    if existing.empty:
        return False, None
    merged = existing.assign(date=pd.to_datetime(existing["date"])).merge(
        fetched[["Date", "Close"]], left_on="date", right_on="Date", how="inner"
    )
    if merged.empty:
        return False, None
    relative = (merged["Close"] / merged["close"] - 1.0).abs()
    if (relative > ADJUSTMENT_TOLERANCE).any():
        return True, f"historical prices revised (max diff {relative.max():.2%})"
    return False, None


def extract_ticker(ticker: str, partition: date, ctx: RunContext) -> dict:
    last_date = last_stored_date(ticker)
    start = HISTORY_START if last_date is None else last_date - timedelta(days=OVERLAP_DAYS)
    frame = download(ticker, start)
    mode = "full" if last_date is None else "incremental"

    if last_date is not None:
        refresh, reason = needs_full_refresh(frame, stored_closes(ticker, start), last_date)
        if not refresh and has_legacy_rows(ticker):
            refresh, reason = True, "backfilling dividends/splits for legacy rows"
        if refresh:
            ctx.warn(f"{ticker}: full history refresh ({reason})")
            frame = download(ticker, HISTORY_START)
            mode = "full_refresh"

    frame["ticker"] = ticker
    frame["fetch_mode"] = mode
    frame["fetched_at"] = datetime.now(timezone.utc)
    location = get_object_store().put_parquet(raw_key(ticker, partition), frame)
    return {"ticker": ticker, "rows": len(frame), "mode": mode, "first": str(frame["Date"].min().date()),
            "last": str(frame["Date"].max().date()), "object": location}


def main() -> None:
    partition = run_date()
    universe = tickers()
    with track_run("market_data", "extract_prices") as ctx:
        results, failures = [], []
        for ticker in universe:
            try:
                result = extract_ticker(ticker, partition, ctx)
                results.append(result)
                ctx.add_rows(result["rows"])
                logger.info("Extracted %s", ticker, extra=result)
            except Exception as error:  # noqa: BLE001
                failures.append({"ticker": ticker, "error": str(error)[:500]})
                ctx.warn(f"{ticker}: extraction failed: {error}")
            time.sleep(0.5)  # be polite to the data provider
        ctx.details.update({"run_date": partition.isoformat(), "extracted": results, "failed": failures})
        if len(failures) > MAX_FAILURE_RATIO * len(universe):
            raise RuntimeError(f"Too many extraction failures: {[f['ticker'] for f in failures]}")


if __name__ == "__main__":
    main()
