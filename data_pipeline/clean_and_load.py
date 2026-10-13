"""
Market data transformation: MinIO raw -> Polars clean/validate/dedupe -> PostgreSQL.

Cleaning rules (all in Polars):
    * exact column mapping (no fuzzy matching that could confuse Close / Adj Close)
    * drop rows without a date or a positive close
    * de-duplicate on date (keep the last delivered record)
    * reject OHLC rows that are internally inconsistent (high < low, etc.)
    * Yahoo reports split ratio 0 when there is no split -> normalised to 1

Validation (hard failures stop the load for that ticker):
    * unique dates, positive prices, non-negative volume, consistent OHLC
    * Great Expectations suite (when installed) as an independent check
Warnings (recorded in pipeline_runs):
    * daily moves > 40% that are not explained by a split
    * stale data (latest bar older than 5 business days)

Loading is an idempotent UPSERT on (ticker, date) that only rewrites rows
whose values actually changed.

Run: python -m data_pipeline.clean_and_load
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import polars as pl
from sqlalchemy import text

from common.db import get_engine
from common.logs import get_logger
from common.market_time import business_days_between, last_completed_session
from common.runs import track_run
from common.storage import get_object_store
from common.universe import tickers
from data_pipeline.market_ingestion import raw_key, run_date

logger = get_logger(__name__)

COLUMN_MAP = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividend",
    "Stock Splits": "split_ratio",
}
PRICE_COLUMNS = ["open", "high", "low", "close"]
EXTREME_MOVE = 0.40


@dataclass
class QualityReport:
    ticker: str
    input_rows: int = 0
    output_rows: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def standardize(raw: pd.DataFrame, ticker: str) -> pl.DataFrame:
    missing = [column for column in ("Date", "Open", "High", "Low", "Close", "Volume") if column not in raw.columns]
    if missing:
        raise ValueError(f"{ticker}: raw payload missing columns {missing}")
    frame = raw[[column for column in COLUMN_MAP if column in raw.columns]].rename(columns=COLUMN_MAP)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    df = pl.from_pandas(frame)
    for column in ("adj_close", "dividend", "split_ratio"):
        if column not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    return df.with_columns(
        [pl.col(column).cast(pl.Float64) for column in [*PRICE_COLUMNS, "adj_close", "dividend", "split_ratio"]]
        + [pl.col("volume").cast(pl.Float64), pl.lit(ticker).alias("ticker")]
    )


def clean(df: pl.DataFrame, report: QualityReport, last_session: date | None = None) -> pl.DataFrame:
    """Clean one ticker. Bars after ``last_session`` (incomplete intraday bars) are dropped."""
    report.input_rows = df.height
    last_session = last_session or last_completed_session()

    def drop(frame: pl.DataFrame, condition: pl.Expr, reason: str) -> pl.DataFrame:
        kept = frame.filter(condition)
        removed = frame.height - kept.height
        if removed:
            report.dropped[reason] = report.dropped.get(reason, 0) + removed
        return kept

    df = drop(df, pl.col("date").is_not_null(), "missing_date")
    df = drop(df, pl.col("date") <= last_session, "incomplete_session")
    df = drop(df, pl.col("close").is_not_null() & (pl.col("close") > 0), "invalid_close")
    df = drop(
        df,
        pl.all_horizontal([pl.col(column).is_null() | (pl.col(column) > 0) for column in ("open", "high", "low")]),
        "non_positive_price",
    )
    df = drop(df, pl.col("volume").is_null() | (pl.col("volume") >= 0), "negative_volume")

    df = df.with_columns(
        pl.col("open").fill_null(pl.col("close")),
        pl.col("high").fill_null(pl.max_horizontal("open", "close")),
        pl.col("low").fill_null(pl.min_horizontal("open", "close")),
        pl.col("volume").fill_null(0.0),
        pl.col("dividend").fill_null(0.0),
        pl.when(pl.col("split_ratio").is_null() | (pl.col("split_ratio") == 0))
        .then(1.0)
        .otherwise(pl.col("split_ratio"))
        .alias("split_ratio"),
    )
    tolerance = 1e-6
    df = drop(
        df,
        (pl.col("high") >= pl.col("low") * (1 - tolerance))
        & (pl.col("high") >= pl.max_horizontal("open", "close") * (1 - 1e-3))
        & (pl.col("low") <= pl.min_horizontal("open", "close") * (1 + 1e-3)),
        "inconsistent_ohlc",
    )

    before = df.height
    df = df.sort("date").unique(subset=["date"], keep="last", maintain_order=True)
    if before != df.height:
        report.dropped["duplicate_date"] = before - df.height

    report.output_rows = df.height
    return df


def validate(df: pl.DataFrame, report: QualityReport, as_of: date | None = None) -> None:
    if df.height == 0:
        report.errors.append("no valid rows")
        return
    if df["date"].n_unique() != df.height:
        report.errors.append("duplicate dates after cleaning")
    if (df["close"] <= 0).any():
        report.errors.append("non-positive close")
    if (df["volume"] < 0).any():
        report.errors.append("negative volume")

    returns = df.sort("date").with_columns((pl.col("close") / pl.col("close").shift(1) - 1).alias("ret"))
    extreme = returns.filter((pl.col("ret").abs() > EXTREME_MOVE) & (pl.col("split_ratio") == 1.0))
    if extreme.height:
        report.warnings.append(
            f"{extreme.height} unexplained daily moves > {EXTREME_MOVE:.0%}: "
            f"{[str(value) for value in extreme['date'].to_list()[:5]]}"
        )

    latest = df["date"].max()
    session = as_of or last_completed_session()
    lag = business_days_between(latest, session)
    if lag > 5:
        report.warnings.append(f"stale data: latest bar {latest} is {lag} business days old")


def great_expectations_check(df: pl.DataFrame, ticker: str, report: QualityReport) -> None:
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")
    try:
        from data_pipeline.validate_market_data import validate_market_data
    except ImportError:
        report.warnings.append("great_expectations not installed; GX suite skipped")
        return
    try:
        validate_market_data(df.select(["ticker", "date", "open", "high", "low", "close", "volume"]), ticker)
    except ValueError as error:
        report.errors.append(str(error))


UPSERT_SQL = text(
    """
    INSERT INTO market_data (ticker, date, open, high, low, close, adj_close, volume, dividend, split_ratio, source, ingested_at)
    VALUES (:ticker, :date, :open, :high, :low, :close, :adj_close, :volume, :dividend, :split_ratio, 'yfinance', now())
    ON CONFLICT (ticker, date) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        adj_close = EXCLUDED.adj_close,
        volume = EXCLUDED.volume,
        dividend = EXCLUDED.dividend,
        split_ratio = EXCLUDED.split_ratio,
        source = EXCLUDED.source,
        ingested_at = now()
    WHERE market_data.close IS DISTINCT FROM EXCLUDED.close
       OR market_data.open IS DISTINCT FROM EXCLUDED.open
       OR market_data.high IS DISTINCT FROM EXCLUDED.high
       OR market_data.low IS DISTINCT FROM EXCLUDED.low
       OR market_data.volume IS DISTINCT FROM EXCLUDED.volume
       OR market_data.dividend IS DISTINCT FROM EXCLUDED.dividend
       OR market_data.split_ratio IS DISTINCT FROM EXCLUDED.split_ratio
       OR market_data.adj_close IS DISTINCT FROM EXCLUDED.adj_close
    """
)


def to_records(df: pl.DataFrame) -> list[dict]:
    frame = df.select(
        ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "dividend", "split_ratio"]
    ).to_pandas()
    frame["volume"] = frame["volume"].round().astype("int64")
    frame = frame.replace([np.inf, -np.inf], np.nan).astype(object).where(frame.notna(), None)
    return frame.to_dict(orient="records")


def upsert(df: pl.DataFrame) -> int:
    records = to_records(df)
    changed = 0
    with get_engine().begin() as conn:
        for start in range(0, len(records), 1000):
            result = conn.execute(UPSERT_SQL, records[start:start + 1000])
            changed += max(result.rowcount or 0, 0)
    return changed


def upsert_asset_metadata() -> None:
    from common.universe import get_universe

    with get_engine().begin() as conn:
        for asset in get_universe():
            conn.execute(
                text(
                    """
                    INSERT INTO assets (ticker, name, asset_class, sector, is_active, updated_at)
                    VALUES (:ticker, :name, :asset_class, :sector, TRUE, now())
                    ON CONFLICT (ticker) DO UPDATE SET
                        name = EXCLUDED.name, asset_class = EXCLUDED.asset_class,
                        sector = EXCLUDED.sector, is_active = TRUE, updated_at = now()
                    """
                ),
                {"ticker": asset.ticker, "name": asset.name, "asset_class": asset.asset_class, "sector": asset.sector},
            )
        conn.execute(
            text(
                """
                UPDATE assets a SET first_trade_date = m.first_date
                FROM (SELECT ticker, MIN(date) AS first_date FROM market_data GROUP BY ticker) m
                WHERE a.ticker = m.ticker
                """
            )
        )


def process_ticker(ticker: str, partition: date) -> QualityReport:
    report = QualityReport(ticker=ticker)
    store = get_object_store()
    key = raw_key(ticker, partition)
    if not store.exists(key):
        report.warnings.append(f"no raw partition {key}; ticker skipped")
        return report
    df = clean(standardize(store.get_parquet(key), ticker), report)
    validate(df, report)
    if not report.errors:
        great_expectations_check(df, ticker, report)
    if report.errors:
        raise ValueError(f"{ticker}: validation failed: {report.errors}")
    changed = upsert(df)
    logger.info("Loaded %s", ticker, extra={"rows": df.height, "changed": changed, "dropped": report.dropped})
    report.dropped["rows_changed_in_db"] = changed
    return report


def main() -> None:
    partition = run_date()
    with track_run("market_data", "clean_load_prices") as ctx:
        reports, failures = [], []
        for ticker in tickers():
            try:
                report = process_ticker(ticker, partition)
                reports.append(report.__dict__)
                ctx.add_rows(report.output_rows)
                for warning in report.warnings:
                    ctx.warn(f"{ticker}: {warning}")
            except ValueError as error:
                failures.append(str(error))
                ctx.warn(str(error))
        upsert_asset_metadata()
        ctx.details.update({"run_date": partition.isoformat(), "reports": reports, "failures": failures})
        if failures:
            raise RuntimeError(f"Validation failures: {failures}")


if __name__ == "__main__":
    main()
