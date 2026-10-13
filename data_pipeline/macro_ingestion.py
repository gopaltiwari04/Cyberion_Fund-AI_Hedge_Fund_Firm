"""
Macroeconomic data: FRED -> MinIO raw -> PostgreSQL macro_data.

Uses the official FRED API when FRED_API_KEY is configured and falls back to
the public fredgraph CSV endpoint otherwise. Each run lands the raw series
in MinIO (deterministic key per run date) and upserts observations, so the
task is safe to re-run.

Point-in-time note: daily FRED values for day t are published on t+1, so the
feature builder lags every series by one business day before use.

Run: python -m data_pipeline.macro_ingestion
"""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import polars as pl
import requests
from sqlalchemy import text

from common.config import get_settings
from common.db import get_engine
from common.logs import get_logger
from common.runs import track_run
from common.storage import get_object_store
from data_pipeline.market_ingestion import run_date

logger = get_logger(__name__)

SERIES = {
    "DFF": "Effective federal funds rate (%)",
    "DGS10": "10-year Treasury constant maturity (%)",
    "DGS2": "2-year Treasury constant maturity (%)",
    "T10Y2Y": "10y minus 2y Treasury spread (%)",
    "VIXCLS": "CBOE VIX close",
    "BAA10Y": "Moody's Baa corporate minus 10y Treasury (%)",
    "CPIAUCSL": "CPI, all urban consumers (index)",
    "UNRATE": "Unemployment rate (%)",
}
START = "2014-01-01"


def fetch_series(series_id: str) -> pd.DataFrame:
    key = get_settings().fred_api_key
    for attempt in range(1, 5):
        try:
            if key:
                response = requests.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={"series_id": series_id, "api_key": key, "file_type": "json", "observation_start": START},
                    timeout=30,
                )
                response.raise_for_status()
                rows = response.json()["observations"]
                frame = pd.DataFrame(rows)[["date", "value"]]
            else:
                response = requests.get(
                    "https://fred.stlouisfed.org/graph/fredgraph.csv",
                    params={"id": series_id, "cosd": START},
                    timeout=30,
                )
                response.raise_for_status()
                frame = pd.read_csv(io.StringIO(response.text))
                frame.columns = ["date", "value"]
            return frame
        except (requests.RequestException, KeyError, ValueError) as error:
            logger.warning("FRED fetch failed for %s (attempt %s): %s", series_id, attempt, error)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"FRED series {series_id} unavailable")


def clean_series(raw: pd.DataFrame, series_id: str) -> pl.DataFrame:
    values = pd.to_numeric(raw["value"].replace({".": None, "": None}), errors="coerce")
    frame = pl.DataFrame({"date": raw["date"].astype(str).tolist(), "value": values.tolist()})
    return (
        frame.with_columns(
            pl.col("date").str.to_date(strict=False),
            pl.col("value").cast(pl.Float64).fill_nan(None),
            pl.lit(series_id).alias("series_id"),
        )
        .filter(pl.col("date").is_not_null())
        .unique(subset=["date"], keep="last")
        .sort("date")
    )


def upsert(df: pl.DataFrame) -> int:
    frame = df.select(["series_id", "date", "value"]).to_pandas()
    frame = frame.astype(object).where(frame.notna(), None)
    records = frame.to_dict(orient="records")
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO macro_data (series_id, date, value, source, ingested_at)
                VALUES (:series_id, :date, :value, 'fred', now())
                ON CONFLICT (series_id, date) DO UPDATE SET value = EXCLUDED.value, ingested_at = now()
                WHERE macro_data.value IS DISTINCT FROM EXCLUDED.value
                """
            ),
            records,
        )
    return len(records)


def main() -> None:
    partition = run_date()
    store = get_object_store()
    with track_run("macro_data", "ingest_fred") as ctx:
        summary = {}
        for series_id in SERIES:
            try:
                raw = fetch_series(series_id)
                store.put_parquet(f"macro/fred/{series_id}/run_date={partition.isoformat()}/observations.parquet", raw)
                cleaned = clean_series(raw, series_id)
                if cleaned.height == 0 or cleaned["value"].drop_nulls().len() == 0:
                    raise ValueError("no numeric observations")
                values = cleaned["value"].drop_nulls().to_numpy()
                if not np.isfinite(values).all():
                    raise ValueError("non-finite values")
                ctx.add_rows(upsert(cleaned))
                summary[series_id] = {"rows": cleaned.height, "last": str(cleaned["date"].max())}
            except Exception as error:  # noqa: BLE001
                ctx.warn(f"{series_id}: {error}")
        ctx.details["series"] = summary
        if len(summary) < len(SERIES) / 2:
            raise RuntimeError("Most FRED series failed to load")


if __name__ == "__main__":
    main()
