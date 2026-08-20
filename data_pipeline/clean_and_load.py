import io

import boto3
import pandas as pd
import polars as pl

from sqlalchemy import create_engine


# ============================================================
# CONNECTIONS
# ============================================================

s3_client = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
)

BUCKET_NAME = "raw-market-data"

DB_URL = "postgresql://quant_user:quant_password@localhost:5432/quant_db"

engine = create_engine(DB_URL)


# ============================================================
# CLEAN AND LOAD ONE TICKER
# ============================================================

def clean_and_load_ticker(ticker):

    print(f"\nProcessing cleaning pipeline for {ticker}...")

    # --------------------------------------------------------
    # 1. Find the latest parquet file in MinIO
    # --------------------------------------------------------

    response = s3_client.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=f"yfinance/{ticker}/",
    )

    if "Contents" not in response:
        print(f"No raw data found for {ticker}")
        return

    latest_file = sorted(
        response["Contents"],
        key=lambda x: x["LastModified"],
        reverse=True,
    )[0]["Key"]

    print(f"Latest file: {latest_file}")

    # --------------------------------------------------------
    # 2. Download parquet file from MinIO
    # --------------------------------------------------------

    obj = s3_client.get_object(
        Bucket=BUCKET_NAME,
        Key=latest_file,
    )

    parquet_bytes = obj["Body"].read()

    # --------------------------------------------------------
    # 3. Read parquet
    # --------------------------------------------------------

    df_pd = pd.read_parquet(
        io.BytesIO(parquet_bytes)
    )

    # yfinance often stores Date in the index
    df_pd = df_pd.reset_index()

    # --------------------------------------------------------
    # 4. Convert Pandas -> Polars
    # --------------------------------------------------------

    df = pl.from_pandas(df_pd)

    print(f"Columns found: {df.columns}")

    # --------------------------------------------------------
    # 5. Standardize column names
    # --------------------------------------------------------

    rename_map = {}

    for column in df.columns:

        if column == "Date":
            rename_map[column] = "date"

        elif column == "Open":
            rename_map[column] = "open"

        elif column == "High":
            rename_map[column] = "high"

        elif column == "Low":
            rename_map[column] = "low"

        elif column == "Close":
            rename_map[column] = "close"

        elif column == "Volume":
            rename_map[column] = "volume"

    df = df.rename(rename_map)

    # --------------------------------------------------------
    # 6. Make sure required columns exist
    # --------------------------------------------------------

    required_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns for {ticker}: {missing_columns}"
        )

    # --------------------------------------------------------
    # 7. Sort by date
    # --------------------------------------------------------

    df = df.sort("date")

    # --------------------------------------------------------
    # 8. Remove invalid prices
    # --------------------------------------------------------

    df = df.filter(
        pl.col("close").is_not_null()
        & (pl.col("close") > 0)
    )

    df = df.filter(
        pl.col("open").is_null()
        | (pl.col("open") > 0)
    )

    df = df.filter(
        pl.col("high").is_null()
        | (pl.col("high") > 0)
    )

    df = df.filter(
        pl.col("low").is_null()
        | (pl.col("low") > 0)
    )

    # --------------------------------------------------------
    # 9. Remove negative volume
    # --------------------------------------------------------

    df = df.filter(
        pl.col("volume").is_null()
        | (pl.col("volume") >= 0)
    )

    # --------------------------------------------------------
    # 10. Fill missing values
    # --------------------------------------------------------

    df = df.with_columns(
        [
            pl.col("open").forward_fill(),
            pl.col("high").forward_fill(),
            pl.col("low").forward_fill(),
            pl.col("close").forward_fill(),
            pl.col("volume").fill_null(0),
        ]
    )

    # --------------------------------------------------------
    # 11. Data validation
    # --------------------------------------------------------

    if df.height == 0:
        raise ValueError(
            f"Validation failed: no valid rows for {ticker}"
        )

    if df["close"].min() <= 0:
        raise ValueError(
            f"Validation failed: invalid close price for {ticker}"
        )

    if df["volume"].min() < 0:
        raise ValueError(
            f"Validation failed: negative volume for {ticker}"
        )

    # --------------------------------------------------------
    # 12. Add ticker
    # --------------------------------------------------------

    df = df.with_columns(
        pl.lit(ticker).alias("ticker")
    )

    # --------------------------------------------------------
    # 13. Select database columns
    # --------------------------------------------------------

    df_clean = df.select(
        [
            "ticker",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    print(f"Clean rows: {df_clean.height}")

    # --------------------------------------------------------
    # 14. Convert Polars -> Pandas
    # --------------------------------------------------------

    final_pd = df_clean.to_pandas()

    final_pd["date"] = pd.to_datetime(
        final_pd["date"]
    ).dt.date

    # --------------------------------------------------------
    # 15. Load into PostgreSQL
    # --------------------------------------------------------

    final_pd.to_sql(
        "market_data",
        engine,
        if_exists="append",
        index=False,
        method="multi",
    )

    print(
        f"Successfully loaded {len(final_pd)} rows "
        f"for {ticker} into PostgreSQL."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    target_tickers = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "SPY",
    ]

    for ticker in target_tickers:

        clean_and_load_ticker(ticker)