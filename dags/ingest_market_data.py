import os
import time
import io
from datetime import datetime

import yfinance as yf
import pandas as pd
import boto3


# --------------------------------------------------------
# MinIO Configuration
# --------------------------------------------------------

MINIO_ENDPOINT = os.getenv(
    "MINIO_URL",
    "http://localhost:9000"
)

s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin"
)

BUCKET_NAME = "raw-market-data"


# --------------------------------------------------------
# MinIO Bucket
# --------------------------------------------------------

def create_bucket_if_not_exists():

    try:
        s3_client.head_bucket(
            Bucket=BUCKET_NAME
        )

    except Exception:
        print(
            f"Bucket {BUCKET_NAME} does not exist. "
            f"Creating it..."
        )

        s3_client.create_bucket(
            Bucket=BUCKET_NAME
        )


# --------------------------------------------------------
# Download one ticker
# --------------------------------------------------------

def download_ticker(ticker, max_retries=3):

    for attempt in range(1, max_retries + 1):

        print(
            f"Fetching {ticker} "
            f"(attempt {attempt}/{max_retries})..."
        )

        try:

            df = yf.download(
                ticker,
                period="10y",
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False
            )

            if df is not None and not df.empty:

                print(
                    f"Successfully downloaded "
                    f"{len(df)} rows for {ticker}"
                )

                return df

            print(
                f"No data returned for {ticker}"
            )

        except Exception as e:

            print(
                f"Error downloading {ticker}: {e}"
            )

        # Don't hammer Yahoo
        if attempt < max_retries:

            wait_seconds = 30 * attempt

            print(
                f"Waiting {wait_seconds} seconds "
                f"before retrying {ticker}..."
            )

            time.sleep(wait_seconds)

    print(
        f"FAILED: Could not download {ticker} "
        f"after {max_retries} attempts."
    )

    return None


# --------------------------------------------------------
# Fetch and store data
# --------------------------------------------------------

def fetch_and_store_data(tickers):

    create_bucket_if_not_exists()

    successful = []
    failed = []

    for ticker in tickers:

        df = download_ticker(ticker)

        # Never overwrite MinIO with empty data
        if df is None or df.empty:

            failed.append(ticker)

            print(
                f"Skipping {ticker}. "
                f"Existing MinIO data will remain untouched."
            )

            continue

        # ------------------------------------------------
        # Convert to Parquet
        # ------------------------------------------------

        parquet_buffer = io.BytesIO()

        df.to_parquet(
            parquet_buffer,
            engine="pyarrow"
        )

        parquet_buffer.seek(0)

        # ------------------------------------------------
        # Upload to MinIO
        # ------------------------------------------------

        file_key = (
            f"yfinance/{ticker}/"
            f"{datetime.now().strftime('%Y-%m-%d')}.parquet"
        )

        s3_client.upload_fileobj(
            parquet_buffer,
            BUCKET_NAME,
            file_key
        )

        print(
            f"Uploaded {file_key} to MinIO"
        )

        successful.append(ticker)

    # ----------------------------------------------------
    # Summary
    # ----------------------------------------------------

    print("\n========== INGESTION SUMMARY ==========")

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )

    if failed:
        print(
            f"WARNING: Market data ingestion failed for: "
            f"{', '.join(failed)}"
        )
        print(
            "Existing data in MinIO will be used by the "
            "cleaning pipeline."
        )

    else:
        print(
            "Market data ingestion completed successfully."
        )


# --------------------------------------------------------
# Main
# --------------------------------------------------------

if __name__ == "__main__":

    target_tickers = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "SPY"
    ]

    fetch_and_store_data(
        target_tickers
    )