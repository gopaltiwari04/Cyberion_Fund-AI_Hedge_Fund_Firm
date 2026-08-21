import io
import os
from datetime import datetime, timezone

import boto3
import pandas as pd

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
# Fetch FRED Series
# --------------------------------------------------------

def fetch_fred_series(series_id):

    print(f"Fetching FRED series {series_id}...")

    url = (
        "https://fred.stlouisfed.org/graph/"
        f"fredgraph.csv?id={series_id}"
    )

    # FRED currently provides the date column as
    # "observation_date"
    df = pd.read_csv(
        url,
        parse_dates=["observation_date"]
    )

    # Rename the date column to our standard name
    df = df.rename(
        columns={
            "observation_date": "date"
        }
    )

    # Set date as the index
    df = df.set_index("date")

    # Rename the value column to the series ID
    df.columns = [series_id]

    print(f"Downloaded {len(df)} observations for {series_id}")

    # ----------------------------------------------------
    # Convert to Parquet
    # ----------------------------------------------------

    parquet_buffer = io.BytesIO()

    df.to_parquet(
        parquet_buffer,
        engine="pyarrow"
    )

    parquet_buffer.seek(0)

    # ----------------------------------------------------
    # Upload to MinIO
    # ----------------------------------------------------

    file_key = (
        f"fred/{series_id}/"
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.parquet"
    )

    s3_client.upload_fileobj(
        parquet_buffer,
        BUCKET_NAME,
        file_key
    )

    print(
        f"Uploaded FRED series {series_id} "
        f"to MinIO: {file_key}"
    )


# --------------------------------------------------------
# Main
# --------------------------------------------------------

if __name__ == "__main__":

    # DFF = Federal Funds Rate
    # GDP = Gross Domestic Product

    for series in ["DFF", "GDP"]:
        fetch_fred_series(series)