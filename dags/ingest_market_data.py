import os
import yfinance as yf
import pandas as pd
import boto3
import io
from datetime import datetime

# MinIO (S3) Configuration
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

BUCKET_NAME = 'raw-market-data'

def create_bucket_if_not_exists():
    try:
        s3_client.head_bucket(Bucket=BUCKET_NAME)
    except Exception:
        s3_client.create_bucket(Bucket=BUCKET_NAME)

def fetch_and_store_data(tickers):
    create_bucket_if_not_exists()
    
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        # Download 10 years of daily data
        df = yf.download(ticker, period="10y", interval="1d", progress=False)
        
        if df.empty:
            print(f"No data found for {ticker}")
            continue

        # Convert to Parquet (efficient columnar storage)
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, engine='pyarrow')
        parquet_buffer.seek(0)
        
        # Upload to MinIO Data Lake
        file_key = f"yfinance/{ticker}/{datetime.now().strftime('%Y-%m-%d')}.parquet"
        s3_client.upload_fileobj(parquet_buffer, BUCKET_NAME, file_key)
        print(f"Uploaded {file_key} to MinIO")

if __name__ == "__main__":
    target_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY'] # Start small for testing
    fetch_and_store_data(target_tickers)