import os
import requests
import pandas as pd
import boto3
import io
from datetime import datetime

s3_client = boto3.client('s3', endpoint_url='http://localhost:9000', aws_access_key_id='minioadmin', aws_secret_access_key='minioadmin')
BUCKET_NAME = 'raw-market-data'

def fetch_fred_series(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url, parse_dates=['DATE'], index_col='DATE')
    df.columns = [series_id]
    
    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer, engine='pyarrow')
    parquet_buffer.seek(0)
    
    file_key = f"fred/{series_id}/{datetime.now().strftime('%Y-%m-%d')}.parquet"
    s3_client.upload_fileobj(parquet_buffer, BUCKET_NAME, file_key)
    print(f"Uploaded FRED series {series_id} to MinIO")

if __name__ == "__main__":
    # DFF = Federal Funds Rate, GDP = Gross Domestic Product
    for series in ['DFF', 'GDP']:
        fetch_fred_series(series)