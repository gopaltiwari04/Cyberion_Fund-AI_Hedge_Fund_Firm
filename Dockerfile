FROM apache/airflow:2.8.1-python3.10

RUN pip install --no-cache-dir \
    "apache-airflow==2.8.1" \
    yfinance==0.2.38 \
    pandas \
    pyarrow \
    boto3 \
    requests \
    polars \
    sqlalchemy \
    psycopg2-binary \
    great-expectations==1.21.0