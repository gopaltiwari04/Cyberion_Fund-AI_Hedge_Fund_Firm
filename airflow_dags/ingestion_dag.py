from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

# Point to your project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

default_args = {
    'owner': 'quant_dev',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'daily_market_data_ingestion',
    default_args=default_args,
    description='Fetches daily OHLCV and Macro data',
    schedule_interval='0 18 * * 1-5', # Run at 6 PM UTC Mon-Fri
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    ingest_market_data = BashOperator(
        task_id='ingest_yfinance',
        bash_command='cd /opt/airflow/dags/project && python data_pipeline/ingest_market_data.py'
    )

    ingest_macro_data = BashOperator(
        task_id='ingest_fred',
        bash_command='cd /opt/airflow/dags/project && python data_pipeline/ingest_fred.py'
    )

    # Run them in parallel
    [ingest_market_data, ingest_macro_data]