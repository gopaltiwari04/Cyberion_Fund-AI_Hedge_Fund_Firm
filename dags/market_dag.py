from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Define the scheduling rules
default_args = {
    'owner': 'quant_user',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Instantiate the DAG so Airflow's scanner can see it
with DAG(
    'daily_market_data_ingestion',
    default_args=default_args,
    description='Automated pipeline for YFinance and FRED data',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['finance'],
) as dag:

    # Task 1
    ingest_yfinance = BashOperator(
        task_id='ingest_yfinance_data',
        bash_command='python /opt/airflow/dags/ingest_market_data.py',
    )

    # Task 2
    ingest_fred = BashOperator(
        task_id='ingest_fred_data',
        bash_command='python /opt/airflow/dags/ingest_fred.py',
    )

    # Set the execution order (YFinance runs first, then FRED)
    ingest_yfinance >> ingest_fred