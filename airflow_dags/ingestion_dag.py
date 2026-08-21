from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta


default_args = {
    "owner": "quant_dev",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    "daily_market_data_ingestion",
    default_args=default_args,
    description="Ingests macro data, then cleans and loads market data",
    schedule_interval="0 18 * * 1-5",
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    # --------------------------------------------------------
    # 1. Ingest macro data from FRED -> MinIO
    # --------------------------------------------------------

    ingest_macro_data = BashOperator(
        task_id="ingest_fred",

        bash_command=(
            "cd /opt/airflow/dags/project && "
            "python dags/ingest_fred.py"
        ),

        env={
            "MINIO_URL": "http://minio:9000",
        },
    )

    # --------------------------------------------------------
    # 2. Clean + Validate + Load -> PostgreSQL
    # --------------------------------------------------------

    clean_data = BashOperator(
        task_id="clean_and_load_postgres",

        bash_command=(
            "cd /opt/airflow/dags/project && "
            "python data_pipeline/clean_and_load.py"
        ),

        env={
            "MINIO_ENDPOINT": "http://minio:9000",

            "DB_URL": (
                "postgresql://quant_user:quant_password"
                "@postgres:5432/quant_db"
            ),
        },
    )

    # --------------------------------------------------------
    # Dependency
    # --------------------------------------------------------

    ingest_macro_data >> clean_data