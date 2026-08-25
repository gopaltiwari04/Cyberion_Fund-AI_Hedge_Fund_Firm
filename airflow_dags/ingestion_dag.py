from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.bash import BashOperator


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
    description="Ingests market data, financial text, embeddings, and sentiment",
    schedule_interval="0 18 * * 1-5",
    start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
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
    # 3. Ingest financial news + SEC filings
    # --------------------------------------------------------

    ingest_text = BashOperator(
        task_id="ingest_financial_text",

        bash_command=(
            "cd /opt/airflow/dags/project && "
            "python ml_core/nlp/ingest_text.py"
        ),

        env={
            "DB_URL": (
                "postgresql://quant_user:quant_password"
                "@postgres:5432/quant_db"
            ),
            "REDIS_HOST": "redis",
            "REDIS_PORT": "6379",
        },
    )

    # --------------------------------------------------------
    # 4. Clean financial text + generate embeddings
    # --------------------------------------------------------

    process_text = BashOperator(
        task_id="process_and_embed_text",

        bash_command=(
            "cd /opt/airflow/dags/project && "
            "python ml_core/nlp/process_text.py"
        ),

        env={
            "DB_URL": (
                "postgresql://quant_user:quant_password"
                "@postgres:5432/quant_db"
            ),
        },
    )

    # --------------------------------------------------------
    # 5. FinBERT sentiment analysis
    # --------------------------------------------------------

    score_sentiment = BashOperator(
        task_id="score_financial_sentiment",

        bash_command=(
            "cd /opt/airflow/dags/project && "
            "python ml_core/nlp/sentiment_analyzer.py"
        ),

        env={
            "DB_URL": (
                "postgresql://quant_user:quant_password"
                "@postgres:5432/quant_db"
            ),

            "REDIS_HOST": "redis",
            "REDIS_PORT": "6379",
        },
    )

    # --------------------------------------------------------
    # DAG DEPENDENCIES
    # --------------------------------------------------------

    ingest_macro_data >> clean_data

    clean_data >> ingest_text >> process_text >> score_sentiment