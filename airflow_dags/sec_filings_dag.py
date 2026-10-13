"""Daily SEC EDGAR pipeline: register/download filings -> RAG index -> 8-K FinBERT sentiment."""

from datetime import datetime, timedelta

from airflow import DAG

from common_dag import DEFAULT_ARGS, pipeline_task

with DAG(
    dag_id="cyberion_sec_filings",
    description="SEC 10-K/10-Q/8-K ingestion, chunking, embeddings and filing sentiment",
    default_args=DEFAULT_ARGS,
    schedule="15 11 * * *",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=3),
    tags=["cyberion", "sec", "rag"],
) as dag:
    ingest = pipeline_task("ingest_filings", "data_pipeline.sec_ingestion", timeout_minutes=90)
    index = pipeline_task("index_filings", "ml_core.rag.indexer", timeout_minutes=90)
    sentiment = pipeline_task("filing_sentiment", "data_pipeline.sentiment_scoring", timeout_minutes=90)

    ingest >> [index, sentiment]
