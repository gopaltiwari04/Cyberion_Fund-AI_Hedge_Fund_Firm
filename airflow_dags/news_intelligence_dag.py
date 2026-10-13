"""Hourly news pipeline: Marketaux -> FinBERT sentiment -> point-in-time daily aggregates + snapshot."""

from datetime import datetime, timedelta

from airflow import DAG

from common_dag import DEFAULT_ARGS, pipeline_task

with DAG(
    dag_id="cyberion_news_intelligence",
    description="Fetch financial news, score FinBERT sentiment and build news features",
    default_args={**DEFAULT_ARGS, "retries": 2},
    schedule="5 * * * *",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=50),
    tags=["cyberion", "news", "intelligence"],
) as dag:
    ingest_news = pipeline_task("ingest_news", "data_pipeline.news_ingestion", timeout_minutes=10)
    sentiment = pipeline_task("finbert_sentiment", "data_pipeline.sentiment_scoring", "--skip-filings", timeout_minutes=30)
    snapshot = pipeline_task("news_snapshot", "data_pipeline.news_features", timeout_minutes=10)

    ingest_news >> sentiment >> snapshot
