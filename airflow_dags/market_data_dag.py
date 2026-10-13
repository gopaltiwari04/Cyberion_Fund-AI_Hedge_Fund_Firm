"""
Daily market pipeline (after the US close, Mon-Fri).

extract_prices ─> clean_load_prices ─┐
ingest_fred ─────────────────────────┼─> engineer_features ─> live_inference ─> house_portfolio ─> model_monitoring
                                     └─> sync_knowledge_graph                                      └─> publish_to_production
"""

from datetime import datetime, timedelta

from airflow import DAG

from common_dag import DEFAULT_ARGS, pipeline_task

with DAG(
    dag_id="cyberion_market_pipeline",
    description="Prices, macro, features, forecasts, risk-constrained portfolio and monitoring",
    default_args=DEFAULT_ARGS,
    schedule="30 22 * * 1-5",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=4),
    tags=["cyberion", "market", "daily"],
) as dag:
    extract_prices = pipeline_task("extract_prices", "data_pipeline.market_ingestion", timeout_minutes=45)
    load_prices = pipeline_task("clean_load_prices", "data_pipeline.clean_and_load", timeout_minutes=30)
    ingest_fred = pipeline_task("ingest_fred", "data_pipeline.macro_ingestion", timeout_minutes=15)
    features = pipeline_task("engineer_features", "ml_core.feature_store.engineer_features", timeout_minutes=45)
    graph = pipeline_task("sync_knowledge_graph", "ml_core.graph.build_graph", timeout_minutes=20)
    inference = pipeline_task("live_inference", "ml_core.models.inference", timeout_minutes=20)
    portfolio = pipeline_task("house_portfolio", "ml_core.portfolio.portfolio_optimizer", timeout_minutes=20)
    monitoring = pipeline_task("model_monitoring", "ml_core.monitoring.drift", timeout_minutes=20, retries=1)
    publish = pipeline_task("publish_to_production", "data_pipeline.publish", timeout_minutes=60)

    extract_prices >> load_prices
    [load_prices, ingest_fred] >> features
    load_prices >> graph
    features >> inference >> portfolio >> [monitoring, publish]
