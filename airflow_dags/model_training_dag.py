"""
Weekly model training (Sunday): purged walk-forward XGBoost retraining with registry
promotion gates, TFT challenger evaluation, then refreshed forecasts and portfolio.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.utils.trigger_rule import TriggerRule

from common_dag import DEFAULT_ARGS, pipeline_task

with DAG(
    dag_id="cyberion_model_training",
    description="Walk-forward XGBoost training, TFT challenger, registry promotion",
    default_args={**DEFAULT_ARGS, "retries": 1},
    schedule="0 6 * * 0",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=6),
    tags=["cyberion", "ml", "weekly"],
) as dag:
    train = pipeline_task("train_xgboost", "ml_core.models.forecasting", timeout_minutes=180)
    tft = pipeline_task("tft_challenger", "ml_core.models.tft_challenger", timeout_minutes=180, retries=0)
    inference = pipeline_task("live_inference", "ml_core.models.inference", timeout_minutes=20,
                              trigger_rule=TriggerRule.ALL_DONE)
    portfolio = pipeline_task("house_portfolio", "ml_core.portfolio.portfolio_optimizer", timeout_minutes=20)
    publish = pipeline_task("publish_to_production", "data_pipeline.publish", timeout_minutes=60)

    train >> tft
    train >> inference >> portfolio >> publish
