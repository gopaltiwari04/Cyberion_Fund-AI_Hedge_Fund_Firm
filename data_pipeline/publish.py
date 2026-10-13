"""
Publish serving tables from the pipeline warehouse to the production database.

The Airflow stack (local Docker or a self-hosted worker) owns the warehouse.
The deployed API (Render) reads a production Postgres. When
PRODUCTION_DATABASE_URL is set, this task:
    1. migrates the production schema (idempotent, advisory-locked)
    2. incrementally upserts every serving table using natural keys and
       per-table watermarks (never deletes production-owned data)
    3. reconciles derived data that the warehouse replaced
       (re-indexed SEC chunks, pruned walk-forward predictions, portfolio days)

Production-owned tables (user_profiles, user_holdings, recommendation_runs,
backtest_runs, research_queries) are never touched. Model artifacts (bytea)
are not copied: the API only serves precomputed predictions.

Run: python -m data_pipeline.publish [--full]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from common.config import get_settings
from common.db import get_engine
from common.logs import get_logger
from common.runs import track_run

logger = get_logger(__name__)
BATCH = 2000


@dataclass(frozen=True)
class TableSpec:
    name: str
    keys: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    incremental: str | None = None  # SQL predicate on local rows; :wm is the production watermark
    watermark: str | None = None     # SQL expression evaluated on production for the watermark


TABLES = [
    TableSpec("assets", ("ticker",), exclude=("id",)),
    TableSpec("market_data", ("ticker", "date"), exclude=("id",),
              incremental="ingested_at > :wm", watermark="MAX(ingested_at)"),
    TableSpec("macro_data", ("series_id", "date"), incremental="ingested_at > :wm", watermark="MAX(ingested_at)"),
    TableSpec("feature_store", ("ticker", "date"), exclude=("id",),
              incremental="updated_at > :wm", watermark="MAX(updated_at)"),
    TableSpec("financial_news", ("uuid", "ticker"), exclude=("id",),
              incremental="COALESCE(scored_at, created_at) > :wm", watermark="MAX(COALESCE(scored_at, created_at))"),
    TableSpec("news_features", ("ticker",)),
    TableSpec("news_sentiment_daily", ("ticker", "date"), incremental="updated_at > :wm", watermark="MAX(updated_at)"),
    TableSpec("sec_filings", ("accession_no",),
              incremental="COALESCE(processed_at, ingested_at) > :wm", watermark="MAX(COALESCE(processed_at, ingested_at))"),
    TableSpec("sec_filing_chunks", ("accession_no", "chunk_index"), exclude=("id", "search_vector"),
              incremental="created_at > :wm", watermark="MAX(created_at)"),
    TableSpec("model_registry", ("id",), exclude=("artifact",)),
    TableSpec("model_predictions", ("model_id", "kind", "ticker", "as_of_date", "horizon_days"), exclude=("id",),
              incremental="(created_at > :wm OR as_of_date >= CURRENT_DATE - 60)", watermark="MAX(created_at)"),
    TableSpec("model_monitoring", ("model_id", "as_of_date", "metric"), exclude=("id",),
              incremental="created_at > :wm", watermark="MAX(created_at)"),
    TableSpec("pipeline_runs", ("id",), incremental="started_at > :wm - interval '2 days'", watermark="MAX(started_at)"),
]
PORTFOLIO_TABLES = ("portfolio_allocations", "portfolio_risk_metrics")


def column_types(engine: Engine, table: str) -> dict[str, str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT column_name, udt_name FROM information_schema.columns WHERE table_name = :t ORDER BY ordinal_position"),
            {"t": table},
        ).all()
    return {name: udt for name, udt in rows}


def _select_list(columns: list[str], types: dict[str, str]) -> str:
    parts = []
    for column in columns:
        if types[column] in ("jsonb", "json", "vector"):
            parts.append(f"{column}::text AS {column}")
        else:
            parts.append(column)
    return ", ".join(parts)


def _value_list(columns: list[str], types: dict[str, str]) -> str:
    parts = []
    for column in columns:
        if types[column] in ("jsonb", "json"):
            parts.append(f"CAST(:{column} AS jsonb)")
        elif types[column] == "vector":
            parts.append(f"CAST(:{column} AS vector)")
        else:
            parts.append(f":{column}")
    return ", ".join(parts)


def _records(frame: pd.DataFrame) -> list[dict]:
    frame = frame.astype(object).where(frame.notna(), None)
    return frame.to_dict(orient="records")


def sync_table(local: Engine, prod: Engine, spec: TableSpec, full: bool) -> int:
    local_types = column_types(local, spec.name)
    prod_types = column_types(prod, spec.name)
    columns = [c for c in local_types if c in prod_types and c not in spec.exclude]
    query = f"SELECT {_select_list(columns, local_types)} FROM {spec.name}"
    params: dict = {}
    if spec.incremental and not full:
        with prod.connect() as conn:
            watermark = conn.execute(text(f"SELECT {spec.watermark} FROM {spec.name}")).scalar()
        if watermark is not None:
            query += f" WHERE {spec.incremental}"
            params["wm"] = watermark
    updates = [c for c in columns if c not in spec.keys]
    conflict = f"DO UPDATE SET {', '.join(f'{c} = EXCLUDED.{c}' for c in updates)}" if updates else "DO NOTHING"
    insert = text(
        f"INSERT INTO {spec.name} ({', '.join(columns)}) VALUES ({_value_list(columns, prod_types)}) "
        f"ON CONFLICT ({', '.join(spec.keys)}) {conflict}"
    )
    total = 0
    with local.connect() as source:
        for chunk in pd.read_sql(text(query), source, params=params, chunksize=BATCH):
            records = _records(chunk)
            if not records:  # pandas yields one empty chunk for an empty result
                continue
            for record in records:
                for column in columns:
                    value = record[column]
                    if local_types[column] in ("jsonb", "json") and value is not None and not isinstance(value, str):
                        record[column] = json.dumps(value)
            with prod.begin() as target:
                target.execute(insert, records)
            total += len(records)
    return total


def reconcile(local: Engine, prod: Engine) -> dict:
    """Remove production rows that the warehouse has replaced or pruned."""
    with local.connect() as conn:
        chunk_counts = pd.read_sql(text("SELECT accession_no, chunk_count FROM sec_filings WHERE status = 'indexed'"), conn)
        kept_models = {row[0] for row in conn.execute(text(
            "SELECT DISTINCT model_id FROM model_predictions WHERE kind = 'walk_forward'"))}
    removed = {"chunks": 0, "walk_forward": 0}
    with prod.begin() as conn:
        if not chunk_counts.empty:
            result = conn.execute(
                text(
                    """
                    DELETE FROM sec_filing_chunks c USING (SELECT UNNEST(CAST(:a AS text[])) AS accession_no,
                        UNNEST(CAST(:n AS int[])) AS chunk_count) k
                    WHERE c.accession_no = k.accession_no AND c.chunk_index >= k.chunk_count
                    """
                ),
                {"a": chunk_counts["accession_no"].tolist(), "n": chunk_counts["chunk_count"].astype(int).tolist()},
            )
            removed["chunks"] = result.rowcount or 0
        if kept_models:
            result = conn.execute(
                text("DELETE FROM model_predictions WHERE kind = 'walk_forward' AND NOT (model_id = ANY(:m))"),
                {"m": sorted(kept_models)},
            )
            removed["walk_forward"] = result.rowcount or 0
        conn.execute(text("SELECT setval(pg_get_serial_sequence('model_registry', 'id'), "
                          "GREATEST((SELECT COALESCE(MAX(id), 1) FROM model_registry), 1))"))
        conn.execute(text("SELECT setval(pg_get_serial_sequence('pipeline_runs', 'id'), "
                          "GREATEST((SELECT COALESCE(MAX(id), 1) FROM pipeline_runs), 1))"))
    return removed


def sync_portfolios(local: Engine, prod: Engine, days: int = 10) -> int:
    """Portfolio tables have surrogate keys only: replace whole (date, strategy) days."""
    total = 0
    for table in PORTFOLIO_TABLES:
        local_types = column_types(local, table)
        prod_types = column_types(prod, table)
        columns = [c for c in local_types if c in prod_types and c != "id"]
        with local.connect() as conn:
            frame = pd.read_sql(
                text(f"SELECT {_select_list(columns, local_types)} FROM {table} WHERE date >= CURRENT_DATE - :d"),
                conn, params={"d": days},
            )
        if frame.empty:
            continue
        keys = frame[["date", "strategy_name"]].drop_duplicates()
        with prod.begin() as conn:
            for row in keys.itertuples():
                conn.execute(text(f"DELETE FROM {table} WHERE date = :d AND strategy_name = :s"),
                             {"d": row.date, "s": row.strategy_name})
            conn.execute(text(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({_value_list(columns, prod_types)})"),
                         _records(frame))
        total += len(frame)
    return total


def migrate(url: str) -> None:
    from alembic import command
    from alembic.config import Config

    from common.config import PROJECT_ROOT

    engine = get_engine(url)
    with engine.connect() as conn:
        conn.execute(text("SELECT pg_advisory_lock(727001)"))
        try:
            config = Config(str(PROJECT_ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(PROJECT_ROOT / "db" / "migrations"))
            config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
            config.attributes["skip_env_url"] = True
            command.upgrade(config, "head")
        finally:
            conn.execute(text("SELECT pg_advisory_unlock(727001)"))


def publish(target_url: str, full: bool = False) -> dict:
    migrate(target_url)
    local, prod = get_engine(), get_engine(target_url)
    summary = {}
    for spec in TABLES:
        summary[spec.name] = sync_table(local, prod, spec, full)
        logger.info("Published %s rows to %s", summary[spec.name], spec.name)
    summary["portfolio_rows"] = sync_portfolios(local, prod)
    summary["reconciled"] = reconcile(local, prod)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="ignore watermarks and republish every row")
    parser.add_argument("--target-url", help="override PRODUCTION_DATABASE_URL (e.g. a staging database)")
    args = parser.parse_args()
    target = args.target_url or get_settings().production_database_url
    with track_run("publish", "publish_to_production") as ctx:
        if not target:
            ctx.warn("PRODUCTION_DATABASE_URL not configured; publish skipped")
            return
        if target == get_settings().database_url:
            raise RuntimeError("Production URL equals the warehouse URL; refusing to publish onto itself")
        summary = publish(target, args.full)
        ctx.details.update(summary)
        ctx.add_rows(sum(v for v in summary.values() if isinstance(v, int)))


if __name__ == "__main__":
    main()
