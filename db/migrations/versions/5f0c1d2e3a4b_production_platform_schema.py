"""Production platform schema

Additive and idempotent: every statement uses IF NOT EXISTS (or checks the
catalog first) so it can run against databases that were partially created
outside Alembic (e.g. financial_news / news_features) and against a fresh
database in CI. Nothing is dropped or rewritten.

Revision ID: 5f0c1d2e3a4b
Revises: 408bf28f87df
Create Date: 2026-10-13
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "5f0c1d2e3a4b"
down_revision: Union[str, Sequence[str], None] = "408bf28f87df"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


FEATURE_COLUMNS = [
    "news_sentiment_1d",
    "news_sentiment_7d",
    "news_count_7d",
    "filing_count_30d",
    "days_since_filing",
    "filing_sentiment_90d",
    "fed_funds_rate",
    "treasury_10y",
    "term_spread_10y2y",
    "vix",
    "credit_spread_baa",
    "corr_degree_60d",
    "corr_eigenvector_60d",
    "peer_return_5d",
    "kg_degree",
    "kg_pagerank",
]


def _has_unique_index(conn, table: str, columns: list[str]) -> bool:
    inspector = sa.inspect(conn)
    for constraint in inspector.get_unique_constraints(table):
        if constraint["column_names"] == columns:
            return True
    for index in inspector.get_indexes(table):
        if index.get("unique") and index["column_names"] == columns:
            return True
    for pk in [inspector.get_pk_constraint(table)]:
        if pk and pk.get("constrained_columns") == columns:
            return True
    return False


def _vector_available(conn) -> bool:
    try:
        conn.execute(sa.text("SAVEPOINT vector_ext"))
        conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(sa.text("RELEASE SAVEPOINT vector_ext"))
    except Exception:  # noqa: BLE001 - extension may be unavailable on some hosts
        conn.execute(sa.text("ROLLBACK TO SAVEPOINT vector_ext"))
    return bool(conn.execute(sa.text("SELECT 1 FROM pg_type WHERE typname = 'vector'")).scalar())


def upgrade() -> None:
    conn = op.get_bind()
    execute = op.execute

    # ------------------------------------------------------------------
    # Reference data
    # ------------------------------------------------------------------
    execute(
        """
        CREATE TABLE IF NOT EXISTS assets (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR NOT NULL UNIQUE,
            asset_class VARCHAR,
            sector VARCHAR
        )
        """
    )
    execute(
        """
        ALTER TABLE assets
            ADD COLUMN IF NOT EXISTS name TEXT,
            ADD COLUMN IF NOT EXISTS cik TEXT,
            ADD COLUMN IF NOT EXISTS sic_code TEXT,
            ADD COLUMN IF NOT EXISTS sic_description TEXT,
            ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE,
            ADD COLUMN IF NOT EXISTS first_trade_date DATE,
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        """
    )

    # ------------------------------------------------------------------
    # Market data: corporate actions + lineage
    # ------------------------------------------------------------------
    execute(
        """
        ALTER TABLE market_data
            ADD COLUMN IF NOT EXISTS dividend DOUBLE PRECISION NOT NULL DEFAULT 0,
            ADD COLUMN IF NOT EXISTS split_ratio DOUBLE PRECISION NOT NULL DEFAULT 1,
            ADD COLUMN IF NOT EXISTS adj_close DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS source TEXT,
            ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ DEFAULT now()
        """
    )
    if not _has_unique_index(conn, "market_data", ["ticker", "date"]):
        execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_market_data_ticker_date ON market_data (ticker, date)")

    execute(
        """
        CREATE TABLE IF NOT EXISTS macro_data (
            series_id TEXT NOT NULL,
            date DATE NOT NULL,
            value DOUBLE PRECISION,
            source TEXT NOT NULL DEFAULT 'fred',
            ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (series_id, date)
        )
        """
    )

    # ------------------------------------------------------------------
    # News (tables previously created outside Alembic)
    # ------------------------------------------------------------------
    execute(
        """
        CREATE TABLE IF NOT EXISTS financial_news (
            id BIGSERIAL PRIMARY KEY,
            uuid TEXT NOT NULL,
            ticker TEXT,
            title TEXT,
            description TEXT,
            snippet TEXT,
            url TEXT,
            image_url TEXT,
            source TEXT,
            published_at TIMESTAMPTZ,
            language TEXT,
            entity_name TEXT,
            entity_sentiment DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT now(),
            CONSTRAINT financial_news_uuid_ticker_key UNIQUE (uuid, ticker)
        )
        """
    )
    execute(
        """
        ALTER TABLE financial_news
            ADD COLUMN IF NOT EXISTS finbert_positive DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS finbert_negative DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS finbert_neutral DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS finbert_score DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS scored_at TIMESTAMPTZ
        """
    )
    execute("CREATE INDEX IF NOT EXISTS ix_financial_news_ticker_published ON financial_news (ticker, published_at DESC)")
    execute(
        """
        CREATE TABLE IF NOT EXISTS news_features (
            ticker TEXT PRIMARY KEY,
            feature_timestamp TIMESTAMPTZ NOT NULL,
            news_count_24h INTEGER NOT NULL,
            news_sentiment_mean_24h DOUBLE PRECISION NOT NULL,
            news_sentiment_mean_7d DOUBLE PRECISION NOT NULL,
            news_positive_ratio_24h DOUBLE PRECISION NOT NULL,
            news_negative_ratio_24h DOUBLE PRECISION NOT NULL,
            news_sentiment_momentum DOUBLE PRECISION NOT NULL
        )
        """
    )
    execute(
        """
        CREATE TABLE IF NOT EXISTS news_sentiment_daily (
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            article_count INTEGER NOT NULL,
            finbert_sentiment_mean DOUBLE PRECISION,
            marketaux_sentiment_mean DOUBLE PRECISION,
            positive_ratio DOUBLE PRECISION,
            negative_ratio DOUBLE PRECISION,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (ticker, date)
        )
        """
    )

    # ------------------------------------------------------------------
    # SEC filings + RAG chunks
    # ------------------------------------------------------------------
    execute(
        """
        CREATE TABLE IF NOT EXISTS sec_filings (
            accession_no TEXT PRIMARY KEY,
            cik TEXT NOT NULL,
            ticker TEXT NOT NULL,
            form_type TEXT NOT NULL,
            filing_date DATE NOT NULL,
            accepted_at TIMESTAMPTZ,
            report_date DATE,
            items TEXT,
            primary_document TEXT,
            document_url TEXT NOT NULL,
            raw_object_key TEXT,
            text_chars INTEGER,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            finbert_score DOUBLE PRECISION,
            finbert_positive DOUBLE PRECISION,
            finbert_negative DOUBLE PRECISION,
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT,
            ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            processed_at TIMESTAMPTZ
        )
        """
    )
    execute("CREATE INDEX IF NOT EXISTS ix_sec_filings_ticker_date ON sec_filings (ticker, filing_date DESC)")
    execute("CREATE INDEX IF NOT EXISTS ix_sec_filings_status ON sec_filings (status)")
    execute(
        """
        CREATE TABLE IF NOT EXISTS sec_filing_chunks (
            id BIGSERIAL PRIMARY KEY,
            accession_no TEXT NOT NULL REFERENCES sec_filings (accession_no) ON DELETE CASCADE,
            ticker TEXT NOT NULL,
            form_type TEXT NOT NULL,
            filing_date DATE NOT NULL,
            section TEXT,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            embedding_model TEXT,
            search_vector TSVECTOR GENERATED ALWAYS AS (
                to_tsvector('english', coalesce(section, '') || ' ' || content)
            ) STORED,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_sec_chunk UNIQUE (accession_no, chunk_index)
        )
        """
    )
    execute("CREATE INDEX IF NOT EXISTS ix_sec_chunks_search ON sec_filing_chunks USING gin (search_vector)")
    execute("CREATE INDEX IF NOT EXISTS ix_sec_chunks_ticker_date ON sec_filing_chunks (ticker, filing_date DESC)")
    if _vector_available(conn):
        execute("ALTER TABLE sec_filing_chunks ADD COLUMN IF NOT EXISTS embedding vector(384)")
        execute(
            "CREATE INDEX IF NOT EXISTS ix_sec_chunks_embedding ON sec_filing_chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )

    # ------------------------------------------------------------------
    # Feature store: sentiment, filing, macro and graph signals
    # ------------------------------------------------------------------
    execute(
        "ALTER TABLE feature_store "
        + ", ".join(f"ADD COLUMN IF NOT EXISTS {column} DOUBLE PRECISION" for column in FEATURE_COLUMNS)
        + ", ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now()"
    )

    # ------------------------------------------------------------------
    # Model registry, predictions and monitoring
    # ------------------------------------------------------------------
    execute(
        """
        CREATE TABLE IF NOT EXISTS model_registry (
            id SERIAL PRIMARY KEY,
            model_name TEXT NOT NULL,
            version TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            target TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'candidate',
            trained_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            train_start DATE,
            train_end DATE,
            data_end DATE,
            feature_columns JSONB NOT NULL,
            params JSONB,
            metrics JSONB,
            calibration JSONB,
            feature_importance JSONB,
            data_hash TEXT,
            code_version TEXT,
            artifact_key TEXT,
            artifact BYTEA,
            artifact_sha256 TEXT,
            notes TEXT,
            CONSTRAINT uq_model_version UNIQUE (model_name, version)
        )
        """
    )
    execute("CREATE INDEX IF NOT EXISTS ix_model_registry_status ON model_registry (status, horizon_days)")
    execute(
        """
        CREATE TABLE IF NOT EXISTS model_predictions (
            id BIGSERIAL PRIMARY KEY,
            model_id INTEGER NOT NULL REFERENCES model_registry (id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            ticker TEXT NOT NULL,
            as_of_date DATE NOT NULL,
            horizon_days INTEGER NOT NULL,
            pred_return DOUBLE PRECISION,
            pred_lower DOUBLE PRECISION,
            pred_upper DOUBLE PRECISION,
            interval_level DOUBLE PRECISION,
            pred_volatility DOUBLE PRECISION,
            train_end DATE,
            realized_return DOUBLE PRECISION,
            realized_volatility DOUBLE PRECISION,
            explanation JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_model_prediction UNIQUE (model_id, kind, ticker, as_of_date, horizon_days)
        )
        """
    )
    execute("CREATE INDEX IF NOT EXISTS ix_model_predictions_lookup ON model_predictions (kind, horizon_days, as_of_date DESC)")
    execute(
        """
        CREATE TABLE IF NOT EXISTS model_monitoring (
            id BIGSERIAL PRIMARY KEY,
            model_id INTEGER REFERENCES model_registry (id) ON DELETE CASCADE,
            as_of_date DATE NOT NULL,
            metric TEXT NOT NULL,
            value DOUBLE PRECISION,
            details JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_model_monitoring UNIQUE (model_id, as_of_date, metric)
        )
        """
    )

    # ------------------------------------------------------------------
    # Portfolio outputs: transparent risk reporting
    # ------------------------------------------------------------------
    execute(
        """
        ALTER TABLE portfolio_allocations
            ADD COLUMN IF NOT EXISTS model_id INTEGER,
            ADD COLUMN IF NOT EXISTS horizon_days INTEGER,
            ADD COLUMN IF NOT EXISTS predicted_volatility DOUBLE PRECISION
        """
    )
    execute(
        """
        ALTER TABLE portfolio_risk_metrics
            ADD COLUMN IF NOT EXISTS model_id INTEGER,
            ADD COLUMN IF NOT EXISTS horizon_days INTEGER,
            ADD COLUMN IF NOT EXISTS cvar_limit DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS solver_status TEXT,
            ADD COLUMN IF NOT EXISTS feasible BOOLEAN,
            ADD COLUMN IF NOT EXISTS turnover DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS transaction_cost DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS assumptions JSONB,
            ADD COLUMN IF NOT EXISTS diagnostics JSONB
        """
    )

    # ------------------------------------------------------------------
    # Users, recommendations and backtests (written by the API)
    # ------------------------------------------------------------------
    execute(
        """
        CREATE TABLE IF NOT EXISTS user_profiles (
            email TEXT PRIMARY KEY,
            display_name TEXT,
            risk_profile TEXT NOT NULL DEFAULT 'balanced',
            cvar_limit DOUBLE PRECISION NOT NULL DEFAULT 0.025,
            max_weight DOUBLE PRECISION NOT NULL DEFAULT 0.30,
            max_turnover DOUBLE PRECISION NOT NULL DEFAULT 0.50,
            transaction_cost_bps DOUBLE PRECISION NOT NULL DEFAULT 5,
            cash_balance DOUBLE PRECISION NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    execute(
        """
        CREATE TABLE IF NOT EXISTS user_holdings (
            email TEXT NOT NULL REFERENCES user_profiles (email) ON DELETE CASCADE,
            ticker TEXT NOT NULL,
            quantity DOUBLE PRECISION NOT NULL CHECK (quantity >= 0),
            average_cost DOUBLE PRECISION CHECK (average_cost IS NULL OR average_cost >= 0),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (email, ticker)
        )
        """
    )
    execute(
        """
        CREATE TABLE IF NOT EXISTS recommendation_runs (
            id UUID PRIMARY KEY,
            email TEXT NOT NULL,
            model_id INTEGER,
            request JSONB NOT NULL,
            result JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    execute("CREATE INDEX IF NOT EXISTS ix_recommendation_runs_email ON recommendation_runs (email, created_at DESC)")
    execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id UUID PRIMARY KEY,
            email TEXT NOT NULL,
            status TEXT NOT NULL,
            config JSONB NOT NULL,
            config_hash TEXT NOT NULL,
            result JSONB,
            error TEXT,
            code_version TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ
        )
        """
    )
    execute("CREATE INDEX IF NOT EXISTS ix_backtest_runs_email ON backtest_runs (email, created_at DESC)")
    execute(
        """
        CREATE TABLE IF NOT EXISTS research_queries (
            id BIGSERIAL PRIMARY KEY,
            email TEXT NOT NULL,
            ticker TEXT,
            question TEXT NOT NULL,
            answer JSONB,
            latency_ms INTEGER,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id BIGSERIAL PRIMARY KEY,
            pipeline TEXT NOT NULL,
            task TEXT NOT NULL,
            run_id TEXT,
            status TEXT NOT NULL,
            started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            finished_at TIMESTAMPTZ,
            rows_processed INTEGER NOT NULL DEFAULT 0,
            details JSONB,
            error TEXT
        )
        """
    )
    execute("CREATE INDEX IF NOT EXISTS ix_pipeline_runs_task ON pipeline_runs (pipeline, task, started_at DESC)")


def downgrade() -> None:
    # Intentionally non-destructive: production data in these tables must not be
    # dropped by an automated downgrade. Remove objects manually if required.
    pass
