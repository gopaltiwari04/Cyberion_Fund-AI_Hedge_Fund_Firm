"""
Centralised configuration.

Every process (Airflow tasks, training jobs, the API) reads settings from
environment variables through this module so local Docker, Render and CI
stay consistent. A local ``.env`` file is loaded when python-dotenv is
installed, but real environment variables always take precedence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:  # Optional in production images.
    from dotenv import load_dotenv

    # ENV_FILE (e.g. ".env.neon") is loaded first so it can point a local process at a
    # different database without editing .env, which Docker/Airflow also use.
    if os.getenv("ENV_FILE"):
        load_dotenv(PROJECT_ROOT / os.environ["ENV_FILE"], override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:  # pragma: no cover - dotenv is a dev convenience
    pass


LOCAL_DATABASE_URL = "postgresql://quant_user:quant_password@localhost:5432/quant_db"


def normalize_database_url(url: str) -> str:
    """Render and Heroku style URLs use ``postgres://`` which SQLAlchemy rejects."""
    url = url.strip()
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def direct_database_url(url: str) -> str:
    """
    Neon exposes a PgBouncer (transaction-mode) endpoint whose host contains "-pooler".
    Migrations and publishing take session-level advisory locks, which need the direct
    endpoint; derive it when only the pooled URL is configured.
    """
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    host = parts.hostname or ""
    if "-pooler." not in host:
        return url
    return urlunsplit(parts._replace(netloc=parts.netloc.replace(host, host.replace("-pooler.", ".", 1))))


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return float(value) if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value is not None else default


@dataclass(frozen=True)
class Settings:
    environment: str
    database_url: str
    migration_database_url: str
    production_database_url: str | None

    storage_backend: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    local_lake_dir: Path

    redis_url: str | None
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    sec_user_agent: str | None
    marketaux_api_token: str | None
    fred_api_key: str | None
    gemini_api_key: str | None
    llm_model: str
    llm_fallback_models: tuple[str, ...]
    embedding_model: str
    rag_embeddings: str

    internal_api_secret: str | None
    frontend_origins: tuple[str, ...]
    alert_webhook_url: str | None
    run_migrations_on_startup: bool

    risk_free_fallback: float
    universe_override: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    database_url = normalize_database_url(
        _env("DATABASE_URL") or _env("DB_URL") or LOCAL_DATABASE_URL
    )
    production_url = _env("PRODUCTION_DATABASE_URL")

    origins = tuple(
        origin.strip().rstrip("/")
        for origin in (_env("FRONTEND_ORIGINS", "http://localhost:3000") or "").split(",")
        if origin.strip()
    )

    universe = tuple(
        ticker.strip().upper()
        for ticker in (_env("UNIVERSE_TICKERS", "") or "").split(",")
        if ticker.strip()
    )

    return Settings(
        environment=(_env("ENVIRONMENT", "development") or "development").lower(),
        database_url=database_url,
        migration_database_url=normalize_database_url(_env("MIGRATION_DATABASE_URL") or direct_database_url(database_url)),
        production_database_url=normalize_database_url(production_url) if production_url else None,
        storage_backend=(_env("STORAGE_BACKEND", "s3") or "s3").lower(),
        minio_endpoint=_env("MINIO_URL") or _env("MINIO_ENDPOINT") or "http://localhost:9000",
        minio_access_key=_env("MINIO_ACCESS_KEY") or _env("MINIO_ROOT_USER") or "minioadmin",
        minio_secret_key=_env("MINIO_SECRET_KEY") or _env("MINIO_ROOT_PASSWORD") or "minioadmin",
        minio_bucket=_env("MINIO_BUCKET", "raw-market-data") or "raw-market-data",
        local_lake_dir=Path(_env("LOCAL_LAKE_DIR", str(PROJECT_ROOT / "data_lake")) or "data_lake"),
        redis_url=_env("REDIS_URL"),
        neo4j_uri=_env("NEO4J_URI", "bolt://localhost:7687") or "bolt://localhost:7687",
        neo4j_user=_env("NEO4J_USER", "neo4j") or "neo4j",
        neo4j_password=_env("NEO4J_PASSWORD", "quant_password") or "quant_password",
        sec_user_agent=_env("SEC_USER_AGENT"),
        marketaux_api_token=_env("MARKETAUX_API_TOKEN"),
        fred_api_key=_env("FRED_API_KEY"),
        gemini_api_key=_env("GEMINI_API_KEY"),
        llm_model=_env("LLM_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash",
        llm_fallback_models=tuple(
            model.strip()
            for model in (_env("LLM_FALLBACK_MODELS", "gemini-flash-latest") or "").split(",")
            if model.strip()
        ),
        embedding_model=_env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5") or "BAAI/bge-small-en-v1.5",
        rag_embeddings=(_env("RAG_EMBEDDINGS", "auto") or "auto").lower(),
        internal_api_secret=_env("INTERNAL_API_SECRET"),
        frontend_origins=origins,
        alert_webhook_url=_env("ALERT_WEBHOOK_URL"),
        run_migrations_on_startup=_env_bool("RUN_MIGRATIONS_ON_STARTUP", False),
        risk_free_fallback=_env_float("RISK_FREE_FALLBACK", 0.0),
        universe_override=universe,
    )


def env_int(name: str, default: int) -> int:
    return _env_int(name, default)
