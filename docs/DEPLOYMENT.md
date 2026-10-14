# Deployment

Three deployable parts share one environment contract (`.env.example`):

| Component | Where | Entry point |
|---|---|---|
| API | Render web service | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` (unchanged) |
| Production DB | Neon PostgreSQL (project `cyberion-fund`, pgvector 0.8.6) | — |
| Dashboard | Vercel (root directory `frontend/`) | `next build` |
| Pipelines | Docker host running `docker compose up -d` (Airflow) | DAGs in `airflow_dags/` |

## 0. Production database: Neon

The original Render PostgreSQL (`aihedgefund-db`) expired. Production now uses Neon:

* **Pooled URL** (host contains `-pooler`, PgBouncer transaction mode) for the API (`DATABASE_URL`).
* **Direct URL** for migrations and publishing — derived automatically from the pooled URL
  (`common.config.direct_database_url`) or set explicitly as `MIGRATION_DATABASE_URL`. Session-level
  advisory locks used by migrations do not work through transaction pooling.
* Always `sslmode=require` (Neon URLs include it; TLS 1.3 verified). Neon suspends idle computes, so the
  engine pings and recycles connections (`DB_POOL_RECYCLE`, keepalives).
* Capacity: the serving data set is ~232 MB on Neon (24k SEC chunks with 384-d vectors are 177 MB of it).
  Check the project's storage limit before widening `SEC_RAG_YEARS` or the universe.

Initial load (done 2026-10-15, verified row-for-row): `python -m data_pipeline.publish --full` with
`PRODUCTION_DATABASE_URL` set to the Neon URL. It migrates the schema to head, then upserts serving tables in
500-row batches (one round trip per batch).

Local development against Neon, without touching `.env` (which Docker/Airflow use):

```bash
# .env.neon (git- and docker-ignored):  DATABASE_URL=<neon connection string>
ENV_FILE=.env.neon uvicorn api.main:app --host 127.0.0.1 --port 8001
# frontend/.env.local: FASTAPI_URL=http://127.0.0.1:8001 ; dashboard on http://localhost:3001
```

## 1. Render API

Settings (dashboard or `render.yaml` blueprint):

* Build: `pip install -r requirements-api.txt` (lean: no torch/xgboost; forecasts are precomputed)
* Start: `uvicorn api.main:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips='*'`
* Health check path: `/health`
* Environment:
  * `ENVIRONMENT=production`
  * `DATABASE_URL` — **Neon pooled connection string** (`postgres://` is normalised automatically)
  * `DB_POOL_SIZE=3`, `DB_MAX_OVERFLOW=2`, `DB_POOL_RECYCLE=240`, `BACKTEST_WORKERS=1` on the free instance
  * `RUN_MIGRATIONS_ON_STARTUP=true` — applies Alembic migrations under a Postgres advisory lock. The
    production migration is additive and idempotent (`IF NOT EXISTS`); it never drops data.
  * `INTERNAL_API_SECRET` — same value as Vercel. **Until it is set, production serves only public
    read-only endpoints and returns 503 for user features** (holdings, recommendations, analyst, backtests).
  * `GEMINI_API_KEY` (analyst generation), `FRONTEND_ORIGINS`, `PYTHON_VERSION=3.12.8`
  * `RAG_EMBEDDINGS=auto` (or `off` on 512 MB instances to skip the ~150 MB ONNX embedding model; the
    analyst then uses full-text retrieval only).

Verify after deploy:

```bash
curl https://<service>.onrender.com/health          # {"status":"healthy","database":"connected"}
curl https://<service>.onrender.com/health/ready    # schema_version 5f0c1d2e3a4b + dataset freshness
```

## 2. Vercel dashboard

Project root `frontend/`. Environment variables:

* `AUTH_SECRET`, `AUTH_GOOGLE_ID`, `AUTH_GOOGLE_SECRET` (Google OAuth redirect URI:
  `https://<app>.vercel.app/api/auth/callback/google`)
* `FASTAPI_URL=https://<service>.onrender.com`
* Google OAuth client: add `https://<app>.vercel.app/api/auth/callback/google` (and keep
  `http://localhost:3001/api/auth/callback/google` for local development)
* `INTERNAL_API_SECRET` (same as Render)

All browser API calls go to `/api/*` on Vercel; the route handler checks the NextAuth session, allow-lists
the path, signs the request and forwards it. The API is never called directly from the browser.

**Deployment order for the auth change:** set `INTERNAL_API_SECRET` on Render and Vercel first, then deploy
both. The API stays backward compatible for read-only endpoints in the meantime.

## 3. Pipelines and data publishing

The Airflow stack writes to its local warehouse database. To feed production:

1. Set `PRODUCTION_DATABASE_URL` in `.env` to the Neon connection string (pooled or direct; the direct
   endpoint is used automatically).
2. `docker compose up -d airflow`. The `publish_to_production` task (after the daily market pipeline and
   weekly training) migrates the production schema and incrementally upserts serving tables using natural
   keys and watermarks. User-owned tables are never touched and model artifacts are not copied.
3. First publish: `python -m data_pipeline.publish --full` (or trigger the DAG task).

Alternatively point Airflow's `DATABASE_URL` straight at the production database (simpler, but the full
SEC text and raw history then live in the production database; check its storage plan first).

## 4. Rollback

* API/dashboard: redeploy the previous commit in Render/Vercel.
* Models: `UPDATE model_registry SET status='active' WHERE id=<previous id>` after archiving the current one
  (or call `ml_core.models.registry.promote(<id>)`), then rerun `ml_core.models.inference`.
* Schema: migrations are additive; the downgrade is intentionally a no-op to protect production data.

## 5. Local Docker notes

* Airflow now uses a Postgres metadata DB (`airflow`, created by the `airflow-db-init` service) and the
  LocalExecutor, so DAG state survives container recreation.
* Pipeline tasks run in `/opt/pipeline-venv` (SQLAlchemy 2, pandas 2.3, CPU torch) isolated from Airflow's
  own dependencies.
* Host ports are configurable (`POSTGRES_PORT`, `AIRFLOW_PORT`, …) if other local stacks use the defaults.
* `docker compose --profile api up -d api` runs the API image used for Render parity checks.
