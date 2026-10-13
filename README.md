# Cyberion — AI Quant Research Platform

An end-to-end quantitative research and decision-support platform: automated market, macro, news and SEC
data pipelines; multi-horizon return/volatility forecasting with honest out-of-sample evaluation; a
CVaR-constrained portfolio optimizer; SHAP explanations; a cited SEC research analyst; an event-driven
backtester; and an authenticated Next.js dashboard.

> **Disclaimer.** This is educational decision-support software. Forecasts are uncertain statistical
> estimates, not guaranteed returns. Nothing produced by the platform is personalised investment advice
> or an offer to buy or sell securities. Backtests are hypothetical and subject to the biases documented below.

## Architecture

```
            ┌─────────────── Airflow (Docker, LocalExecutor, pipeline venv) ───────────────┐
 yfinance ─▶│ extract_prices ─▶ clean_load (Polars + Great Expectations) ─┐                │
 FRED ─────▶│ ingest_fred ────────────────────────────────────────────────┼▶ features ─▶ inference ─▶ house portfolio ─▶ monitoring
 Marketaux ▶│ ingest_news ─▶ FinBERT ─▶ daily point-in-time sentiment ────┤        (TreeSHAP)   (CVaR optimizer)   (PSI, rolling IC)
 SEC EDGAR ▶│ register/download filings ─▶ chunk + embed (pgvector) ─▶ 8-K FinBERT            │
            │ weekly: purged walk-forward XGBoost ─▶ registry gates ─▶ TFT challenger           │
            └──────────── raw payloads ─▶ MinIO │ knowledge graph ─▶ Neo4j │ publish ─▶ prod DB ┘
                                                 ▼
                         PostgreSQL + pgvector (warehouse)  ──publish──▶  Render Postgres
                                                                              │
     Browser ─▶ Vercel (Next.js, Google auth) ─▶ /api/[...path] (session check + HMAC signing) ─▶ Render FastAPI
```

| Layer | Location | Notes |
|---|---|---|
| Shared runtime (config, DB, storage, run tracking, alerts) | `common/` | One `DATABASE_URL`/env contract for Docker, Render and CI |
| Pipelines | `data_pipeline/` | Idempotent, retryable, tracked in `pipeline_runs` |
| Features, graph, NLP, RAG, models, portfolio, backtest, monitoring | `ml_core/` | |
| DAGs | `airflow_dags/` | `cyberion_market_pipeline`, `cyberion_news_intelligence`, `cyberion_sec_filings`, `cyberion_model_training` |
| API | `api/` | FastAPI routers, signed-request auth, rate limits, safe errors |
| Dashboard | `frontend/` | Next.js 16, NextAuth (Google), Recharts |
| Schema | `db/migrations/` | Alembic; additive, idempotent production migration |
| Tests | `tests/`, `frontend/tests/` | Unit, end-to-end integration, frontend |

Detailed docs: [Data & pipelines](docs/DATA_PIPELINES.md) · [Models & evaluation](docs/MODEL_CARD.md) ·
[Risk, portfolio & backtesting](docs/PORTFOLIO_AND_BACKTESTING.md) · [Deployment](docs/DEPLOYMENT.md) ·
[Operations & runbook](docs/OPERATIONS.md) · [Security](docs/SECURITY.md) · [Limitations](docs/LIMITATIONS.md)

## Capabilities

* **Data engineering** — 12-asset universe (9 large caps + SPY, TLT, GLD) since 2015 with dividends/splits
  and automatic full re-fetch when a split re-adjusts history; FRED macro (fed funds, 2y/10y, curve, VIX,
  credit spread, CPI, unemployment); Marketaux news; SEC 10-K/10-Q/8-K with real EDGAR acceptance
  timestamps. Raw payloads land in MinIO under deterministic keys; Polars cleaning, de-duplication and
  validation (plus Great Expectations) before idempotent upserts. Every task records status, rows,
  warnings and errors in `pipeline_runs` and alerts via webhook on failure.
* **Predictive AI** — XGBoost bundles for 1/5/21-day horizons: cross-sectionally demeaned mean model,
  quantile model with split-conformal calibration (90% intervals), and a realised-volatility model.
  Purged + embargoed expanding walk-forward evaluation, configuration chosen on pre-2020 folds only,
  registry with promotion gates, data hashes and code versions. FinBERT sentiment, filing signals, macro
  and Neo4j knowledge-graph / correlation-network features — all point-in-time. A darts Temporal Fusion
  Transformer is trained weekly as a challenger on identical holdout rows.
* **Risk & portfolio** — CVXPY optimiser with a hard Rockafellar–Uryasev CVaR limit (verified post-solve),
  position/sector caps, turnover cap, amortised transaction costs, Ledoit–Wolf covariance, a cash asset at
  the fed funds rate, and forecast shrinkage by demonstrated skill. Infeasible problems return no trades plus
  the minimum achievable CVaR and whether relaxing turnover would help.
* **Explainability** — exact TreeSHAP (`pred_contribs`) per forecast, stored with sentiment/filing/graph
  signal values and shown next to every recommendation.
* **SEC analyst** — hybrid retrieval (pgvector + Postgres full-text, RRF), Gemini with a strict JSON schema,
  server-side citation validation, fact vs inference labelling, numeric grounding checks, extractive fallback.
* **Backtesting** — event-driven (next-open fills, slippage, commissions, ex-date dividends, cash interest),
  runtime look-ahead assertions, CVaR-limited rebalances, benchmarks (SPY buy & hold, equal weight),
  CAGR/Sharpe/Sortino/max drawdown/CVaR/turnover/costs, trade log, reproducible config hashes.
* **Dashboard** — overview, markets (forecasts, intervals, SHAP, news), personal holdings → recommendations
  with P/L scenarios and stress tests, model diagnostics and drift, SEC analyst chat, backtest lab, system
  freshness and pipeline health. Loading, empty and error states throughout.

## Quick start (local)

Prerequisites: Docker Desktop, Python 3.12, Node 22.

```bash
cp .env.example .env                 # fill SEC_USER_AGENT, MARKETAUX_API_TOKEN, GEMINI_API_KEY, INTERNAL_API_SECRET
cp frontend/.env.example frontend/.env.local
docker compose up -d                 # postgres, redis, minio, neo4j, airflow (http://localhost:8080)
python -m venv venv && venv/Scripts/pip install -r requirements-pipeline.txt -r requirements-api.txt
venv/Scripts/python -m alembic upgrade head
```

First data load (or trigger the DAGs in the Airflow UI):

```bash
python -m data_pipeline.market_ingestion && python -m data_pipeline.clean_and_load
python -m data_pipeline.macro_ingestion
python -m data_pipeline.sec_ingestion --max-downloads 3000
python -m ml_core.rag.indexer --limit 2000
python -m data_pipeline.news_ingestion && python -m data_pipeline.sentiment_scoring
python -m ml_core.feature_store.engineer_features
python -m ml_core.graph.build_graph
python -m ml_core.models.forecasting          # ~15-30 min on CPU
python -m ml_core.models.inference
python -m ml_core.portfolio.portfolio_optimizer
python -m ml_core.monitoring.drift
python -m ml_core.models.tft_challenger       # optional, CPU-heavy
```

Run the API and dashboard:

```bash
uvicorn api.main:app --reload --port 8000
cd frontend && npm install && npm run dev     # http://localhost:3000
```

## Testing

```bash
pytest tests/unit                                                   # fast, no services
TEST_DATABASE_URL=postgresql://quant_user:quant_password@localhost:5432/quant_test pytest tests/integration
cd frontend && npm test && npx tsc --noEmit && npx eslint . && npm run build
```

CI (`.github/workflows/ci.yml`) runs ruff, unit tests with coverage, the end-to-end integration test on a
pgvector service container, an API-only dependency import check (Render parity), and the frontend
lint/typecheck/tests/build.

## Project history

The original prototype (5 tickers, a single 5-day XGBoost model, mean-variance optimiser) is preserved in
git history. Legacy research scripts (`ml_core/models/train_*.py`, `model_tournament.py`,
`ml_core/nlp/ingest_text.py` etc.) remain for reference but are not scheduled. See
[docs/LIMITATIONS.md](docs/LIMITATIONS.md) for why the earlier 63.7% directional-accuracy figure was
withdrawn (label leakage).
