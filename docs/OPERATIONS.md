# Operations & runbook

## Schedules (UTC)

| DAG | Schedule | Tasks |
|---|---|---|
| `cyberion_market_pipeline` | 22:30 Mon–Fri (after the US close) | extract_prices → clean_load_prices, ingest_fred → engineer_features → live_inference → house_portfolio → model_monitoring, publish_to_production; sync_knowledge_graph |
| `cyberion_news_intelligence` | hourly :05 | ingest_news → finbert_sentiment → news_snapshot |
| `cyberion_sec_filings` | 11:15 daily | ingest_filings → index_filings, filing_sentiment |
| `cyberion_model_training` | 06:00 Sunday | train_xgboost → tft_challenger; live_inference → house_portfolio → publish_to_production |

Retries: 3 with exponential backoff (max 1h); per-task execution timeouts; `max_active_runs=1`.
Final failures call `ALERT_WEBHOOK_URL`. Pipelines also alert from `common/runs.py`.

## Observability

* `pipeline_runs` — every task: status (`success`, `success_with_warnings`, `failed`, `abandoned`), rows,
  duration, warnings, error. Runs killed without cleanup are marked `abandoned` on the next start.
* `GET /system/status` — dataset freshness against SLAs (business days), latest run per task, failures in
  24h, SEC index stats. Shown on the dashboard System page.
* `GET /health` (liveness, DB) and `GET /health/ready` (schema version + dataset presence).
* Structured JSON logs with request ids (`X-Request-ID`).
* Model monitoring (`model_monitoring`): PSI drift of top features, rolling 63-day IC and interval coverage.

## Common incidents

| Symptom | Action |
|---|---|
| `market_prices` stale | Check `market_data.extract_prices` warnings (yfinance throttling). Re-run the DAG; the task is idempotent. |
| Split / revised history | Ingestion detects it and re-fetches full history automatically (warning in run details). |
| `forecasts` stale but prices fresh | Check `live_inference`; ensure an `active` model exists (`/system/models`). |
| House portfolio `infeasible` | The CVaR limit can't be met under the turnover cap; diagnostics give the minimum achievable CVaR. No trades are made. |
| Marketaux HTTP 402 | Daily request quota used; news resumes next day. Reduce `MARKETAUX_PAGES`. |
| SEC 403/429 | Ensure `SEC_USER_AGENT` is "Company email"; the client already rate-limits to ~6.6 req/s. |
| Significant drift alert | Review `/system/monitoring`; trigger `cyberion_model_training` manually. |
| Analyst returns "extractive" | Gemini unavailable or key missing; passages are still cited. |

## Manual operations

```bash
python -m data_pipeline.sec_ingestion --max-downloads 3000   # backfill
python -m data_pipeline.publish --full                         # republish everything
python -m ml_core.models.forecasting --horizons 5              # retrain one horizon
```

## Backups

`docker compose exec -T postgres pg_dump -U quant_user -d quant_db -Fc > backups/quant_db_$(date +%F).dump`
(`backups/` is git-ignored). Render PostgreSQL has its own backups per plan.
