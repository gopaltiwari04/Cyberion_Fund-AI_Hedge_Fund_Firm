# Data & pipelines

All tasks are `python -m` modules, run by Airflow inside the pipeline virtualenv, tracked in
`pipeline_runs`, idempotent and safe to re-run.

## Market data (`data_pipeline/market_ingestion.py`, `clean_and_load.py`)

* Source: yfinance daily bars with corporate actions (`auto_adjust=False`, `actions=True`): split-adjusted
  close, raw dividends (split-adjusted by the source), split ratios, adjusted close for reference.
* Incremental fetch from `last_date - 20 days`; full history re-fetch when a new split appears, when the
  overlap disagrees with stored closes by >0.5% (back-adjustment), or for legacy rows lacking lineage.
* Raw landing: `market/yfinance/<ticker>/run_date=<date>/prices.parquet` (MinIO, deterministic key).
* Polars cleaning: exact column mapping, invalid prices/volume removal, OHLC consistency, de-duplication,
  dropping bars for sessions that have not closed yet; validation + Great Expectations; conditional upsert
  (`IS DISTINCT FROM`) so unchanged rows are not rewritten.
* Total returns everywhere downstream: `(close_t + dividend_t) / close_{t-1} - 1`.

## Macro (`macro_ingestion.py`)

FRED: DFF, DGS10, DGS2, T10Y2Y, VIXCLS, BAA10Y, CPIAUCSL, UNRATE (official API with `FRED_API_KEY`,
public CSV otherwise). Features use values strictly before the feature date (FRED publishes day-t
values on t+1), forward-filled at most 10 observations.

## News (`news_ingestion.py`, `sentiment_scoring.py`, `news_features.py`)

Marketaux (entity-tagged, one row per article × ticker, upsert on `(uuid, ticker)`), raw JSON to MinIO.
FinBERT (ProsusAI/finbert) scores headline + description. Daily aggregates are keyed to the trading
session in which the article became public (after 16:00 ET → next session). Earlier NewsAPI articles
stored in `financial_text` were migrated into `financial_news`.

## SEC filings (`sec_ingestion.py`, `ml_core/rag/indexer.py`)

* EDGAR submissions API (all pages back to 2016), `company_tickers.json` CIK mapping, additional CIKs
  for reorganised registrants (XOM 2026 holding company). Timestamps = EDGAR `acceptanceDateTime`.
* Downloads: 8-Ks since 2016 (primary document + EX-99 press releases) for filing-sentiment features;
  10-K/10-Q for the last `SEC_RAG_YEARS` (3). HTML → text with hidden XBRL removed; raw + text in MinIO.
* Indexing: section-aware chunking (10-K Items 1/1A/1C/2/3/5/7/7A/9A, 10-Q MD&A/market risk/controls/
  legal/risk factors/other; heading-based fallback for bank-style layouts), ~1.2k-char chunks with
  overlap, `BAAI/bge-small-en-v1.5` embeddings (fastembed, 384-d) + generated `tsvector`.
* 8-K sentiment: FinBERT over up to six substantive passages (exhibits preferred).

## Features (`ml_core/feature_store/engineer_features.py`, `external_signals.py`, `ml_core/graph/build_graph.py`)

26 technical/market-relative features (unchanged logic, point-in-time), plus macro (5), news (3),
filing (3: count 30d, days since filing, 8-K sentiment 90d) and graph (5) signals, plus 4
cross-sectional ranks computed at training time. Missing information stays `NULL`.

Graph signals: rolling 60-day correlation network (degree share, eigenvector centrality, mean 5-day
return of the 3 most-correlated peers) and knowledge-graph centrality (sector + SEC SIC industry edges and
news co-mention edges within the trailing 90 days). The Neo4j graph (`build_graph.py`) is built from the
same sourced data; the earlier hand-written demo relationships were removed.

## Publishing (`publish.py`)

See [DEPLOYMENT.md](DEPLOYMENT.md#3-pipelines-and-data-publishing).
