# Known limitations and honest caveats

## Findings from the audit (now fixed)

* **Label leakage in the original XGBoost model.** `forward_return_5d` was built with `shift(-4)`, so the
  5-day target window included the current day's `return_1d`, which was also an input feature. The
  reported 63.7% directional accuracy and 0.389 correlation were inflated by this. The new labels use
  returns from t+1 to t+h only, with purged/embargoed walk-forward evaluation (see MODEL_CARD.md for the
  honest numbers).
* The original optimiser annualised 5-day forecasts by compounding them ~50× (e.g. 190% "expected"
  returns), stored VaR/CVaR as NULL and had no CVaR, turnover or cost constraints. Those rows remain in
  the database for history but are no longer served.
* Sentiment and graph features in the original feature store were populated on only two rows (effectively
  constant), and the knowledge graph used hand-written "demo" relationships. Both were replaced with
  sourced, point-in-time data.
* SEC filing timestamps were the download time; they are now EDGAR acceptance times.
* The FastAPI service had no authentication and returned raw exception messages; `/api/*` rewrites on
  Vercel bypassed the login check.

## Data

* **Survivorship bias.** The universe is 12 assets chosen today (9 mega/large caps + SPY, TLT, GLD).
  Backtests therefore overstate what a contemporaneous investor could have selected.
* yfinance is an unofficial Yahoo interface without an SLA; the pipeline retries, validates and detects
  revisions, but a licensed feed would be needed for institutional use. Stooq now requires a browser
  challenge and is not used.
* News coverage is thin: the Marketaux free plan returns 3 articles per request (≈48 requests/day are
  scheduled), and the NewsAPI history covers only a few days in Aug–Sep 2026. News sentiment is therefore
  missing for most of the training history; the model treats it as missing rather than neutral.
* FRED values are latest vintages (not ALFRED real-time vintages); revisions to monthly series (CPI,
  unemployment) are not point-in-time, which is why only daily market-based macro series are model features.
* The RAG index covers narrative sections of 10-K/10-Q filings and full 8-Ks for the last three years;
  financial statement tables are not indexed.

## Models

* Daily/weekly equity return predictability is weak. Expect rank ICs of a few hundredths, frequently
  not statistically significant, and out-of-sample R² near or below zero. The optimiser scales forecasts
  by demonstrated holdout skill, so weak models produce near risk-only portfolios.
* Conformal intervals assume exchangeability; in time series their coverage drifts with regime changes.
  Coverage is measured out-of-sample and monitored in production.
* Hyperparameters are fixed (selected among four candidates on pre-2020 folds), not exhaustively tuned.
* The TFT challenger is trained on CPU with a small configuration; it is evaluated, not served.

## Portfolio & backtesting

* Historical-scenario CVaR assumes the last ~3 years are representative; it understates tail risk in
  unprecedented regimes (stress replays are shown to compensate).
* Proportional slippage only; no market impact, borrow, taxes or intraday liquidity modelling.
  Fractional shares are allowed. Prices are split-adjusted, so share counts are in adjusted units.
* Long-only; no leverage or shorting.

## Platform

* Rate limits are in-process unless `REDIS_URL` is configured (fine for a single Render instance).
* Backtests run in a bounded in-process worker pool; a restart marks running jobs as failed (users can
  resubmit). A dedicated queue (e.g. Celery/RQ) would be needed at scale.
* Pipelines run on the Docker host; production freshness depends on that host being online and
  `PRODUCTION_DATABASE_URL` being configured for publishing.
