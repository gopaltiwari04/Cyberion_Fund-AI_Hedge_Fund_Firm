You're right. Here's the **entire README in ONE single copy-paste block**. Just copy everything inside this block into `README.md`.

````markdown
# AI Hedge Fund Firm

An end-to-end quantitative investment platform designed to combine automated financial data ingestion, machine learning, portfolio optimization, explainable AI, qualitative financial research, and institutional-grade backtesting.

The platform is being developed as a modular AI-driven investment research and decision-support system.

---

## Overview

The goal of this project is to build an automated quantitative investment platform capable of:

- Collecting and processing financial market and alternative financial data
- Engineering quantitative features from historical market data
- Predicting future asset returns using machine learning
- Optimizing portfolio allocations based on predicted returns and risk
- Providing transparent explanations for model decisions
- Answering qualitative financial research questions using financial documents
- Backtesting investment strategies under realistic market conditions

The current implementation provides the core quantitative infrastructure, including market-data storage, feature engineering, an XGBoost return prediction model, portfolio optimization, risk analytics, and a web-based dashboard.

Additional institutional capabilities are planned as future development phases.

---

# Current System

## 1. Market Data Infrastructure

Historical market data is stored in PostgreSQL and forms the foundation of the quantitative pipeline.

The current system includes market data for:

- AAPL
- AMZN
- GOOGL
- MSFT
- SPY

Stored market fields include:

- Open
- High
- Low
- Close
- Volume
- Trading date
- Ticker

The database is designed to support further expansion into additional equities, ETFs, macroeconomic indicators, financial news, and regulatory filings.

---

## 2. Quantitative Feature Engineering

The platform contains an automated feature-engineering layer that transforms raw market data into quantitative model features.

Current features include:

### Return Features

- 1-day return
- 5-day return
- 10-day return
- 20-day return
- 60-day return

### Technical Indicators

- RSI
- MACD
- ATR
- SMA 20 distance
- SMA 50 distance
- SMA 200 distance

### Volatility Features

- 5-day volatility
- 20-day volatility
- 60-day volatility

### Drawdown Features

- 20-day drawdown
- 60-day drawdown

### Volume Features

- 5-day volume change
- 20-day volume z-score

### Market-relative Features

- Market 5-day return
- Relative 5-day return
- 60-day beta
- 60-day correlation with SPY

### Price Structure Features

- Intraday range
- Close position
- Gap return

### Additional Intelligence Features

- Sentiment score
- Degree centrality
- PageRank
- Market regime

The current model uses 29 features.

---

# 3. Machine Learning

## XGBoost Return Prediction

The platform currently includes an XGBoost regression model for predicting forward asset returns.

### Prediction Target

The current target is:

`forward_return_5d`

The model therefore attempts to predict an asset's return over the following five trading days.

### Model

Current model:

`XGBRegressor`

Model artifact:

`ml_core/models/xgboost_return_model.joblib`

The pipeline uses:

- Chronological train/validation/test splitting
- Training-set-only imputation
- Regularized XGBoost
- 29 quantitative features
- Out-of-sample predictions

---

## Model Evaluation

The current model has been evaluated using out-of-sample predictions.

Current evaluation metrics include:

| Metric | Result |
|---|---:|
| Mean Absolute Error | 0.0286 |
| Root Mean Squared Error | 0.0410 |
| Directional Accuracy | 63.71% |
| Prediction Correlation | 0.389 |

Dataset split:

| Dataset | Observations |
|---|---:|
| Total rows | 12,562 |
| Target rows | 12,535 |
| Training | 8,770 |
| Validation | 1,880 |
| Test | 1,885 |

The split is chronological to reduce the risk of future information entering earlier training observations.

Out-of-sample predictions are stored in:

`ml_core/models/xgboost_oos_predictions.csv`

---

# 4. Portfolio Optimization

The platform currently feeds machine-learning return predictions into a portfolio optimization layer.

The optimizer uses predicted asset returns and historical market relationships to determine portfolio weights.

Current strategy:

`xgboost_mean_variance_v1`

The system currently supports:

- Model-driven expected returns
- Portfolio volatility estimation
- Mean-variance optimization
- Asset allocation
- Risk metrics
- Sharpe ratio calculation
- Portfolio allocation persistence in PostgreSQL

Example current output:

```text
AAPL   35%
AMZN   35%
MSFT   30%
GOOGL   0%
SPY     0%
````

These allocations are model-generated outputs and are not presented as guaranteed investment results.

---

# 5. Risk Analytics

The portfolio layer currently calculates:

* Expected annual return
* Expected annual volatility
* Sharpe ratio
* VaR 95%
* CVaR 95%

The risk metrics are exposed through the FastAPI backend and displayed in the web dashboard.

---

# 6. Backend API

The platform uses FastAPI as the application backend.

Current API endpoints include:

```text
GET /health
GET /portfolio
GET /risk
GET /markets
GET /markets/{ticker}/history
GET /research/model
GET /research/predictions
GET /research/features
```

The API provides a controlled interface between the quantitative infrastructure, PostgreSQL database, ML artifacts, and frontend application.

---

# 7. Database

PostgreSQL is used as the primary structured data store.

Current database:

```text
Database: quant_db
User: quant_user
```

Important tables include:

```text
market_data
feature_store
financial_text
assets
portfolio_allocations
portfolio_risk_metrics
```

The database layer is designed to support expansion into financial text, news, regulatory filings, model metadata, and additional quantitative datasets.

---

# 8. Automated Pipeline Infrastructure

The project uses Apache Airflow as the orchestration layer for automated data and machine-learning workflows.

The intended pipeline architecture is:

```text
Data Sources
     ↓
Airflow
     ↓
Data Validation
     ↓
PostgreSQL
     ↓
Feature Engineering
     ↓
Machine Learning
     ↓
Portfolio Optimization
     ↓
Dashboard
```

Airflow and supporting infrastructure are currently being prepared for the complete automated investment pipeline.

Additional data sources and production scheduling are part of the future roadmap.

---

# 9. Frontend Dashboard

The platform includes a Next.js web application designed as an institutional quantitative investment dashboard.

The interface currently contains:

## Overview

Provides a high-level view of:

* Current strategy
* Expected annual return
* Expected annual volatility
* Current portfolio allocation
* Portfolio risk metrics

## Portfolio

Provides:

* Current model allocation
* Individual holdings
* Portfolio weights
* Expected asset returns
* Risk contributions
* Portfolio risk metrics

## Markets

Provides real market-data information including:

* Latest prices
* Daily returns
* 5-day returns
* RSI
* Volatility
* Market regime
* Historical price charts
* Beta vs SPY
* Correlation vs SPY
* Relative market returns

## Research

Provides:

* Model specification
* Model artifact status
* Out-of-sample evaluation metrics
* Dataset information
* Actual vs predicted return visualization
* Saved model feature set

## System

Provides:

* API health
* Database connectivity
* Basic infrastructure status

---

# Architecture

The current architecture is approximately:

```text
                         ┌─────────────────────┐
                         │     Next.js UI      │
                         │                     │
                         │ Overview            │
                         │ Portfolio           │
                         │ Markets             │
                         │ Research            │
                         │ System              │
                         └──────────┬──────────┘
                                    │
                                    ↓
                         ┌─────────────────────┐
                         │      FastAPI        │
                         │        API          │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  ↓                 ↓                 ↓
           ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
           │ PostgreSQL  │   │ ML Artifacts│   │    Redis    │
           │             │   │             │   │             │
           │ Market Data │   │ XGBoost     │   │ Caching     │
           │ Features    │   │ Predictions │   │             │
           │ Portfolio   │   │             │   │             │
           └─────────────┘   └─────────────┘   └─────────────┘
                                    ↑
                                    │
                         ┌─────────────────────┐
                         │   Feature Engine    │
                         └──────────┬──────────┘
                                    ↑
                                    │
                         ┌─────────────────────┐
                         │      Airflow        │
                         │   Orchestration     │
                         └─────────────────────┘
```

---

# Technology Stack

## Backend

* Python
* FastAPI
* SQLAlchemy
* Alembic

## Database

* PostgreSQL
* Redis

## Machine Learning

* XGBoost
* pandas
* NumPy
* scikit-learn
* Technical analysis libraries

## Data Engineering

* Apache Airflow
* Great Expectations
* MinIO

## Quantitative Finance

* Mean-variance optimization
* Portfolio risk analytics
* Return forecasting
* Volatility analysis

## Frontend

* Next.js
* React
* TypeScript
* Tailwind CSS
* Recharts

## Development Infrastructure

* Docker
* Docker Compose
* Git

---

# Future Development Roadmap

The current system provides the core quantitative infrastructure. The following capabilities are planned as future development phases.

---

# Phase 1 — Automated Intelligence Gathering

The platform will evolve into a fully automated financial intelligence pipeline.

The goal is for the system to automatically ingest new information every night and transform it into structured, model-ready data.

Planned data sources include:

### Market Data

* Stock prices
* Trading volume
* Market indices
* ETF data

### Macroeconomic Data

* Interest rates
* Inflation
* GDP
* Employment indicators
* Economic activity indicators
* Other relevant macroeconomic variables

### Financial News

* Breaking financial news
* Company news
* Industry news
* Market-moving events
* News sentiment

### SEC Regulatory Filings

* 10-K
* 10-Q
* 8-K
* Other relevant SEC filings

The planned workflow is:

```text
Nightly Schedule
      ↓
Market Data
      ↓
Macro Data
      ↓
Financial News
      ↓
SEC Filings
      ↓
Data Validation
      ↓
Data Cleaning
      ↓
PostgreSQL
      ↓
Feature Engineering
      ↓
Model Pipeline
```

The objective is to minimize manual intervention and ensure that the investment system starts each trading day with refreshed information.

---

# Phase 2 — Predictive Alpha Generation

The current system uses XGBoost for 5-day return prediction.

Future versions will expand the predictive layer to include multiple models.

## Deep Learning Transformers

A Transformer-based model will be investigated for financial time-series and/or financial-text prediction.

Potential inputs include:

* Historical price sequences
* Technical indicators
* News information
* Financial text
* Macro indicators
* Market context

The Transformer and XGBoost models can eventually operate as complementary predictive models.

---

## Probabilistic Predictions

The future prediction system will move beyond a single expected return.

Instead of only producing:

```text
Expected 5-day return:
+2.1%
```

the system will attempt to estimate uncertainty around the prediction.

Conceptually:

```text
Expected return
      +
Prediction interval
      +
Probability distribution / quantiles
```

For example:

```text
Expected 5-day return: +2.1%

Estimated prediction interval:
-1.8% to +5.9%
```

The exact statistical methodology will be determined during implementation.

These values will represent estimated model uncertainty rather than guaranteed probability bounds.

---

# Phase 3 — Advanced Risk-Managed Asset Allocation

The current system already uses machine-learning predictions as inputs to portfolio optimization.

Future versions will introduce more sophisticated portfolio constraints.

Potential constraints include:

* Maximum position size
* Maximum sector exposure
* Portfolio volatility limit
* Maximum drawdown constraints
* Downside-risk constraints
* Value-at-Risk constraints
* CVaR constraints
* Turnover constraints
* Transaction costs
* Market liquidity constraints
* Cash allocation

The objective will be to transform model predictions into an actionable target portfolio while explicitly controlling portfolio risk.

Conceptually:

```text
Model Predictions
       ↓
Expected Return Distribution
       ↓
Risk Model
       ↓
Portfolio Constraints
       ↓
Convex Optimization
       ↓
Target Portfolio
```

Example future output:

```text
AAPL    15%
MSFT    10%
AMZN    20%
GOOGL    5%
SPY      0%
CASH    50%
```

The optimizer will determine the allocation based on the configured investment mandate and risk constraints.

---

# Phase 4 — Explainable AI / Glass-Box Reasoning

Future versions will integrate Explainable AI techniques such as SHAP.

The goal is to make model decisions transparent.

Instead of simply showing:

```text
AAPL expected return: +2.1%
```

the system will explain the major factors contributing to the prediction.

For example:

```text
AAPL

Expected return: +2.1%

Primary model drivers:

5D momentum          ↑
RSI                  ↑
Market return        ↑
Volatility           ↓
SMA distance         ↑
Relative performance ↑
```

The final implementation will derive these explanations directly from the model rather than manually assigning importance.

The dashboard will allow users to inspect the factors influencing individual predictions and portfolio decisions.

---

# Phase 5 — On-Demand Qualitative AI Analyst

A future version will include an interactive ChatGPT-style financial research assistant.

The user will be able to ask questions such as:

```text
What material risks did Nvidia report yesterday?
```

or:

```text
What changed in Microsoft's latest 10-Q?
```

or:

```text
Summarize the latest regulatory risks for Apple.
```

The system will retrieve relevant financial documents and provide a synthesized answer.

Planned architecture:

```text
User Question
      ↓
Query Understanding
      ↓
SEC Filings / Financial News
      ↓
Document Retrieval
      ↓
Relevant Evidence
      ↓
LLM Analysis
      ↓
Answer
      ↓
Source Citations
```

The system will be designed to cite the underlying sources used to generate the answer.

This will allow users to distinguish between:

* Retrieved facts
* Source information
* Model interpretation
* Analyst-style synthesis

---

# Phase 6 — Institutional Backtesting Engine

A future institutional-grade backtesting engine will allow users to test investment strategies against historical market data.

Users will be able to specify:

* Strategy
* Universe
* Historical period
* Rebalancing frequency
* Initial capital
* Position limits
* Transaction costs
* Slippage
* Other market-friction assumptions

Example:

```text
Strategy:
XGBoost + Portfolio Optimizer

Period:
2016–2026

Initial Capital:
$100,000

Transaction Costs:
10 bps

Rebalancing:
Weekly
```

The simulator will calculate metrics such as:

* CAGR
* Total return
* Annualized volatility
* Sharpe ratio
* Sortino ratio
* Maximum drawdown
* Value at Risk
* CVaR
* Turnover
* Win rate
* Transaction costs

The objective is to evaluate whether a strategy would have been viable under realistic historical conditions.

---

# Point-in-Time Data Integrity

A major design goal of the future backtesting and prediction infrastructure is avoiding look-ahead bias.

Historical simulations must only use information that would actually have been available at the time of the simulated decision.

Conceptually:

```text
Information available on Day T
             ↓
       Model prediction
             ↓
       Portfolio decision
             ↓
      Future market return
```

Future information must never be allowed to influence a historical decision.

This principle will apply to:

* Feature engineering
* Model training
* Backtesting
* News
* SEC filings
* Macro data
* Sentiment
* Portfolio optimization

---

# Long-Term Platform Architecture

The long-term system is planned to evolve toward:

```text
                         GLOBAL FINANCIAL DATA
                                  │
             ┌────────────────────┼────────────────────┐
             ↓                    ↓                    ↓
        Market Data           Macro Data          News / SEC
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  ↓
                         AUTOMATED INGESTION
                                  │
                               Airflow
                                  │
                                  ↓
                         DATA VALIDATION
                                  │
                                  ↓
                            DATA LAKE / DB
                                  │
                                  ↓
                       FEATURE ENGINEERING
                                  │
              ┌───────────────────┼───────────────────┐
              ↓                   ↓                   ↓
         Technical            Sentiment           Graph/NLP
          Features             Features            Features
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  ↓
                         MODELING LAYER
                                  │
                    ┌─────────────┴─────────────┐
                    ↓                           ↓
                 XGBoost                  Transformer
                    │                           │
                    └─────────────┬─────────────┘
                                  ↓
                       PROBABILISTIC FORECAST
                                  │
                                  ↓
                         RISK MODEL
                                  │
                                  ↓
                       CONVEX OPTIMIZER
                                  │
                                  ↓
                         TARGET PORTFOLIO
                                  │
              ┌───────────────────┼───────────────────┐
              ↓                   ↓                   ↓
          SHAP / XAI          Dashboard           AI Analyst
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  ↓
                           USER DECISION

                                  ↕
                         BACKTESTING ENGINE
```

---

# Project Objectives

The long-term objective is to build a system that combines:

1. **Automated intelligence gathering**
2. **Machine-learning-based alpha generation**
3. **Probabilistic return forecasting**
4. **Risk-aware portfolio optimization**
5. **Explainable AI**
6. **AI-assisted qualitative research**
7. **Institutional backtesting**
8. **Automated daily investment workflows**

The final platform is intended to function as an integrated quantitative research and investment decision-support environment rather than a standalone prediction model.

---

# Current Limitations

The current implementation is still under development.

Important current limitations include:

* Automated nightly ingestion of all planned data sources is not yet complete.
* Transformer-based prediction is not yet implemented.
* Probabilistic prediction intervals are not yet implemented.
* Advanced downside-risk constraints are not yet implemented.
* SHAP-based explanations are not yet implemented.
* The interactive qualitative AI analyst is not yet implemented.
* SEC/news retrieval and source citation workflows are not yet fully integrated.
* Institutional backtesting with transaction costs and slippage is not yet implemented.
* Current model-derived portfolio expectations should not be interpreted as guaranteed future performance.

These are planned development areas rather than claims about the current system.

---

# Development Philosophy

The platform is being developed incrementally.

The strategy is:

```text
Build
  ↓
Validate
  ↓
Integrate
  ↓
Deploy
  ↓
Improve
```

The system prioritizes a working end-to-end architecture before progressively adding more sophisticated modeling, data sources, risk controls, explainability, research capabilities, and backtesting.

---

# Disclaimer

This project is a research and software-engineering project.

The predictions, portfolio allocations, risk metrics, and model outputs produced by the system are experimental and should not be interpreted as financial advice, investment recommendations, or guarantees of future performance.

Machine-learning predictions are inherently uncertain, and historical or out-of-sample performance does not guarantee future results.

---

# Status

## Current

* [x] PostgreSQL database
* [x] Historical market data
* [x] Feature store
* [x] Quantitative feature engineering
* [x] XGBoost return prediction
* [x] Out-of-sample evaluation
* [x] Portfolio optimization
* [x] Portfolio risk metrics
* [x] FastAPI backend
* [x] Next.js frontend
* [x] Overview dashboard
* [x] Portfolio dashboard
* [x] Markets dashboard
* [x] Research dashboard
* [x] System health dashboard
* [x] Historical market visualization
* [x] Model diagnostics

## Future

* [ ] Fully automated nightly intelligence ingestion
* [ ] Expanded macroeconomic ingestion
* [ ] Automated financial news ingestion
* [ ] Automated SEC filing ingestion
* [ ] Transformer-based prediction model
* [ ] Probabilistic return forecasting
* [ ] Prediction intervals / uncertainty estimates
* [ ] Advanced downside-risk constraints
* [ ] Cash-aware portfolio optimization
* [ ] Transaction-cost-aware optimization
* [ ] SHAP explainability
* [ ] Per-trade model reasoning
* [ ] AI qualitative research analyst
* [ ] SEC/news retrieval and source citations
* [ ] Institutional backtesting engine
* [ ] Transaction costs and slippage simulation
* [ ] Point-in-time historical simulation
* [ ] Production deployment
* [ ] Automated daily portfolio generation

---

# Project Vision

The ultimate goal is to develop an AI-powered quantitative investment platform that can move through the complete investment research workflow:

```text
Observe
   ↓
Understand
   ↓
Predict
   ↓
Explain
   ↓
Optimize
   ↓
Backtest
   ↓
Monitor
```

The system is designed to combine quantitative models with qualitative financial intelligence while maintaining transparency, reproducibility, and strong data-integrity principles.

```
```
