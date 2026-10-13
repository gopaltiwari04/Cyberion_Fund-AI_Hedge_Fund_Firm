# Risk, portfolio construction & backtesting

## Optimiser (`ml_core/portfolio/optimizer.py`)

Long-only, daily units, risky assets + CASH (earns the lagged effective fed funds rate):

```
maximise   a·μ_h/h·w + r_f·w_cash − (λ/2)·wᵀΣw − (c/h)·Σ|w − w₀|
s.t.       Σw + w_cash = 1, 0 ≤ w ≤ max_weight, min_cash ≤ w_cash ≤ max_cash,
           sector weight ≤ max_sector_weight, Σ|w − w₀| ≤ max_turnover (if w₀ given),
           CVaR₉₅(1-day loss) ≤ limit         (Rockafellar–Uryasev on historical scenarios)
```

* μ_h — model forecast of h-day total return; **a** = forecast weight in [0, 1] = clip(holdout IC t-stat / 3).
  With no demonstrated skill the optimiser becomes a pure risk model rather than chasing noise.
* Σ — Ledoit–Wolf shrinkage covariance of 756 daily total returns; CVaR uses the same scenarios.
* c — proportional costs (commission + slippage) amortised over the horizon.
* Post-solve verification recomputes CVaR, weights and turnover in numpy; a solution that breaches any
  limit is rejected.
* Infeasible: returns `status="infeasible"`, **no weights**, the minimum achievable CVaR, and whether the
  turnover cap is the cause. The minimum-CVaR portfolio is reported for transparency only.

Risk profiles: conservative (λ=10, CVaR 1.5%, max 20%/position, sector 40%, ≥5% cash), balanced
(λ=5, 2.5%, 30%, 60%), aggressive (λ=2, 4%, 40%, 80%). Users can override CVaR, max weight, turnover, costs.

## Recommendations (`ml_core/portfolio/recommend.py`)

Holdings → current weights (latest closes) → optimiser → trades in currency and shares, per-asset
forecast, interval, volatility, risk contribution and SHAP explanation. P/L scenarios for current vs
proposed portfolios: model expectation with a Gaussian 90% band, empirical percentiles of rolling h-day
returns since 2015, and stress replays (Q4 2018, COVID-19 crash, 2022 rate shock, March 2023 bank stress,
April 2025 tariff shock). Every run is stored in `recommendation_runs`.

## Backtester (`ml_core/backtest/engine.py`)

Event loop per day: fill pending orders at the open (± slippage, commission; sells before buys; buys
scaled to cash) → ex-date dividends → cash interest → mark to market at the close → on rebalance days,
compute targets with information available at that close only. Runtime assertions reject any prediction
whose model saw labels beyond the decision date and any scenario window that extends past it.

Strategies: `ml_cvar`, `min_cvar` (risk only), `equal_weight`, `spy_buy_and_hold`; benchmarks run
through the same engine over the same window. Metrics: total return, CAGR, volatility, Sharpe and Sortino
(excess of cash), max drawdown and duration, Calmar, 1-day VaR/CVaR, annualised turnover, costs, trades,
beta/alpha/tracking error/information ratio vs benchmarks. Results store config hash, model version,
data snapshot and code version; identical inputs reproduce identical results (tested).

Known biases (shown with every result): survivorship (fixed present-day universe), proportional slippage
only, fractional shares, no taxes. Forecasts used are walk-forward out-of-sample; the risk model uses only
trailing data.
