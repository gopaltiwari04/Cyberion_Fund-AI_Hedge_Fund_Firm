"""
Portfolio Optimizer

Uses:
- Real historical market data from PostgreSQL
- Historical covariance matrix
- Temporary expected-return predictions
- CVXPY constrained optimization

Constraints:
- Long-only
- Fully invested
- Maximum 35% per asset

Results are stored in:
- portfolio_allocations
- portfolio_risk_metrics
"""

from datetime import date

import numpy as np
import pandas as pd
import cvxpy as cp
from sqlalchemy import create_engine, text


# ============================================================
# CONFIGURATION
# ============================================================

PG_URL = "postgresql://quant_user:quant_password@localhost:5432/quant_db"

ASSETS = [
    "AAPL",
    "AMZN",
    "GOOGL",
    "MSFT",
    "SPY",
]

STRATEGY_NAME = "mean_variance_v1"

MAX_WEIGHT = 0.35

TRADING_DAYS = 252

RISK_FREE_RATE = 0.0

# Temporary expected-return predictions.
#
# These are NOT actual ML predictions yet.
# Later these values will come from XGBoost / TFT.
EXPECTED_DAILY_RETURNS = {
    "AAPL": 0.0008,
    "AMZN": 0.0007,
    "GOOGL": 0.0006,
    "MSFT": 0.0009,
    "SPY": 0.0004,
}


# ============================================================
# DATABASE
# ============================================================

engine = create_engine(PG_URL)


# ============================================================
# LOAD MARKET DATA
# ============================================================

def load_market_data() -> pd.DataFrame:
    """
    Load historical closing prices for the portfolio universe.
    """

    print("=" * 60)
    print("LOADING MARKET DATA")
    print("=" * 60)

    placeholders = ", ".join(f":ticker_{i}" for i in range(len(ASSETS)))

    params = {
        f"ticker_{i}": ticker
        for i, ticker in enumerate(ASSETS)
    }

    query = text(
        f"""
        SELECT
            ticker,
            date,
            close
        FROM market_data
        WHERE ticker IN ({placeholders})
          AND close IS NOT NULL
        ORDER BY date, ticker
        """
    )

    df = pd.read_sql(query, engine, params=params)

    if df.empty:
        raise RuntimeError("No market data found.")

    df["date"] = pd.to_datetime(df["date"])

    print(f"Rows loaded: {len(df)}")
    print(f"Assets found: {sorted(df['ticker'].unique())}")
    print(
        f"Date range: "
        f"{df['date'].min().date()} -> {df['date'].max().date()}"
    )

    return df


# ============================================================
# PREPARE PRICE MATRIX
# ============================================================

def prepare_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-form market data into a date x ticker price matrix.
    """

    prices = df.pivot(
        index="date",
        columns="ticker",
        values="close",
    )

    missing_assets = [
        ticker for ticker in ASSETS
        if ticker not in prices.columns
    ]

    if missing_assets:
        raise RuntimeError(
            f"Missing assets from market data: {missing_assets}"
        )

    prices = prices[ASSETS]

    # We need synchronized observations across all assets
    prices = prices.dropna()

    if len(prices) < 252:
        raise RuntimeError(
            f"Insufficient synchronized price history: "
            f"{len(prices)} rows."
        )

    print(f"Synchronized trading days: {len(prices)}")

    return prices


# ============================================================
# CALCULATE RETURNS
# ============================================================

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily percentage returns.
    """

    returns = prices.pct_change().dropna()

    print(f"Return observations: {len(returns)}")

    return returns


# ============================================================
# COVARIANCE MATRIX
# ============================================================

def calculate_covariance_matrix(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate annualized covariance matrix.
    """

    covariance = returns.cov() * TRADING_DAYS

    print("\nANNUALIZED COVARIANCE MATRIX")
    print("-" * 60)
    print(covariance)

    return covariance


# ============================================================
# EXPECTED RETURNS
# ============================================================

def get_expected_returns() -> np.ndarray:
    """
    Return temporary daily expected-return predictions.

    IMPORTANT:
    These are placeholders for the future ML prediction layer.
    """

    expected_daily = np.array(
        [
            EXPECTED_DAILY_RETURNS[ticker]
            for ticker in ASSETS
        ],
        dtype=float,
    )

    expected_annual = expected_daily * TRADING_DAYS

    print("\nEXPECTED ANNUAL RETURNS")
    print("-" * 60)

    for ticker, value in zip(ASSETS, expected_annual):
        print(f"{ticker:<6} | {value:.4%}")

    return expected_annual


# ============================================================
# PORTFOLIO OPTIMIZATION
# ============================================================

def optimize_portfolio(
    expected_returns: np.ndarray,
    covariance: pd.DataFrame,
) -> np.ndarray:
    """
    Maximize expected return while penalizing portfolio variance.

    Constraints:
        sum(weights) == 1
        0 <= weights <= MAX_WEIGHT
    """

    print("\n")
    print("=" * 60)
    print("OPTIMIZING PORTFOLIO")
    print("=" * 60)

    n_assets = len(ASSETS)

    weights = cp.Variable(n_assets)

    covariance_matrix = covariance.values

    # Risk-aversion parameter.
    #
    # Higher values place more emphasis on reducing variance.
    risk_aversion = 1.0

    expected_portfolio_return = expected_returns @ weights

    portfolio_variance = cp.quad_form(
        weights,
        covariance_matrix,
    )

    objective = cp.Maximize(
        expected_portfolio_return
        - risk_aversion * portfolio_variance
    )

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= MAX_WEIGHT,
    ]

    problem = cp.Problem(
        objective,
        constraints,
    )

    problem.solve()

    if problem.status not in (
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    ):
        raise RuntimeError(
            f"Portfolio optimization failed. "
            f"Solver status: {problem.status}"
        )

    optimized_weights = np.asarray(
        weights.value,
        dtype=float,
    )

    # Numerical cleanup.
    optimized_weights[
        np.abs(optimized_weights) < 1e-10
    ] = 0.0

    # Normalize to exactly 100%.
    optimized_weights /= optimized_weights.sum()

    print("\nOPTIMAL ALLOCATION")
    print("-" * 60)

    for ticker, weight in zip(ASSETS, optimized_weights):
        print(f"{ticker:<6} | {weight:.4%}")

    print("-" * 60)
    print(f"Total   | {optimized_weights.sum():.4%}")
    print(f"Maximum | {optimized_weights.max():.4%}")

    return optimized_weights


# ============================================================
# PORTFOLIO METRICS
# ============================================================

def calculate_portfolio_metrics(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance: pd.DataFrame,
    returns: pd.DataFrame,
) -> dict:
    """
    Calculate expected return, volatility and Sharpe ratio.
    """

    covariance_matrix = covariance.values

    expected_return = float(
        weights @ expected_returns
    )

    variance = float(
        weights @ covariance_matrix @ weights
    )

    volatility = float(
        np.sqrt(max(variance, 0.0))
    )

    sharpe = (
        (expected_return - RISK_FREE_RATE)
        / volatility
        if volatility > 0
        else 0.0
    )

    print("\n")
    print("=" * 60)
    print("PORTFOLIO METRICS")
    print("=" * 60)

    print(
        f"Expected annual return : {expected_return:.4%}"
    )

    print(
        f"Expected annual volatility : {volatility:.4%}"
    )

    print(
        f"Sharpe ratio : {sharpe:.4f}"
    )

    return {
        "expected_annual_return": expected_return,
        "expected_annual_volatility": volatility,
        "sharpe_ratio": sharpe,
    }


# ============================================================
# RISK CONTRIBUTION
# ============================================================

def calculate_risk_contributions(
    weights: np.ndarray,
    covariance: pd.DataFrame,
) -> np.ndarray:
    """
    Calculate each asset's contribution to portfolio volatility.
    """

    covariance_matrix = covariance.values

    portfolio_variance = (
        weights
        @ covariance_matrix
        @ weights
    )

    portfolio_volatility = np.sqrt(
        max(portfolio_variance, 0.0)
    )

    if portfolio_volatility == 0:
        return np.zeros(len(weights))

    marginal_contribution = (
        covariance_matrix @ weights
    ) / portfolio_volatility

    contribution = (
        weights * marginal_contribution
    )

    return contribution


# ============================================================
# SAVE ALLOCATIONS
# ============================================================

def save_allocations(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    risk_contributions: np.ndarray,
) -> None:
    """
    Store optimized asset weights in PostgreSQL.
    """

    today = date.today()

    print("\nSaving portfolio allocations...")

    insert_query = text(
        """
        INSERT INTO portfolio_allocations (
            date,
            strategy_name,
            ticker,
            weight,
            expected_return,
            risk_contribution
        )
        VALUES (
            :date,
            :strategy_name,
            :ticker,
            :weight,
            :expected_return,
            :risk_contribution
        )
        """
    )

    with engine.begin() as conn:

        # Remove an existing allocation for this
        # strategy/date so rerunning the optimizer
        # remains idempotent.
        conn.execute(
            text(
                """
                DELETE FROM portfolio_allocations
                WHERE date = :date
                  AND strategy_name = :strategy_name
                """
            ),
            {
                "date": today,
                "strategy_name": STRATEGY_NAME,
            },
        )

        for ticker, weight, expected_return, risk in zip(
            ASSETS,
            weights,
            expected_returns,
            risk_contributions,
        ):
            conn.execute(
                insert_query,
                {
                    "date": today,
                    "strategy_name": STRATEGY_NAME,
                    "ticker": ticker,
                    "weight": float(weight),
                    "expected_return": float(
                        expected_return
                    ),
                    "risk_contribution": float(risk),
                },
            )

    print(
        f"Saved {len(ASSETS)} portfolio allocations."
    )


# ============================================================
# SAVE RISK METRICS
# ============================================================

def save_risk_metrics(metrics: dict) -> None:
    """
    Store portfolio-level metrics in PostgreSQL.
    """

    today = date.today()

    print("Saving portfolio risk metrics...")

    with engine.begin() as conn:

        conn.execute(
            text(
                """
                DELETE FROM portfolio_risk_metrics
                WHERE date = :date
                  AND strategy_name = :strategy_name
                """
            ),
            {
                "date": today,
                "strategy_name": STRATEGY_NAME,
            },
        )

        conn.execute(
            text(
                """
                INSERT INTO portfolio_risk_metrics (
                    date,
                    strategy_name,
                    expected_annual_return,
                    expected_annual_volatility,
                    sharpe_ratio,
                    var_95,
                    cvar_95
                )
                VALUES (
                    :date,
                    :strategy_name,
                    :expected_annual_return,
                    :expected_annual_volatility,
                    :sharpe_ratio,
                    :var_95,
                    :cvar_95
                )
                """
            ),
            {
                "date": today,
                "strategy_name": STRATEGY_NAME,
                "expected_annual_return": metrics[
                    "expected_annual_return"
                ],
                "expected_annual_volatility": metrics[
                    "expected_annual_volatility"
                ],
                "sharpe_ratio": metrics[
                    "sharpe_ratio"
                ],
                # VaR / CVaR will be populated by
                # the dedicated risk engine.
                "var_95": None,
                "cvar_95": None,
            },
        )

    print("Portfolio risk metrics saved.")


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_optimizer() -> None:

    print("\n")
    print("=" * 60)
    print("PORTFOLIO OPTIMIZER")
    print("=" * 60)

    # 1. Load market data
    market_data = load_market_data()

    # 2. Build synchronized price matrix
    prices = prepare_prices(market_data)

    # 3. Calculate daily returns
    returns = calculate_returns(prices)

    # 4. Calculate annualized covariance
    covariance = calculate_covariance_matrix(
        returns
    )

    # 5. Get expected returns
    expected_returns = get_expected_returns()

    # 6. Optimize
    weights = optimize_portfolio(
        expected_returns,
        covariance,
    )

    # 7. Calculate risk contribution
    risk_contributions = (
        calculate_risk_contributions(
            weights,
            covariance,
        )
    )

    # 8. Calculate portfolio metrics
    metrics = calculate_portfolio_metrics(
        weights,
        expected_returns,
        covariance,
        returns,
    )

    # 9. Save allocations
    save_allocations(
        weights,
        expected_returns,
        risk_contributions,
    )

    # 10. Save portfolio-level metrics
    save_risk_metrics(metrics)

    print("\n")
    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_optimizer()