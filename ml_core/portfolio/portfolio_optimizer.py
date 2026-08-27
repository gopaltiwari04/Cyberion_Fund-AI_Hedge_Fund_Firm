"""
ML-DRIVEN PORTFOLIO OPTIMIZER

Pipeline:

    PostgreSQL feature_store
            ↓
    trained XGBoost model
            ↓
    predicted 5-day returns
            ↓
    annualized expected returns
            ↓
    covariance matrix from market_data
            ↓
    CVXPY portfolio optimization
            ↓
    portfolio_allocations
    portfolio_risk_metrics

Constraints:
    - Long-only
    - Fully invested
    - Maximum 35% per asset
"""

from __future__ import annotations

import os
from datetime import date

import joblib
import numpy as np
import pandas as pd
import cvxpy as cp

from sqlalchemy import create_engine, text


# ============================================================
# CONFIGURATION
# ============================================================

PG_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://quant_user:quant_password@localhost:5432/quant_db",
)

MODEL_PATH = os.getenv(
    "XGBOOST_MODEL_PATH",
    os.path.join(
        "ml_core",
        "models",
        "xgboost_return_model.joblib",
    ),
)

ASSETS = [
    "AAPL",
    "AMZN",
    "GOOGL",
    "MSFT",
    "SPY",
]

STRATEGY_NAME = "xgboost_mean_variance_v1"

MAX_WEIGHT = 0.35

TRADING_DAYS = 252

# XGBoost predicts a 5-day forward return.
PREDICTION_HORIZON_DAYS = 5

RISK_FREE_RATE = 0.0

# Same feature order used by xgboost_model.py.
FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "rsi_14",
    "macd",
    "volatility_5d",
    "volatility_20d",
    "volatility_60d",
    "atr_14_pct",
    "sma_20_distance",
    "sma_50_distance",
    "sma_200_distance",
    "drawdown_20d",
    "drawdown_60d",
    "volume_change_5d",
    "volume_zscore_20d",
    "market_return_5d",
    "relative_return_5d",
    "beta_60d",
    "correlation_spy_60d",
    "intraday_range",
    "close_position",
    "gap_return",
    "regime",
    "sentiment_score",
    "degree_centrality",
    "pagerank",
]


# ============================================================
# DATABASE
# ============================================================

engine = create_engine(
    PG_URL,
    pool_pre_ping=True,
)


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

def load_model():
    """
    Load the trained XGBoost model.
    """

    print()
    print("=" * 60)
    print("LOADING XGBOOST MODEL")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"XGBoost model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print(f"Model loaded: {MODEL_PATH}")

    return model


# ============================================================
# LOAD LATEST FEATURES
# ============================================================

def load_latest_features() -> pd.DataFrame:
    """
    Load the most recent feature_store observation for every asset.
    """

    print()
    print("=" * 60)
    print("LOADING LATEST FEATURES")
    print("=" * 60)

    placeholders = ", ".join(
        f":ticker_{i}"
        for i in range(len(ASSETS))
    )

    params = {
        f"ticker_{i}": ticker
        for i, ticker in enumerate(ASSETS)
    }

    feature_sql = ",\n            ".join(
        FEATURE_COLUMNS
    )

    query = text(
        f"""
        SELECT
            ticker,
            date,
            {feature_sql}
        FROM feature_store
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, date DESC
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params=params,
        )

    if df.empty:
        raise RuntimeError(
            "No rows found in feature_store."
        )

    # One latest observation per asset.
    df = (
        df.sort_values(
            ["ticker", "date"],
            ascending=[True, False],
        )
        .groupby("ticker", as_index=False)
        .first()
    )

    missing_assets = [
        ticker
        for ticker in ASSETS
        if ticker not in set(df["ticker"])
    ]

    if missing_assets:
        raise RuntimeError(
            f"Missing latest features for: {missing_assets}"
        )

    df = df.set_index("ticker").loc[ASSETS].reset_index()

    print(
        f"Latest feature dates: "
        f"{df[['ticker', 'date']].to_dict('records')}"
    )

    return df


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare latest feature rows for XGBoost inference.

    Missing values are filled using the median of the available
    latest cross-section. This is only a deployment safeguard.
    """

    print()
    print("=" * 60)
    print("PREPARING XGBOOST INPUT")
    print("=" * 60)

    X = df[FEATURE_COLUMNS].copy()

    # Replace infinities.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    missing_before = int(
        X.isna().sum().sum()
    )

    print(
        f"Missing feature values before "
        f"deployment imputation: {missing_before}"
    )

    # The model was trained with imputation.
    # For deployment, use cross-sectional medians.
    for column in FEATURE_COLUMNS:
        if X[column].isna().any():
            median = X[column].median()

            if pd.isna(median):
                median = 0.0

            X[column] = X[column].fillna(median)

    missing_after = int(
        X.isna().sum().sum()
    )

    print(
        f"Missing feature values after "
        f"deployment imputation: {missing_after}"
    )

    if missing_after:
        raise RuntimeError(
            "Unable to produce complete XGBoost input."
        )

    return X


# ============================================================
# XGBOOST PREDICTIONS
# ============================================================

def predict_returns(
    model,
    X: pd.DataFrame,
    latest_features: pd.DataFrame,
) -> np.ndarray:
    """
    Predict 5-day forward returns.

    Returns are converted into annualized expected returns
    for portfolio optimization.

    IMPORTANT:

    The XGBoost model predicts a 5-day return.

    We annualize using:

        (1 + predicted_5d) ** (252 / 5) - 1

    rather than simply multiplying by 252.
    """

    print()
    print("=" * 60)
    print("GENERATING XGBOOST RETURN PREDICTIONS")
    print("=" * 60)

    predictions = model.predict(X)

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    if predictions.shape[0] != len(ASSETS):
        raise RuntimeError(
            "XGBoost prediction count does not match "
            "portfolio asset count."
        )

    if not np.all(np.isfinite(predictions)):
        raise RuntimeError(
            "XGBoost produced non-finite predictions."
        )

    # Guard against mathematically invalid annualization.
    predictions = np.maximum(
        predictions,
        -0.999,
    )

    annualized = (
        np.power(
            1.0 + predictions,
            TRADING_DAYS / PREDICTION_HORIZON_DAYS,
        )
        - 1.0
    )

    print()
    print("MODEL PREDICTIONS")
    print("-" * 60)

    for ticker, prediction, annual in zip(
        ASSETS,
        predictions,
        annualized,
    ):
        print(
            f"{ticker:<6} | "
            f"5-day={prediction:>9.4%} | "
            f"annualized={annual:>9.4%}"
        )

    return annualized


# ============================================================
# LOAD MARKET DATA
# ============================================================

def load_market_data() -> pd.DataFrame:
    """
    Load historical closing prices for portfolio covariance.
    """

    print()
    print("=" * 60)
    print("LOADING MARKET DATA")
    print("=" * 60)

    placeholders = ", ".join(
        f":ticker_{i}"
        for i in range(len(ASSETS))
    )

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

    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params=params,
        )

    if df.empty:
        raise RuntimeError(
            "No market data found."
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} -> "
        f"{df['date'].max().date()}"
    )

    return df


# ============================================================
# PREPARE PRICE MATRIX
# ============================================================

def prepare_prices(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert market data to date x ticker price matrix.
    """

    prices = df.pivot(
        index="date",
        columns="ticker",
        values="close",
    )

    missing_assets = [
        ticker
        for ticker in ASSETS
        if ticker not in prices.columns
    ]

    if missing_assets:
        raise RuntimeError(
            f"Missing assets from market data: "
            f"{missing_assets}"
        )

    prices = prices[ASSETS]

    prices = prices.dropna()

    if len(prices) < 252:
        raise RuntimeError(
            f"Insufficient synchronized price history: "
            f"{len(prices)} rows."
        )

    print(
        f"Synchronized trading days: "
        f"{len(prices):,}"
    )

    return prices


# ============================================================
# CALCULATE RETURNS
# ============================================================

def calculate_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate daily percentage returns.
    """

    returns = (
        prices
        .pct_change()
        .dropna()
    )

    print(
        f"Return observations: "
        f"{len(returns):,}"
    )

    return returns


# ============================================================
# COVARIANCE
# ============================================================

def calculate_covariance_matrix(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate annualized covariance matrix.
    """

    covariance = (
        returns.cov()
        * TRADING_DAYS
    )

    # Numerical stabilization.
    covariance = (
        covariance
        + covariance.T
    ) / 2.0

    print()
    print("ANNUALIZED COVARIANCE MATRIX")
    print("-" * 60)
    print(covariance)

    return covariance


# ============================================================
# PORTFOLIO OPTIMIZATION
# ============================================================

def optimize_portfolio(
    expected_returns: np.ndarray,
    covariance: pd.DataFrame,
) -> np.ndarray:
    """
    Maximize expected return while penalizing variance.

    Constraints:
        sum(weights) == 1
        0 <= weights <= MAX_WEIGHT
    """

    print()
    print("=" * 60)
    print("OPTIMIZING ML-DRIVEN PORTFOLIO")
    print("=" * 60)

    n_assets = len(ASSETS)

    weights = cp.Variable(
        n_assets
    )

    covariance_matrix = covariance.values

    # Keep risk aversion moderate.
    risk_aversion = 1.0

    expected_portfolio_return = (
        expected_returns @ weights
    )

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

    if not np.all(
        np.isfinite(optimized_weights)
    ):
        raise RuntimeError(
            "Optimizer produced invalid weights."
        )

    optimized_weights[
        np.abs(optimized_weights) < 1e-10
    ] = 0.0

    total = optimized_weights.sum()

    if total <= 0:
        raise RuntimeError(
            "Optimizer produced zero total allocation."
        )

    optimized_weights /= total

    print()
    print("OPTIMAL ALLOCATION")
    print("-" * 60)

    for ticker, weight in zip(
        ASSETS,
        optimized_weights,
    ):
        print(
            f"{ticker:<6} | {weight:.4%}"
        )

    print("-" * 60)
    print(
        f"Total   | "
        f"{optimized_weights.sum():.4%}"
    )
    print(
        f"Maximum | "
        f"{optimized_weights.max():.4%}"
    )

    return optimized_weights


# ============================================================
# PORTFOLIO METRICS
# ============================================================

def calculate_portfolio_metrics(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance: pd.DataFrame,
) -> dict:
    """
    Calculate expected portfolio return,
    volatility and Sharpe ratio.
    """

    covariance_matrix = covariance.values

    expected_return = float(
        weights @ expected_returns
    )

    variance = float(
        weights
        @ covariance_matrix
        @ weights
    )

    volatility = float(
        np.sqrt(
            max(variance, 0.0)
        )
    )

    sharpe = (
        (expected_return - RISK_FREE_RATE)
        / volatility
        if volatility > 0
        else 0.0
    )

    print()
    print("=" * 60)
    print("PORTFOLIO METRICS")
    print("=" * 60)

    print(
        f"Expected annual return     : "
        f"{expected_return:.4%}"
    )

    print(
        f"Expected annual volatility : "
        f"{volatility:.4%}"
    )

    print(
        f"Sharpe ratio               : "
        f"{sharpe:.4f}"
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
        return np.zeros(
            len(weights)
        )

    marginal_contribution = (
        covariance_matrix @ weights
    ) / portfolio_volatility

    contribution = (
        weights
        * marginal_contribution
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

    print()
    print("Saving portfolio allocations...")

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

        # Idempotent reruns.
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

        for (
            ticker,
            weight,
            expected_return,
            risk,
        ) in zip(
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
                    "risk_contribution": float(
                        risk
                    ),
                },
            )

    print(
        f"Saved {len(ASSETS)} portfolio allocations."
    )


# ============================================================
# SAVE RISK METRICS
# ============================================================

def save_risk_metrics(
    metrics: dict,
) -> None:
    """
    Store portfolio-level metrics.
    """

    today = date.today()

    print(
        "Saving portfolio risk metrics..."
    )

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
                "var_95": None,
                "cvar_95": None,
            },
        )

    print(
        "Portfolio risk metrics saved."
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_optimizer() -> None:

    print()
    print("=" * 60)
    print("ML-DRIVEN PORTFOLIO OPTIMIZER")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load trained ML model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # 2. Load latest feature_store observations
    # --------------------------------------------------------

    latest_features = (
        load_latest_features()
    )

    # --------------------------------------------------------
    # 3. Prepare model input
    # --------------------------------------------------------

    X = prepare_model_input(
        latest_features
    )

    # --------------------------------------------------------
    # 4. Generate ML expected returns
    # --------------------------------------------------------

    expected_returns = predict_returns(
        model,
        X,
        latest_features,
    )

    # --------------------------------------------------------
    # 5. Load historical market data
    # --------------------------------------------------------

    market_data = load_market_data()

    # --------------------------------------------------------
    # 6. Build synchronized prices
    # --------------------------------------------------------

    prices = prepare_prices(
        market_data
    )

    # --------------------------------------------------------
    # 7. Calculate daily returns
    # --------------------------------------------------------

    returns = calculate_returns(
        prices
    )

    # --------------------------------------------------------
    # 8. Calculate covariance
    # --------------------------------------------------------

    covariance = calculate_covariance_matrix(
        returns
    )

    # --------------------------------------------------------
    # 9. Optimize
    # --------------------------------------------------------

    weights = optimize_portfolio(
        expected_returns,
        covariance,
    )

    # --------------------------------------------------------
    # 10. Risk contributions
    # --------------------------------------------------------

    risk_contributions = (
        calculate_risk_contributions(
            weights,
            covariance,
        )
    )

    # --------------------------------------------------------
    # 11. Portfolio metrics
    # --------------------------------------------------------

    metrics = calculate_portfolio_metrics(
        weights,
        expected_returns,
        covariance,
    )

    # --------------------------------------------------------
    # 12. Save allocations
    # --------------------------------------------------------

    save_allocations(
        weights,
        expected_returns,
        risk_contributions,
    )

    # --------------------------------------------------------
    # 13. Save risk metrics
    # --------------------------------------------------------

    save_risk_metrics(
        metrics
    )

    print()
    print("=" * 60)
    print("ML PORTFOLIO OPTIMIZATION COMPLETED")
    print("=" * 60)

    print()
    print(
        f"Strategy : {STRATEGY_NAME}"
    )

    print(
        f"Model    : {MODEL_PATH}"
    )

    print(
        f"Date     : {date.today()}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_optimizer()