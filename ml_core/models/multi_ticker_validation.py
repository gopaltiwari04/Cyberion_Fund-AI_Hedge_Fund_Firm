import os

import mlflow
import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

DB_URL = os.getenv(
    "DB_URL",
    "postgresql://quant_user:quant_password@localhost:5432/quant_db",
)

MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "SPY",
]

FEATURES = [
    "return_1d",
    "return_5d",
    "rsi_14",
    "macd",
    "volatility_20d",
    "regime",
]

INITIAL_TRAIN_SIZE = 1000
TEST_SIZE = 100
PURGE_DAYS = 5


engine = create_engine(DB_URL)

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(
    "multi_ticker_validation"
)


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(ticker):

    query = text("""
        SELECT
            m.date,
            m.close,
            f.return_1d,
            f.return_5d,
            f.rsi_14,
            f.macd,
            f.volatility_20d,
            f.regime
        FROM market_data m
        INNER JOIN feature_store f
            ON m.ticker = f.ticker
            AND m.date = f.date
        WHERE m.ticker = :ticker
        ORDER BY m.date ASC
    """)

    df = pd.read_sql(
        query,
        engine,
        params={"ticker": ticker},
    )

    if df.empty:
        raise ValueError(
            f"No data found for {ticker}"
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    # --------------------------------------------------------
    # Correct 5-day forward cumulative return
    # --------------------------------------------------------

    df["target_5d"] = (
        df["close"].shift(-5)
        / df["close"]
        - 1
    )

    df = df.dropna(
        subset=FEATURES + ["target_5d"]
    ).reset_index(drop=True)

    return df


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
):

    mse = mean_squared_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    direction_accuracy = np.mean(
        np.sign(y_true)
        == np.sign(predictions)
    )

    if (
        np.std(predictions) > 0
        and np.std(y_true) > 0
    ):
        correlation = np.corrcoef(
            y_true,
            predictions,
        )[0, 1]
    else:
        correlation = 0.0

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "direction_accuracy": float(
            direction_accuracy
        ),
        "correlation": float(
            correlation
        ),
    }


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def walk_forward(
    model,
    df,
):

    predictions = []
    actuals = []

    fold_metrics = []

    n = len(df)

    train_end = INITIAL_TRAIN_SIZE

    fold_number = 1

    while train_end + PURGE_DAYS < n:

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_stop = (
            train_end - PURGE_DAYS
        )

        test_start = train_end

        test_end = min(
            test_start + TEST_SIZE,
            n,
        )

        train_df = df.iloc[
            :train_stop
        ]

        test_df = df.iloc[
            test_start:test_end
        ]

        if test_df.empty:
            break

        X_train = train_df[
            FEATURES
        ]

        y_train = train_df[
            "target_5d"
        ]

        X_test = test_df[
            FEATURES
        ]

        y_test = test_df[
            "target_5d"
        ]

        # ----------------------------------------------------
        # Fit
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,
        )

        preds = model.predict(
            X_test
        )

        metrics = calculate_metrics(
            y_test.to_numpy(),
            preds,
        )

        metrics["fold"] = fold_number

        fold_metrics.append(
            metrics
        )

        predictions.extend(
            preds
        )

        actuals.extend(
            y_test.to_numpy()
        )

        fold_number += 1

        # Expanding window
        train_end = test_end

    overall = calculate_metrics(
        np.array(actuals),
        np.array(predictions),
    )

    return (
        overall,
        pd.DataFrame(fold_metrics),
    )


# ============================================================
# NAIVE BASELINE
# ============================================================

def evaluate_naive(df):

    n = len(df)

    train_end = INITIAL_TRAIN_SIZE

    predictions = []
    actuals = []

    while train_end + PURGE_DAYS < n:

        train_stop = (
            train_end - PURGE_DAYS
        )

        test_start = train_end

        test_end = min(
            test_start + TEST_SIZE,
            n,
        )

        train_df = df.iloc[
            :train_stop
        ]

        test_df = df.iloc[
            test_start:test_end
        ]

        if test_df.empty:
            break

        # Mean target observed in training
        prediction = train_df[
            "target_5d"
        ].mean()

        predictions.extend(
            [prediction] * len(test_df)
        )

        actuals.extend(
            test_df[
                "target_5d"
            ].to_numpy()
        )

        train_end = test_end

    return calculate_metrics(
        np.array(actuals),
        np.array(predictions),
    )


# ============================================================
# RUN ONE TICKER
# ============================================================

def run_ticker(ticker):

    print("\n")
    print("=" * 70)
    print(f"VALIDATING {ticker}")
    print("=" * 70)

    df = load_dataset(ticker)

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"→ "
        f"{df['date'].max().date()}"
    )

    results = []

    # ========================================================
    # NAIVE
    # ========================================================

    naive_metrics = evaluate_naive(
        df
    )

    results.append({
        "ticker": ticker,
        "model": "Naive",
        **naive_metrics,
    })

    print("\nNAIVE")
    print(naive_metrics)

    # ========================================================
    # RIDGE
    # ========================================================

    ridge = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            Ridge(alpha=1.0),
        ),
    ])

    with mlflow.start_run(
        run_name=f"{ticker}_Ridge"
    ):

        metrics, folds = walk_forward(
            ridge,
            df,
        )

        mlflow.log_params({
            "ticker": ticker,
            "model_type": "Ridge",
            "alpha": 1.0,
            "initial_train_size":
                INITIAL_TRAIN_SIZE,
            "test_size": TEST_SIZE,
            "purge_days": PURGE_DAYS,
            "features": ",".join(
                FEATURES
            ),
        })

        mlflow.log_metrics(
            metrics
        )

        results.append({
            "ticker": ticker,
            "model": "Ridge",
            **metrics,
        })

    print("\nRIDGE")
    print(metrics)

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )

    with mlflow.start_run(
        run_name=f"{ticker}_RandomForest"
    ):

        metrics, folds = walk_forward(
            rf,
            df,
        )

        mlflow.log_params({
            "ticker": ticker,
            "model_type": "RandomForest",
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_leaf": 10,
            "initial_train_size":
                INITIAL_TRAIN_SIZE,
            "test_size": TEST_SIZE,
            "purge_days": PURGE_DAYS,
            "features": ",".join(
                FEATURES
            ),
        })

        mlflow.log_metrics(
            metrics
        )

        results.append({
            "ticker": ticker,
            "model": "RandomForest",
            **metrics,
        })

    print("\nRANDOM FOREST")
    print(metrics)

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MULTI-TICKER PURGED WALK-FORWARD VALIDATION")
    print("=" * 70)

    all_results = []

    for ticker in TICKERS:

        try:

            results = run_ticker(
                ticker
            )

            all_results.extend(
                results
            )

        except Exception as e:

            print(
                f"\nERROR processing "
                f"{ticker}: {e}"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    if results_df.empty:
        raise RuntimeError(
            "No validation results generated."
        )

    print("\n")
    print("=" * 70)
    print("FINAL MULTI-TICKER RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "ticker",
                "model",
                "rmse",
                "mae",
                "direction_accuracy",
                "correlation",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Average performance by model
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("AVERAGE PERFORMANCE BY MODEL")
    print("=" * 70)

    summary = (
        results_df
        .groupby("model")[
            [
                "rmse",
                "mae",
                "direction_accuracy",
                "correlation",
            ]
        ]
        .mean()
        .sort_values("rmse")
    )

    print(
        summary.to_string()
    )

    # --------------------------------------------------------
    # Save local CSV
    # --------------------------------------------------------

    output_path = (
        "ml_core/models/"
        "multi_ticker_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved results to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()