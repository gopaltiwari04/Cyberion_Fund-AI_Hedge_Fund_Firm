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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIG
# ============================================================

DB_URL = os.getenv(
    "DB_URL",
    "postgresql://quant_user:quant_password@localhost:5432/quant_db",
)

MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

TICKER = "AAPL"

FEATURES = [
    "return_1d",
    "return_5d",
    "rsi_14",
    "macd",
    "volatility_20d",
    "regime",
]


engine = create_engine(DB_URL)

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("baseline_models")


# ============================================================
# LOAD DATA
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
            f"No data available for {ticker}"
        )

    df["date"] = pd.to_datetime(df["date"])

    # --------------------------------------------------------
    # 5-day FORWARD cumulative return
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
# EVALUATION
# ============================================================

def evaluate_model(model, X_train, y_train, X_test, y_test):

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    return mse, rmse, mae


# ============================================================
# TRAIN
# ============================================================

def train_baselines():

    df = load_dataset(TICKER)

    print("=" * 60)
    print(f"BASELINE MODELING — {TICKER}")
    print("=" * 60)

    print(
        f"Rows       : {len(df)}"
    )

    print(
        f"Date range : "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    split_idx = int(
        len(df) * 0.8
    )

    train_df = df.iloc[
        :split_idx
    ].copy()

    test_df = df.iloc[
        split_idx:
    ].copy()

    X_train = train_df[FEATURES]
    y_train = train_df["target_5d"]

    X_test = test_df[FEATURES]
    y_test = test_df["target_5d"]

    print(
        f"Training rows: {len(train_df)}"
    )

    print(
        f"Testing rows : {len(test_df)}"
    )

    # ========================================================
    # NAIVE BASELINE
    # ========================================================

    naive_prediction = np.repeat(
        y_train.mean(),
        len(y_test),
    )

    naive_mse = mean_squared_error(
        y_test,
        naive_prediction,
    )

    naive_rmse = np.sqrt(
        naive_mse
    )

    naive_mae = mean_absolute_error(
        y_test,
        naive_prediction,
    )

    print(
        f"\nNaive baseline"
        f"\nMSE : {naive_mse:.8f}"
        f"\nRMSE: {naive_rmse:.8f}"
        f"\nMAE : {naive_mae:.8f}"
    )

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
        run_name="Ridge_Baseline"
    ):

        mse, rmse, mae = evaluate_model(
            ridge,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        mlflow.log_param(
            "model_type",
            "Ridge",
        )

        mlflow.log_param(
            "ticker",
            TICKER,
        )

        mlflow.log_param(
            "alpha",
            1.0,
        )

        mlflow.log_param(
            "features",
            ",".join(FEATURES),
        )

        mlflow.log_param(
            "train_rows",
            len(train_df),
        )

        mlflow.log_param(
            "test_rows",
            len(test_df),
        )

        mlflow.log_metric(
            "mse",
            mse,
        )

        mlflow.log_metric(
            "rmse",
            rmse,
        )

        mlflow.log_metric(
            "mae",
            mae,
        )

        print(
            f"\nRidge"
            f"\nMSE : {mse:.8f}"
            f"\nRMSE: {rmse:.8f}"
            f"\nMAE : {mae:.8f}"
        )

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
        run_name="RandomForest_Baseline"
    ):

        mse, rmse, mae = evaluate_model(
            rf,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        mlflow.log_param(
            "model_type",
            "RandomForest",
        )

        mlflow.log_param(
            "ticker",
            TICKER,
        )

        mlflow.log_param(
            "n_estimators",
            100,
        )

        mlflow.log_param(
            "max_depth",
            5,
        )

        mlflow.log_param(
            "min_samples_leaf",
            10,
        )

        mlflow.log_param(
            "features",
            ",".join(FEATURES),
        )

        mlflow.log_metric(
            "mse",
            mse,
        )

        mlflow.log_metric(
            "rmse",
            rmse,
        )

        mlflow.log_metric(
            "mae",
            mae,
        )

        print(
            f"\nRandom Forest"
            f"\nMSE : {mse:.8f}"
            f"\nRMSE: {rmse:.8f}"
            f"\nMAE : {mae:.8f}"
        )


if __name__ == "__main__":
    train_baselines()