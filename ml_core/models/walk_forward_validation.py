import os

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

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

# Number of observations in the initial training set.
INITIAL_TRAIN_SIZE = 1000

# Number of observations evaluated in each validation window.
TEST_SIZE = 100

# Because target_5d uses the next 5 trading days,
# purge the final 5 training observations before each test set.
PURGE_DAYS = 5


engine = create_engine(DB_URL)

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(
    "walk_forward_validation"
)


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
            f"No data found for {ticker}"
        )

    df["date"] = pd.to_datetime(df["date"])

    # 5-day cumulative forward return
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

def calculate_metrics(y_true, predictions):

    mse = mean_squared_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    # Directional accuracy
    direction_accuracy = np.mean(
        np.sign(y_true)
        == np.sign(predictions)
    )

    # Prediction/target correlation
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
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "direction_accuracy": direction_accuracy,
        "correlation": correlation,
    }


# ============================================================
# WALK-FORWARD EVALUATION
# ============================================================

def walk_forward(model_name, model, df):

    predictions = []
    actuals = []
    fold_results = []

    n = len(df)

    train_end = INITIAL_TRAIN_SIZE
    fold_number = 1

    while train_end + PURGE_DAYS < n:

        # ----------------------------------------------------
        # Training data ends before the purge window.
        # ----------------------------------------------------

        train_start = 0

        train_stop = (
            train_end - PURGE_DAYS
        )

        test_start = train_end

        test_end = min(
            test_start + TEST_SIZE,
            n,
        )

        train_df = df.iloc[
            train_start:train_stop
        ]

        test_df = df.iloc[
            test_start:test_end
        ]

        if len(test_df) == 0:
            break

        X_train = train_df[FEATURES]
        y_train = train_df["target_5d"]

        X_test = test_df[FEATURES]
        y_test = test_df["target_5d"]

        print(
            f"\nFold {fold_number}"
        )

        print(
            f"Train: "
            f"{train_df['date'].iloc[0].date()} "
            f"→ "
            f"{train_df['date'].iloc[-1].date()}"
        )

        print(
            f"Purge: "
            f"{df['date'].iloc[train_stop].date()} "
            f"→ "
            f"{df['date'].iloc[test_start - 1].date()}"
        )

        print(
            f"Test : "
            f"{test_df['date'].iloc[0].date()} "
            f"→ "
            f"{test_df['date'].iloc[-1].date()}"
        )

        # ----------------------------------------------------
        # Fit only on past observations.
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

        fold_results.append(
            metrics
        )

        predictions.extend(
            preds
        )

        actuals.extend(
            y_test.to_numpy()
        )

        print(
            f"RMSE: "
            f"{metrics['rmse']:.6f}"
        )

        print(
            f"MAE: "
            f"{metrics['mae']:.6f}"
        )

        print(
            f"Direction: "
            f"{metrics['direction_accuracy']:.2%}"
        )

        print(
            f"Correlation: "
            f"{metrics['correlation']:.4f}"
        )

        fold_number += 1

        # Expanding window
        train_end = test_end

    overall = calculate_metrics(
        np.array(actuals),
        np.array(predictions),
    )

    return overall, pd.DataFrame(
        fold_results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PURGED WALK-FORWARD VALIDATION")
    print("=" * 60)

    df = load_dataset(TICKER)

    print(
        f"Dataset rows: {len(df)}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"→ "
        f"{df['date'].max().date()}"
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
        run_name="Ridge_Purged_WalkForward"
    ):

        metrics, _folds = walk_forward(
            "Ridge",
            ridge,
            df,
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
            "initial_train_size",
            INITIAL_TRAIN_SIZE,
        )

        mlflow.log_param(
            "test_size",
            TEST_SIZE,
        )

        mlflow.log_param(
            "purge_days",
            PURGE_DAYS,
        )

        for name, value in metrics.items():
            mlflow.log_metric(
                name,
                float(value),
            )

        print("\nRIDGE OVERALL")
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
        run_name="RandomForest_Purged_WalkForward"
    ):

        metrics, _folds = walk_forward(
            "RandomForest",
            rf,
            df,
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
            "initial_train_size",
            INITIAL_TRAIN_SIZE,
        )

        mlflow.log_param(
            "test_size",
            TEST_SIZE,
        )

        mlflow.log_param(
            "purge_days",
            PURGE_DAYS,
        )

        for name, value in metrics.items():
            mlflow.log_metric(
                name,
                float(value),
            )

        print("\nRANDOM FOREST OVERALL")
        print(metrics)


if __name__ == "__main__":
    main()