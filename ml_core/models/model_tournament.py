"""
Model Tournament
================

Standardized comparison of:

    1. Naive baseline
    2. Ridge Regression
    3. Random Forest
    4. Optimized XGBoost

All models use:
    - identical dataset
    - identical features
    - identical target
    - identical purged walk-forward folds
    - identical test observations

The final holdout is NOT used for model selection.
"""

import os
import sys

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sqlalchemy import create_engine

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from ml_core.models.cv_utils import purged_time_series_split

# ============================================================
# CONFIGURATION
# ============================================================

DB_URL = "postgresql://quant_user:quant_password@localhost:5432/quant_db"

TICKER = "AAPL"

FEATURES = [
    "rsi_14",
    "macd",
    "volatility_20d",
    "regime",
    "return_1d",
]

TARGET = "target_5d"

N_SPLITS = 3
PURGE_GAP = 5

# Final untouched holdout
HOLDOUT_RATIO = 0.20

RANDOM_STATE = 42

engine = create_engine(DB_URL)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("model_tournament")


# ============================================================
# XGBOOST PARAMETERS
# ============================================================

# These are the parameters found by the Week 6 Optuna run.

XGBOOST_PARAMS = {
    "n_estimators": 50,
    "max_depth": 7,
    "learning_rate": 0.00889784997053407,
    "subsample": 0.9446654644176917,
    "colsample_bytree": 0.944307996244671,
    "min_child_weight": 6,
    "reg_alpha": 0.0001057862064721877,
    "reg_lambda": 0.5866811502741054,
    "random_state": RANDOM_STATE,
    "objective": "reg:squarederror",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_data(ticker=TICKER):

    query = """
        SELECT
            f.date,
            m.close,
            f.rsi_14,
            f.macd,
            f.volatility_20d,
            f.regime,
            f.return_1d
        FROM feature_store f
        JOIN market_data m
            ON f.ticker = m.ticker
            AND f.date = m.date
        WHERE f.ticker = %(ticker)s
        ORDER BY f.date ASC
    """

    df = pd.read_sql(
        query,
        engine,
        params={"ticker": ticker},
    )

    if df.empty:
        raise ValueError(
            f"No joined market/feature data found for {ticker}"
        )

    df["date"] = pd.to_datetime(df["date"])

    # --------------------------------------------------------
    # 5-day forward return
    #
    # Target at time t:
    #
    # Close[t+5] / Close[t] - 1
    #
    # The final 5 observations naturally have no target.
    # --------------------------------------------------------

    df[TARGET] = (
        df["close"].shift(-5) / df["close"]
    ) - 1.0

    df = df.dropna().reset_index(drop=True)

    return df

# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, predictions):

    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)

    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, predictions)

    # Directional accuracy
    direction_accuracy = np.mean(
        np.sign(y_true) == np.sign(predictions)
    )

    # Pearson correlation
    if (
        np.std(y_true) == 0
        or np.std(predictions) == 0
    ):
        correlation = 0.0
    else:
        correlation = np.corrcoef(
            y_true,
            predictions
        )[0, 1]

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "direction_accuracy": float(direction_accuracy),
        "correlation": float(correlation),
    }


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(model_name):

    if model_name == "Ridge":

        return Ridge(
            alpha=1.0
        )

    elif model_name == "RandomForest":

        return RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif model_name == "XGBoost":

        return xgb.XGBRegressor(
            **XGBOOST_PARAMS
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


# ============================================================
# PURGED WALK-FORWARD EVALUATION
# ============================================================

def evaluate_model(
    model_name,
    X,
    y,
    dates,
):

    print()
    print("=" * 70)
    print(f"{model_name.upper()} — PURGED WALK-FORWARD")
    print("=" * 70)

    fold_predictions = []

    fold_metrics = []

    # --------------------------------------------------------
    # Same folds for EVERY model.
    # This is essential for a fair comparison.
    # --------------------------------------------------------

    for fold_number, (train_idx, test_idx) in enumerate(
        purged_time_series_split(
            X,
            n_splits=N_SPLITS,
            purge_gap=PURGE_GAP,
        ),
        start=1,
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        train_start = dates.iloc[train_idx[0]]
        train_end = dates.iloc[train_idx[-1]]

        test_start = dates.iloc[test_idx[0]]
        test_end = dates.iloc[test_idx[-1]]

        purge_start = dates.iloc[
            train_idx[-1] + 1
        ]

        purge_end = dates.iloc[
            test_idx[0] - 1
        ]

        print()
        print(f"Fold {fold_number}")
        print("-" * 50)

        print(
            f"Train : {train_start.date()} → "
            f"{train_end.date()}"
        )

        print(
            f"Purge : {purge_start.date()} → "
            f"{purge_end.date()}"
        )

        print(
            f"Test  : {test_start.date()} → "
            f"{test_end.date()}"
        )

        # ----------------------------------------------------
        # Naive baseline
        #
        # Predict the mean return observed in the TRAINING
        # portion only.
        # ----------------------------------------------------

        if model_name == "Naive":

            prediction_value = y_train.mean()

            predictions = np.full(
                len(y_test),
                prediction_value,
            )

        else:

            model = create_model(model_name)

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(
                X_test
            )

        metrics = calculate_metrics(
            y_test,
            predictions,
        )

        print(
            f"RMSE        : {metrics['rmse']:.6f}"
        )

        print(
            f"MAE         : {metrics['mae']:.6f}"
        )

        print(
            f"Direction   : "
            f"{metrics['direction_accuracy']:.2%}"
        )

        print(
            f"Correlation : "
            f"{metrics['correlation']:.4f}"
        )

        fold_metrics.append(metrics)

        # ----------------------------------------------------
        # Save every prediction.
        # This allows us to compare models on exactly the
        # same observations later.
        # ----------------------------------------------------

        fold_prediction_df = pd.DataFrame({
            "ticker": TICKER,
            "date": dates.iloc[test_idx].values,
            "model": model_name,
            "actual": y_test.values,
            "prediction": predictions,
            "fold": fold_number,
        })

        fold_predictions.append(
            fold_prediction_df
        )

    # --------------------------------------------------------
    # Combine all out-of-sample predictions
    # --------------------------------------------------------

    predictions_df = pd.concat(
        fold_predictions,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Overall metrics are calculated from ALL out-of-sample
    # predictions, not by simply averaging fold metrics.
    # --------------------------------------------------------

    overall_metrics = calculate_metrics(
        predictions_df["actual"],
        predictions_df["prediction"],
    )

    return (
        overall_metrics,
        predictions_df,
    )


# ============================================================
# FINAL HOLDOUT
# ============================================================

def evaluate_final_holdout(
    model_name,
    X_train,
    y_train,
    X_holdout,
    y_holdout,
):

    if model_name == "Naive":

        prediction_value = y_train.mean()

        predictions = np.full(
            len(y_holdout),
            prediction_value,
        )

    else:

        model = create_model(model_name)

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_holdout
        )

    return calculate_metrics(
        y_holdout,
        predictions,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MODEL TOURNAMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_data(TICKER)

    print()
    print(f"Ticker       : {TICKER}")
    print(f"Rows         : {len(df)}")
    print(
        f"Date range   : "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Final holdout split
    #
    # The final 20% is completely untouched during CV/model
    # selection.
    # --------------------------------------------------------

    split_idx = int(
        len(df) * (1 - HOLDOUT_RATIO)
    )

    development_df = df.iloc[
        :split_idx
    ].copy()

    holdout_df = df.iloc[
        split_idx:
    ].copy()

    print()
    print(
        f"Development rows : "
        f"{len(development_df)}"
    )

    print(
        f"Holdout rows     : "
        f"{len(holdout_df)}"
    )

    print(
        f"Holdout period   : "
        f"{holdout_df['date'].min().date()} → "
        f"{holdout_df['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Feature matrices
    # --------------------------------------------------------

    X_dev = development_df[FEATURES]
    y_dev = development_df[TARGET]

    dates_dev = development_df["date"]

    X_holdout = holdout_df[FEATURES]
    y_holdout = holdout_df[TARGET]

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = [
        "Naive",
        "Ridge",
        "RandomForest",
        "XGBoost",
    ]

    all_results = []
    all_predictions = []

    # ========================================================
    # WALK-FORWARD TOURNAMENT
    # ========================================================

    for model_name in models:

        metrics, predictions = evaluate_model(
            model_name,
            X_dev,
            y_dev,
            dates_dev,
        )

        metrics["ticker"] = TICKER
        metrics["model"] = model_name

        all_results.append(metrics)

        all_predictions.append(
            predictions
        )

        # ----------------------------------------------------
        # MLflow
        # ----------------------------------------------------

        with mlflow.start_run(
            run_name=f"{TICKER}_{model_name}_Tournament"
        ):

            mlflow.log_param(
                "ticker",
                TICKER,
            )

            mlflow.log_param(
                "model",
                model_name,
            )

            mlflow.log_param(
                "n_splits",
                N_SPLITS,
            )

            mlflow.log_param(
                "purge_gap",
                PURGE_GAP,
            )

            mlflow.log_param(
                "holdout_ratio",
                HOLDOUT_RATIO,
            )

            mlflow.log_param(
                "target",
                TARGET,
            )

            for feature in FEATURES:

                mlflow.log_param(
                    f"feature_{feature}",
                    True,
                )

            for metric_name, value in metrics.items():

                if metric_name not in [
                    "ticker",
                    "model",
                ]:

                    mlflow.log_metric(
                        metric_name,
                        value,
                    )

            if model_name == "XGBoost":

                mlflow.log_params(
                    {
                        f"xgb_{key}": value
                        for key, value
                        in XGBOOST_PARAMS.items()
                    }
                )

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_df = results_df[
        [
            "ticker",
            "model",
            "mse",
            "rmse",
            "mae",
            "direction_accuracy",
            "correlation",
        ]
    ]

    # Sort primarily by RMSE
    results_df = results_df.sort_values(
        "rmse"
    ).reset_index(drop=True)

    # ========================================================
    # PREDICTIONS
    # ========================================================

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # ========================================================
    # SAVE WALK-FORWARD RESULTS
    # ========================================================

    results_path = (
        "ml_core/models/"
        "model_tournament_results.csv"
    )

    predictions_path = (
        "ml_core/models/"
        "model_tournament_predictions.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    # ========================================================
    # PRINT TOURNAMENT RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("WALK-FORWARD TOURNAMENT RESULTS")
    print("=" * 70)

    display_df = results_df.copy()

    display_df["direction_accuracy"] = (
        display_df["direction_accuracy"] * 100
    )

    print(
        display_df.to_string(
            index=False,
            formatters={
                "mse": "{:.8f}".format,
                "rmse": "{:.6f}".format,
                "mae": "{:.6f}".format,
                "direction_accuracy": "{:.2f}%".format,
                "correlation": "{:.4f}".format,
            },
        )
    )

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    best_model = results_df.iloc[0]["model"]

    print()
    print(
        f"Best walk-forward model by RMSE: "
        f"{best_model}"
    )

    # ========================================================
    # FINAL HOLDOUT
    #
    # IMPORTANT:
    # We only evaluate the holdout AFTER model selection.
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL HOLDOUT EVALUATION")
    print("=" * 70)

    holdout_results = []

    for model_name in models:

        metrics = evaluate_final_holdout(
            model_name,
            X_dev,
            y_dev,
            X_holdout,
            y_holdout,
        )

        metrics["ticker"] = TICKER
        metrics["model"] = model_name

        holdout_results.append(
            metrics
        )

        print()
        print(model_name)

        print(
            f"RMSE        : "
            f"{metrics['rmse']:.6f}"
        )

        print(
            f"MAE         : "
            f"{metrics['mae']:.6f}"
        )

        print(
            f"Direction   : "
            f"{metrics['direction_accuracy']:.2%}"
        )

        print(
            f"Correlation : "
            f"{metrics['correlation']:.4f}"
        )

    holdout_df_results = pd.DataFrame(
        holdout_results
    )

    holdout_path = (
        "ml_core/models/"
        "model_tournament_holdout_results.csv"
    )

    holdout_df_results.to_csv(
        holdout_path,
        index=False,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print()
    print(
        "Walk-forward winner:"
    )

    print(
        f"  {best_model}"
    )

    print()
    print(
        "Files saved:"
    )

    print(
        f"  {results_path}"
    )

    print(
        f"  {predictions_path}"
    )

    print(
        f"  {holdout_path}"
    )

    print()
    print("=" * 70)
    print("MODEL TOURNAMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()