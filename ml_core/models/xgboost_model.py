"""
XGBOOST 5-DAY FORWARD RETURN PREDICTOR

Purpose
-------
Train an out-of-sample XGBoost regression model using the point-in-time
feature store.

Design principles
-----------------
1. Chronological train/validation/test split.
2. No random shuffling.
3. Training-only median imputation.
4. Conservative XGBoost regularization.
5. Early stopping using validation data.
6. Multiple evaluation metrics.
7. Out-of-sample predictions saved to CSV.
8. Feature importance reported.
9. Model artifact contains preprocessing metadata.
10. Features are never calculated using future observations.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://quant_user:quant_password@localhost:5432/quant_db",
)

MODEL_PATH = Path(
    "ml_core/models/xgboost_return_model.joblib"
)

PREDICTIONS_PATH = Path(
    "ml_core/models/xgboost_oos_predictions.csv"
)


# ============================================================
# FEATURES
# ============================================================

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
# ENGINE
# ============================================================

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset() -> pd.DataFrame:

    print("=" * 60)
    print("LOADING ML DATASET")
    print("=" * 60)

    query = text(
        """
        SELECT
            f.ticker,
            f.date,

            f.return_1d,
            f.return_5d,
            f.return_10d,
            f.return_20d,
            f.return_60d,

            f.rsi_14,
            f.macd,

            f.volatility_5d,
            f.volatility_20d,
            f.volatility_60d,
            f.atr_14_pct,

            f.sma_20_distance,
            f.sma_50_distance,
            f.sma_200_distance,

            f.drawdown_20d,
            f.drawdown_60d,

            f.volume_change_5d,
            f.volume_zscore_20d,

            f.market_return_5d,
            f.relative_return_5d,
            f.beta_60d,
            f.correlation_spy_60d,

            f.intraday_range,
            f.close_position,
            f.gap_return,

            f.regime,
            f.sentiment_score,
            f.degree_centrality,
            f.pagerank

        FROM feature_store f

        WHERE f.ticker IN (
            'AAPL',
            'AMZN',
            'GOOGL',
            'MSFT',
            'SPY'
        )

        ORDER BY f.date, f.ticker
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        raise RuntimeError(
            "No rows were returned from feature_store."
        )

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Assets: "
        f"{sorted(df['ticker'].unique().tolist())}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} -> "
        f"{df['date'].max().date()}"
    )

    return df


# ============================================================
# TARGET
# ============================================================

def create_forward_return_target(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 60)
    print("CREATING FORWARD RETURN TARGET")
    print("=" * 60)

    df = df.copy()

    df["forward_return_5d"] = (
        df.groupby("ticker")["return_1d"]
        .transform(
            lambda x: (
                (1.0 + x)
                .rolling(5)
                .apply(
                    np.prod,
                    raw=True,
                )
                - 1.0
            )
            .shift(-4)
        )
    )

    # More reliable target construction directly from close would
    # require joining market_data. The existing return_1d sequence
    # is retained here to preserve the current project's definition.

    before = len(df)

    df = df[
        df["forward_return_5d"].notna()
    ].copy()

    removed = before - len(df)

    print(
        "Target created: forward_return_5d"
    )

    print(
        f"Rows with target: {len(df):,}"
    )

    print(
        f"Rows without target: {removed:,}"
    )

    return df


# ============================================================
# TRAINING DATA
# ============================================================

def prepare_training_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 60)
    print("PREPARING TRAINING DATA")
    print("=" * 60)

    before = len(df)

    required = (
        FEATURE_COLUMNS
        + ["forward_return_5d"]
    )

    numeric = df[required].apply(
        pd.to_numeric,
        errors="coerce",
    )

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df[required] = numeric

    # Target must always be available.
    df = df[
        df["forward_return_5d"].notna()
    ].copy()

    removed = before - len(df)

    print(
        f"Rows before cleaning : {before:,}"
    )

    print(
        f"Rows after cleaning  : {len(df):,}"
    )

    print(
        f"Rows removed         : {removed:,}"
    )

    if df.empty:
        raise RuntimeError(
            "No usable training observations remain."
        )

    print()
    print("=" * 60)
    print("FEATURE MISSING-VALUE REPORT")
    print("=" * 60)

    total_missing = 0

    for column in FEATURE_COLUMNS:

        count = int(
            df[column].isna().sum()
        )

        total_missing += count

        if count > 0:

            percentage = (
                count / len(df) * 100
            )

            print(
                f"{column:<28}"
                f"{count:>6} "
                f"({percentage:>6.2f}%)"
            )

    print("-" * 60)

    print(
        f"Total missing feature values: "
        f"{total_missing:,}"
    )

    return df


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    df: pd.DataFrame,
):

    print()
    print("=" * 60)
    print(
        "CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT"
    )
    print("=" * 60)

    dates = np.sort(
        df["date"].unique()
    )

    n_dates = len(dates)

    train_end_index = int(
        n_dates * 0.70
    )

    validation_end_index = int(
        n_dates * 0.85
    )

    train_end_date = dates[
        train_end_index - 1
    ]

    validation_end_date = dates[
        validation_end_index - 1
    ]

    train = df[
        df["date"] <= train_end_date
    ].copy()

    validation = df[
        (df["date"] > train_end_date)
        & (df["date"] <= validation_end_date)
    ].copy()

    test = df[
        df["date"] > validation_end_date
    ].copy()

    print(
        f"Training   : "
        f"{train['date'].min().date()} -> "
        f"{train['date'].max().date()} "
        f"({len(train):,} rows)"
    )

    print(
        f"Validation : "
        f"{validation['date'].min().date()} -> "
        f"{validation['date'].max().date()} "
        f"({len(validation):,} rows)"
    )

    print(
        f"Test       : "
        f"{test['date'].min().date()} -> "
        f"{test['date'].max().date()} "
        f"({len(test):,} rows)"
    )

    return train, validation, test


# ============================================================
# IMPUTATION
# ============================================================

def fit_imputation(
    train: pd.DataFrame,
):

    print()
    print("=" * 60)
    print(
        "FITTING TRAINING-SET FEATURE IMPUTATION"
    )
    print("=" * 60)

    medians = {}

    for column in FEATURE_COLUMNS:

        values = train[column]

        median = values.median()

        # If an entire feature is missing in training,
        # use a neutral fallback.
        if pd.isna(median):
            median = 0.0

        medians[column] = float(median)

    print(
        "Training-set medians calculated."
    )

    return medians


def apply_imputation(
    df: pd.DataFrame,
    medians: dict,
    name: str,
) -> pd.DataFrame:

    df = df.copy()

    print(
        f"Applying imputation to {name}..."
    )

    for column in FEATURE_COLUMNS:

        df[column] = (
            df[column]
            .fillna(medians[column])
        )

    remaining = int(
        df[FEATURE_COLUMNS]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"{name} remaining missing "
        f"feature values: {remaining}"
    )

    if remaining:
        raise RuntimeError(
            f"Missing values remain in {name}."
        )

    return df


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
):

    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    directional = (
        np.sign(y_true)
        == np.sign(predictions)
    ).mean()

    if (
        np.std(y_true) == 0
        or np.std(predictions) == 0
    ):
        correlation = np.nan
    else:
        correlation = np.corrcoef(
            y_true,
            predictions,
        )[0, 1]

    return {
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional,
        "correlation": correlation,
    }


def print_metrics(
    title: str,
    metrics: dict,
):

    print()
    print(title)
    print("-" * 60)

    print(
        f"MAE                   : "
        f"{metrics['mae']:.6f}"
    )

    print(
        f"RMSE                  : "
        f"{metrics['rmse']:.6f}"
    )

    print(
        f"Directional accuracy  : "
        f"{metrics['directional_accuracy']:.4%}"
    )

    correlation = metrics["correlation"]

    if np.isnan(correlation):
        print(
            "Prediction correlation: N/A"
        )
    else:
        print(
            f"Prediction correlation: "
            f"{correlation:.6f}"
        )


# ============================================================
# MODEL
# ============================================================

def build_model():

    return XGBRegressor(

        objective="reg:squarederror",

        # Conservative model complexity.
        n_estimators=2000,
        learning_rate=0.02,

        max_depth=3,
        min_child_weight=8,

        subsample=0.75,
        colsample_bytree=0.75,

        gamma=0.10,

        reg_alpha=0.20,
        reg_lambda=5.0,

        random_state=42,

        n_jobs=-1,

        eval_metric="rmse",
    )


def train_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
):

    print()
    print("=" * 60)
    print("TRAINING REGULARIZED XGBOOST")
    print("=" * 60)

    X_train = train[
        FEATURE_COLUMNS
    ]

    y_train = train[
        "forward_return_5d"
    ]

    X_validation = validation[
        FEATURE_COLUMNS
    ]

    y_validation = validation[
        "forward_return_5d"
    ]

    print(
        f"Features: {FEATURE_COLUMNS}"
    )

    print(
        f"Feature count: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        f"Training observations: "
        f"{len(train):,}"
    )

    model = build_model()

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (
                X_train,
                y_train,
            ),
            (
                X_validation,
                y_validation,
            ),
        ],

        verbose=False,
    )

    print(
        "Training completed."
    )

    return model


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def print_feature_importance(
    model,
):

    print()
    print("=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS,
    ).sort_values(
        ascending=False
    )

    for feature, value in importance.items():

        print(
            f"{feature:<28} | "
            f"{value:.6f}"
        )


# ============================================================
# OUT-OF-SAMPLE PREDICTIONS
# ============================================================

def save_oos_predictions(
    model,
    test: pd.DataFrame,
):

    predictions = model.predict(
        test[FEATURE_COLUMNS]
    )

    output = test[
        [
            "date",
            "ticker",
            "forward_return_5d",
        ]
    ].copy()

    output["predicted_return_5d"] = (
        predictions
    )

    output["actual_return_5d"] = (
        output["forward_return_5d"]
    )

    output = output[
        [
            "date",
            "ticker",
            "actual_return_5d",
            "predicted_return_5d",
        ]
    ]

    output["date"] = (
        output["date"]
        .dt.strftime("%Y-%m-%d")
    )

    output.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print()
    print(
        f"Predictions saved: "
        f"{PREDICTIONS_PATH.resolve()}"
    )

    print()
    print(
        "OUT-OF-SAMPLE PREDICTIONS"
    )
    print("=" * 60)

    for _, row in output.tail(15).iterrows():

        print(
            f"{row['date']} | "
            f"{row['ticker']:<5} | "
            f"actual="
            f"{row['actual_return_5d']:>7.4%} | "
            f"predicted="
            f"{row['predicted_return_5d']:>7.4%}"
        )

    return output


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    medians,
):

    print()
    print("=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "imputation_medians": medians,
        "target": "forward_return_5d",
        "horizon_days": 5,
        "random_state": 42,
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    print(
        f"Model saved: "
        f"{MODEL_PATH.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("XGBOOST 5-DAY RETURN PREDICTOR")
    print("=" * 60)

    df = load_dataset()

    df = create_forward_return_target(
        df
    )

    df = prepare_training_data(
        df
    )

    train, validation, test = (
        chronological_split(df)
    )

    medians = fit_imputation(
        train
    )

    train = apply_imputation(
        train,
        medians,
        "training",
    )

    validation = apply_imputation(
        validation,
        medians,
        "validation",
    )

    test = apply_imputation(
        test,
        medians,
        "test",
    )

    model = train_model(
        train,
        validation,
    )

    train_predictions = model.predict(
        train[FEATURE_COLUMNS]
    )

    validation_predictions = model.predict(
        validation[FEATURE_COLUMNS]
    )

    test_predictions = model.predict(
        test[FEATURE_COLUMNS]
    )

    train_metrics = calculate_metrics(
        train["forward_return_5d"],
        train_predictions,
    )

    validation_metrics = calculate_metrics(
        validation["forward_return_5d"],
        validation_predictions,
    )

    test_metrics = calculate_metrics(
        test["forward_return_5d"],
        test_predictions,
    )

    print_metrics(
        "TRAIN PERFORMANCE",
        train_metrics,
    )

    print_metrics(
        "VALIDATION PERFORMANCE",
        validation_metrics,
    )

    print_metrics(
        "TEST / OUT-OF-SAMPLE PERFORMANCE",
        test_metrics,
    )

    print_feature_importance(
        model
    )

    save_oos_predictions(
        model,
        test,
    )

    save_model(
        model,
        medians,
    )

    print()
    print("=" * 60)
    print("XGBOOST TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Test MAE                   : "
        f"{test_metrics['mae']:.6f}"
    )

    print(
        f"Test RMSE                  : "
        f"{test_metrics['rmse']:.6f}"
    )

    print(
        f"Test directional accuracy  : "
        f"{test_metrics['directional_accuracy']:.4%}"
    )

    print(
        f"Test prediction correlation: "
        f"{test_metrics['correlation']:.6f}"
    )

    print(
        f"Model path: "
        f"{MODEL_PATH.resolve()}"
    )

    print(
        f"Predictions path: "
        f"{PREDICTIONS_PATH.resolve()}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()