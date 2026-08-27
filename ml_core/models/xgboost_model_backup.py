"""
XGBOOST 5-DAY FORWARD RETURN PREDICTOR

Expanded point-in-time feature set with leakage-safe median imputation.

Pipeline:
    1. Load feature_store data
    2. Create 5-day forward-return target
    3. Keep rows with a valid target
    4. Chronological train/validation/test split
    5. Fit feature medians on TRAINING DATA ONLY
    6. Apply those medians to validation/test
    7. Train XGBoost
    8. Evaluate train/validation/test
    9. Report feature importance
   10. Save model + preprocessing metadata
   11. Save out-of-sample predictions

Important:
    No random split is used.

    No future observations are used to calculate the target.

    Imputation statistics are learned from the training set only.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

PG_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://quant_user:quant_password@localhost:5432/quant_db",
)

engine = create_engine(PG_URL)


MARKET_TICKERS = [
    "AAPL",
    "AMZN",
    "GOOGL",
    "MSFT",
    "SPY",
]


# ============================================================
# FEATURE SET
# ============================================================

FEATURE_COLUMNS = [
    # Momentum
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",

    # Technical indicators
    "rsi_14",
    "macd",

    # Volatility
    "volatility_5d",
    "volatility_20d",
    "volatility_60d",
    "atr_14_pct",

    # Trend
    "sma_20_distance",
    "sma_50_distance",
    "sma_200_distance",

    # Drawdown
    "drawdown_20d",
    "drawdown_60d",

    # Volume
    "volume_change_5d",
    "volume_zscore_20d",

    # Market-relative
    "market_return_5d",
    "relative_return_5d",
    "beta_60d",
    "correlation_spy_60d",

    # Price action
    "intraday_range",
    "close_position",
    "gap_return",

    # Regime / NLP / graph
    "regime",
    "sentiment_score",
    "degree_centrality",
    "pagerank",
]


TARGET_COLUMN = "forward_return_5d"


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "xgboost_return_model.joblib"
)

PREDICTIONS_PATH = (
    Path(__file__).resolve().parent
    / "xgboost_oos_predictions.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset() -> pd.DataFrame:

    print("=" * 60)
    print("LOADING ML DATASET")
    print("=" * 60)

    feature_sql = ",\n            ".join(
        f"fs.{column}"
        for column in FEATURE_COLUMNS
    )

    query = text(
        f"""
        SELECT
            fs.ticker,
            fs.date,
            {feature_sql}
        FROM feature_store fs
        WHERE fs.ticker = ANY(:tickers)
        ORDER BY fs.date, fs.ticker
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={
                "tickers": MARKET_TICKERS,
            },
        )

    if df.empty:
        raise RuntimeError(
            "No rows found in feature_store."
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        ["ticker", "date"]
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
# CREATE FORWARD TARGET
# ============================================================

def create_forward_target(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 60)
    print("CREATING FORWARD RETURN TARGET")
    print("=" * 60)

    df = df.copy()

    # return_5d at t+5 represents:
    #
    # close[t+5] / close[t] - 1
    #
    # Therefore shifting return_5d backward by 5 rows
    # gives the 5-day forward return from date t.

    df[TARGET_COLUMN] = (
        df.groupby("ticker")["return_5d"]
        .shift(-5)
    )

    target_count = (
        df[TARGET_COLUMN]
        .notna()
        .sum()
    )

    missing_count = (
        df[TARGET_COLUMN]
        .isna()
        .sum()
    )

    print(
        f"Target created: {TARGET_COLUMN}"
    )

    print(
        f"Rows with target: {target_count:,}"
    )

    print(
        f"Rows without target: {missing_count:,}"
    )

    return df


# ============================================================
# FEATURE QUALITY REPORT
# ============================================================

def print_missing_feature_report(
    df: pd.DataFrame,
) -> None:

    print()
    print("=" * 60)
    print("FEATURE MISSING-VALUE REPORT")
    print("=" * 60)

    missing = df[
        FEATURE_COLUMNS
    ].isna().sum()

    total = len(df)

    for feature in FEATURE_COLUMNS:

        count = int(
            missing[feature]
        )

        if count > 0:

            percentage = (
                count / total * 100
            )

            print(
                f"{feature:<25} "
                f"{count:>6} "
                f"({percentage:>6.2f}%)"
            )

    total_missing = int(
        missing.sum()
    )

    print("-" * 60)

    print(
        f"Total missing feature values: "
        f"{total_missing:,}"
    )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_training_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 60)
    print("PREPARING TRAINING DATA")
    print("=" * 60)

    before = len(df)

    # Replace infinities first.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Only the target is mandatory at this stage.
    #
    # Feature NaNs are intentionally preserved and will be
    # imputed using training-set medians after the chronological
    # split.

    df = df.dropna(
        subset=[TARGET_COLUMN]
    ).copy()

    df = df.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    after = len(df)

    print(
        f"Rows before cleaning : {before:,}"
    )

    print(
        f"Rows after cleaning  : {after:,}"
    )

    print(
        f"Rows removed         : "
        f"{before - after:,}"
    )

    if after == 0:
        raise RuntimeError(
            "No rows with a valid target remain."
        )

    print_missing_feature_report(
        df
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
    print("CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 60)

    unique_dates = sorted(
        df["date"].unique()
    )

    n_dates = len(unique_dates)

    train_end_index = int(
        n_dates * 0.70
    )

    validation_end_index = int(
        n_dates * 0.85
    )

    train_end_date = (
        unique_dates[train_end_index - 1]
    )

    validation_end_date = (
        unique_dates[validation_end_index - 1]
    )

    train = df[
        df["date"] <= train_end_date
    ].copy()

    validation = df[
        (df["date"] > train_end_date)
        & (
            df["date"]
            <= validation_end_date
        )
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

    if train.empty:
        raise RuntimeError(
            "Training split is empty."
        )

    if validation.empty:
        raise RuntimeError(
            "Validation split is empty."
        )

    if test.empty:
        raise RuntimeError(
            "Test split is empty."
        )

    return (
        train,
        validation,
        test,
    )


# ============================================================
# TRAINING-ONLY MEDIAN IMPUTATION
# ============================================================

def fit_imputation(
    train: pd.DataFrame,
):
    print()
    print("=" * 60)
    print("FITTING TRAINING-SET FEATURE IMPUTATION")
    print("=" * 60)

    medians = {}

    for feature in FEATURE_COLUMNS:

        median = train[
            feature
        ].median()

        # Extremely defensive fallback.
        # This should only happen if a feature is completely
        # unavailable in the training period.
        if pd.isna(median):
            median = 0.0

        medians[feature] = float(
            median
        )

    print(
        "Training-set medians calculated."
    )

    return medians


def apply_imputation(
    df: pd.DataFrame,
    medians: dict,
    dataset_name: str,
) -> pd.DataFrame:

    print(
        f"Applying imputation to "
        f"{dataset_name}..."
    )

    df = df.copy()

    for feature in FEATURE_COLUMNS:

        df[feature] = (
            df[feature]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(
                medians[feature]
            )
        )

    remaining_missing = int(
        df[
            FEATURE_COLUMNS
        ].isna().sum().sum()
    )

    print(
        f"{dataset_name} remaining missing "
        f"feature values: "
        f"{remaining_missing:,}"
    )

    if remaining_missing:
        raise RuntimeError(
            f"Missing feature values remain "
            f"after imputation in {dataset_name}."
        )

    return df


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    directional_accuracy = (
        np.sign(y_true)
        == np.sign(y_pred)
    ).mean()

    if (
        np.std(y_true) == 0
        or np.std(y_pred) == 0
    ):
        correlation = np.nan
    else:
        correlation = np.corrcoef(
            y_true,
            y_pred,
        )[0, 1]

    return (
        mae,
        rmse,
        directional_accuracy,
        correlation,
    )


def print_metrics(
    name: str,
    y_true,
    y_pred,
):

    (
        mae,
        rmse,
        directional_accuracy,
        correlation,
    ) = calculate_metrics(
        y_true,
        y_pred,
    )

    print()
    print(
        f"{name.upper()} PERFORMANCE"
    )
    print("-" * 60)

    print(
        f"MAE                   : "
        f"{mae:.6f}"
    )

    print(
        f"RMSE                  : "
        f"{rmse:.6f}"
    )

    print(
        f"Directional accuracy  : "
        f"{directional_accuracy:.4%}"
    )

    print(
        f"Prediction correlation: "
        f"{correlation:.6f}"
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy":
            directional_accuracy,
        "correlation":
            correlation,
    }


# ============================================================
# TRAIN XGBOOST
# ============================================================

def train_model(
    train: pd.DataFrame,
):

    print()
    print("=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)

    X_train = train[
        FEATURE_COLUMNS
    ]

    y_train = train[
        TARGET_COLUMN
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

    model = XGBRegressor(
        objective="reg:squarederror",

        n_estimators=600,

        learning_rate=0.03,

        max_depth=4,

        min_child_weight=8,

        subsample=0.80,

        colsample_bytree=0.80,

        reg_alpha=0.10,

        reg_lambda=2.0,

        gamma=0.0,

        random_state=42,

        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training completed."
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
):

    results = {}

    for name, dataset in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:

        X = dataset[
            FEATURE_COLUMNS
        ]

        y = dataset[
            TARGET_COLUMN
        ]

        predictions = model.predict(
            X
        )

        results[name] = {
            "actual":
                y.to_numpy(),
            "predicted":
                predictions,
        }

        print_metrics(
            name,
            y,
            predictions,
        )

    return results


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
    )

    importance = importance.sort_values(
        ascending=False
    )

    for feature, value in importance.items():

        print(
            f"{feature:<25} | "
            f"{value:.6f}"
        )


# ============================================================
# OUT-OF-SAMPLE PREDICTIONS
# ============================================================

def save_out_of_sample_predictions(
    model,
    test: pd.DataFrame,
):

    print()
    print("=" * 60)
    print("OUT-OF-SAMPLE PREDICTIONS")
    print("=" * 60)

    output = test[
        [
            "date",
            "ticker",
            TARGET_COLUMN,
        ]
    ].copy()

    output[
        "predicted_return_5d"
    ] = model.predict(
        test[
            FEATURE_COLUMNS
        ]
    )

    output = output.rename(
        columns={
            TARGET_COLUMN:
                "actual_return_5d"
        }
    )

    output = output.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    # Show the latest 15 observations.
    latest = output.tail(15)

    for _, row in latest.iterrows():

        print(
            f"{row['date'].date()} | "
            f"{row['ticker']:<5} | "
            f"actual="
            f"{row['actual_return_5d']: .4%} | "
            f"predicted="
            f"{row['predicted_return_5d']: .4%}"
        )

    output.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print()
    print(
        f"Predictions saved: "
        f"{PREDICTIONS_PATH}"
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

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_package = {
        "model": model,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "tickers": MARKET_TICKERS,
        "imputation_medians": medians,
    }

    joblib.dump(
        model_package,
        MODEL_PATH,
    )

    print(
        f"Model saved: "
        f"{MODEL_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("XGBOOST 5-DAY RETURN PREDICTOR")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load feature store
    # --------------------------------------------------------
    df = load_dataset()

    # --------------------------------------------------------
    # 2. Create forward target
    # --------------------------------------------------------
    df = create_forward_target(
        df
    )

    # --------------------------------------------------------
    # 3. Remove only rows without target
    # --------------------------------------------------------
    df = prepare_training_data(
        df
    )

    # --------------------------------------------------------
    # 4. Chronological split
    # --------------------------------------------------------
    (
        train,
        validation,
        test,
    ) = chronological_split(
        df
    )

    # --------------------------------------------------------
    # 5. Fit imputation ONLY on training data
    # --------------------------------------------------------
    medians = fit_imputation(
        train
    )

    # --------------------------------------------------------
    # 6. Apply same training medians everywhere
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 7. Train
    # --------------------------------------------------------
    model = train_model(
        train
    )

    # --------------------------------------------------------
    # 8. Evaluate
    # --------------------------------------------------------
    results = evaluate_model(
        model,
        train,
        validation,
        test,
    )

    # --------------------------------------------------------
    # 9. Feature importance
    # --------------------------------------------------------
    print_feature_importance(
        model
    )

    # --------------------------------------------------------
    # 10. Save OOS predictions
    # --------------------------------------------------------
    save_out_of_sample_predictions(
        model,
        test,
    )

    # --------------------------------------------------------
    # 11. Save model + preprocessing
    # --------------------------------------------------------
    save_model(
        model,
        medians,
    )

    # --------------------------------------------------------
    # 12. Final summary
    # --------------------------------------------------------
    test_metrics = calculate_metrics(
        results["test"]["actual"],
        results["test"]["predicted"],
    )

    print()
    print("=" * 60)
    print("XGBOOST TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Test MAE                   : "
        f"{test_metrics[0]:.6f}"
    )

    print(
        f"Test RMSE                  : "
        f"{test_metrics[1]:.6f}"
    )

    print(
        f"Test directional accuracy  : "
        f"{test_metrics[2]:.4%}"
    )

    print(
        f"Test prediction correlation: "
        f"{test_metrics[3]:.6f}"
    )

    print(
        f"Model path: "
        f"{MODEL_PATH}"
    )

    print(
        f"Predictions path: "
        f"{PREDICTIONS_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()