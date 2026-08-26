"""
XGBoost Return Prediction Model

Purpose:
    Train an XGBoost regression model to predict the
    5-trading-day forward return of each asset.

Features:
    - return_1d
    - return_5d
    - rsi_14
    - macd
    - volatility_20d
    - regime
    - sentiment_score

Target:
    5-trading-day forward return

Important:
    - Chronological train/validation/test split
    - No random shuffling
    - No graph features yet
    - No future information used as model features
    - Test set remains completely out-of-sample
"""

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

PG_URL = "postgresql://quant_user:quant_password@localhost:5432/quant_db"

ASSETS = [
    "AAPL",
    "AMZN",
    "GOOGL",
    "MSFT",
    "SPY",
]

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "rsi_14",
    "macd",
    "volatility_20d",
    "regime",
    "sentiment_score",
]

TARGET_COLUMN = "forward_return_5d"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

MODEL_DIR = Path("ml_core/models")
MODEL_PATH = MODEL_DIR / "xgboost_return_model.joblib"

engine = create_engine(PG_URL)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load feature-store data and market prices.

    The feature store contains information known at date t.
    Market prices are used to construct the future 5-day
    return target.
    """

    print("=" * 60)
    print("LOADING ML DATASET")
    print("=" * 60)

    query = text(
        """
        SELECT
            fs.ticker,
            fs.date,
            fs.return_1d,
            fs.return_5d,
            fs.rsi_14,
            fs.macd,
            fs.volatility_20d,
            fs.regime,
            fs.sentiment_score,
            md.close
        FROM feature_store fs
        INNER JOIN market_data md
            ON fs.ticker = md.ticker
           AND fs.date = md.date
        WHERE fs.ticker IN (
            'AAPL',
            'AMZN',
            'GOOGL',
            'MSFT',
            'SPY'
        )
        AND md.close IS NOT NULL
        ORDER BY fs.date, fs.ticker
        """
    )

    df = pd.read_sql(query, engine)

    if df.empty:
        raise RuntimeError("No training data found.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"Rows loaded: {len(df)}")
    print(f"Assets: {sorted(df['ticker'].unique())}")
    print(
        f"Date range: "
        f"{df['date'].min().date()} -> "
        f"{df['date'].max().date()}"
    )

    return df


# ============================================================
# CREATE TARGET
# ============================================================

def create_forward_return_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the 5-trading-day forward return.

    For a row at date t:

        target =
            close(t+5) / close(t) - 1

    The shift is performed independently for each ticker.
    """

    print("\n")
    print("=" * 60)
    print("CREATING FORWARD RETURN TARGET")
    print("=" * 60)

    df = df.copy()

    df["future_close_5d"] = (
        df.groupby("ticker")["close"]
        .shift(-5)
    )

    df[TARGET_COLUMN] = (
        df["future_close_5d"] / df["close"] - 1.0
    )

    df = df.drop(columns=["future_close_5d"])

    print(
        f"Target created: {TARGET_COLUMN}"
    )

    print(
        f"Rows with target: "
        f"{df[TARGET_COLUMN].notna().sum()}"
    )

    print(
        f"Rows without target: "
        f"{df[TARGET_COLUMN].isna().sum()}"
    )

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where required features or target are unavailable.
    """

    print("\n")
    print("=" * 60)
    print("PREPARING TRAINING DATA")
    print("=" * 60)

    required_columns = [
        "ticker",
        "date",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]

    before = len(df)

    df = df.dropna(
        subset=required_columns
    ).copy()

    after = len(df)

    print(f"Rows before cleaning : {before}")
    print(f"Rows after cleaning  : {after}")
    print(f"Rows removed         : {before - after}")

    if len(df) < 100:
        raise RuntimeError(
            "Insufficient clean observations for model training."
        )

    return df


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    df: pd.DataFrame,
):
    """
    Split the dataset chronologically.

    No random shuffling is performed.

    The split is based on unique calendar dates so that
    observations from the same date remain in the same set.
    """

    print("\n")
    print("=" * 60)
    print("CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 60)

    dates = np.sort(
        df["date"].unique()
    )

    n_dates = len(dates)

    train_end = int(
        n_dates * TRAIN_RATIO
    )

    validation_end = int(
        n_dates *
        (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_end_date = dates[train_end - 1]
    validation_end_date = dates[validation_end - 1]

    train_mask = (
        df["date"] <= train_end_date
    )

    validation_mask = (
        (df["date"] > train_end_date)
        & (df["date"] <= validation_end_date)
    )

    test_mask = (
        df["date"] > validation_end_date
    )

    train_df = df.loc[train_mask].copy()
    validation_df = df.loc[validation_mask].copy()
    test_df = df.loc[test_mask].copy()

    print(
        f"Training   : {train_df['date'].min().date()} "
        f"-> {train_df['date'].max().date()} "
        f"({len(train_df)} rows)"
    )

    print(
        f"Validation : {validation_df['date'].min().date()} "
        f"-> {validation_df['date'].max().date()} "
        f"({len(validation_df)} rows)"
    )

    print(
        f"Test       : {test_df['date'].min().date()} "
        f"-> {test_df['date'].max().date()} "
        f"({len(test_df)} rows)"
    )

    if (
        train_df.empty
        or validation_df.empty
        or test_df.empty
    ):
        raise RuntimeError(
            "One of the chronological dataset splits is empty."
        )

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# MODEL
# ============================================================

def create_model() -> XGBRegressor:
    """
    Create the baseline XGBoost regression model.
    """

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(
    name: str,
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict:
    """
    Calculate regression and directional metrics.
    """

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

    actual_direction = (
        np.asarray(y_true) > 0
    )

    predicted_direction = (
        np.asarray(predictions) > 0
    )

    directional_accuracy = np.mean(
        actual_direction
        == predicted_direction
    )

    correlation = np.nan

    if (
        np.std(y_true) > 0
        and np.std(predictions) > 0
    ):
        correlation = np.corrcoef(
            np.asarray(y_true),
            np.asarray(predictions),
        )[0, 1]

    print("\n")
    print("=" * 60)
    print(f"{name} PERFORMANCE")
    print("=" * 60)

    print(
        f"MAE                 : {mae:.6f}"
    )

    print(
        f"RMSE                : {rmse:.6f}"
    )

    print(
        f"Directional accuracy: "
        f"{directional_accuracy:.4%}"
    )

    if np.isnan(correlation):
        print(
            "Prediction correlation: N/A"
        )
    else:
        print(
            f"Prediction correlation: "
            f"{correlation:.6f}"
        )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "directional_accuracy": float(
            directional_accuracy
        ),
        "correlation": (
            None
            if np.isnan(correlation)
            else float(correlation)
        ),
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def print_feature_importance(
    model: XGBRegressor,
) -> None:
    """
    Display XGBoost feature importance.
    """

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS,
    ).sort_values(
        ascending=False
    )

    print("\n")
    print("=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    for feature, value in importance.items():
        print(
            f"{feature:<20} | {value:.6f}"
        )


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

def print_sample_predictions(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
) -> None:
    """
    Print several out-of-sample predictions.
    """

    output = test_df[
        ["ticker", "date", TARGET_COLUMN]
    ].copy()

    output["predicted_return_5d"] = predictions

    output = output.sort_values(
        "date"
    )

    print("\n")
    print("=" * 60)
    print("OUT-OF-SAMPLE PREDICTIONS")
    print("=" * 60)

    for _, row in output.tail(15).iterrows():

        print(
            f"{row['date'].date()} | "
            f"{row['ticker']:<5} | "
            f"actual="
            f"{row[TARGET_COLUMN]: .4%} | "
            f"predicted="
            f"{row['predicted_return_5d']: .4%}"
        )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model: XGBRegressor,
) -> None:
    """
    Save trained model to disk.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print("\n")
    print("=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(
        f"Path: {MODEL_PATH}"
    )


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def train_model() -> None:

    print("\n")
    print("=" * 60)
    print("XGBOOST 5-DAY RETURN PREDICTOR")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # 2. Create future target
    # --------------------------------------------------------

    df = create_forward_return_target(
        df
    )

    # --------------------------------------------------------
    # 3. Clean data
    # --------------------------------------------------------

    df = prepare_training_data(
        df
    )

    # --------------------------------------------------------
    # 4. Chronological split
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = chronological_split(df)

    # --------------------------------------------------------
    # 5. Prepare matrices
    # --------------------------------------------------------

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        TARGET_COLUMN
    ]

    X_validation = validation_df[
        FEATURE_COLUMNS
    ]

    y_validation = validation_df[
        TARGET_COLUMN
    ]

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        TARGET_COLUMN
    ]

    # --------------------------------------------------------
    # 6. Create model
    # --------------------------------------------------------

    model = create_model()

    print("\n")
    print("=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)

    print(
        f"Features: {FEATURE_COLUMNS}"
    )

    print(
        f"Training observations: "
        f"{len(X_train)}"
    )

    # --------------------------------------------------------
    # 7. Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        verbose=False,
    )

    print(
        "Training completed."
    )

    # --------------------------------------------------------
    # 8. Predictions
    # --------------------------------------------------------

    train_predictions = (
        model.predict(X_train)
    )

    validation_predictions = (
        model.predict(X_validation)
    )

    test_predictions = (
        model.predict(X_test)
    )

    # --------------------------------------------------------
    # 9. Evaluate
    # --------------------------------------------------------

    train_metrics = evaluate_predictions(
        "TRAIN",
        y_train,
        train_predictions,
    )

    validation_metrics = evaluate_predictions(
        "VALIDATION",
        y_validation,
        validation_predictions,
    )

    test_metrics = evaluate_predictions(
        "TEST / OUT-OF-SAMPLE",
        y_test,
        test_predictions,
    )

    # --------------------------------------------------------
    # 10. Feature importance
    # --------------------------------------------------------

    print_feature_importance(
        model
    )

    # --------------------------------------------------------
    # 11. Sample predictions
    # --------------------------------------------------------

    print_sample_predictions(
        test_df,
        test_predictions,
    )

    # --------------------------------------------------------
    # 12. Save model
    # --------------------------------------------------------

    save_model(
        model
    )

    # --------------------------------------------------------
    # 13. Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("XGBOOST TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Test MAE                 : "
        f"{test_metrics['mae']:.6f}"
    )

    print(
        f"Test RMSE                : "
        f"{test_metrics['rmse']:.6f}"
    )

    print(
        f"Test directional accuracy: "
        f"{test_metrics['directional_accuracy']:.4%}"
    )

    correlation = test_metrics[
        "correlation"
    ]

    if correlation is not None:
        print(
            f"Test prediction correlation: "
            f"{correlation:.6f}"
        )

    print(
        f"Model path: {MODEL_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    train_model()