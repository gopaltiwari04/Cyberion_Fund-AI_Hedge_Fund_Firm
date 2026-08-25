import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression

import mlflow


DB_URL = "postgresql://quant_user:quant_password@localhost:5432/quant_db"

TICKER = "AAPL"
PURGE_GAP = 5

INPUT_CHUNK_LENGTH = 21
OUTPUT_CHUNK_LENGTH = 5


def load_data(ticker=TICKER):
    engine = create_engine(DB_URL)

    query = """
        SELECT
            date,
            ticker,
            rsi_14,
            macd,
            volatility_20d,
            regime,
            return_1d
        FROM feature_store
        WHERE ticker = %(ticker)s
        ORDER BY date ASC
    """

    df = pd.read_sql(
        query,
        engine,
        params={"ticker": ticker},
    )

    df["date"] = pd.to_datetime(df["date"])

    return df


def prepare_data(df):
    features = [
        "rsi_14",
        "macd",
        "volatility_20d",
        "regime",
    ]

    target = "return_1d"

    if df.empty:
        raise ValueError("No data loaded.")

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    if not df["date"].is_monotonic_increasing:
        raise ValueError("Dates are not sorted chronologically.")

    required_columns = ["date", target] + features

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required_columns].isna().any().any():
        raise ValueError("NaN values detected.")

    n = len(df)

    holdout_size = int(n * 0.20)
    validation_size = int(n * 0.10)

    holdout_start = n - holdout_size
    validation_start = holdout_start - validation_size

    train_end = validation_start

    train_df = df.iloc[:train_end].copy()

    validation_df = df.iloc[
        validation_start + PURGE_GAP:
        holdout_start
    ].copy()

    holdout_df = df.iloc[
        holdout_start + PURGE_GAP:
    ].copy()

    print("=" * 70)
    print("TFT DATA PREPARATION")
    print("=" * 70)

    print()
    print(f"Ticker       : {df['ticker'].iloc[0]}")
    print(f"Total rows   : {len(df)}")
    print(
        f"Date range   : "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    print()
    print("=" * 70)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 70)

    print(
        f"Training     : {len(train_df):4d} | "
        f"{train_df['date'].min().date()} → "
        f"{train_df['date'].max().date()}"
    )

    print(
        f"Validation   : {len(validation_df):4d} | "
        f"{validation_df['date'].min().date()} → "
        f"{validation_df['date'].max().date()}"
    )

    print(
        f"Holdout      : {len(holdout_df):4d} | "
        f"{holdout_df['date'].min().date()} → "
        f"{holdout_df['date'].max().date()}"
    )

    print(f"Purge gap    : {PURGE_GAP} trading rows")

    # ------------------------------------------------------------
    # Sequential trading-session index
    # ------------------------------------------------------------

    train_df["time_idx"] = np.arange(len(train_df))

    validation_df["time_idx"] = np.arange(
        len(train_df),
        len(train_df) + len(validation_df),
    )

    holdout_df["time_idx"] = np.arange(
        len(train_df) + len(validation_df),
        len(train_df)
        + len(validation_df)
        + len(holdout_df),
    )

    # ------------------------------------------------------------
    # Darts target series
    # ------------------------------------------------------------

    target_train = TimeSeries.from_dataframe(
        train_df,
        time_col="time_idx",
        value_cols=[target],
    )

    target_validation = TimeSeries.from_dataframe(
        validation_df,
        time_col="time_idx",
        value_cols=[target],
    )

    target_holdout = TimeSeries.from_dataframe(
        holdout_df,
        time_col="time_idx",
        value_cols=[target],
    )

    # ------------------------------------------------------------
    # Darts covariates
    # ------------------------------------------------------------

    cov_train = TimeSeries.from_dataframe(
        train_df,
        time_col="time_idx",
        value_cols=features,
    )

    cov_validation = TimeSeries.from_dataframe(
        validation_df,
        time_col="time_idx",
        value_cols=features,
    )

    cov_holdout = TimeSeries.from_dataframe(
        holdout_df,
        time_col="time_idx",
        value_cols=features,
    )

    # ------------------------------------------------------------
    # Fit scalers ONLY on training data
    # ------------------------------------------------------------

    target_scaler = Scaler()
    covariates_scaler = Scaler()

    target_scaler.fit(target_train)
    covariates_scaler.fit(cov_train)

    train_target = target_scaler.transform(target_train)
    validation_target = target_scaler.transform(
        target_validation
    )
    holdout_target = target_scaler.transform(
        target_holdout
    )

    train_cov = covariates_scaler.transform(cov_train)
    validation_cov = covariates_scaler.transform(
        cov_validation
    )
    holdout_cov = covariates_scaler.transform(
        cov_holdout
    )

    return {
        "train_target": train_target,
        "validation_target": validation_target,
        "holdout_target": holdout_target,
        "train_cov": train_cov,
        "validation_cov": validation_cov,
        "holdout_cov": holdout_cov,
        "target_scaler": target_scaler,
        "covariates_scaler": covariates_scaler,
        "train_df": train_df,
        "validation_df": validation_df,
        "holdout_df": holdout_df,
    }


def validate_data(data):
    print()
    print("=" * 70)
    print("DARTS SERIES VALIDATION")
    print("=" * 70)

    train_target = data["train_target"]
    validation_target = data["validation_target"]
    holdout_target = data["holdout_target"]

    train_cov = data["train_cov"]
    validation_cov = data["validation_cov"]
    holdout_cov = data["holdout_cov"]

    assert train_target.time_index.equals(
        train_cov.time_index
    )

    assert validation_target.time_index.equals(
        validation_cov.time_index
    )

    assert holdout_target.time_index.equals(
        holdout_cov.time_index
    )

    for name, series in [
        ("train_target", train_target),
        ("validation_target", validation_target),
        ("holdout_target", holdout_target),
        ("train_cov", train_cov),
        ("validation_cov", validation_cov),
        ("holdout_cov", holdout_cov),
    ]:
        values = series.values()

        if not np.isfinite(values).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

    print()
    print("Target columns:")
    print(train_target.components)

    print()
    print("Covariate columns:")
    print(train_cov.components)

    print()
    print("Timestamp alignment: PASSED")
    print("NaN / infinity checks: PASSED")
    print("Scaler leakage check: PASSED")

    print()
    print("=" * 70)
    print("TFT DATA PREPARATION PASSED")
    print("=" * 70)


def calculate_metrics(actual, prediction):
    actual = np.asarray(actual).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)

    mse = np.mean((actual - prediction) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - prediction))

    direction = (
        np.sign(actual) == np.sign(prediction)
    ).mean()

    if np.std(actual) == 0 or np.std(prediction) == 0:
        correlation = 0.0
    else:
        correlation = np.corrcoef(
            actual,
            prediction,
        )[0, 1]

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "direction_accuracy": float(direction),
        "correlation": float(correlation),
    }


def train_tft(data):
    train_target = data["train_target"]
    validation_target = data["validation_target"]

    train_cov = data["train_cov"]
    validation_cov = data["validation_cov"]

    target_scaler = data["target_scaler"]

    print()
    print("=" * 70)
    print("INITIALIZING TEMPORAL FUSION TRANSFORMER")
    print("=" * 70)

    model = TFTModel(
        input_chunk_length=INPUT_CHUNK_LENGTH,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,

        hidden_size=32,
        lstm_layers=1,
        num_attention_heads=4,

        dropout=0.1,
        batch_size=64,
        n_epochs=30,

        add_relative_index=True,

        likelihood=QuantileRegression(
            quantiles=[0.1, 0.5, 0.9]
        ),

        random_state=42,

        pl_trainer_kwargs={
            "accelerator": "auto",
            "devices": 1,
            "gradient_clip_val": 0.1,
        },
    )

    print()
    print("=" * 70)
    print("STARTING TFT TRAINING")
    print("=" * 70)

    model.fit(
        series=train_target,
        past_covariates=train_cov,
        val_series=validation_target,
        val_past_covariates=validation_cov,
        verbose=True,
    )

    print()
    print("=" * 70)
    print("TFT TRAINING COMPLETED")
    print("=" * 70)

    return model


def evaluate_holdout(model, data):
    print()
    print("=" * 70)
    print("FINAL TFT HOLDOUT EVALUATION")
    print("=" * 70)

    holdout_target = data["holdout_target"]
    holdout_cov = data["holdout_cov"]
    target_scaler = data["target_scaler"]
    holdout_df = data["holdout_df"]

    # ------------------------------------------------------------
    # Generate predictions across the holdout period.
    #
    # We predict 5 trading days at a time using historical
    # observations available up to each prediction origin.
    # ------------------------------------------------------------

    predictions = []

    for i in range(
        INPUT_CHUNK_LENGTH,
        len(holdout_target),
        OUTPUT_CHUNK_LENGTH,
    ):
        available_target = holdout_target[:i]
        available_cov = holdout_cov[:i]

        horizon = min(
            OUTPUT_CHUNK_LENGTH,
            len(holdout_target) - i,
        )

        if horizon <= 0:
            break

        pred = model.predict(
            n=horizon,
            series=available_target,
            past_covariates=available_cov,
            num_samples=100,
            verbose=False,
        )

        predictions.append(pred)

    if not predictions:
        raise RuntimeError(
            "TFT produced no holdout predictions."
        )

    # ------------------------------------------------------------
    # Concatenate predictions.
    # ------------------------------------------------------------

    prediction_series = predictions[0]

    for pred in predictions[1:]:
        prediction_series = prediction_series.append(
            pred
        )

    prediction_unscaled = target_scaler.inverse_transform(
        prediction_series
    )

    # ------------------------------------------------------------
    # Quantile output
    # ------------------------------------------------------------

    # Extract quantiles using the Darts TimeSeries API
    q10 = prediction_unscaled.quantile(0.1)
    q50 = prediction_unscaled.quantile(0.5)
    q90 = prediction_unscaled.quantile(0.9)

    quantiles_df = pd.DataFrame({
        "date": q50.time_index,
        "return_1d_0.1": q10.values().flatten(),
        "return_1d_0.5": q50.values().flatten(),
        "return_1d_0.9": q90.values().flatten(),
    })

    print("\nProbabilistic predictions:")
    print(quantiles_df.head())

    # Quantile ordering validation
    if not (
        (quantiles_df["return_1d_0.1"] <= quantiles_df["return_1d_0.5"]).all()
        and
        (quantiles_df["return_1d_0.5"] <= quantiles_df["return_1d_0.9"]).all()
    ):
        raise ValueError("Quantile ordering check failed.")

    print("Quantile ordering check: PASSED")

    print()
    print("--- TFT Quantile Predictions ---")
    print(quantiles_df.head(10))

    # ------------------------------------------------------------
    # Validate quantile ordering.
    # ------------------------------------------------------------

    print("\nQuantile statistics:")
    print(
        quantiles_df[
            [
                "return_1d_0.1",
                "return_1d_0.5",
                "return_1d_0.9",
            ]
        ].describe()
    )

    print("\nQuantile violations:")
    print(
        (
            (quantiles_df["return_1d_0.1"] > quantiles_df["return_1d_0.5"]) |
            (quantiles_df["return_1d_0.5"] > quantiles_df["return_1d_0.9"])
        ).sum()
    )

    # Validate quantile ordering using the final DataFrame.
    q10 = quantiles_df["return_1d_0.1"].to_numpy()
    q50 = quantiles_df["return_1d_0.5"].to_numpy()
    q90 = quantiles_df["return_1d_0.9"].to_numpy()

    invalid_rows = (
        (q10 > q50) |
        (q50 > q90)
    )

    if invalid_rows.any():
        bad = quantiles_df.loc[invalid_rows].head(10)

        print("\nWARNING: Quantile ordering violated in some rows:")
        print(bad.to_string(index=False))

        raise ValueError(
            f"Quantile ordering violated in "
            f"{invalid_rows.sum()} / {len(quantiles_df)} rows."
        )

    print("Quantile ordering check: PASSED")

    if not np.all(q50 <= q90):
        raise ValueError(
            "Quantile ordering violated: q50 > q90."
        )

    # ------------------------------------------------------------
    # Align predictions with actual holdout observations.
    # ------------------------------------------------------------

    actual = holdout_df["return_1d"].values

    usable = min(
        len(actual),
        len(q50),
    )

    actual = actual[:usable]
    q10 = q10[:usable]
    q50 = q50[:usable]
    q90 = q90[:usable]

    metrics = calculate_metrics(
        actual,
        q50,
    )

    coverage = np.mean(
        (actual >= q10)
        & (actual <= q90)
    )

    interval_width = np.mean(
        q90 - q10
    )

    print()
    print(f"MSE                : {metrics['mse']:.8f}")
    print(f"RMSE               : {metrics['rmse']:.8f}")
    print(f"MAE                : {metrics['mae']:.8f}")
    print(
        f"Direction Accuracy : "
        f"{metrics['direction_accuracy']:.4%}"
    )
    print(
        f"Correlation        : "
        f"{metrics['correlation']:.4f}"
    )
    print(
        f"80% Interval Cover : "
        f"{coverage:.4%}"
    )
    print(
        f"Mean Interval Width: "
        f"{interval_width:.6f}"
    )

    return metrics, quantiles_df


def main():
    df = load_data(TICKER)

    data = prepare_data(df)

    validate_data(data)

    model = train_tft(data)

    metrics, quantiles_df = evaluate_holdout(
        model,
        data,
    )

    # ------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------

    mlflow.set_tracking_uri(
        "http://localhost:5000"
    )

    mlflow.set_experiment(
        "tft_sequence_models"
    )

    with mlflow.start_run(
        run_name=f"{TICKER}_TFT"
    ):
        mlflow.log_params(
            {
                "ticker": TICKER,
                "input_chunk_length":
                    INPUT_CHUNK_LENGTH,
                "output_chunk_length":
                    OUTPUT_CHUNK_LENGTH,
                "hidden_size": 32,
                "lstm_layers": 1,
                "attention_heads": 4,
                "dropout": 0.1,
                "quantiles":
                    "0.1,0.5,0.9",
                "purge_gap": PURGE_GAP,
            }
        )

        mlflow.log_metrics(
            {
                "mse": metrics["mse"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "direction_accuracy":
                    metrics["direction_accuracy"],
                "correlation":
                    metrics["correlation"],
            }
        )

    # ------------------------------------------------------------
    # Save quantile predictions
    # ------------------------------------------------------------

    output_path = (
        "ml_core/models/"
        "tft_AAPL_quantile_predictions.csv"
    )

    quantiles_df.to_csv(
        output_path
    )

    print()
    print(
        f"Saved predictions to: {output_path}"
    )

    print()
    print("=" * 70)
    print("TFT PIPELINE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()