import os
import random

import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from sequence_dataset import TimeSeriesDataset
from lstm_model import QuantLSTM


# ============================================================
# CONFIG
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

SEQUENCE_LENGTH = 21
BATCH_SIZE = 64

PURGE_GAP = 5

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.10

MAX_EPOCHS = 50
LEARNING_RATE = 1e-3

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA LOADING
# ============================================================

def load_data():

    engine = create_engine(DB_URL)

    query = """
        SELECT
            date,
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
        params={"ticker": TICKER},
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # target = return over the next 5 trading days
    #
    # This is:
    #
    # Close[t+5] / Close[t] - 1
    #
    # We need the price series to construct it.
    # --------------------------------------------------------

    price_query = """
        SELECT
            date,
            close
        FROM market_data
        WHERE ticker = %(ticker)s
        ORDER BY date ASC
    """

    prices = pd.read_sql(
        price_query,
        engine,
        params={"ticker": TICKER},
    )

    prices["date"] = pd.to_datetime(prices["date"])
    df["date"] = pd.to_datetime(df["date"])

    df = df.merge(
        prices,
        on="date",
        how="inner",
    )

    df["target_5d"] = (
        df["close"].shift(-5) / df["close"] - 1.0
    )

    df = df.dropna(
        subset=FEATURES + ["target_5d"]
    ).reset_index(drop=True)

    return df


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def train_lstm_pipeline():

    set_seed()

    print("=" * 70)
    print("LSTM SEQUENCE MODEL")
    print("=" * 70)

    df = load_data()

    print()
    print(f"Ticker       : {TICKER}")
    print(f"Rows         : {len(df)}")
    print(
        f"Date range   : "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    n = len(df)

    train_end = int(n * TRAIN_RATIO)

    validation_end = int(
        n * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_df = df.iloc[:train_end]

    validation_start = train_end + PURGE_GAP

    validation_df = df.iloc[
        validation_start:validation_end
    ]

    holdout_start = validation_end + PURGE_GAP

    holdout_df = df.iloc[holdout_start:]

    print()
    print("=" * 70)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 70)

    print(
        f"Train      : {len(train_df):4d} | "
        f"{train_df['date'].min().date()} → "
        f"{train_df['date'].max().date()}"
    )

    print(
        f"Purged gap : {PURGE_GAP} trading days"
    )

    print(
        f"Validation : {len(validation_df):4d} | "
        f"{validation_df['date'].min().date()} → "
        f"{validation_df['date'].max().date()}"
    )

    print(
        f"Holdout    : {len(holdout_df):4d} | "
        f"{holdout_df['date'].min().date()} → "
        f"{holdout_df['date'].max().date()}"
    )

    # ========================================================
    # SCALE FEATURES
    # ========================================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train_df[FEATURES]
    )

    X_validation = scaler.transform(
        validation_df[FEATURES]
    )

    X_holdout = scaler.transform(
        holdout_df[FEATURES]
    )

    y_train = train_df["target_5d"].values.astype(
        np.float32
    )

    y_validation = validation_df["target_5d"].values.astype(
        np.float32
    )

    y_holdout = holdout_df["target_5d"].values.astype(
        np.float32
    )

    # ========================================================
    # SEQUENCE DATASETS
    # ========================================================

    train_dataset = TimeSeriesDataset(
        X_train,
        y_train,
        sequence_length=SEQUENCE_LENGTH,
    )

    validation_dataset = TimeSeriesDataset(
        X_validation,
        y_validation,
        sequence_length=SEQUENCE_LENGTH,
    )

    holdout_dataset = TimeSeriesDataset(
        X_holdout,
        y_holdout,
        sequence_length=SEQUENCE_LENGTH,
    )

    print()
    print("=" * 70)
    print("SEQUENCE DATASETS")
    print("=" * 70)

    print(
        f"Sequence length : {SEQUENCE_LENGTH}"
    )

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Validation samp.: {len(validation_dataset)}"
    )

    print(
        f"Holdout samples : {len(holdout_dataset)}"
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = QuantLSTM(
        input_size=len(FEATURES),
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=LEARNING_RATE,
    )

    # ========================================================
    # MLFLOW
    # ========================================================

    mlflow_logger = MLFlowLogger(
        experiment_name="lstm_sequence_models",
        tracking_uri="http://localhost:5000",
        run_name=f"{TICKER}_LSTM",
    )

    # ========================================================
    # CALLBACKS
    # ========================================================

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-lstm-{epoch:02d}-{val_loss:.6f}",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
    )

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=mlflow_logger,
        callbacks=[
            checkpoint_callback,
            early_stopping,
        ],
        gradient_clip_val=1.0,
        accelerator="auto",
        devices=1,
        deterministic=True,
    )

    print()
    print("=" * 70)
    print("STARTING LSTM TRAINING")
    print("=" * 70)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_model_path = checkpoint_callback.best_model_path

    print()
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"Checkpoint: {best_model_path}"
    )

    print(
        f"Best val loss: "
        f"{checkpoint_callback.best_model_score:.8f}"
    )

    # ========================================================
    # FINAL HOLDOUT
    # ========================================================

    best_model = QuantLSTM.load_from_checkpoint(
        best_model_path
    )

    best_model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():

        for X_batch, y_batch in holdout_loader:

            preds = best_model(X_batch)

            predictions.extend(
                preds.cpu().numpy()
            )

            actuals.extend(
                y_batch.cpu().numpy()
            )

    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)

    mse = np.mean(
        (actuals - predictions) ** 2
    )

    rmse = np.sqrt(mse)

    mae = np.mean(
        np.abs(actuals - predictions)
    )

    direction_accuracy = np.mean(
        np.sign(actuals) == np.sign(predictions)
    )

    correlation = np.corrcoef(
        actuals,
        predictions,
    )[0, 1]

    print()
    print("=" * 70)
    print("FINAL LSTM HOLDOUT RESULTS")
    print("=" * 70)

    print(f"MSE                : {mse:.8f}")
    print(f"RMSE               : {rmse:.8f}")
    print(f"MAE                : {mae:.8f}")
    print(
        f"Direction Accuracy : "
        f"{direction_accuracy:.4%}"
    )
    print(
        f"Correlation        : "
        f"{correlation:.4f}"
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    prediction_dates = holdout_df.iloc[
        SEQUENCE_LENGTH - 1:
    ]["date"].values

    prediction_df = pd.DataFrame(
        {
            "ticker": TICKER,
            "date": prediction_dates,
            "actual": actuals,
            "prediction": predictions,
        }
    )

    output_path = (
        "ml_core/models/"
        "lstm_AAPL_predictions.csv"
    )

    prediction_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Saved predictions to: {output_path}"
    )

    print()
    print("=" * 70)
    print("LSTM TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    train_lstm_pipeline()