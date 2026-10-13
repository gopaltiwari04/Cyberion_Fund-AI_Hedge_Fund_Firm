"""
Temporal Fusion Transformer challenger (darts).

A global TFT is trained on all tickers' daily total returns with past
covariates, then evaluated on the most recent year using rolling origins
(the model only sees data up to each forecast origin). The 5-day forecast is
the compounded median of the next five daily return forecasts.

It is compared with the active XGBoost model's walk-forward predictions on
exactly the same (ticker, date) rows and recorded in the model registry as a
challenger. It is not served: promotion would require beating the incumbent
on rank IC and error over the identical holdout, and serving it would need a
torch runtime the API deliberately does not carry.

Run: python -m ml_core.models.tft_challenger
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from sqlalchemy import text

from common.db import get_engine
from common.logs import get_logger
from common.runs import track_run
from ml_core.models import registry
from ml_core.models.forecasting import EMBARGO_DAYS, MODEL_NAME, load_frame

logger = get_logger(__name__)

HORIZON = 5
HOLDOUT_DAYS = 252
INPUT_CHUNK = 42
COVARIATES = ["return_5d", "return_20d", "volatility_20d", "rsi_14", "relative_return_5d", "sma_50_distance",
              "vix", "term_spread_10y2y", "cs_rank_return_20d"]
PARAMS = {"input_chunk_length": INPUT_CHUNK, "output_chunk_length": HORIZON, "hidden_size": 16, "lstm_layers": 1,
          "num_attention_heads": 2, "dropout": 0.1, "batch_size": 256, "n_epochs": int(os.getenv("TFT_EPOCHS", "8")),
          "quantiles": [0.1, 0.5, 0.9]}


def build_series(frame: pd.DataFrame):
    from darts import TimeSeries

    targets, covariates, tickers = [], [], []
    for ticker, group in frame.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        group["time_index"] = np.arange(len(group))
        target_values = (group["total_return_1d"].fillna(0.0) * 100).to_frame("ret")
        cov_values = group[COVARIATES].ffill().fillna(0.0)
        targets.append(TimeSeries.from_dataframe(target_values.astype("float32")))
        covariates.append(TimeSeries.from_dataframe(cov_values.astype("float32")))
        tickers.append((ticker, group["date"].to_numpy()))
    return targets, covariates, tickers


def train_and_evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from darts.dataprocessing.transformers import Scaler
    from darts.models import TFTModel
    from darts.utils.likelihood_models import QuantileRegression

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning").setLevel(logging.ERROR)

    dates = pd.DatetimeIndex(frame["date"].unique()).sort_values()
    labelled_end = dates[-(HORIZON + 1)]
    test_dates = dates[(dates <= labelled_end)][-HOLDOUT_DAYS:]
    train_end = dates[dates.get_loc(test_dates[0]) - HORIZON - EMBARGO_DAYS]

    targets, covariates, tickers = build_series(frame)
    split = [int(np.searchsorted(ticker_dates, np.datetime64(train_end), side="right")) for _, ticker_dates in tickers]
    train_targets = [series[:position] for series, position in zip(targets, split)]
    train_covs = [series[:position] for series, position in zip(covariates, split)]

    scaler = Scaler()
    scaler.fit(train_covs)
    scaled_covs = scaler.transform(covariates)
    scaled_train_covs = scaler.transform(train_covs)

    model = TFTModel(
        input_chunk_length=PARAMS["input_chunk_length"], output_chunk_length=PARAMS["output_chunk_length"],
        hidden_size=PARAMS["hidden_size"], lstm_layers=PARAMS["lstm_layers"],
        num_attention_heads=PARAMS["num_attention_heads"], dropout=PARAMS["dropout"],
        batch_size=PARAMS["batch_size"], n_epochs=PARAMS["n_epochs"], add_relative_index=True,
        likelihood=QuantileRegression(quantiles=PARAMS["quantiles"]), random_state=42,
        pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False, "enable_model_summary": False},
    )
    model.fit(series=train_targets, past_covariates=scaled_train_covs, verbose=False)

    rows = []
    for (ticker, ticker_dates), series, cov in zip(tickers, targets, scaled_covs):
        index = pd.DatetimeIndex(ticker_dates)
        positions = [index.get_loc(day) for day in test_dates if day in index]
        if not positions:
            continue
        forecasts = model.historical_forecasts(
            series=series, past_covariates=cov, start=positions[0] + 1, forecast_horizon=HORIZON, stride=1,
            retrain=False, last_points_only=False, verbose=False, num_samples=1,
        )
        for origin, forecast in zip(positions, forecasts):
            daily = forecast.values().flatten() / 100.0
            rows.append({"ticker": ticker, "date": index[origin], "tft_pred_return": float(np.prod(1 + daily) - 1)})
    predictions = pd.DataFrame(rows)
    info = {"train_end": str(train_end.date()), "test_start": str(test_dates[0].date()), "test_end": str(test_dates[-1].date())}
    return predictions, info


def compare_with_incumbent(tft: pd.DataFrame, frame: pd.DataFrame) -> dict:
    with get_engine().connect() as conn:
        incumbent = pd.read_sql(
            text(
                """
                SELECT p.ticker, p.as_of_date AS date, p.pred_return AS xgb_pred_return
                FROM model_predictions p JOIN model_registry r ON r.id = p.model_id
                WHERE r.model_name = :n AND r.horizon_days = :h AND r.status = 'active' AND p.kind = 'walk_forward'
                """
            ),
            conn, params={"n": MODEL_NAME, "h": HORIZON},
        )
    incumbent["date"] = pd.to_datetime(incumbent["date"])
    data = (tft.merge(incumbent, on=["ticker", "date"], how="inner")
            .merge(frame[["ticker", "date", f"fwd_return_{HORIZON}d"]], on=["ticker", "date"])
            .dropna())
    y = data[f"fwd_return_{HORIZON}d"].to_numpy()

    def score(column: str) -> dict:
        p = data[column].to_numpy()
        ic = data.groupby("date").apply(lambda g: g[column].rank().corr(g[f"fwd_return_{HORIZON}d"].rank()),
                                        include_groups=False).dropna()
        return {"mae": float(np.mean(np.abs(y - p))), "r2_oos_vs_zero": float(1 - np.sum((y - p) ** 2) / np.sum(y ** 2)),
                "directional_accuracy": float(np.mean(np.sign(y) == np.sign(p))), "ic_mean": float(ic.mean()),
                "ic_tstat_overlap_adjusted": float(ic.mean() / ic.std() * np.sqrt(len(ic) / HORIZON)) if ic.std() > 0 else None}

    return {"observations": int(len(data)), "tft": score("tft_pred_return"), "xgboost": score("xgb_pred_return"),
            "mae_zero_baseline": float(np.mean(np.abs(y)))}


def main() -> None:
    with track_run("model_training", "tft_challenger") as ctx:
        frame = load_frame()
        predictions, info = train_and_evaluate(frame)
        comparison = compare_with_incumbent(predictions, frame)
        tft_better = (
            comparison["observations"] > 0
            and comparison["tft"]["ic_mean"] > comparison["xgboost"]["ic_mean"]
            and comparison["tft"]["mae"] < comparison["xgboost"]["mae"]
        )
        metrics = {**comparison, **info, "tft_beats_incumbent": bool(tft_better)}
        registry.register_model(
            model_name="tft_challenger", algorithm="darts.TFTModel", horizon_days=HORIZON,
            target=f"total_return_{HORIZON}d", artifact=None, feature_columns=COVARIATES, params=PARAMS,
            metrics=metrics, calibration={}, feature_importance={},
            train_start=frame["date"].min().date(), train_end=pd.Timestamp(info["train_end"]).date(),
            data_end=frame["date"].max().date(), data_hash=registry.frame_hash(frame[["ticker", "date", *COVARIATES]]),
            status="candidate",
            notes=("Challenger evaluated on the same holdout rows as the active XGBoost model; not served. "
                   + ("Outperformed the incumbent on IC and MAE - review before promotion." if tft_better
                      else "Did not outperform the incumbent.")),
        )
        ctx.details.update(metrics)


if __name__ == "__main__":
    main()
