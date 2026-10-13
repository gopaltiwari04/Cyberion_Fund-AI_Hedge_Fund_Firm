"""
Multi-horizon return and volatility forecasting (XGBoost).

For each horizon h in (1, 5, 21) trading days, a model bundle is trained:
    * mean model      E[r_{t->t+h}]  (squared error on winsorised total returns)
    * quantile model  5% / 50% / 95% quantiles (reg:quantileerror)
                      + split-conformal adjustment (CQR) -> calibrated 90% interval
    * volatility model log realised volatility over the next max(h, 5) days

Leakage prevention
    * Labels are built from *future* total returns r_{t+1..t+h} (close-to-close
      including dividends); every feature is known at the close of t.
      (The legacy model's label overlapped the current day's return feature.)
    * Walk-forward evaluation with expanding windows. Each test block starts at
      t0; training rows must satisfy t + h + embargo < t0 (purging + embargo),
      and the conformal calibration set is purged from the fitting set too.
    * Imputation is unnecessary: XGBoost handles missing values natively, so no
      statistics from the future leak through fill values.

Outputs
    * model_registry rows (one active bundle per horizon) with walk-forward
      metrics, calibration diagnostics and SHAP feature importance
    * model_predictions(kind='walk_forward'): every out-of-sample prediction,
      used for evaluation, monitoring and leakage-free backtests

Run: python -m ml_core.models.forecasting [--horizons 1 5 21]
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import text

from common.db import get_engine
from common.logs import get_logger
from common.runs import track_run
from common.universe import tickers
from ml_core.feature_store.external_signals import EXTERNAL_FEATURES
from ml_core.models import registry

logger = get_logger(__name__)

MODEL_NAME = "xgb_forecaster"
HORIZONS = (1, 5, 21)
EMBARGO_DAYS = 5
MIN_TRAIN_DAYS = int(os.getenv("FORECAST_MIN_TRAIN_DAYS", "756"))
REFIT_EVERY = int(os.getenv("FORECAST_REFIT_EVERY", "63"))
CAL_DAYS = 126
INTERVAL_ALPHA = 0.10
TRADING_DAYS = 252
KEEP_WALK_FORWARD_VERSIONS = 3

TECHNICAL_FEATURES = [
    "return_1d", "return_5d", "return_10d", "return_20d", "return_60d",
    "rsi_14", "macd", "volatility_5d", "volatility_20d", "volatility_60d", "atr_14_pct",
    "sma_20_distance", "sma_50_distance", "sma_200_distance", "drawdown_20d", "drawdown_60d",
    "volume_change_5d", "volume_zscore_20d", "market_return_5d", "relative_return_5d",
    "beta_60d", "correlation_spy_60d", "intraday_range", "close_position", "gap_return", "regime",
]
CROSS_SECTIONAL = {
    "cs_rank_return_20d": "return_20d",
    "cs_rank_return_60d": "return_60d",
    "cs_rank_volatility_20d": "volatility_20d",
    "cs_rank_rsi_14": "rsi_14",
}
MODEL_FEATURES = TECHNICAL_FEATURES + EXTERNAL_FEATURES + list(CROSS_SECTIONAL)

BASE_PARAMS = {
    "objective": "reg:squarederror", "eta": 0.03, "max_depth": 4, "min_child_weight": 50,
    "subsample": 0.8, "colsample_bytree": 0.7, "lambda": 5.0, "tree_method": "hist", "seed": 42,
}
# Mean-model configuration selected on validation folds whose test dates are
# before 2020-01-01 only (candidates: baseline / regularised / demeaned target /
# demeaned + regularised). 2020 onward is reported separately as an untouched
# holdout. Strong regularisation + a cross-sectionally demeaned target reduced
# forecast dispersion, which otherwise made errors worse than a zero forecast.
MEAN_PARAMS = {**BASE_PARAMS, "eta": 0.02, "max_depth": 3, "min_child_weight": 300, "subsample": 0.7, "lambda": 20.0}
QUANTILE_PARAMS = {
    **BASE_PARAMS, "objective": "reg:quantileerror", "quantile_alpha": np.array([0.05, 0.5, 0.95]),
}
VOL_PARAMS = {**BASE_PARAMS, "max_depth": 5, "min_child_weight": 20}
ROUNDS = {"mean": 200, "quantile": 300, "vol": 400}
SELECTION_CUTOFF = "2020-01-01"


def vol_window(horizon: int) -> int:
    return max(horizon, 5)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def add_targets(prices: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """prices: ticker, date, close, dividend -> adds total return, forward returns and forward vol."""
    frames = []
    for _, group in prices.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        group = group.copy()
        total = (group["close"] + group["dividend"].fillna(0.0)) / group["close"].shift(1) - 1.0
        group["total_return_1d"] = total
        log_cum = np.log1p(total.fillna(0.0)).cumsum()
        for h in horizons:
            group[f"fwd_return_{h}d"] = np.expm1(log_cum.shift(-h) - log_cum)
            window = vol_window(h)
            future_std = total.iloc[::-1].rolling(window, min_periods=window).std().iloc[::-1].shift(-1)
            group[f"fwd_vol_{h}d"] = future_std * math.sqrt(TRADING_DAYS)
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def add_cross_sectional(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for name, source in CROSS_SECTIONAL.items():
        frame[name] = frame.groupby("date")[source].rank(pct=True)
    return frame


def load_frame(universe: list[str] | None = None) -> pd.DataFrame:
    universe = universe or tickers()
    columns = ", ".join(f"f.{column}" for column in TECHNICAL_FEATURES + EXTERNAL_FEATURES)
    with get_engine().connect() as conn:
        features = pd.read_sql(
            text(f"SELECT f.ticker, f.date, {columns} FROM feature_store f WHERE f.ticker = ANY(:t)"),
            conn, params={"t": universe},
        )
        prices = pd.read_sql(
            text("SELECT ticker, date, close, dividend FROM market_data WHERE ticker = ANY(:t)"),
            conn, params={"t": universe},
        )
    prices = add_targets(prices)
    frame = features.merge(prices, on=["ticker", "date"], how="inner")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = add_cross_sectional(frame)
    frame[MODEL_FEATURES] = frame[MODEL_FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# Model bundle
# ----------------------------------------------------------------------

def _matrix(frame: pd.DataFrame, label: np.ndarray | None = None) -> xgb.DMatrix:
    return xgb.DMatrix(frame[MODEL_FEATURES], label=label, missing=np.nan, feature_names=MODEL_FEATURES)


def fit_bundle(fit: pd.DataFrame, cal: pd.DataFrame, horizon: int) -> dict:
    target = f"fwd_return_{horizon}d"
    vol_target = f"fwd_vol_{horizon}d"
    fit = fit.dropna(subset=[target])
    lo, hi = fit[target].quantile([0.01, 0.99]).tolist()
    y = fit[target].clip(lo, hi).to_numpy()

    # Mean model: predict return relative to the universe average on the same date,
    # then add a prior equal to the historical average universe return (known at fit time).
    universe_mean = fit.groupby("date")[target].transform("mean")
    relative = fit[target] - universe_mean
    rel_lo, rel_hi = relative.quantile([0.01, 0.99]).tolist()
    prior = float(fit.groupby("date")[target].mean().mean())
    mean_model = xgb.train(MEAN_PARAMS, _matrix(fit, relative.clip(rel_lo, rel_hi).to_numpy()), num_boost_round=ROUNDS["mean"])
    quantile_model = xgb.train(QUANTILE_PARAMS, _matrix(fit, y), num_boost_round=ROUNDS["quantile"])
    vol_rows = fit.dropna(subset=[vol_target])
    vol_rows = vol_rows[vol_rows[vol_target] > 0]
    vol_model = xgb.train(VOL_PARAMS, _matrix(vol_rows, np.log(vol_rows[vol_target].to_numpy())), num_boost_round=ROUNDS["vol"])

    bundle = {
        "horizon": horizon, "features": MODEL_FEATURES, "mean": mean_model, "quantile": quantile_model,
        "vol": vol_model, "winsor": (lo, hi), "interval_level": 1 - INTERVAL_ALPHA,
        "conformal_q": 0.0, "vol_smearing": 1.0, "prior": prior,
    }

    cal = cal.dropna(subset=[target])
    if len(cal) > 50:
        quantiles = np.sort(quantile_model.predict(_matrix(cal)), axis=1)
        actual = cal[target].to_numpy()
        scores = np.maximum(quantiles[:, 0] - actual, actual - quantiles[:, 2])
        level = min(1.0, math.ceil((len(scores) + 1) * (1 - INTERVAL_ALPHA)) / len(scores))
        bundle["conformal_q"] = float(np.quantile(scores, level, method="higher"))
        cal_vol = cal.dropna(subset=[vol_target])
        cal_vol = cal_vol[cal_vol[vol_target] > 0]
        if len(cal_vol) > 50:
            residual = np.log(cal_vol[vol_target].to_numpy()) - vol_model.predict(_matrix(cal_vol))
            bundle["vol_smearing"] = float(np.mean(np.exp(residual)))
    return bundle


def predict_bundle(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    matrix = _matrix(frame)
    point = bundle["mean"].predict(matrix) + bundle.get("prior", 0.0)
    quantiles = np.sort(bundle["quantile"].predict(matrix), axis=1)
    lower = np.minimum(quantiles[:, 0] - bundle["conformal_q"], point)
    upper = np.maximum(quantiles[:, 2] + bundle["conformal_q"], point)
    volatility = np.exp(bundle["vol"].predict(matrix)) * bundle["vol_smearing"]
    return pd.DataFrame(
        {"pred_return": point, "pred_lower": lower, "pred_upper": upper, "pred_volatility": volatility},
        index=frame.index,
    )


def shap_contributions(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """
    Exact TreeSHAP values of the mean model (xgboost pred_contribs). The last
    column is the base value; it includes the return prior so that
    contributions sum exactly to the predicted return.
    """
    contributions = bundle["mean"].predict(_matrix(frame), pred_contribs=True)
    contributions[:, -1] += bundle.get("prior", 0.0)
    return contributions


# ----------------------------------------------------------------------
# Walk-forward
# ----------------------------------------------------------------------

@dataclass
class Fold:
    train_end: pd.Timestamp
    test_dates: pd.DatetimeIndex


def make_folds(dates: pd.DatetimeIndex, horizon: int) -> list[Fold]:
    folds = []
    for start in range(MIN_TRAIN_DAYS, len(dates), REFIT_EVERY):
        cut = start - horizon - EMBARGO_DAYS
        if cut <= CAL_DAYS * 2:
            continue
        folds.append(Fold(train_end=dates[cut - 1], test_dates=dates[start:start + REFIT_EVERY]))
    return folds


def split_fit_cal(train: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(train["date"].unique()).sort_values()
    cal_start = dates[-CAL_DAYS]
    fit_end = dates[-(CAL_DAYS + horizon + EMBARGO_DAYS)]
    return train[train["date"] <= fit_end], train[train["date"] >= cal_start]


def walk_forward(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    target = f"fwd_return_{horizon}d"
    labelled = frame.dropna(subset=[target])
    dates = pd.DatetimeIndex(labelled["date"].unique()).sort_values()
    results = []
    for number, fold in enumerate(make_folds(dates, horizon), start=1):
        train = labelled[labelled["date"] <= fold.train_end]
        test = frame[frame["date"].isin(fold.test_dates)]
        # Purging invariant: every training label is realised (h trading days) plus an
        # embargo before the first test date.
        gap = dates.searchsorted(fold.test_dates[0]) - dates.searchsorted(train["date"].max())
        if gap <= horizon + EMBARGO_DAYS - 1:
            raise AssertionError(f"Purge violated: gap {gap} trading days for horizon {horizon}")
        fit, cal = split_fit_cal(train, horizon)
        bundle = fit_bundle(fit, cal, horizon)
        predictions = predict_bundle(bundle, test)
        block = test[["ticker", "date", target, f"fwd_vol_{horizon}d", "volatility_20d"]].join(predictions)
        block["train_end"] = fold.train_end
        results.append(block)
        logger.info("h=%s fold %s: train_end=%s test=%s..%s", horizon, number, fold.train_end.date(),
                    pd.Timestamp(fold.test_dates[0]).date(), pd.Timestamp(fold.test_dates[-1]).date())
    out = pd.concat(results, ignore_index=True)
    return out.rename(columns={target: "realized_return", f"fwd_vol_{horizon}d": "realized_volatility"})


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate(oos: pd.DataFrame, horizon: int) -> tuple[dict, dict]:
    data = oos.dropna(subset=["realized_return"])
    y, p = data["realized_return"].to_numpy(), data["pred_return"].to_numpy()
    sse = float(np.sum((y - p) ** 2))
    metrics: dict = {
        "observations": int(len(data)),
        "test_start": str(data["date"].min().date()),
        "test_end": str(data["date"].max().date()),
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae_zero_baseline": float(np.mean(np.abs(y))),
        "r2_oos_vs_zero": float(1 - sse / np.sum(y ** 2)),
        "directional_accuracy": float(np.mean(np.sign(y) == np.sign(p))),
        "base_rate_positive": float(np.mean(y > 0)),
        "prediction_correlation": float(np.corrcoef(y, p)[0, 1]),
    }

    daily_ic = data.groupby("date").apply(
        lambda g: g["pred_return"].rank().corr(g["realized_return"].rank()) if len(g) >= 5 else np.nan,
        include_groups=False,
    ).dropna()
    effective_n = max(len(daily_ic) / horizon, 1)
    metrics.update(
        {
            "ic_mean": float(daily_ic.mean()),
            "ic_std": float(daily_ic.std()),
            "ic_tstat_overlap_adjusted": float(daily_ic.mean() / daily_ic.std() * math.sqrt(effective_n)),
            "ic_hit_rate": float((daily_ic > 0).mean()),
        }
    )
    by_year = daily_ic.groupby(daily_ic.index.year).mean()
    metrics["ic_by_year"] = {str(year): round(float(value), 4) for year, value in by_year.items()}

    def spread(group: pd.DataFrame) -> float:
        ranked = group["pred_return"].rank(pct=True)
        return group.loc[ranked > 2 / 3, "realized_return"].mean() - group.loc[ranked <= 1 / 3, "realized_return"].mean()

    metrics["top_minus_bottom_tercile_return"] = float(data.groupby("date").apply(spread, include_groups=False).mean())

    inside = (data["realized_return"] >= data["pred_lower"]) & (data["realized_return"] <= data["pred_upper"])
    metrics["interval_level"] = 1 - INTERVAL_ALPHA
    metrics["interval_coverage"] = float(inside.mean())
    metrics["interval_mean_width"] = float((data["pred_upper"] - data["pred_lower"]).mean())

    vol = data.dropna(subset=["realized_volatility", "pred_volatility"])
    baseline = vol["volatility_20d"] * math.sqrt(TRADING_DAYS)
    valid = baseline.notna()
    metrics["vol_mae"] = float(np.mean(np.abs(vol["realized_volatility"] - vol["pred_volatility"])))
    metrics["vol_mae_trailing20d_baseline"] = float(np.mean(np.abs(vol.loc[valid, "realized_volatility"] - baseline[valid])))
    metrics["vol_correlation"] = float(np.corrcoef(vol["realized_volatility"], vol["pred_volatility"])[0, 1])

    deciles = pd.qcut(data["pred_return"], 10, labels=False, duplicates="drop")
    table = data.groupby(deciles).agg(
        predicted=("pred_return", "mean"), realized=("realized_return", "mean"), count=("pred_return", "size")
    )
    slope, intercept = np.polyfit(p, y, 1)
    calibration = {
        "interval_level": 1 - INTERVAL_ALPHA,
        "interval_coverage": metrics["interval_coverage"],
        "reliability_slope": float(slope),
        "reliability_intercept": float(intercept),
        "decile_table": [
            {"decile": int(k) + 1, "predicted": float(v.predicted), "realized": float(v.realized), "count": int(v["count"])}
            for k, v in table.iterrows()
        ],
    }
    return metrics, calibration


def activation_gates(metrics: dict) -> tuple[bool, list[str]]:
    reasons = []
    if metrics["observations"] < 2000:
        reasons.append("too few out-of-sample observations")
    if not 0.75 <= metrics["interval_coverage"] <= 0.98:
        reasons.append(f"interval coverage {metrics['interval_coverage']:.2f} outside [0.75, 0.98]")
    if not np.isfinite(metrics["mae"]):
        reasons.append("non-finite errors")
    return not reasons, reasons


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------

def store_walk_forward(model_id: int, horizon: int, oos: pd.DataFrame) -> int:
    frame = oos.copy()
    frame["as_of_date"] = frame["date"].dt.date
    frame["train_end"] = pd.to_datetime(frame["train_end"]).dt.date
    columns = ["ticker", "as_of_date", "pred_return", "pred_lower", "pred_upper", "pred_volatility", "train_end",
               "realized_return", "realized_volatility"]
    records = frame[columns].astype(object).where(frame[columns].notna(), None).to_dict(orient="records")
    for record in records:
        record.update({"model_id": model_id, "horizon": horizon, "level": 1 - INTERVAL_ALPHA})
    with get_engine().begin() as conn:
        for start in range(0, len(records), 2000):
            conn.execute(
                text(
                    """
                    INSERT INTO model_predictions (model_id, kind, ticker, as_of_date, horizon_days, pred_return,
                        pred_lower, pred_upper, interval_level, pred_volatility, train_end, realized_return, realized_volatility)
                    VALUES (:model_id, 'walk_forward', :ticker, :as_of_date, :horizon, :pred_return, :pred_lower,
                        :pred_upper, :level, :pred_volatility, :train_end, :realized_return, :realized_volatility)
                    ON CONFLICT (model_id, kind, ticker, as_of_date, horizon_days) DO NOTHING
                    """
                ),
                records[start:start + 2000],
            )
    return len(records)


def prune_old_walk_forward(horizon: int) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM model_predictions WHERE kind = 'walk_forward' AND horizon_days = :h AND model_id IN (
                    SELECT id FROM model_registry WHERE model_name = :name AND horizon_days = :h AND status = 'archived'
                    ORDER BY trained_at DESC OFFSET :keep)
                """
            ),
            {"h": horizon, "name": MODEL_NAME, "keep": KEEP_WALK_FORWARD_VERSIONS - 1},
        )


def global_importance(bundle: dict, frame: pd.DataFrame) -> dict:
    contributions = shap_contributions(bundle, frame)[:, :-1]
    importance = np.mean(np.abs(contributions), axis=0)
    total = importance.sum() or 1.0
    ranked = sorted(zip(MODEL_FEATURES, importance / total), key=lambda item: -item[1])
    return {name: round(float(value), 5) for name, value in ranked}


def train_horizon(frame: pd.DataFrame, horizon: int, data_hash: str) -> dict:
    target = f"fwd_return_{horizon}d"
    oos = walk_forward(frame, horizon)
    metrics, calibration = evaluate(oos, horizon)
    holdout = oos[oos["date"] >= pd.Timestamp(SELECTION_CUTOFF)]
    if len(holdout.dropna(subset=["realized_return"])) > 500:
        holdout_metrics, _ = evaluate(holdout, horizon)
        metrics["holdout_since_2020"] = {
            key: holdout_metrics[key]
            for key in ("observations", "r2_oos_vs_zero", "mae", "mae_zero_baseline", "ic_mean",
                        "ic_tstat_overlap_adjusted", "directional_accuracy", "interval_coverage", "vol_mae",
                        "vol_mae_trailing20d_baseline", "top_minus_bottom_tercile_return")
        }

    labelled = frame.dropna(subset=[target])
    fit, cal = split_fit_cal(labelled, horizon)
    final = fit_bundle(fit, cal, horizon)
    recent = frame[frame["date"] >= frame["date"].max() - pd.Timedelta(days=365)]
    importance = global_importance(final, recent)

    ok, reasons = activation_gates(metrics)
    model_id = registry.register_model(
        model_name=MODEL_NAME,
        algorithm="xgboost",
        horizon_days=horizon,
        target=f"total_return_{horizon}d",
        artifact=final,
        feature_columns=MODEL_FEATURES,
        params={
            "mean": {k: v for k, v in MEAN_PARAMS.items()},
            "mean_target": "cross-sectionally demeaned return + historical universe-mean prior",
            "selection": f"configuration chosen on validation folds with test dates < {SELECTION_CUTOFF}",
            "quantile_alpha": [0.05, 0.5, 0.95],
            "rounds": ROUNDS,
            "embargo_days": EMBARGO_DAYS,
            "min_train_days": MIN_TRAIN_DAYS,
            "refit_every": REFIT_EVERY,
            "calibration_days": CAL_DAYS,
            "winsor": final["winsor"],
            "conformal_q": final["conformal_q"],
            "vol_window_days": vol_window(horizon),
        },
        metrics=metrics,
        calibration=calibration,
        feature_importance=importance,
        train_start=labelled["date"].min().date(),
        train_end=labelled["date"].max().date(),
        data_end=frame["date"].max().date(),
        data_hash=data_hash,
        status="active" if ok else "candidate",
        notes=None if ok else "; ".join(reasons),
    )
    store_walk_forward(model_id, horizon, oos)
    prune_old_walk_forward(horizon)
    return {"model_id": model_id, "active": ok, "reasons": reasons, **{k: metrics[k] for k in (
        "observations", "ic_mean", "ic_tstat_overlap_adjusted", "directional_accuracy", "interval_coverage",
        "vol_mae", "vol_mae_trailing20d_baseline")}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    args = parser.parse_args()
    with track_run("model_training", "xgboost_walk_forward") as ctx:
        frame = load_frame()
        data_hash = registry.frame_hash(frame[["ticker", "date", *MODEL_FEATURES]])
        ctx.details.update({"rows": len(frame), "data_end": str(frame["date"].max().date()), "data_hash": data_hash})
        for horizon in args.horizons:
            summary = train_horizon(frame, horizon, data_hash)
            ctx.details[f"h{horizon}"] = summary
            if not summary["active"]:
                ctx.warn(f"h={horizon} model kept as candidate: {summary['reasons']}")
        ctx.add_rows(len(frame))


if __name__ == "__main__":
    main()
