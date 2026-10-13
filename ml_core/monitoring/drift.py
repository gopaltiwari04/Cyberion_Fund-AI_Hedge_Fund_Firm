"""
Model monitoring: feature drift + rolling out-of-sample performance.

For each active horizon model:
    * feature drift: Population Stability Index (PSI) of the 15 most important
      features, last 63 trading days vs. the 504 days before the model's
      training cut-off (<0.10 stable, 0.10-0.25 moderate, >0.25 significant)
    * performance: rolling 63-day rank IC and interval coverage of predictions
      whose outcomes are now known (walk-forward + realised live predictions)
Results are written to model_monitoring; significant drift or degraded
performance raises an alert via the pipeline-run alerting hook.

Run: python -m ml_core.monitoring.drift
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sqlalchemy import text

from common.db import get_engine
from common.runs import send_alert, track_run
from ml_core.models.forecasting import HORIZONS, MODEL_NAME, load_frame

RECENT_DAYS = 63
REFERENCE_DAYS = 504
PSI_MODERATE, PSI_SIGNIFICANT = 0.10, 0.25


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float | None:
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < 50 or len(current) < 20:
        return None
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(current, edges)[0] / len(current)
    ref_share = np.clip(ref_share, 1e-4, None)
    cur_share = np.clip(cur_share, 1e-4, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def active_models() -> list[dict]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, horizon_days, train_end, feature_importance, metrics FROM model_registry "
                "WHERE model_name = :n AND status = 'active'"
            ),
            {"n": MODEL_NAME},
        ).mappings().all()
    return [dict(row) for row in rows]


def realised_predictions(model_id: int) -> pd.DataFrame:
    with get_engine().connect() as conn:
        frame = pd.read_sql(
            text(
                """
                SELECT as_of_date, ticker, pred_return, pred_lower, pred_upper, realized_return FROM model_predictions
                WHERE model_id = :m AND realized_return IS NOT NULL ORDER BY as_of_date
                """
            ),
            conn,
            params={"m": model_id},
        )
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"])
    return frame


def monitor_model(model: dict, frame: pd.DataFrame) -> list[dict]:
    as_of = frame["date"].max()
    results = []

    importance = model["feature_importance"] or {}
    features = [name for name in list(importance)[:15] if name in frame.columns]
    train_end = pd.Timestamp(model["train_end"])
    dates = pd.DatetimeIndex(frame["date"].unique()).sort_values()
    reference_dates = dates[dates <= train_end][-REFERENCE_DAYS:]
    recent_dates = dates[-RECENT_DAYS:]
    reference = frame[frame["date"].isin(reference_dates)]
    recent = frame[frame["date"].isin(recent_dates)]
    scores = {feature: psi(reference[feature].to_numpy(float), recent[feature].to_numpy(float)) for feature in features}
    valid = {k: v for k, v in scores.items() if v is not None}
    if valid:
        worst = max(valid, key=valid.get)
        results.append({"metric": "feature_psi_max", "value": valid[worst],
                        "details": {"feature": worst, "per_feature": valid,
                                    "reference": f"{reference_dates.min().date()}..{reference_dates.max().date()}",
                                    "recent": f"{recent_dates.min().date()}..{recent_dates.max().date()}"}})

    realised = realised_predictions(model["id"])
    window = realised[realised["as_of_date"] >= realised["as_of_date"].max() - pd.Timedelta(days=95)] if len(realised) else realised
    if len(window) > 100:
        ic = window.groupby("as_of_date").apply(
            lambda g: g["pred_return"].rank().corr(g["realized_return"].rank()) if len(g) >= 5 else np.nan,
            include_groups=False,
        ).dropna()
        coverage = ((window["realized_return"] >= window["pred_lower"]) & (window["realized_return"] <= window["pred_upper"])).mean()
        baseline = (model["metrics"] or {}).get("ic_mean")
        results.append({"metric": "rolling_ic_63d", "value": float(ic.mean()),
                        "details": {"observations": int(len(window)), "baseline_walk_forward_ic": baseline,
                                    "window_end": str(window["as_of_date"].max().date())}})
        results.append({"metric": "rolling_interval_coverage_63d", "value": float(coverage),
                        "details": {"nominal": 0.9, "observations": int(len(window))}})

    with get_engine().begin() as conn:
        for item in results:
            conn.execute(
                text(
                    """
                    INSERT INTO model_monitoring (model_id, as_of_date, metric, value, details)
                    VALUES (:m, :d, :metric, :v, CAST(:details AS jsonb))
                    ON CONFLICT (model_id, as_of_date, metric) DO UPDATE SET value = EXCLUDED.value, details = EXCLUDED.details
                    """
                ),
                {"m": model["id"], "d": as_of.date(), "metric": item["metric"], "v": item["value"],
                 "details": json.dumps(item["details"], default=str)},
            )
    return results


def main() -> None:
    with track_run("model_monitoring", "drift_and_performance") as ctx:
        frame = load_frame()
        alerts = []
        for model in active_models():
            results = monitor_model(model, frame)
            summary = {item["metric"]: round(item["value"], 4) for item in results}
            ctx.details[f"h{model['horizon_days']}"] = summary
            if summary.get("feature_psi_max", 0) > PSI_SIGNIFICANT:
                alerts.append(f"h={model['horizon_days']}: significant feature drift (PSI {summary['feature_psi_max']:.2f})")
            if summary.get("rolling_interval_coverage_63d", 1) < 0.75:
                alerts.append(f"h={model['horizon_days']}: interval coverage {summary['rolling_interval_coverage_63d']:.0%}")
        for alert in alerts:
            ctx.warn(alert)
        if alerts:
            send_alert("Model monitoring warning", "\n".join(alerts))
        if not active_models():
            ctx.warn("No active models to monitor")
        ctx.details["horizons_expected"] = list(HORIZONS)


if __name__ == "__main__":
    main()
