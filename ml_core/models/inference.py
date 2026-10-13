"""
Daily inference: active model bundles -> live forecasts + TreeSHAP explanations.

For the latest feature date, every active horizon bundle produces a point
forecast, a conformal 90% interval and a volatility forecast per ticker. The
mean model's exact TreeSHAP decomposition (xgboost ``pred_contribs``) is
stored with each prediction, together with the sentiment / filing / graph
signal values that fed it, so the dashboard can explain every recommendation.

It also back-fills realised outcomes for earlier live predictions, which the
monitoring job uses to track live performance.

Run: python -m ml_core.models.inference
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sqlalchemy import text

from common.db import get_engine
from common.logs import get_logger
from common.runs import track_run
from ml_core.models import registry
from ml_core.models.feature_catalog import SIGNAL_FEATURES, describe
from ml_core.models.forecasting import HORIZONS, MODEL_NAME, load_frame, predict_bundle, shap_contributions

logger = get_logger(__name__)
TOP_CONTRIBUTIONS = 10


def _clean(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def explanation(features: list[str], row: pd.Series, contributions: np.ndarray) -> dict:
    values = contributions[:-1]
    order = np.argsort(-np.abs(values))[:TOP_CONTRIBUTIONS]
    return {
        "method": "TreeSHAP (exact, xgboost pred_contribs) on the mean-return model",
        "base_value": _clean(contributions[-1]),
        "prediction": _clean(contributions.sum()),
        "top_contributions": [
            {**describe(features[i]), "value": _clean(row[features[i]]), "contribution": _clean(values[i])}
            for i in order
        ],
        "signals": {name: _clean(row.get(name)) for name in SIGNAL_FEATURES},
    }


def run_inference(frame: pd.DataFrame | None = None) -> dict:
    frame = frame if frame is not None else load_frame()
    latest_date = frame["date"].max()
    latest = frame[frame["date"] == latest_date].reset_index(drop=True)
    summary = {"as_of_date": str(latest_date.date()), "horizons": {}}
    for horizon in HORIZONS:
        loaded = registry.load_active(MODEL_NAME, horizon)
        if loaded is None:
            summary["horizons"][horizon] = "no active model"
            continue
        model_id, bundle, _ = loaded
        predictions = predict_bundle(bundle, latest)
        contributions = shap_contributions(bundle, latest)
        records = []
        for index, row in latest.iterrows():
            pred = predictions.loc[index]
            records.append(
                {
                    "model_id": model_id,
                    "ticker": row["ticker"],
                    "as_of": latest_date.date(),
                    "h": horizon,
                    "pred_return": _clean(pred["pred_return"]),
                    "pred_lower": _clean(pred["pred_lower"]),
                    "pred_upper": _clean(pred["pred_upper"]),
                    "level": bundle["interval_level"],
                    "pred_volatility": _clean(pred["pred_volatility"]),
                    "explanation": json.dumps(explanation(bundle["features"], row, contributions[index])),
                }
            )
        with get_engine().begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO model_predictions (model_id, kind, ticker, as_of_date, horizon_days, pred_return,
                        pred_lower, pred_upper, interval_level, pred_volatility, explanation)
                    VALUES (:model_id, 'live', :ticker, :as_of, :h, :pred_return, :pred_lower, :pred_upper, :level,
                        :pred_volatility, CAST(:explanation AS jsonb))
                    ON CONFLICT (model_id, kind, ticker, as_of_date, horizon_days) DO UPDATE SET
                        pred_return = EXCLUDED.pred_return, pred_lower = EXCLUDED.pred_lower,
                        pred_upper = EXCLUDED.pred_upper, pred_volatility = EXCLUDED.pred_volatility,
                        explanation = EXCLUDED.explanation, created_at = now()
                    """
                ),
                records,
            )
        summary["horizons"][horizon] = {"model_id": model_id, "predictions": len(records)}
    summary["realized_backfilled"] = backfill_realized(frame)
    return summary


def backfill_realized(frame: pd.DataFrame) -> int:
    """Attach realised returns/volatility to earlier live predictions once their horizon has elapsed."""
    updated = 0
    with get_engine().begin() as conn:
        for horizon in HORIZONS:
            realized = frame[["ticker", "date", f"fwd_return_{horizon}d", f"fwd_vol_{horizon}d"]].dropna()
            if realized.empty:
                continue
            pending = pd.read_sql(
                text(
                    "SELECT ticker, as_of_date FROM model_predictions "
                    "WHERE kind = 'live' AND horizon_days = :h AND realized_return IS NULL"
                ),
                conn,
                params={"h": horizon},
            )
            if pending.empty:
                continue
            pending["date"] = pd.to_datetime(pending["as_of_date"])
            matched = pending.merge(realized, on=["ticker", "date"])
            if matched.empty:
                continue
            conn.execute(
                text(
                    """
                    UPDATE model_predictions SET realized_return = :r, realized_volatility = :v
                    WHERE kind = 'live' AND horizon_days = :h AND ticker = :ticker AND as_of_date = :d
                    """
                ),
                [
                    {"r": float(row[f"fwd_return_{horizon}d"]), "v": float(row[f"fwd_vol_{horizon}d"]), "h": horizon,
                     "ticker": row["ticker"], "d": row["date"].date()}
                    for _, row in matched.iterrows()
                ],
            )
            updated += len(matched)
    return updated


def main() -> None:
    with track_run("model_inference", "live_predictions") as ctx:
        summary = run_inference()
        ctx.details.update(summary)
        missing = [h for h, value in summary["horizons"].items() if value == "no active model"]
        if len(missing) == len(HORIZONS):
            raise RuntimeError("No active models in the registry; run ml_core.models.forecasting first")
        for horizon in missing:
            ctx.warn(f"No active model for horizon {horizon}")


if __name__ == "__main__":
    main()
