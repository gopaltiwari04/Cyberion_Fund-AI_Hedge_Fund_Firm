"""Model research diagnostics, backed by the model registry and walk-forward predictions."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.deps import get_db, jsonable
from api.security import optional_user
from ml_core.models.feature_catalog import describe

router = APIRouter(prefix="/research", tags=["research"], dependencies=[Depends(optional_user)])


def _active(db: Session, horizon: int) -> dict:
    row = db.execute(
        text(
            """
            SELECT id, model_name, version, algorithm, horizon_days, target, trained_at, train_start, train_end, data_end,
                   feature_columns, params, metrics, calibration, feature_importance, data_hash, code_version, status, notes
            FROM model_registry WHERE model_name = 'xgb_forecaster' AND horizon_days = :h
            ORDER BY (status = 'active') DESC, trained_at DESC LIMIT 1
            """
        ),
        {"h": horizon},
    ).mappings().one_or_none()
    if row is None:
        raise HTTPException(status_code=503, detail="No trained model is registered yet")
    return dict(row)


@router.get("/model")
def get_research_model(horizon: int = Query(5, ge=1, le=21), db: Session = Depends(get_db)):
    model = _active(db, horizon)
    metrics = model["metrics"] or {}
    counts = db.execute(
        text("SELECT COUNT(*), MIN(as_of_date), MAX(as_of_date) FROM model_predictions WHERE model_id = :m AND kind = 'walk_forward'"),
        {"m": model["id"]},
    ).one()
    return jsonable({
        "model": {"name": "XGBoost multi-horizon forecaster", "type": model["algorithm"], "target": model["target"],
                  "horizon_days": model["horizon_days"], "version": model["version"], "status": model["status"],
                  "trained_at": model["trained_at"], "notes": model["notes"]},
        "dataset": {"rows": None, "target_rows": None, "train_rows": None, "validation_rows": None,
                    "test_rows": counts[0], "train_start": model["train_start"], "train_end": model["train_end"],
                    "oos_start": counts[1], "oos_end": counts[2]},
        "metrics": {"mae": metrics.get("mae"), "rmse": metrics.get("rmse"),
                    "directional_accuracy": metrics.get("directional_accuracy"),
                    "prediction_correlation": metrics.get("prediction_correlation")},
        "evaluation": metrics,
        "calibration": model["calibration"],
        "feature_count": len(model["feature_columns"] or []),
        "data_hash": model["data_hash"],
        "code_version": model["code_version"],
        "validation": "Purged, embargoed expanding-window walk-forward; configuration selected on pre-2020 folds only",
    })


@router.get("/models")
def get_all_models(db: Session = Depends(get_db)):
    rows = db.execute(
        text(
            """
            SELECT DISTINCT ON (horizon_days) id, horizon_days, version, status, trained_at, metrics, calibration,
                   feature_importance, notes
            FROM model_registry WHERE model_name = 'xgb_forecaster'
            ORDER BY horizon_days, (status = 'active') DESC, trained_at DESC
            """
        )
    ).mappings().all()
    challengers = db.execute(
        text(
            """
            SELECT DISTINCT ON (horizon_days) id, model_name, algorithm, horizon_days, version, status, trained_at, metrics, notes
            FROM model_registry WHERE model_name <> 'xgb_forecaster' ORDER BY horizon_days, trained_at DESC
            """
        )
    ).mappings().all()
    models = []
    for row in rows:
        item = dict(row)
        importance = item.pop("feature_importance") or {}
        ranked = sorted(importance.items(), key=lambda pair: -pair[1])[:20]  # jsonb does not preserve key order
        item["feature_importance"] = [{**describe(name), "importance": value} for name, value in ranked]
        models.append(jsonable(item))
    return {"models": models, "challengers": [jsonable(dict(row)) for row in challengers]}


@router.get("/predictions")
def get_research_predictions(limit: int = Query(100, ge=1, le=5000), horizon: int = Query(5, ge=1, le=21),
                             ticker: str | None = Query(None, pattern=r"^[A-Za-z]{1,6}$"), db: Session = Depends(get_db)):
    model = _active(db, horizon)
    params = {"m": model["id"], "limit": limit}
    where = ""
    if ticker:
        where = " AND ticker = :ticker"
        params["ticker"] = ticker.upper()
    rows = db.execute(
        text(
            f"""
            SELECT as_of_date AS date, ticker, realized_return AS actual_return_{horizon}d,
                   pred_return AS predicted_return_{horizon}d, pred_lower, pred_upper
            FROM model_predictions WHERE model_id = :m AND kind = 'walk_forward' AND realized_return IS NOT NULL {where}
            ORDER BY as_of_date DESC, ticker LIMIT :limit
            """
        ),
        params,
    ).mappings().all()
    columns = ["date", "ticker", f"actual_return_{horizon}d", f"predicted_return_{horizon}d", "pred_lower", "pred_upper"]
    return {"available_columns": columns, "observations": [jsonable(dict(row)) for row in rows]}


@router.get("/features")
def get_research_features(horizon: int = Query(5, ge=1, le=21), db: Session = Depends(get_db)):
    model = _active(db, horizon)
    features = model["feature_columns"] or []
    importance = model["feature_importance"] or {}
    return {
        "feature_count": len(features),
        "features": features,
        "details": [{**describe(name), "importance": importance.get(name)} for name in features],
    }
