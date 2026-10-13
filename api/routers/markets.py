"""Market monitoring: prices, features, live forecasts with explanations, news."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.deps import get_db, jsonable
from api.security import optional_user
from common.universe import asset_lookup

router = APIRouter(tags=["markets"], dependencies=[Depends(optional_user)])

TickerPath = Path(..., min_length=1, max_length=6, pattern=r"^[A-Za-z]{1,6}$")


def _ticker(value: str) -> str:
    ticker = value.upper()
    if ticker not in asset_lookup():
        raise HTTPException(status_code=404, detail=f"{ticker} is not in the supported universe")
    return ticker


@router.get("/markets")
def get_markets(db: Session = Depends(get_db)):
    rows = db.execute(
        text(
            """
            WITH latest_market AS (
                SELECT DISTINCT ON (ticker) ticker, date, open, high, low, close, volume
                FROM market_data ORDER BY ticker, date DESC
            ), latest_features AS (
                SELECT DISTINCT ON (ticker) ticker, date AS feature_date, return_1d, return_5d, rsi_14, macd,
                    volatility_20d, regime, relative_return_5d, beta_60d, correlation_spy_60d,
                    news_sentiment_7d, news_count_7d, filing_sentiment_90d, vix
                FROM feature_store WHERE return_1d IS NOT NULL ORDER BY ticker, date DESC
            ), active AS (
                SELECT id FROM model_registry
                WHERE model_name = 'xgb_forecaster' AND horizon_days = 5 AND status = 'active'
                ORDER BY trained_at DESC LIMIT 1
            ), forecasts AS (
                SELECT DISTINCT ON (p.ticker) p.ticker, p.as_of_date, p.pred_return, p.pred_lower, p.pred_upper,
                    p.pred_volatility
                FROM model_predictions p JOIN active a ON a.id = p.model_id
                WHERE p.kind = 'live' ORDER BY p.ticker, p.as_of_date DESC
            )
            SELECT m.*, f.feature_date, f.return_1d, f.return_5d, f.rsi_14, f.macd, f.volatility_20d, f.regime,
                f.relative_return_5d, f.beta_60d, f.correlation_spy_60d, f.news_sentiment_7d, f.news_count_7d,
                f.filing_sentiment_90d, p.as_of_date AS forecast_date, p.pred_return AS forecast_return_5d,
                p.pred_lower AS forecast_lower_5d, p.pred_upper AS forecast_upper_5d,
                p.pred_volatility AS forecast_volatility
            FROM latest_market m
            LEFT JOIN latest_features f ON f.ticker = m.ticker
            LEFT JOIN forecasts p ON p.ticker = m.ticker
            ORDER BY m.ticker
            """
        )
    ).mappings().all()
    assets = asset_lookup()
    rows = [dict(row) for row in rows if row["ticker"] in assets]
    latest_date = max((row["date"] for row in rows), default=None)
    markets = []
    for row in rows:
        asset = assets[row["ticker"]]
        markets.append(jsonable({**row, "name": asset.name, "sector": asset.sector, "asset_class": asset.asset_class}))
    return {"date": str(latest_date) if latest_date else None, "markets": markets}


@router.get("/markets/{ticker}/history")
def get_market_history(ticker: str = TickerPath, days: int = Query(252, ge=1, le=5000), db: Session = Depends(get_db)):
    ticker = _ticker(ticker)
    rows = db.execute(
        text(
            """
            SELECT date, open, high, low, close, volume, dividend FROM (
                SELECT date, open, high, low, close, volume, dividend FROM market_data
                WHERE ticker = :ticker ORDER BY date DESC LIMIT :days
            ) recent ORDER BY date ASC
            """
        ),
        {"ticker": ticker, "days": days},
    ).mappings().all()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No market data found for ticker {ticker}")
    return {"ticker": ticker, "days": len(rows), "history": [jsonable(dict(row)) for row in rows]}


@router.get("/markets/{ticker}/forecast")
def get_forecast(ticker: str = TickerPath, db: Session = Depends(get_db)):
    ticker = _ticker(ticker)
    rows = db.execute(
        text(
            """
            SELECT DISTINCT ON (p.horizon_days) p.horizon_days, p.as_of_date, p.pred_return, p.pred_lower, p.pred_upper,
                p.interval_level, p.pred_volatility, p.explanation, r.id AS model_id, r.version, r.metrics, r.trained_at
            FROM model_predictions p JOIN model_registry r ON r.id = p.model_id
            WHERE p.ticker = :ticker AND p.kind = 'live' AND r.status = 'active'
            ORDER BY p.horizon_days, p.as_of_date DESC
            """
        ),
        {"ticker": ticker},
    ).mappings().all()
    if not rows:
        raise HTTPException(status_code=404, detail="No live forecasts available yet")
    forecasts = []
    for row in rows:
        metrics = row["metrics"] or {}
        holdout = metrics.get("holdout_since_2020") or {}
        forecasts.append(jsonable({
            "horizon_days": row["horizon_days"], "as_of_date": row["as_of_date"],
            "expected_return": row["pred_return"], "interval": [row["pred_lower"], row["pred_upper"]],
            "interval_level": row["interval_level"], "volatility_annualized": row["pred_volatility"],
            "explanation": row["explanation"],
            "model": {"id": row["model_id"], "version": row["version"], "trained_at": row["trained_at"],
                      "holdout_ic": holdout.get("ic_mean"), "holdout_interval_coverage": holdout.get("interval_coverage"),
                      "holdout_directional_accuracy": holdout.get("directional_accuracy")},
        }))
    asset = asset_lookup()[ticker]
    return {"ticker": ticker, "name": asset.name, "sector": asset.sector, "forecasts": forecasts,
            "disclaimer": "Model forecasts are uncertain statistical estimates, not guarantees or advice."}


@router.get("/news")
def get_news(ticker: str | None = Query(None, pattern=r"^[A-Za-z]{1,6}$"), limit: int = Query(30, ge=1, le=100),
             db: Session = Depends(get_db)):
    params: dict = {"limit": limit}
    where = "WHERE ticker IS NOT NULL"
    if ticker:
        params["ticker"] = _ticker(ticker)
        where += " AND ticker = :ticker"
    rows = db.execute(
        text(
            f"""
            SELECT uuid, ticker, title, description, url, source, published_at, finbert_score, entity_sentiment
            FROM financial_news {where} ORDER BY published_at DESC NULLS LAST LIMIT :limit
            """
        ),
        params,
    ).mappings().all()
    return {"articles": [jsonable(dict(row)) for row in rows]}
