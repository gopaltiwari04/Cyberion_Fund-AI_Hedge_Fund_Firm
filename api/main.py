import csv
import math
from collections import defaultdict
from pathlib import Path as FilePath

import joblib
from fastapi import Depends, FastAPI, HTTPException, Path, Query
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

DATABASE_URL = "postgresql://quant_user:quant_password@localhost:5432/quant_db"

engine = create_engine(DATABASE_URL)

PROJECT_ROOT = FilePath(__file__).resolve().parent.parent
MODEL_ARTIFACT_PATH = PROJECT_ROOT / "ml_core" / "models" / "xgboost_return_model.joblib"
OOS_PREDICTIONS_PATH = PROJECT_ROOT / "ml_core" / "models" / "xgboost_oos_predictions.csv"

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

app = FastAPI(title="Quant Platform API")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected"
        }
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/portfolio")
def get_portfolio(db: Session = Depends(get_db)):
    try:
        query = text("""
            SELECT
                date,
                strategy_name,
                ticker,
                weight,
                expected_return,
                risk_contribution
            FROM portfolio_allocations
            WHERE date = (
                SELECT MAX(date)
                FROM portfolio_allocations
            )
            ORDER BY weight DESC
        """)

        result = db.execute(query)

        rows = result.mappings().all()

        if not rows:
            raise HTTPException(
                status_code=404,
                detail="No portfolio allocations found"
            )

        return {
            "date": str(rows[0]["date"]),
            "strategy": rows[0]["strategy_name"],
            "allocations": [
                {
                    "ticker": row["ticker"],
                    "weight": row["weight"],
                    "expected_return": row["expected_return"],
                    "risk_contribution": row["risk_contribution"]
                }
                for row in rows
            ]
        }

    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/risk")
def get_risk_metrics(db: Session = Depends(get_db)):
    try:
        query = text("""
            SELECT
                date,
                strategy_name,
                expected_annual_return,
                expected_annual_volatility,
                sharpe_ratio,
                var_95,
                cvar_95
            FROM portfolio_risk_metrics
            WHERE date = (
                SELECT MAX(date)
                FROM portfolio_risk_metrics
            )
            ORDER BY date DESC
            LIMIT 1
        """)

        result = db.execute(query)
        row = result.mappings().first()

        if not row:
            raise HTTPException(
                status_code=404,
                detail="No portfolio risk metrics found"
            )

        return {
            "date": str(row["date"]),
            "strategy": row["strategy_name"],
            "expected_annual_return": row["expected_annual_return"],
            "expected_annual_volatility": row["expected_annual_volatility"],
            "sharpe_ratio": row["sharpe_ratio"],
            "var_95": row["var_95"],
            "cvar_95": row["cvar_95"]
        }

    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/markets")
def get_markets(db: Session = Depends(get_db)):
    try:
        query = text("""
            WITH latest_market AS (
                SELECT DISTINCT ON (ticker)
                    ticker,
                    date,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM market_data
                ORDER BY ticker, date DESC
            ), latest_features AS (
                SELECT DISTINCT ON (ticker)
                    ticker,
                    date AS feature_date,
                    return_1d,
                    return_5d,
                    rsi_14,
                    macd,
                    volatility_20d,
                    regime,
                    relative_return_5d,
                    beta_60d,
                    correlation_spy_60d
                FROM feature_store
                ORDER BY ticker, date DESC
            )
            SELECT
                market.ticker,
                market.date,
                market.open,
                market.high,
                market.low,
                market.close,
                market.volume,
                features.return_1d,
                features.return_5d,
                features.rsi_14,
                features.macd,
                features.volatility_20d,
                features.regime,
                features.relative_return_5d,
                features.beta_60d,
                features.correlation_spy_60d
            FROM latest_market AS market
            LEFT JOIN latest_features AS features
                ON features.ticker = market.ticker
            ORDER BY market.ticker
        """)
        rows = db.execute(query).mappings().all()

        latest_date = max((row["date"] for row in rows), default=None)

        return {
            "date": str(latest_date) if latest_date else None,
            "markets": [
                {
                    "ticker": row["ticker"],
                    "date": str(row["date"]),
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                    "return_1d": row["return_1d"],
                    "return_5d": row["return_5d"],
                    "rsi_14": row["rsi_14"],
                    "macd": row["macd"],
                    "volatility_20d": row["volatility_20d"],
                    "regime": row["regime"],
                    "relative_return_5d": row["relative_return_5d"],
                    "beta_60d": row["beta_60d"],
                    "correlation_spy_60d": row["correlation_spy_60d"],
                }
                for row in rows
            ],
        }
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/markets/{ticker}/history")
def get_market_history(
    ticker: str = Path(..., min_length=1, max_length=16),
    days: int = Query(252, ge=1, le=5000),
    db: Session = Depends(get_db),
):
    try:
        query = text("""
            SELECT date, open, high, low, close, volume
            FROM (
                SELECT date, open, high, low, close, volume
                FROM market_data
                WHERE ticker = :ticker
                ORDER BY date DESC
                LIMIT :days
            ) AS recent_history
            ORDER BY date ASC
        """)
        rows = db.execute(
            query,
            {"ticker": ticker.upper(), "days": days},
        ).mappings().all()

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"No market data found for ticker {ticker.upper()}",
            )

        return {
            "ticker": ticker.upper(),
            "days": len(rows),
            "history": [
                {
                    "date": str(row["date"]),
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                }
                for row in rows
            ],
        }
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))


def load_research_artifact():
    if not MODEL_ARTIFACT_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact is unavailable")

    try:
        artifact = joblib.load(MODEL_ARTIFACT_PATH)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Model artifact could not be read: {error}") from error

    if not isinstance(artifact, dict):
        raise HTTPException(status_code=500, detail="Model artifact has an unsupported format")

    return artifact


def load_oos_predictions():
    if not OOS_PREDICTIONS_PATH.exists():
        raise HTTPException(status_code=503, detail="OOS predictions artifact is unavailable")

    try:
        with OOS_PREDICTIONS_PATH.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            columns = reader.fieldnames or []
            rows = list(reader)
    except (OSError, csv.Error) as error:
        raise HTTPException(status_code=500, detail=f"OOS predictions could not be read: {error}") from error

    return columns, rows


def parse_prediction_value(value: str | None):
    if value is None or value == "":
        return None

    try:
        return float(value)
    except ValueError:
        return value


def find_column(columns: list[str], names: tuple[str, ...]):
    normalized = {column.lower(): column for column in columns}
    for name in names:
        if name in normalized:
            return normalized[name]
    return None


def get_research_dataset_counts(db: Session, tickers: set[str]):
    if not tickers:
        return None

    try:
        rows = db.execute(
            text("""
                SELECT ticker, date, return_1d
                FROM feature_store
                WHERE ticker = ANY(:tickers)
                ORDER BY date, ticker
            """),
            {"tickers": list(tickers)},
        ).mappings().all()
    except SQLAlchemyError:
        return None

    by_ticker = defaultdict(list)
    for row in rows:
        by_ticker[row["ticker"]].append(row)

    target_dates = []
    for sequence in by_ticker.values():
        for index in range(len(sequence)):
            values = [item["return_1d"] for item in sequence[index:index + 5]]
            if len(values) == 5 and all(value is not None and math.isfinite(value) for value in values):
                target_dates.append(sequence[index]["date"])

    if not target_dates:
        return {
            "rows": len(rows),
            "target_rows": 0,
            "train_rows": 0,
            "validation_rows": 0,
            "test_rows": 0,
        }

    dates = sorted(set(target_dates))
    train_end = dates[int(len(dates) * 0.70) - 1]
    validation_end = dates[int(len(dates) * 0.85) - 1]
    return {
        "rows": len(rows),
        "target_rows": len(target_dates),
        "train_rows": sum(date <= train_end for date in target_dates),
        "validation_rows": sum(train_end < date <= validation_end for date in target_dates),
        "test_rows": sum(date > validation_end for date in target_dates),
    }


@app.get("/research/model")
def get_research_model(db: Session = Depends(get_db)):
    artifact = load_research_artifact()
    columns, rows = load_oos_predictions()
    dataset_counts = get_research_dataset_counts(db, {row.get("ticker", "") for row in rows})
    actual_column = find_column(columns, ("actual_return_5d", "actual", "target"))
    predicted_column = find_column(columns, ("predicted_return_5d", "predicted", "prediction"))

    paired_values = []
    if actual_column and predicted_column:
        for row in rows:
            actual = parse_prediction_value(row.get(actual_column))
            predicted = parse_prediction_value(row.get(predicted_column))
            if isinstance(actual, (int, float)) and isinstance(predicted, (int, float)):
                if math.isfinite(actual) and math.isfinite(predicted):
                    paired_values.append((actual, predicted))

    metrics = {"mae": None, "rmse": None, "directional_accuracy": None, "prediction_correlation": None}
    if paired_values:
        errors = [predicted - actual for actual, predicted in paired_values]
        metrics["mae"] = sum(abs(error) for error in errors) / len(errors)
        metrics["rmse"] = math.sqrt(sum(error * error for error in errors) / len(errors))
        metrics["directional_accuracy"] = sum((actual >= 0) == (predicted >= 0) for actual, predicted in paired_values) / len(paired_values)
        actual_mean = sum(actual for actual, _ in paired_values) / len(paired_values)
        predicted_mean = sum(predicted for _, predicted in paired_values) / len(paired_values)
        actual_variance = sum((actual - actual_mean) ** 2 for actual, _ in paired_values)
        predicted_variance = sum((predicted - predicted_mean) ** 2 for _, predicted in paired_values)
        covariance = sum((actual - actual_mean) * (predicted - predicted_mean) for actual, predicted in paired_values)
        metrics["prediction_correlation"] = covariance / math.sqrt(actual_variance * predicted_variance) if actual_variance and predicted_variance else None

    model = artifact.get("model")
    feature_columns = artifact.get("feature_columns")
    return {
        "model": {
            "name": type(model).__name__ if model is not None else None,
            "type": getattr(model, "objective", None),
            "target": artifact.get("target"),
            "horizon_days": artifact.get("horizon_days"),
        },
        "dataset": dataset_counts or {
            "rows": None,
            "target_rows": len(paired_values) or None,
            "train_rows": None,
            "validation_rows": None,
            "test_rows": len(paired_values) or None,
        },
        "metrics": metrics,
        "feature_count": len(feature_columns) if isinstance(feature_columns, list) else None,
    }


@app.get("/research/predictions")
def get_research_predictions(limit: int = Query(100, ge=1, le=5000)):
    columns, rows = load_oos_predictions()
    date_column = find_column(columns, ("date", "timestamp", "datetime"))
    if date_column:
        rows.sort(key=lambda row: row.get(date_column, ""), reverse=True)
    else:
        rows = list(reversed(rows))

    return {
        "available_columns": columns,
        "observations": [
            {column: parse_prediction_value(row.get(column)) for column in columns}
            for row in rows[:limit]
        ],
    }


@app.get("/research/features")
def get_research_features():
    artifact = load_research_artifact()
    feature_columns = artifact.get("feature_columns")
    if not isinstance(feature_columns, list):
        raise HTTPException(status_code=500, detail="Model artifact does not contain feature columns")

    return {
        "feature_count": len(feature_columns),
        "features": feature_columns,
    }