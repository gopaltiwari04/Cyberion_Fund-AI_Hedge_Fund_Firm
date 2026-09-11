from fastapi import Depends, FastAPI, HTTPException, Path, Query
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

DATABASE_URL = "postgresql://quant_user:quant_password@localhost:5432/quant_db"

engine = create_engine(DATABASE_URL)

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