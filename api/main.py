from fastapi import Depends, FastAPI, HTTPException
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