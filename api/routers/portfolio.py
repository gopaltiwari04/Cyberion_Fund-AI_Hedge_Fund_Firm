"""House model portfolio (read-only)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.deps import get_db, jsonable
from api.security import optional_user

router = APIRouter(tags=["portfolio"], dependencies=[Depends(optional_user)])
PREFERRED_STRATEGY = "xgb_cvar_v2"


def _strategy(db: Session, table: str) -> str:
    # Legacy strategies (e.g. xgboost_mean_variance_v1) stored annualised-by-compounding
    # forecasts and no CVaR; their rows are kept for history but not served.
    return PREFERRED_STRATEGY


@router.get("/portfolio")
def get_portfolio(db: Session = Depends(get_db)):
    strategy = _strategy(db, "portfolio_allocations")
    rows = db.execute(
        text(
            """
            SELECT date, strategy_name, ticker, weight, expected_return, risk_contribution, horizon_days, predicted_volatility
            FROM portfolio_allocations
            WHERE strategy_name = :s AND date = (SELECT MAX(date) FROM portfolio_allocations WHERE strategy_name = :s)
            ORDER BY weight DESC
            """
        ),
        {"s": strategy},
    ).mappings().all()
    if not rows:
        raise HTTPException(status_code=404, detail="No portfolio allocations found")
    return {
        "date": str(rows[0]["date"]),
        "strategy": rows[0]["strategy_name"],
        "horizon_days": rows[0]["horizon_days"],
        "allocations": [jsonable({k: row[k] for k in ("ticker", "weight", "expected_return", "risk_contribution",
                                                      "predicted_volatility")}) for row in rows],
    }


@router.get("/risk")
def get_risk_metrics(db: Session = Depends(get_db)):
    strategy = _strategy(db, "portfolio_risk_metrics")
    row = db.execute(
        text(
            """
            SELECT * FROM portfolio_risk_metrics WHERE strategy_name = :s ORDER BY date DESC, id DESC LIMIT 1
            """
        ),
        {"s": strategy},
    ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="No portfolio risk metrics found")
    diagnostics = row.get("diagnostics") or {}
    return jsonable({
        "date": str(row["date"]),
        "strategy": row["strategy_name"],
        "expected_annual_return": row["expected_annual_return"],
        "expected_annual_volatility": row["expected_annual_volatility"],
        "sharpe_ratio": row["sharpe_ratio"],
        "var_95": row["var_95"],
        "cvar_95": row["cvar_95"],
        "cvar_limit": row.get("cvar_limit"),
        "feasible": row.get("feasible"),
        "solver_status": row.get("solver_status"),
        "turnover": row.get("turnover"),
        "transaction_cost": row.get("transaction_cost"),
        "horizon_days": row.get("horizon_days"),
        "message": diagnostics.get("message"),
        "binding_constraints": diagnostics.get("binding_constraints"),
        "alpha_confidence": diagnostics.get("alpha_confidence"),
        "metrics": diagnostics.get("metrics"),
        "scenarios": diagnostics.get("scenarios"),
        "assumptions": row.get("assumptions"),
    })


@router.get("/portfolio/history")
def portfolio_history(db: Session = Depends(get_db)):
    rows = db.execute(
        text(
            """
            SELECT date, expected_annual_return, expected_annual_volatility, var_95, cvar_95, cvar_limit, turnover, feasible
            FROM portfolio_risk_metrics WHERE strategy_name = :s ORDER BY date
            """
        ),
        {"s": PREFERRED_STRATEGY},
    ).mappings().all()
    return {"strategy": PREFERRED_STRATEGY, "history": [jsonable(dict(row)) for row in rows]}
