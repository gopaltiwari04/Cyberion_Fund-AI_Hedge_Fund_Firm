"""
House model portfolio (daily job).

Runs the shared recommendation engine for a notional portfolio that starts
from yesterday's house allocation (so turnover limits and transaction costs
apply) and stores the result:
    portfolio_allocations   one row per asset incl. CASH, with forecasts
    portfolio_risk_metrics  expected return/vol, VaR/CVaR, turnover, costs,
                            solver status, feasibility, assumptions, diagnostics

The earlier strategy ("xgboost_mean_variance_v1") annualised 5-day forecasts
by compounding them 50x and stored no VaR/CVaR. Its historical rows are left
untouched; this job writes a new strategy name.

Run: python -m ml_core.portfolio.portfolio_optimizer
"""

from __future__ import annotations

import json
from datetime import date

from sqlalchemy import text

from common.db import get_engine
from common.runs import track_run
from ml_core.portfolio.optimizer import CASH
from ml_core.portfolio.recommend import DEFAULT_HORIZON, build_recommendation, constraints_for

STRATEGY_NAME = "xgb_cvar_v2"
PROFILE = "balanced"
HOUSE_MAX_TURNOVER = 0.60


def previous_weights() -> dict[str, float]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT ticker, weight FROM portfolio_allocations
                WHERE strategy_name = :s AND date = (SELECT MAX(date) FROM portfolio_allocations WHERE strategy_name = :s)
                """
            ),
            {"s": STRATEGY_NAME},
        ).all()
    return {row.ticker: float(row.weight) for row in rows}


def save(result: dict, as_of: date) -> None:
    metrics = (result.get("proposal") or {}).get("metrics") or {}
    with get_engine().begin() as conn:
        conn.execute(
            text("DELETE FROM portfolio_allocations WHERE date = :d AND strategy_name = :s"),
            {"d": as_of, "s": STRATEGY_NAME},
        )
        conn.execute(
            text("DELETE FROM portfolio_risk_metrics WHERE date = :d AND strategy_name = :s"),
            {"d": as_of, "s": STRATEGY_NAME},
        )
        if result["feasible"]:
            weights = result["proposal"]["weights"]
            rows = []
            for asset in result["assets"]:
                rows.append(
                    {
                        "d": as_of, "s": STRATEGY_NAME, "t": asset["ticker"], "w": weights[asset["ticker"]],
                        "er": asset["forecast"]["expected_return"], "rc": asset["risk_contribution"],
                        "m": result["model"]["id"], "h": result["model"]["horizon_days"],
                        "v": asset["forecast"]["volatility_annualized"],
                    }
                )
            rows.append({"d": as_of, "s": STRATEGY_NAME, "t": CASH, "w": weights[CASH],
                         "er": result["risk_free"]["daily"] * DEFAULT_HORIZON, "rc": 0.0,
                         "m": result["model"]["id"], "h": DEFAULT_HORIZON, "v": 0.0})
            conn.execute(
                text(
                    """
                    INSERT INTO portfolio_allocations (date, strategy_name, ticker, weight, expected_return,
                        risk_contribution, model_id, horizon_days, predicted_volatility)
                    VALUES (:d, :s, :t, :w, :er, :rc, :m, :h, :v)
                    """
                ),
                rows,
            )
        conn.execute(
            text(
                """
                INSERT INTO portfolio_risk_metrics (date, strategy_name, expected_annual_return, expected_annual_volatility,
                    sharpe_ratio, var_95, cvar_95, model_id, horizon_days, cvar_limit, solver_status, feasible, turnover,
                    transaction_cost, assumptions, diagnostics)
                VALUES (:d, :s, :er, :vol, :sharpe, :var, :cvar, :m, :h, :limit, :status, :feasible, :turnover, :cost,
                    CAST(:assumptions AS jsonb), CAST(:diagnostics AS jsonb))
                """
            ),
            {
                "d": as_of, "s": STRATEGY_NAME,
                "er": metrics.get("expected_return_annualized_linear"),
                "vol": metrics.get("volatility_annualized"),
                "sharpe": metrics.get("sharpe_annualized_model"),
                "var": metrics.get("var_95_1d"),
                "cvar": metrics.get("cvar_95_1d"),
                "m": result["model"]["id"], "h": result["model"]["horizon_days"],
                "limit": result["constraints"]["cvar_limit"],
                "status": result["status"], "feasible": result["feasible"],
                "turnover": metrics.get("turnover"), "cost": metrics.get("estimated_transaction_cost"),
                "assumptions": json.dumps(result["assumptions"]),
                "diagnostics": json.dumps(
                    {**result["diagnostics"], "message": result["message"], "binding_constraints": result["binding_constraints"],
                     "metrics": metrics, "alpha_confidence": result["model"]["alpha_confidence"],
                     "scenarios": result["scenarios"]["proposed"]},
                    default=str,
                ),
            },
        )


def run_optimizer() -> None:
    with track_run("portfolio", "house_portfolio") as ctx:
        previous = previous_weights()
        holdings = {t: w for t, w in previous.items() if t != CASH}
        cash = previous.get(CASH, 1.0 if not previous else 0.0)
        overrides = {"max_turnover": HOUSE_MAX_TURNOVER} if previous else {}
        constraints_for(PROFILE, **overrides)  # validate early
        result = build_recommendation(holdings, cash, PROFILE, DEFAULT_HORIZON, overrides, treat_as_weights=True)
        if result["status"] == "unavailable":
            raise RuntimeError(result["message"])
        as_of = date.fromisoformat(result["as_of_date"])
        save(result, as_of)
        ctx.details.update({"as_of": str(as_of), "status": result["status"], "message": result["message"],
                            "weights": (result.get("proposal") or {}).get("weights")})
        if not result["feasible"]:
            ctx.warn(result["message"])


if __name__ == "__main__":
    run_optimizer()
