"""
Constrained portfolio optimisation (CVXPY).

Problem (daily units, long-only, weights include a CASH asset that earns the
risk-free rate):

    maximise   a * mu_h / h . w  +  rf * w_cash  -  (lambda / 2) w' S w  -  (c / h) * sum|w - w0|
    subject to sum(w) + w_cash = 1,  0 <= w_i <= max_weight,  min_cash <= w_cash <= max_cash
               sector exposure <= max_sector_weight
               sum|w - w0| <= max_turnover                      (if current weights given)
               CVaR_95(daily loss) <= cvar_limit                (Rockafellar-Uryasev, historical scenarios)

* mu_h: model forecasts of h-day total return; a in [0, 1] shrinks them by
  the model's demonstrated out-of-sample skill (a = 0 -> pure risk model).
* S: Ledoit-Wolf shrinkage covariance of daily total returns.
* c: proportional transaction cost (commission + spread/slippage), amortised
  over the forecast horizon.
* CVaR is enforced as a hard constraint. After solving, the realised scenario
  CVaR is recomputed with numpy and the solution is rejected if it violates
  the limit by more than a numerical tolerance.

Infeasibility handling: if the problem is infeasible the optimiser solves a
minimum-CVaR problem under the remaining constraints to report the lowest
achievable CVaR, checks whether relaxing the turnover cap would make the
limit attainable, and returns NO recommended allocation (status
"infeasible") together with that diagnostic - it never silently returns a
portfolio that breaches the risk limit.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import cvxpy as cp
import numpy as np

from common.serialization import to_builtin

CASH = "CASH"
TOLERANCE = 1e-5


@dataclass
class OptimizationConstraints:
    max_weight: float = 0.30
    min_cash: float = 0.0
    max_cash: float = 1.0
    max_sector_weight: float = 0.60
    max_turnover: float | None = None
    transaction_cost_bps: float = 5.0
    cvar_alpha: float = 0.95
    cvar_limit: float = 0.025
    risk_aversion: float = 5.0

    def validate(self) -> None:
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if not 0 <= self.min_cash <= self.max_cash <= 1:
            raise ValueError("cash bounds must satisfy 0 <= min_cash <= max_cash <= 1")
        if not 0 < self.max_sector_weight <= 1:
            raise ValueError("max_sector_weight must be in (0, 1]")
        if self.max_turnover is not None and not 0 <= self.max_turnover <= 2:
            raise ValueError("max_turnover must be in [0, 2]")
        if not 0 <= self.transaction_cost_bps <= 500:
            raise ValueError("transaction_cost_bps must be in [0, 500]")
        if not 0.5 < self.cvar_alpha < 1:
            raise ValueError("cvar_alpha must be in (0.5, 1)")
        if not 0 < self.cvar_limit <= 0.5:
            raise ValueError("cvar_limit must be in (0, 0.5]")
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion must be >= 0")


@dataclass
class OptimizationInput:
    tickers: list[str]
    expected_returns: np.ndarray  # forecast h-day total return per risky asset
    horizon_days: int
    scenarios: np.ndarray  # T x N historical daily total returns of the risky assets
    sectors: list[str]
    current_weights: np.ndarray | None = None  # risky weights; cash is the remainder
    risk_free_daily: float = 0.0
    alpha_confidence: float = 1.0


@dataclass
class OptimizationResult:
    status: str  # optimal | infeasible | error
    feasible: bool
    message: str
    weights: dict[str, float] | None
    metrics: dict = field(default_factory=dict)
    binding_constraints: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    assumptions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return to_builtin(asdict(self))


# ----------------------------------------------------------------------
# Risk helpers
# ----------------------------------------------------------------------

def ledoit_wolf(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage towards a scaled identity (Ledoit & Wolf, 2004)."""
    x = returns - returns.mean(axis=0)
    t, n = x.shape
    sample = x.T @ x / t
    mu = np.trace(sample) / n
    target = mu * np.eye(n)
    delta = np.sum((sample - target) ** 2) / n
    beta = sum(np.sum((np.outer(row, row) - sample) ** 2) for row in x) / (t ** 2 * n)
    shrinkage = 0.0 if delta == 0 else min(1.0, beta / delta)
    covariance = shrinkage * target + (1 - shrinkage) * sample
    return (covariance + covariance.T) / 2


def historical_var_cvar(portfolio_returns: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    """Empirical VaR / CVaR of losses (positive numbers = loss)."""
    losses = -np.asarray(portfolio_returns)
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    return var, float(tail.mean()) if tail.size else var


def ru_cvar(portfolio_returns: np.ndarray, alpha: float = 0.95) -> float:
    """Rockafellar-Uryasev CVaR (the quantity constrained in the optimiser)."""
    losses = -np.asarray(portfolio_returns)
    var = np.quantile(losses, alpha)
    return float(var + np.mean(np.maximum(losses - var, 0.0)) / (1 - alpha))


# ----------------------------------------------------------------------
# Optimisation
# ----------------------------------------------------------------------

def _solve(problem: cp.Problem) -> str:
    for solver in (cp.CLARABEL, cp.ECOS, cp.SCS):
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver)
        except cp.SolverError:
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
            return problem.status
    return problem.status or "solver_error"


def _build(data: OptimizationInput, constraints: OptimizationConstraints, covariance: np.ndarray,
           objective_kind: str, enforce_cvar: bool, enforce_turnover: bool):
    n = len(data.tickers)
    w = cp.Variable(n, nonneg=True)
    cash = cp.Variable(nonneg=True)
    scenario_count = data.scenarios.shape[0]
    zeta = cp.Variable()
    excess = cp.Variable(scenario_count, nonneg=True)
    portfolio_scenarios = data.scenarios @ w + data.risk_free_daily * cash
    cvar_expr = zeta + cp.sum(excess) / ((1 - constraints.cvar_alpha) * scenario_count)

    cons = [
        cp.sum(w) + cash == 1,
        w <= constraints.max_weight,
        cash >= constraints.min_cash,
        cash <= constraints.max_cash,
        excess >= -portfolio_scenarios - zeta,
    ]
    for sector in sorted(set(data.sectors)):
        members = [i for i, value in enumerate(data.sectors) if value == sector]
        cons.append(cp.sum(w[members]) <= constraints.max_sector_weight)

    current = data.current_weights if data.current_weights is not None else np.zeros(n)
    trades = cp.sum(cp.abs(w - current))
    if enforce_turnover and data.current_weights is not None and constraints.max_turnover is not None:
        cons.append(trades <= constraints.max_turnover)
    if enforce_cvar:
        cons.append(cvar_expr <= constraints.cvar_limit)

    if objective_kind == "min_cvar":
        objective = cp.Minimize(cvar_expr)
    else:
        mu_daily = data.alpha_confidence * np.asarray(data.expected_returns) / data.horizon_days
        cost = constraints.transaction_cost_bps / 1e4 / data.horizon_days
        objective = cp.Maximize(
            mu_daily @ w
            + data.risk_free_daily * cash
            - constraints.risk_aversion / 2 * cp.quad_form(w, cp.psd_wrap(covariance))
            - (cost * trades if data.current_weights is not None else 0)
        )
    return cp.Problem(objective, cons), w, cash


def _metrics(data: OptimizationInput, constraints: OptimizationConstraints, covariance: np.ndarray,
             weights: np.ndarray, cash: float) -> dict:
    h = data.horizon_days
    scenarios = data.scenarios @ weights + data.risk_free_daily * cash
    var_1d, cvar_1d = historical_var_cvar(scenarios, constraints.cvar_alpha)
    daily_vol = float(np.sqrt(max(weights @ covariance @ weights, 0.0)))
    expected_h = float(weights @ data.expected_returns + cash * data.risk_free_daily * h)
    marginal = covariance @ weights
    variance = float(weights @ marginal)
    risk_share = (weights * marginal / variance) if variance > 0 else np.zeros_like(weights)
    current = data.current_weights if data.current_weights is not None else np.zeros_like(weights)
    turnover = float(np.abs(weights - current).sum())
    sectors: dict[str, float] = {}
    for sector, weight in zip(data.sectors, weights):
        sectors[sector] = sectors.get(sector, 0.0) + float(weight)
    sectors["Cash"] = float(cash)
    all_weights = np.append(weights, cash)
    return to_builtin({
        "horizon_days": h,
        "expected_return_horizon": expected_h,
        "expected_return_annualized_linear": expected_h * 252 / h,
        "volatility_daily": daily_vol,
        "volatility_horizon": daily_vol * np.sqrt(h),
        "volatility_annualized": daily_vol * np.sqrt(252),
        "sharpe_annualized_model": (
            (expected_h / h - data.risk_free_daily) * 252 / (daily_vol * np.sqrt(252)) if daily_vol > 0 else None
        ),
        "var_95_1d": var_1d,
        "cvar_95_1d": cvar_1d,
        "cvar_95_1d_ru": ru_cvar(scenarios, constraints.cvar_alpha),
        "cvar_limit_1d": constraints.cvar_limit,
        "parametric_var_95_1d": float(1.645 * daily_vol - (weights @ data.scenarios.mean(axis=0))),
        "worst_scenario_1d": float(-scenarios.min()),
        "turnover": turnover,
        "estimated_transaction_cost": turnover * constraints.transaction_cost_bps / 1e4,
        "effective_number_of_assets": float(1 / np.sum(all_weights ** 2)) if np.sum(all_weights ** 2) > 0 else None,
        "max_position": float(weights.max()) if len(weights) else 0.0,
        "risk_contribution": {t: float(r) for t, r in zip(data.tickers, risk_share)},
        "sector_exposure": sectors,
    })


def optimize_portfolio(data: OptimizationInput, constraints: OptimizationConstraints) -> OptimizationResult:
    constraints.validate()
    n = len(data.tickers)
    if data.scenarios.ndim != 2 or data.scenarios.shape[1] != n or data.scenarios.shape[0] < 60:
        raise ValueError("scenarios must be a T x N matrix with at least 60 observations")
    if not np.all(np.isfinite(data.scenarios)) or not np.all(np.isfinite(data.expected_returns)):
        raise ValueError("scenarios and expected returns must be finite")
    if data.current_weights is not None and (np.any(data.current_weights < -1e-9) or data.current_weights.sum() > 1 + 1e-6):
        raise ValueError("current weights must be non-negative and sum to at most 1")

    covariance = ledoit_wolf(data.scenarios)
    assumptions = {
        "objective": "maximize shrunk expected return - risk penalty - amortized transaction costs",
        "alpha_confidence": data.alpha_confidence,
        "risk_aversion": constraints.risk_aversion,
        "covariance": "Ledoit-Wolf shrinkage of daily total returns",
        "scenarios": f"{data.scenarios.shape[0]} historical daily total-return scenarios (equal weight)",
        "cvar": f"historical {int(constraints.cvar_alpha * 100)}% 1-day CVaR (Rockafellar-Uryasev), hard constraint",
        "transaction_cost_bps": constraints.transaction_cost_bps,
        "cash_rate_daily": data.risk_free_daily,
        "long_only": True,
        "forecast_horizon_days": data.horizon_days,
    }

    started = time.perf_counter()
    problem, w, cash = _build(data, constraints, covariance, "max_utility", True, True)
    status = _solve(problem)
    diagnostics = {"solver_status": status, "solve_seconds": round(time.perf_counter() - started, 3)}

    if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and w.value is not None:
        weights = np.clip(np.asarray(w.value, dtype=float), 0.0, None)
        weights[weights < 1e-6] = 0.0
        cash_weight = max(0.0, 1.0 - weights.sum())
        metrics = _metrics(data, constraints, covariance, weights, cash_weight)
        violations = []
        if metrics["cvar_95_1d_ru"] > constraints.cvar_limit * (1 + 1e-3) + TOLERANCE:
            violations.append(f"CVaR {metrics['cvar_95_1d_ru']:.4%} exceeds limit {constraints.cvar_limit:.4%}")
        if weights.max(initial=0) > constraints.max_weight + 1e-4:
            violations.append("position limit exceeded")
        if (
            data.current_weights is not None
            and constraints.max_turnover is not None
            and metrics["turnover"] > constraints.max_turnover + 1e-4
        ):
            violations.append("turnover limit exceeded")
        if violations:
            return OptimizationResult(
                status="error", feasible=False, weights=None, diagnostics=diagnostics, assumptions=assumptions,
                message="Solver returned a solution that fails post-solve verification: " + "; ".join(violations),
            )
        binding = []
        if metrics["cvar_95_1d_ru"] >= constraints.cvar_limit * 0.995:
            binding.append("cvar_limit")
        if np.any(weights >= constraints.max_weight - 1e-4):
            binding.append("max_weight")
        if data.current_weights is not None and constraints.max_turnover is not None and \
                metrics["turnover"] >= constraints.max_turnover - 1e-4:
            binding.append("max_turnover")
        if cash_weight <= constraints.min_cash + 1e-6 and constraints.min_cash > 0:
            binding.append("min_cash")
        result_weights = {t: float(v) for t, v in zip(data.tickers, weights)}
        result_weights[CASH] = float(cash_weight)
        return OptimizationResult(
            status="optimal", feasible=True, weights=result_weights, metrics=metrics, binding_constraints=binding,
            diagnostics=diagnostics, assumptions=assumptions,
            message="Optimal allocation satisfies all constraints including the CVaR limit.",
        )

    # ---------------- infeasible / failed: diagnose, never breach limits ----------------
    min_problem, mw, mcash = _build(data, constraints, covariance, "min_cvar", False, True)
    min_status = _solve(min_problem)
    min_cvar = float(min_problem.value) if min_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) else None
    diagnostics.update({"min_achievable_cvar_95_1d": min_cvar, "min_cvar_status": min_status})
    if data.current_weights is not None and constraints.max_turnover is not None:
        relaxed, _, _ = _build(data, constraints, covariance, "min_cvar", False, False)
        relaxed_status = _solve(relaxed)
        if relaxed_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            diagnostics["min_cvar_without_turnover_cap"] = float(relaxed.value)
    if min_cvar is not None and mw.value is not None:
        fallback = {t: float(max(v, 0.0)) for t, v in zip(data.tickers, mw.value)}
        fallback[CASH] = float(max(mcash.value, 0.0))
        # Reported for transparency only; it is NOT a recommendation.
        diagnostics["minimum_cvar_portfolio"] = fallback

    message = "No allocation satisfies all constraints."
    if min_cvar is not None and min_cvar > constraints.cvar_limit:
        message = (
            f"The CVaR limit of {constraints.cvar_limit:.2%} cannot be met under the other constraints; "
            f"the lowest achievable 1-day CVaR is {min_cvar:.2%}."
        )
        if diagnostics.get("min_cvar_without_turnover_cap", np.inf) <= constraints.cvar_limit:
            message += " Relaxing the turnover cap would make the limit achievable."
    return OptimizationResult(
        status="infeasible", feasible=False, weights=None, diagnostics=diagnostics, assumptions=assumptions,
        message=message + " No trades are recommended.",
    )
