import numpy as np
import pytest

from ml_core.portfolio.optimizer import (
    CASH,
    OptimizationConstraints,
    OptimizationInput,
    historical_var_cvar,
    ledoit_wolf,
    optimize_portfolio,
    ru_cvar,
)

TICKERS = ["A", "B", "C", "D"]
SECTORS = ["Tech", "Tech", "Energy", "Bonds"]


def scenarios(seed=0, t=500):
    rng = np.random.default_rng(seed)
    vols = np.array([0.025, 0.02, 0.018, 0.006])
    corr = np.array([[1, .7, .3, -.2], [.7, 1, .3, -.2], [.3, .3, 1, 0], [-.2, -.2, 0, 1]])
    cov = np.outer(vols, vols) * corr
    return rng.multivariate_normal([0.0006, 0.0005, 0.0003, 0.0001], cov, size=t)


def make_input(**kwargs):
    base = dict(tickers=TICKERS, expected_returns=np.array([0.02, 0.01, 0.005, 0.002]), horizon_days=5,
                scenarios=scenarios(), sectors=SECTORS, risk_free_daily=0.0001)
    base.update(kwargs)
    return OptimizationInput(**base)


def portfolio_cvar(weights, data):
    w = np.array([weights[t] for t in TICKERS])
    return ru_cvar(data.scenarios @ w + data.risk_free_daily * weights[CASH])


@pytest.mark.parametrize("limit", [0.004, 0.01, 0.02])
def test_cvar_limit_is_strictly_enforced(limit):
    data = make_input()
    result = optimize_portfolio(data, OptimizationConstraints(cvar_limit=limit, max_weight=0.6, max_sector_weight=1.0))
    assert result.feasible
    assert portfolio_cvar(result.weights, data) <= limit * 1.001 + 1e-6
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(weight >= -1e-9 for weight in result.weights.values())


def test_tight_limit_moves_into_cash_and_bonds():
    data = make_input()
    loose = optimize_portfolio(data, OptimizationConstraints(cvar_limit=0.05, max_weight=0.6))
    tight = optimize_portfolio(data, OptimizationConstraints(cvar_limit=0.003, max_weight=0.6))
    assert tight.weights[CASH] + tight.weights["D"] > loose.weights[CASH] + loose.weights["D"]
    assert "cvar_limit" in tight.binding_constraints


def test_position_and_sector_limits():
    result = optimize_portfolio(make_input(), OptimizationConstraints(max_weight=0.25, max_sector_weight=0.3, cvar_limit=0.05))
    assert max(v for k, v in result.weights.items() if k != CASH) <= 0.25 + 1e-6
    assert result.weights["A"] + result.weights["B"] <= 0.3 + 1e-6


def test_turnover_cap_respected():
    current = np.array([0.0, 0.0, 0.0, 1.0])
    data = make_input(current_weights=current)
    result = optimize_portfolio(data, OptimizationConstraints(max_turnover=0.2, cvar_limit=0.05, max_weight=1.0, max_sector_weight=1.0))
    assert result.feasible
    traded = sum(abs(result.weights[t] - c) for t, c in zip(TICKERS, current))
    assert traded <= 0.2 + 1e-5


def test_infeasible_cvar_under_turnover_returns_no_allocation_and_diagnosis():
    current = np.array([1.0, 0.0, 0.0, 0.0])  # concentrated, risky book
    data = make_input(current_weights=current)
    result = optimize_portfolio(data, OptimizationConstraints(max_turnover=0.05, cvar_limit=0.004, max_weight=1.0, max_sector_weight=1.0))
    assert not result.feasible and result.status == "infeasible"
    assert result.weights is None
    assert result.diagnostics["min_achievable_cvar_95_1d"] > 0.004
    assert result.diagnostics["min_cvar_without_turnover_cap"] <= 0.004 + 1e-6
    assert "turnover" in result.message


def test_infeasible_min_cash_and_cvar_combination():
    data = make_input()
    result = optimize_portfolio(data, OptimizationConstraints(cvar_limit=0.001, max_weight=0.3, max_cash=0.0, max_sector_weight=1.0))
    assert not result.feasible and result.weights is None


def test_zero_alpha_confidence_ignores_forecasts():
    a = optimize_portfolio(make_input(alpha_confidence=0.0), OptimizationConstraints(cvar_limit=0.05))
    b = optimize_portfolio(make_input(alpha_confidence=0.0, expected_returns=np.array([-0.05, 0.08, 0.0, 0.0])),
                           OptimizationConstraints(cvar_limit=0.05))
    for ticker in TICKERS:
        assert a.weights[ticker] == pytest.approx(b.weights[ticker], abs=1e-4)


def test_transaction_costs_reduce_trading():
    current = np.array([0.25, 0.25, 0.25, 0.25])
    cheap = optimize_portfolio(make_input(current_weights=current), OptimizationConstraints(transaction_cost_bps=0, cvar_limit=0.05, max_weight=1, max_sector_weight=1))
    costly = optimize_portfolio(make_input(current_weights=current), OptimizationConstraints(transaction_cost_bps=300, cvar_limit=0.05, max_weight=1, max_sector_weight=1))
    assert costly.metrics["turnover"] < cheap.metrics["turnover"]


def test_input_validation():
    with pytest.raises(ValueError):
        optimize_portfolio(make_input(scenarios=scenarios(t=10)), OptimizationConstraints())
    with pytest.raises(ValueError):
        optimize_portfolio(make_input(), OptimizationConstraints(cvar_limit=0))
    with pytest.raises(ValueError):
        optimize_portfolio(make_input(expected_returns=np.array([np.nan, 0, 0, 0])), OptimizationConstraints())


def test_risk_helpers():
    returns = np.array([-0.05, -0.03, 0.0, 0.01, 0.02] * 20)
    var, cvar = historical_var_cvar(returns, 0.95)
    assert var == pytest.approx(0.05) and cvar >= var
    cov = ledoit_wolf(scenarios())
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    assert np.allclose(cov, cov.T)
