from datetime import date

import numpy as np
import pandas as pd
import pytest

from ml_core.backtest import engine
from ml_core.backtest.engine import BacktestConfig, LookAheadError, MarketData, run_strategy, target_weights
from ml_core.backtest.metrics import max_drawdown_duration, performance_metrics, relative_metrics

TICKERS = ["AAPL", "MSFT", "SPY"]


def synthetic_market(n=320, seed=1):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2024-01-01", periods=n)
    close = pd.DataFrame({t: 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n)) for t in TICKERS}, index=index)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, (n, len(TICKERS))))
    dividend = pd.DataFrame(0.0, index=index, columns=TICKERS)
    dividend.iloc[200, 0] = 0.5
    returns = (close + dividend) / close.shift(1) - 1
    return MarketData(open_, close, dividend, returns, pd.Series(0.0001, index=index))


def config(**kwargs):
    base = dict(start=date(2024, 1, 1), end=date(2025, 3, 1), strategy="equal_weight", initial_capital=100_000,
                rebalance_every=20, commission_bps=10, slippage_bps=10, tickers=TICKERS, scenario_window=126)
    base.update(kwargs)
    return BacktestConfig(**base)


def test_orders_fill_at_next_open_with_costs():
    market = synthetic_market()
    run = run_strategy("equal_weight", config(), market, TICKERS, None, 0.0)
    first_trade = run["trades"][0]
    first_day = market.close.index[0]
    assert first_trade["date"] == str(market.close.index[1].date())  # decided at close of day 0, filled next open
    expected_fill = market.open.loc[market.close.index[1], first_trade["ticker"]] * 1.001
    assert first_trade["price"] == pytest.approx(expected_fill)
    assert run["equity"].loc[first_day] == pytest.approx(100_000)
    assert run["costs_paid"] > 0


def test_dividends_are_credited():
    market = synthetic_market()
    with_div = run_strategy("equal_weight", config(), market, TICKERS, None, 0.0)
    no_div = MarketData(market.open, market.close, market.dividend * 0, market.returns, market.risk_free_daily)
    without = run_strategy("equal_weight", config(), no_div, TICKERS, None, 0.0)
    assert with_div["equity"].iloc[-1] > without["equity"].iloc[-1]


def test_cash_never_negative_and_equity_consistent():
    market = synthetic_market()
    run = run_strategy("equal_weight", config(rebalance_every=5), market, TICKERS, None, 0.0)
    assert (run["equity"] > 0).all()
    assert run["rebalances_executed"] > 10


def test_buy_and_hold_trades_once():
    market = synthetic_market()
    run = run_strategy("spy_buy_and_hold", config(strategy="spy_buy_and_hold"), market, TICKERS, None, 0.0)
    assert {trade["ticker"] for trade in run["trades"]} == {"SPY"}
    assert len(run["trades"]) == 1


def test_lookahead_prediction_is_rejected(monkeypatch):
    market = synthetic_market()
    day = market.close.index[150]
    bad = pd.DataFrame({"ticker": TICKERS, "as_of_date": [day] * 3, "pred_return": [0.01, 0.0, 0.0],
                        "train_end": [day - pd.Timedelta(days=1)] * 3})
    monkeypatch.setattr(engine, "asset_lookup", lambda: {t: type("A", (), {"sector": "X"})() for t in TICKERS})
    with pytest.raises(LookAheadError):
        target_weights("ml_cvar", day, market, TICKERS, config(strategy="ml_cvar"), bad, {}, 1.0, True)


def test_valid_prediction_is_accepted(monkeypatch):
    market = synthetic_market()
    day = market.close.index[150]
    ok = pd.DataFrame({"ticker": TICKERS, "as_of_date": [day] * 3, "pred_return": [0.01, 0.0, 0.0],
                       "train_end": [market.close.index[130]] * 3})
    monkeypatch.setattr(engine, "asset_lookup", lambda: {t: type("A", (), {"sector": t})() for t in TICKERS})
    weights, info = target_weights("ml_cvar", day, market, TICKERS, config(strategy="ml_cvar", cvar_limit=0.05), ok, {}, 1.0, True)
    assert weights is not None and info["status"] == "optimal"


def test_config_validation():
    with pytest.raises(ValueError):
        config(end=date(2023, 1, 1)).validate()
    with pytest.raises(ValueError):
        config(strategy="magic").validate()
    assert config().config_hash() == config().config_hash()


def test_metrics_on_known_series():
    index = pd.bdate_range("2020-01-01", periods=253)
    equity = pd.Series(np.linspace(100, 110, 253), index=index)
    stats = performance_metrics(equity)
    assert stats["total_return"] == pytest.approx(0.10)
    assert stats["cagr"] == pytest.approx(0.10, rel=0.01)
    assert stats["max_drawdown"] == 0
    dd = pd.Series([100, 90, 95, 100, 80, 120], index=pd.bdate_range("2020-01-01", periods=6))
    assert performance_metrics(dd)["max_drawdown"] == pytest.approx(-0.2)
    assert max_drawdown_duration(dd) == 2
    rel = relative_metrics(equity, equity * 1.0)
    assert rel["beta"] == pytest.approx(1.0)
