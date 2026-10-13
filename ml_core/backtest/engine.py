"""
Event-driven historical backtester.

Event loop per trading day d:
    1. OPEN    pending orders (decided at the previous close) fill at d's open
               price adjusted for slippage; commission is charged. Sells are
               processed before buys and buys are scaled down if cash is short.
    2. DIVIDENDS  cash dividends are credited on the ex-date.
    3. INTEREST   cash accrues the lagged effective fed funds rate.
    4. CLOSE   positions are marked to market at the close.
    5. SIGNAL  on rebalance dates the strategy sees only information available
               at d's close (prices <= d, predictions with as_of_date = d made
               by models whose training labels ended before d) and emits target
               weights -> orders for the next open.

Look-ahead safeguards are asserted at runtime (prediction as_of/train_end,
scenario window end). Prices are split-adjusted, so share counts are in
split-adjusted units; dividends are split-adjusted consistently by the source.

Known limitations (reported with every result):
    * survivorship bias - the universe is today's list of large, liquid assets
    * no intraday liquidity / market-impact model beyond proportional slippage
    * fractional shares, no borrow (long-only), no taxes
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text

from common.db import get_engine
from common.universe import BENCHMARK_TICKER, asset_lookup, tickers as universe_tickers
from ml_core.backtest.metrics import drawdown_series, performance_metrics, relative_metrics
from ml_core.portfolio.market import alpha_confidence, load_prices
from ml_core.portfolio.optimizer import CASH, OptimizationInput, optimize_portfolio
from ml_core.portfolio.recommend import MODEL_NAME, constraints_for

EMBARGO_DAYS = 5
STRATEGIES = ("ml_cvar", "min_cvar", "equal_weight", "spy_buy_and_hold")


@dataclass
class BacktestConfig:
    start: date
    end: date
    strategy: str = "ml_cvar"
    initial_capital: float = 1_000_000.0
    horizon_days: int = 5
    rebalance_every: int = 5
    profile: str = "balanced"
    cvar_limit: float | None = None
    max_weight: float | None = None
    max_turnover: float | None = None
    commission_bps: float = 1.0
    slippage_bps: float = 5.0
    scenario_window: int = 504
    tickers: list[str] = field(default_factory=list)
    benchmarks: list[str] = field(default_factory=lambda: ["spy_buy_and_hold", "equal_weight"])

    def validate(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if self.end <= self.start:
            raise ValueError("end must be after start")
        if not 1_000 <= self.initial_capital <= 1e10:
            raise ValueError("initial_capital out of range")
        if self.horizon_days not in (1, 5, 21):
            raise ValueError("horizon_days must be 1, 5 or 21")
        if not 1 <= self.rebalance_every <= 63:
            raise ValueError("rebalance_every must be between 1 and 63 trading days")
        if not 0 <= self.commission_bps <= 100 or not 0 <= self.slippage_bps <= 200:
            raise ValueError("cost assumptions out of range")
        if not 126 <= self.scenario_window <= 1260:
            raise ValueError("scenario_window must be between 126 and 1260 days")
        unknown = set(self.tickers) - set(universe_tickers())
        if unknown:
            raise ValueError(f"Unsupported tickers: {sorted(unknown)}")
        for benchmark in self.benchmarks:
            if benchmark not in STRATEGIES:
                raise ValueError(f"Unknown benchmark {benchmark}")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["start"], data["end"] = str(self.start), str(self.end)
        return data

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


class LookAheadError(AssertionError):
    pass


@dataclass
class MarketData:
    open: pd.DataFrame
    close: pd.DataFrame
    dividend: pd.DataFrame
    returns: pd.DataFrame
    risk_free_daily: pd.Series


def load_market(tickers: list[str], start: date, end: date, warmup_days: int) -> MarketData:
    prices = load_prices(tickers, end=end)
    open_ = prices.pivot(index="date", columns="ticker", values="open").sort_index()
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    dividend = prices.pivot(index="date", columns="ticker", values="dividend").reindex(close.index).fillna(0.0)
    returns = (close + dividend) / close.shift(1) - 1.0
    first = close.index[close.index >= pd.Timestamp(start)]
    if len(first) == 0:
        raise ValueError("No market data in the requested window")
    keep_from = close.index.get_loc(first[0]) - warmup_days
    index = close.index[max(keep_from, 0):]
    with get_engine().connect() as conn:
        dff = pd.read_sql(text("SELECT date, value FROM macro_data WHERE series_id = 'DFF' ORDER BY date"), conn)
    if dff.empty:
        rf = pd.Series(0.0, index=index)
    else:
        dff["date"] = pd.to_datetime(dff["date"])
        # Lag one observation: the rate for day d is the last value published before d.
        rf_series = dff.set_index("date")["value"].shift(1)
        rf = rf_series.reindex(index.union(rf_series.index)).ffill().reindex(index).fillna(0.0) / 100 / 252
    return MarketData(open_.reindex(index), close.reindex(index), dividend.reindex(index).fillna(0.0),
                      returns.reindex(index), rf)


def load_predictions(model_id: int, horizon: int, tickers: list[str]) -> pd.DataFrame:
    with get_engine().connect() as conn:
        frame = pd.read_sql(
            text(
                """
                SELECT ticker, as_of_date, pred_return, train_end FROM model_predictions
                WHERE model_id = :m AND kind = 'walk_forward' AND horizon_days = :h AND ticker = ANY(:t)
                """
            ),
            conn,
            params={"m": model_id, "h": horizon, "t": tickers},
        )
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"])
    frame["train_end"] = pd.to_datetime(frame["train_end"])
    return frame


def active_model(horizon: int) -> dict:
    with get_engine().connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT id, version, metrics FROM model_registry
                WHERE model_name = :n AND horizon_days = :h AND status = 'active' ORDER BY trained_at DESC LIMIT 1
                """
            ),
            {"n": MODEL_NAME, "h": horizon},
        ).mappings().one_or_none()
    if row is None:
        raise ValueError(f"No active model for horizon {horizon}")
    return dict(row)


class Portfolio:
    def __init__(self, cash: float) -> None:
        self.cash = cash
        self.shares: dict[str, float] = {}
        self.costs_paid = 0.0

    def value(self, prices: pd.Series) -> float:
        return self.cash + sum(qty * prices[t] for t, qty in self.shares.items() if qty)

    def weights(self, prices: pd.Series) -> dict[str, float]:
        total = self.value(prices)
        return {t: qty * prices[t] / total for t, qty in self.shares.items() if qty} if total > 0 else {}


def target_weights(strategy: str, day: pd.Timestamp, market: MarketData, tradable: list[str], config: BacktestConfig,
                   predictions: pd.DataFrame | None, current: dict[str, float], confidence: float,
                   first_rebalance: bool) -> tuple[dict[str, float] | None, dict]:
    if strategy == "spy_buy_and_hold":
        return ({BENCHMARK_TICKER: 1.0} if first_rebalance else None), {}
    if strategy == "equal_weight":
        return {t: 1.0 / len(tradable) for t in tradable}, {}

    window = market.returns.loc[:day, tradable].dropna().tail(config.scenario_window)
    if window.index.max() > day:
        raise LookAheadError("scenario window extends past decision date")
    if len(window) < 126:
        return None, {"skipped": "insufficient scenario history"}

    expected = np.zeros(len(tradable))
    if strategy == "ml_cvar":
        todays = predictions[predictions["as_of_date"] == day].set_index("ticker")
        if todays.empty or not set(tradable) <= set(todays.index):
            return None, {"skipped": "no walk-forward predictions for this date"}
        todays = todays.loc[tradable]
        horizon_offset = pd.tseries.offsets.BDay(config.horizon_days + EMBARGO_DAYS - 1)
        if (todays["train_end"] + horizon_offset > day).any():
            raise LookAheadError(f"prediction on {day.date()} uses a model trained on unrealised labels")
        expected = todays["pred_return"].to_numpy(dtype=float)

    assets = asset_lookup()
    constraints = constraints_for(config.profile, cvar_limit=config.cvar_limit, max_weight=config.max_weight,
                                  max_turnover=None if first_rebalance else config.max_turnover,
                                  transaction_cost_bps=config.commission_bps + config.slippage_bps)
    result = optimize_portfolio(
        OptimizationInput(
            tickers=tradable,
            expected_returns=expected,
            horizon_days=config.horizon_days,
            scenarios=window.to_numpy(),
            sectors=[assets[t].sector for t in tradable],
            current_weights=None if first_rebalance else np.array([current.get(t, 0.0) for t in tradable]),
            risk_free_daily=float(market.risk_free_daily.loc[day]),
            alpha_confidence=confidence if strategy == "ml_cvar" else 0.0,
        ),
        constraints,
    )
    info = {"status": result.status, "binding": result.binding_constraints,
            "cvar_95_1d": (result.metrics or {}).get("cvar_95_1d")}
    if not result.feasible:
        info["message"] = result.message
        return None, info  # keep current holdings; never trade into a limit breach
    return {t: w for t, w in result.weights.items() if t != CASH}, info


def run_strategy(strategy: str, config: BacktestConfig, market: MarketData, tradable_all: list[str],
                 predictions: pd.DataFrame | None, confidence: float) -> dict:
    days = market.close.index[(market.close.index >= pd.Timestamp(config.start)) & (market.close.index <= pd.Timestamp(config.end))]
    if strategy == "ml_cvar" and predictions is not None and not predictions.empty:
        days = days[days >= predictions["as_of_date"].min()]
    if len(days) < 30:
        raise ValueError("Backtest window too short or outside the period covered by out-of-sample predictions")

    portfolio = Portfolio(config.initial_capital)
    pending: dict[str, float] | None = None
    equity, trades, rebalances = {}, [], []
    slippage = config.slippage_bps / 1e4
    commission = config.commission_bps / 1e4
    first = True

    for position, day in enumerate(days):
        opens, closes = market.open.loc[day], market.close.loc[day]

        # 1. OPEN: fill orders decided at the previous close.
        if pending is not None:
            value_at_open = portfolio.value(opens.fillna(closes))
            orders = []
            for ticker in sorted(set(pending) | set(portfolio.shares)):
                price = opens.get(ticker)
                if price is None or not np.isfinite(price):
                    continue
                target_shares = pending.get(ticker, 0.0) * value_at_open / price
                delta = target_shares - portfolio.shares.get(ticker, 0.0)
                if abs(delta * price) > value_at_open * 1e-4:
                    orders.append((ticker, delta, price))
            for ticker, delta, price in sorted(orders, key=lambda order: order[1]):  # sells first
                fill = price * (1 + slippage) if delta > 0 else price * (1 - slippage)
                if delta > 0:
                    affordable = portfolio.cash / (fill * (1 + commission))
                    delta = min(delta, max(affordable, 0.0))
                    if delta <= 0:
                        continue
                notional = abs(delta) * fill
                fee = notional * commission
                portfolio.cash -= delta * fill + fee
                portfolio.shares[ticker] = portfolio.shares.get(ticker, 0.0) + delta
                portfolio.costs_paid += fee + abs(delta) * price * slippage
                trades.append({"date": str(day.date()), "ticker": ticker, "side": "buy" if delta > 0 else "sell",
                               "shares": float(abs(delta)), "price": float(fill), "notional": float(notional),
                               "commission": float(fee)})
            pending = None

        # 2. DIVIDENDS (ex-date) and 3. INTEREST on cash
        for ticker, qty in portfolio.shares.items():
            amount = market.dividend.loc[day].get(ticker, 0.0)
            if qty and amount:
                portfolio.cash += qty * amount
        if position > 0 and portfolio.cash > 0:
            portfolio.cash *= 1 + float(market.risk_free_daily.loc[day])

        # 4. CLOSE: mark to market
        equity[day] = portfolio.value(closes)

        # 5. SIGNAL on rebalance dates (never on the final day: no next open to trade)
        if position % config.rebalance_every == 0 and position < len(days) - 1:
            tradable = [t for t in tradable_all if np.isfinite(closes.get(t, np.nan))]
            current = portfolio.weights(closes)
            weights, info = target_weights(strategy, day, market, tradable, config, predictions, current, confidence, first)
            if weights is not None:
                pending = weights
                first = False
                rebalances.append({"date": str(day.date()), "weights": {t: round(w, 4) for t, w in weights.items() if w > 1e-4},
                                   "turnover": float(sum(abs(weights.get(t, 0) - current.get(t, 0)) for t in set(weights) | set(current))),
                                   **info})
            elif info:
                rebalances.append({"date": str(day.date()), "weights": None, **info})

    curve = pd.Series(equity)
    executed = [r for r in rebalances if r.get("weights")]
    years = max(len(curve) / 252, 1e-9)
    return {
        "equity": curve,
        "trades": trades,
        "rebalances": rebalances,
        "costs_paid": portfolio.costs_paid,
        "turnover_annualized": float(sum(r["turnover"] for r in executed) / years),
        "rebalances_executed": len(executed),
        "rebalances_skipped": len(rebalances) - len(executed),
    }


def run_backtest(config: BacktestConfig) -> dict:
    config.validate()
    tickers = config.tickers or universe_tickers()
    load_list = sorted(set(tickers) | {BENCHMARK_TICKER})
    market = load_market(load_list, config.start, config.end, warmup_days=config.scenario_window + 5)

    model, predictions, confidence = None, None, 0.0
    if config.strategy == "ml_cvar":
        model = active_model(config.horizon_days)
        predictions = load_predictions(model["id"], config.horizon_days, tickers)
        if predictions.empty:
            raise ValueError("The active model has no walk-forward predictions to backtest")
        confidence = alpha_confidence(model["metrics"])

    runs = {config.strategy: run_strategy(config.strategy, config, market, tickers, predictions, confidence)}
    window_start = runs[config.strategy]["equity"].index[0]
    bench_config = BacktestConfig(**{**config.__dict__, "start": window_start.date()})
    for benchmark in config.benchmarks:
        if benchmark != config.strategy:
            runs[benchmark] = run_strategy(benchmark, bench_config, market, tickers, None, 0.0)

    primary = runs[config.strategy]["equity"]
    rf = market.risk_free_daily
    results = {}
    for name, run in runs.items():
        stats = performance_metrics(run["equity"], rf)
        stats.update({
            "turnover_annualized": run["turnover_annualized"],
            "costs_paid": run["costs_paid"],
            "costs_pct_of_initial": run["costs_paid"] / config.initial_capital,
            "trades": len(run["trades"]),
            "rebalances_executed": run["rebalances_executed"],
            "rebalances_skipped": run["rebalances_skipped"],
        })
        if name != config.strategy:
            stats["vs_strategy"] = relative_metrics(primary, run["equity"])
        results[name] = stats

    def curve(series: pd.Series, decimals: int = 2) -> list[dict]:
        sampled = series if len(series) <= 1500 else series.iloc[:: int(np.ceil(len(series) / 1500))]
        return [{"date": str(d.date()), "value": round(float(v), decimals)} for d, v in sampled.items()]

    return {
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "model": {"id": model["id"], "version": model["version"], "alpha_confidence": confidence} if model else None,
        "data_snapshot": {"market_data_end": str(market.close.index.max().date()), "tickers": tickers},
        "metrics": results,
        "relative_to_benchmarks": {name: results[name].get("vs_strategy") for name in results if name != config.strategy},
        "equity_curves": {name: curve(run["equity"]) for name, run in runs.items()},
        "drawdown": curve(drawdown_series(primary), decimals=5),
        "trades": runs[config.strategy]["trades"][-2000:],
        "trades_total": len(runs[config.strategy]["trades"]),
        "rebalances": runs[config.strategy]["rebalances"][-500:],
        "assumptions": {
            "execution": "orders decided at close t fill at open t+1",
            "slippage_bps": config.slippage_bps,
            "commission_bps": config.commission_bps,
            "dividends": "credited in cash on ex-date",
            "cash_rate": "effective fed funds rate, lagged one day",
            "predictions": "walk-forward out-of-sample forecasts; each from a model trained on labels realised before the decision date",
            "risk_model": f"{config.scenario_window}-day historical scenarios, Ledoit-Wolf covariance, CVaR hard limit",
            "infeasible_rebalance": "holdings kept unchanged when no allocation satisfies the CVaR limit",
        },
        "known_biases": [
            "Survivorship bias: the universe is a fixed list of assets that exist today.",
            "Proportional slippage only; no volume-dependent market impact or borrow constraints.",
            "Fractional shares and no taxes.",
        ],
    }
