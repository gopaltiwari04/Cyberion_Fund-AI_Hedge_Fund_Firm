"""Performance and risk statistics for backtests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown_duration(equity: pd.Series) -> int:
    underwater = drawdown_series(equity) < 0
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


def performance_metrics(equity: pd.Series, risk_free_daily: pd.Series | None = None) -> dict:
    equity = equity.dropna()
    returns = equity.pct_change().dropna()
    if len(returns) < 2:
        return {}
    rf = (risk_free_daily.reindex(returns.index).fillna(0.0) if risk_free_daily is not None
          else pd.Series(0.0, index=returns.index))
    excess = returns - rf
    years = len(returns) / TRADING_DAYS
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else None
    vol = float(returns.std() * math.sqrt(TRADING_DAYS))
    downside = excess[excess < 0]
    downside_dev = float(np.sqrt(np.mean(np.minimum(excess, 0) ** 2)) * math.sqrt(TRADING_DAYS))
    losses = -returns
    var95 = float(np.quantile(losses, 0.95))
    cvar95 = float(losses[losses >= var95].mean())
    drawdown = drawdown_series(equity)
    monthly = (1 + returns).groupby(returns.index.to_period("M")).prod() - 1
    return {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "years": round(years, 2),
        "total_return": total,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": float(excess.mean() / excess.std() * math.sqrt(TRADING_DAYS)) if excess.std() > 0 else None,
        "sortino": float(excess.mean() * TRADING_DAYS / downside_dev) if downside_dev > 0 and len(downside) else None,
        "max_drawdown": float(drawdown.min()),
        "max_drawdown_duration_days": max_drawdown_duration(equity),
        "calmar": float(cagr / abs(drawdown.min())) if cagr is not None and drawdown.min() < 0 else None,
        "var_95_1d": var95,
        "cvar_95_1d": cvar95,
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
        "positive_months": float((monthly > 0).mean()) if len(monthly) else None,
    }


def relative_metrics(strategy: pd.Series, benchmark: pd.Series) -> dict:
    joined = pd.concat([strategy.pct_change(), benchmark.pct_change()], axis=1, keys=["s", "b"]).dropna()
    if len(joined) < 20:
        return {}
    beta = float(np.cov(joined["s"], joined["b"])[0, 1] / np.var(joined["b"], ddof=1))
    alpha_daily = float(joined["s"].mean() - beta * joined["b"].mean())
    active = joined["s"] - joined["b"]
    tracking = float(active.std() * math.sqrt(TRADING_DAYS))
    return {
        "beta": beta,
        "alpha_annualized": alpha_daily * TRADING_DAYS,
        "tracking_error": tracking,
        "information_ratio": float(active.mean() * TRADING_DAYS / tracking) if tracking > 0 else None,
        "correlation": float(joined["s"].corr(joined["b"])),
    }
