"""Validated request models (all user input enters through these)."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from common.universe import tickers as universe_tickers

TICKER_PATTERN = r"^[A-Z]{1,6}$"
RiskProfile = Literal["conservative", "balanced", "aggressive"]


def _check_ticker(value: str) -> str:
    value = value.strip().upper()
    if value not in universe_tickers():
        raise ValueError(f"{value} is not in the supported universe")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ProfileUpdate(StrictModel):
    display_name: str | None = Field(default=None, max_length=120)
    risk_profile: RiskProfile = "balanced"
    cvar_limit: float = Field(0.025, ge=0.005, le=0.10, description="Max 1-day 95% CVaR (fraction of value)")
    max_weight: float = Field(0.30, ge=0.05, le=1.0)
    max_turnover: float = Field(0.50, ge=0.0, le=2.0)
    transaction_cost_bps: float = Field(5.0, ge=0.0, le=100.0)
    cash_balance: float = Field(0.0, ge=0.0, le=1e10)


class Holding(StrictModel):
    ticker: str = Field(pattern=TICKER_PATTERN)
    quantity: float = Field(ge=0, le=1e9)
    average_cost: float | None = Field(default=None, ge=0, le=1e7)

    @field_validator("ticker", mode="before")
    @classmethod
    def upper(cls, value: str) -> str:
        return _check_ticker(str(value))


class HoldingsUpdate(StrictModel):
    holdings: list[Holding] = Field(max_length=50)
    cash_balance: float | None = Field(default=None, ge=0, le=1e10)

    @model_validator(mode="after")
    def unique(self) -> "HoldingsUpdate":
        names = [holding.ticker for holding in self.holdings]
        if len(names) != len(set(names)):
            raise ValueError("Each ticker may appear only once")
        return self


class RecommendationRequest(StrictModel):
    horizon_days: Literal[1, 5, 21] = 5
    risk_profile: RiskProfile | None = None
    cvar_limit: float | None = Field(default=None, ge=0.005, le=0.10)
    max_weight: float | None = Field(default=None, ge=0.05, le=1.0)
    max_turnover: float | None = Field(default=None, ge=0.0, le=2.0)
    transaction_cost_bps: float | None = Field(default=None, ge=0.0, le=100.0)


class AnalystQuestion(StrictModel):
    question: str = Field(min_length=5, max_length=1000)
    tickers: list[str] | None = Field(default=None, max_length=5)
    form_types: list[Literal["10-K", "10-Q", "8-K"]] | None = None
    date_from: date | None = None
    date_to: date | None = None

    @field_validator("tickers")
    @classmethod
    def valid_tickers(cls, value: list[str] | None) -> list[str] | None:
        return [_check_ticker(ticker) for ticker in value] if value else None


class BacktestRequest(StrictModel):
    start: date
    end: date
    strategy: Literal["ml_cvar", "min_cvar", "equal_weight", "spy_buy_and_hold"] = "ml_cvar"
    initial_capital: float = Field(1_000_000, ge=1_000, le=1e9)
    horizon_days: Literal[1, 5, 21] = 5
    rebalance_every: int = Field(5, ge=1, le=63)
    risk_profile: RiskProfile = "balanced"
    cvar_limit: float | None = Field(default=None, ge=0.005, le=0.10)
    max_weight: float | None = Field(default=None, ge=0.05, le=1.0)
    max_turnover: float | None = Field(default=None, ge=0.0, le=2.0)
    commission_bps: float = Field(1.0, ge=0, le=100)
    slippage_bps: float = Field(5.0, ge=0, le=200)
    tickers: list[str] | None = Field(default=None, max_length=20)

    @field_validator("tickers")
    @classmethod
    def valid_tickers(cls, value: list[str] | None) -> list[str] | None:
        if not value:
            return None
        cleaned = [_check_ticker(ticker) for ticker in value]
        if len(cleaned) < 3:
            raise ValueError("Select at least three assets")
        return sorted(set(cleaned))

    @model_validator(mode="after")
    def dates(self) -> "BacktestRequest":
        if self.end <= self.start:
            raise ValueError("end must be after start")
        if (self.end - self.start).days < 90:
            raise ValueError("Backtest window must be at least 90 days")
        return self
