"""
Investable universe.

Sector / asset-class labels are static reference data used for exposure
limits. SEC CIK numbers are resolved from the SEC ``company_tickers.json``
mapping at ingestion time rather than hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.config import get_settings

CASH_TICKER = "CASH"
BENCHMARK_TICKER = "SPY"


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    asset_class: str  # equity | etf
    sector: str
    files_with_sec: bool


UNIVERSE: tuple[Asset, ...] = (
    Asset("AAPL", "Apple Inc.", "equity", "Information Technology", True),
    Asset("MSFT", "Microsoft Corp.", "equity", "Information Technology", True),
    Asset("NVDA", "NVIDIA Corp.", "equity", "Information Technology", True),
    Asset("GOOGL", "Alphabet Inc. (Class A)", "equity", "Communication Services", True),
    Asset("META", "Meta Platforms Inc.", "equity", "Communication Services", True),
    Asset("AMZN", "Amazon.com Inc.", "equity", "Consumer Discretionary", True),
    Asset("JPM", "JPMorgan Chase & Co.", "equity", "Financials", True),
    Asset("XOM", "Exxon Mobil Corp.", "equity", "Energy", True),
    Asset("JNJ", "Johnson & Johnson", "equity", "Health Care", True),
    Asset("SPY", "SPDR S&P 500 ETF Trust", "etf", "Broad Market Equity", False),
    Asset("TLT", "iShares 20+ Year Treasury Bond ETF", "etf", "Fixed Income", False),
    Asset("GLD", "SPDR Gold Shares", "etf", "Commodities", False),
)


def get_universe() -> list[Asset]:
    override = get_settings().universe_override
    if not override:
        return list(UNIVERSE)
    by_ticker = {asset.ticker: asset for asset in UNIVERSE}
    unknown = [ticker for ticker in override if ticker not in by_ticker]
    if unknown:
        raise ValueError(f"UNIVERSE_TICKERS contains unsupported tickers: {unknown}")
    if BENCHMARK_TICKER not in override:
        override = (*override, BENCHMARK_TICKER)
    return [by_ticker[ticker] for ticker in override]


def tickers() -> list[str]:
    return [asset.ticker for asset in get_universe()]


def sec_filers() -> list[str]:
    return [asset.ticker for asset in get_universe() if asset.files_with_sec]


def asset_lookup() -> dict[str, Asset]:
    return {asset.ticker: asset for asset in get_universe()}
