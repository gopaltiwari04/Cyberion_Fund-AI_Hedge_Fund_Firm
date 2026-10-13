"""Human-readable metadata for model features (used in explanations and the dashboard)."""

FEATURE_CATALOG: dict[str, tuple[str, str]] = {
    "return_1d": ("1-day return", "momentum"),
    "return_5d": ("5-day return", "momentum"),
    "return_10d": ("10-day return", "momentum"),
    "return_20d": ("20-day return", "momentum"),
    "return_60d": ("60-day return", "momentum"),
    "rsi_14": ("RSI (14)", "technical"),
    "macd": ("MACD", "technical"),
    "volatility_5d": ("5-day volatility", "risk"),
    "volatility_20d": ("20-day volatility", "risk"),
    "volatility_60d": ("60-day volatility", "risk"),
    "atr_14_pct": ("ATR (14) % of price", "risk"),
    "sma_20_distance": ("Distance from 20-day SMA", "trend"),
    "sma_50_distance": ("Distance from 50-day SMA", "trend"),
    "sma_200_distance": ("Distance from 200-day SMA", "trend"),
    "drawdown_20d": ("20-day drawdown", "risk"),
    "drawdown_60d": ("60-day drawdown", "risk"),
    "volume_change_5d": ("5-day volume change", "volume"),
    "volume_zscore_20d": ("Volume z-score (20d)", "volume"),
    "market_return_5d": ("S&P 500 5-day return", "market"),
    "relative_return_5d": ("5-day return vs S&P 500", "market"),
    "beta_60d": ("60-day beta to S&P 500", "market"),
    "correlation_spy_60d": ("60-day correlation to S&P 500", "market"),
    "intraday_range": ("Intraday range", "technical"),
    "close_position": ("Close position in daily range", "technical"),
    "gap_return": ("Overnight gap", "technical"),
    "regime": ("High-volatility regime", "risk"),
    "fed_funds_rate": ("Fed funds rate", "macro"),
    "treasury_10y": ("10-year Treasury yield", "macro"),
    "term_spread_10y2y": ("10y-2y term spread", "macro"),
    "vix": ("VIX", "macro"),
    "credit_spread_baa": ("Baa credit spread", "macro"),
    "news_sentiment_1d": ("News sentiment (FinBERT, today)", "sentiment"),
    "news_sentiment_7d": ("News sentiment (FinBERT, 7d)", "sentiment"),
    "news_count_7d": ("News volume (7d)", "sentiment"),
    "filing_count_30d": ("SEC filings in last 30d", "filings"),
    "days_since_filing": ("Days since last SEC filing", "filings"),
    "filing_sentiment_90d": ("8-K sentiment (FinBERT, 90d)", "filings"),
    "corr_degree_60d": ("Correlation-network degree", "graph"),
    "corr_eigenvector_60d": ("Correlation-network centrality", "graph"),
    "peer_return_5d": ("Peer 5-day return (graph)", "graph"),
    "kg_degree": ("Knowledge-graph degree", "graph"),
    "kg_pagerank": ("Knowledge-graph PageRank", "graph"),
    "cs_rank_return_20d": ("20-day return rank in universe", "cross-section"),
    "cs_rank_return_60d": ("60-day return rank in universe", "cross-section"),
    "cs_rank_volatility_20d": ("Volatility rank in universe", "cross-section"),
    "cs_rank_rsi_14": ("RSI rank in universe", "cross-section"),
}

SIGNAL_FEATURES = [
    "news_sentiment_1d", "news_sentiment_7d", "news_count_7d", "filing_sentiment_90d", "filing_count_30d",
    "kg_degree", "kg_pagerank", "corr_eigenvector_60d", "peer_return_5d",
]


def describe(feature: str) -> dict:
    label, group = FEATURE_CATALOG.get(feature, (feature, "other"))
    return {"feature": feature, "label": label, "group": group}
