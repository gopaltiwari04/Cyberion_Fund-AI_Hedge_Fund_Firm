"""
Point-in-time external signals merged into feature_store.

Every signal for feature date t only uses information public by the 16:00 ET
close of t:
    * macro (FRED): value from the previous available observation strictly
      before t (FRED publishes day-t values on t+1)
    * news: articles mapped to the trading session in which they became public
    * SEC filings: EDGAR acceptance timestamp mapped the same way
    * graph: rolling correlation network + knowledge graph valid at t
Missing information stays NaN (never silently replaced by 0); XGBoost handles
missing values natively and the dashboard shows "no data".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text

from common.db import get_engine
from common.market_time import effective_trading_dates

MACRO_FEATURES = {
    "DFF": "fed_funds_rate",
    "DGS10": "treasury_10y",
    "T10Y2Y": "term_spread_10y2y",
    "VIXCLS": "vix",
    "BAA10Y": "credit_spread_baa",
}
NEWS_FEATURES = ["news_sentiment_1d", "news_sentiment_7d", "news_count_7d"]
FILING_FEATURES = ["filing_count_30d", "days_since_filing", "filing_sentiment_90d"]
GRAPH_FEATURES = ["corr_degree_60d", "corr_eigenvector_60d", "peer_return_5d", "kg_degree", "kg_pagerank"]
EXTERNAL_FEATURES = list(MACRO_FEATURES.values()) + NEWS_FEATURES + FILING_FEATURES + GRAPH_FEATURES


def _read(sql: str, **params) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


# ----------------------------------------------------------------------
# Macro
# ----------------------------------------------------------------------

def macro_features(dates: pd.Series, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    macro = macro if macro is not None else _read(
        "SELECT series_id, date, value FROM macro_data WHERE series_id = ANY(:ids) AND value IS NOT NULL",
        ids=list(MACRO_FEATURES),
    )
    target = pd.DataFrame({"date": pd.to_datetime(pd.Series(sorted(set(dates))))})
    if macro.empty:
        return target.assign(**{column: np.nan for column in MACRO_FEATURES.values()})
    macro["date"] = pd.to_datetime(macro["date"])
    wide = macro.pivot_table(index="date", columns="series_id", values="value").sort_index()
    # Series have different holiday calendars; carry each one's last known value
    # (bounded so a discontinued series cannot go stale silently).
    wide = wide.ffill(limit=10).rename(columns=MACRO_FEATURES).reset_index()
    merged = pd.merge_asof(target.sort_values("date"), wide, on="date", direction="backward", allow_exact_matches=False)
    for column in MACRO_FEATURES.values():
        if column not in merged:
            merged[column] = np.nan
    return merged[["date", *MACRO_FEATURES.values()]]


# ----------------------------------------------------------------------
# News
# ----------------------------------------------------------------------

def news_features(frame: pd.DataFrame, daily: pd.DataFrame | None = None) -> pd.DataFrame:
    daily = daily if daily is not None else _read(
        "SELECT ticker, date, article_count, finbert_sentiment_mean FROM news_sentiment_daily"
    )
    keys = frame[["ticker", "date"]].copy()
    keys["date"] = pd.to_datetime(keys["date"])
    if daily.empty:
        return keys.assign(**{column: np.nan for column in NEWS_FEATURES})
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["weighted"] = daily["finbert_sentiment_mean"] * daily["article_count"]

    out = []
    for ticker, group in keys.groupby("ticker"):
        events = daily[daily["ticker"] == ticker].set_index("date").sort_index()
        index = pd.DatetimeIndex(sorted(set(group["date"]) | set(events.index)))
        aligned = events.reindex(index)
        count_7d = aligned["article_count"].fillna(0).rolling("7D").sum()
        weighted_7d = aligned["weighted"].fillna(0).rolling("7D").sum()
        result = pd.DataFrame(
            {
                "news_sentiment_1d": aligned["finbert_sentiment_mean"],
                "news_count_7d": count_7d,
                "news_sentiment_7d": (weighted_7d / count_7d).where(count_7d > 0),
            },
            index=index,
        ).reindex(pd.DatetimeIndex(group["date"]))
        result["ticker"] = ticker
        out.append(result.rename_axis("date").reset_index())
    combined = pd.concat(out, ignore_index=True)
    first_news = daily["date"].min()
    # Before news coverage starts, article counts are unknown rather than zero.
    combined.loc[combined["date"] < first_news, "news_count_7d"] = np.nan
    return combined[["ticker", "date", *NEWS_FEATURES]]


# ----------------------------------------------------------------------
# SEC filings
# ----------------------------------------------------------------------

def filing_features(frame: pd.DataFrame, filings: pd.DataFrame | None = None, trading_dates=None) -> pd.DataFrame:
    filings = filings if filings is not None else _read(
        "SELECT ticker, form_type, filing_date, accepted_at, finbert_score FROM sec_filings"
    )
    keys = frame[["ticker", "date"]].copy()
    keys["date"] = pd.to_datetime(keys["date"])
    if filings.empty:
        return keys.assign(**{column: np.nan for column in FILING_FEATURES})

    filings = filings.copy()
    # Without an acceptance time, assume the filing became public after the close.
    fallback = pd.to_datetime(filings["filing_date"]).dt.tz_localize("America/New_York") + pd.Timedelta(hours=17)
    timestamps = pd.to_datetime(filings["accepted_at"], utc=True).fillna(fallback.dt.tz_convert("UTC"))
    filings["effective"] = pd.to_datetime(
        effective_trading_dates(timestamps, trading_dates if trading_dates is not None else keys["date"].dt.date.unique())
    )
    filings = filings.dropna(subset=["effective"])
    history_start = pd.to_datetime(filings["filing_date"]).min()

    out = []
    for ticker, group in keys.groupby("ticker"):
        events = filings[filings["ticker"] == ticker]
        dates = pd.DatetimeIndex(group["date"])
        if events.empty:
            out.append(pd.DataFrame({"ticker": ticker, "date": dates, **{c: np.nan for c in FILING_FEATURES}}))
            continue
        daily = events.groupby("effective").agg(count=("form_type", "size"))
        sentiment = events.dropna(subset=["finbert_score"]).groupby("effective")["finbert_score"].agg(["sum", "count"])
        index = pd.DatetimeIndex(sorted(set(dates) | set(daily.index)))
        counts = daily["count"].reindex(index).fillna(0)
        sent_sum = sentiment["sum"].reindex(index).fillna(0).rolling("90D").sum()
        sent_n = sentiment["count"].reindex(index).fillna(0).rolling("90D").sum()
        last_event = pd.Series(index.where(counts > 0), index=index).ffill()
        result = pd.DataFrame(
            {
                "filing_count_30d": counts.rolling("30D").sum(),
                "days_since_filing": (index - pd.DatetimeIndex(last_event)).days.astype(float),
                "filing_sentiment_90d": (sent_sum / sent_n).where(sent_n > 0),
            },
            index=index,
        ).reindex(dates)
        result["ticker"] = ticker
        out.append(result.rename_axis("date").reset_index())
    combined = pd.concat(out, ignore_index=True)
    combined.loc[combined["date"] < history_start + pd.Timedelta(days=30), FILING_FEATURES] = np.nan
    return combined[["ticker", "date", *FILING_FEATURES]]


# ----------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------

def add_external_signals(df: pd.DataFrame, include_graph: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop(columns=[column for column in EXTERNAL_FEATURES if column in df.columns])

    df = df.merge(macro_features(df["date"]), on="date", how="left")
    df = df.merge(news_features(df), on=["ticker", "date"], how="left")
    df = df.merge(filing_features(df, trading_dates=df["date"].dt.date.unique()), on=["ticker", "date"], how="left")
    if include_graph:
        from ml_core.graph.build_graph import graph_features

        df = df.merge(graph_features(df), on=["ticker", "date"], how="left")
    else:
        for column in GRAPH_FEATURES:
            df[column] = np.nan

    # Legacy columns kept for backwards compatibility with earlier consumers.
    df["sentiment_score"] = df["news_sentiment_7d"]
    df["degree_centrality"] = df["kg_degree"]
    df["pagerank"] = df["kg_pagerank"]
    df["date"] = df["date"].dt.date
    return df
