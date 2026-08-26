"""
FEATURE ENGINEERING PIPELINE

Builds point-in-time financial features from market_data and stores them
in PostgreSQL feature_store.

Existing features:
    - return_1d
    - return_5d
    - rsi_14
    - macd
    - volatility_20d
    - regime
    - sentiment_score
    - degree_centrality
    - pagerank

New features:
    Momentum:
        - return_10d
        - return_20d
        - return_60d

    Trend:
        - sma_20_distance
        - sma_50_distance
        - sma_200_distance

    Volatility:
        - volatility_5d
        - volatility_60d
        - atr_14_pct

    Drawdown:
        - drawdown_20d
        - drawdown_60d

    Volume:
        - volume_change_5d
        - volume_zscore_20d

    Market-relative:
        - market_return_5d
        - relative_return_5d
        - beta_60d
        - correlation_spy_60d

    OHLC:
        - intraday_range
        - close_position
        - gap_return

Important:
    Features are calculated using information available at or before each
    observation date. No future observations are used to calculate features.
"""

from __future__ import annotations

import json
import math
import os
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import redis
from sqlalchemy import create_engine, text

try:
    import ta
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
except ImportError as exc:
    raise RuntimeError(
        "The 'ta' package is required. Install it with: pip install ta"
    ) from exc


# ============================================================
# CONFIGURATION
# ============================================================

PG_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://quant_user:quant_password@localhost:5432/quant_db",
)

REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0",
)

MARKET_TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "SPY"]

engine = create_engine(PG_URL)

try:
    redis_client = redis.Redis.from_url(
        REDIS_URL,
        decode_responses=True,
    )
    redis_client.ping()
    REDIS_AVAILABLE = True
except Exception as exc:
    print(f"[WARN] Redis unavailable: {exc}")
    redis_client = None
    REDIS_AVAILABLE = False


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value) -> Optional[float]:
    """
    Convert a value to a finite Python float.

    NaN and +/-inf are converted to None so PostgreSQL receives NULL.
    """
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace infinities with NaN and sort chronologically.
    """
    df = df.copy()

    df = df.replace([np.inf, -np.inf], np.nan)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    return df


# ============================================================
# LOAD MARKET DATA
# ============================================================

def load_market_data() -> pd.DataFrame:
    """
    Load all market data required for cross-asset features.
    """
    print("=" * 60)
    print("LOADING MARKET DATA")
    print("=" * 60)

    query = text(
        """
        SELECT
            ticker,
            date,
            open,
            high,
            low,
            close,
            volume
        FROM market_data
        WHERE ticker = ANY(:tickers)
        ORDER BY ticker, date
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={"tickers": MARKET_TICKERS},
        )

    if df.empty:
        raise RuntimeError("No market data found.")

    df = clean_dataframe(df)

    print(f"Rows loaded: {len(df):,}")
    print(f"Assets found: {sorted(df['ticker'].unique().tolist())}")
    print(
        f"Date range: "
        f"{df['date'].min()} -> {df['date'].max()}"
    )

    return df


# ============================================================
# BASIC / MOMENTUM FEATURES
# ============================================================

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add historical return features.

    All calculations use current and historical prices only.
    """
    df = df.copy()

    grouped = df.groupby("ticker", group_keys=False)

    df["return_1d"] = grouped["close"].pct_change(1)
    df["return_5d"] = grouped["close"].pct_change(5)
    df["return_10d"] = grouped["close"].pct_change(10)
    df["return_20d"] = grouped["close"].pct_change(20)
    df["return_60d"] = grouped["close"].pct_change(60)

    return df


# ============================================================
# TREND FEATURES
# ============================================================

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add moving-average distance features.

    Distance is expressed as:

        (close / SMA) - 1

    Therefore:
        +0.05 = price is 5% above SMA
        -0.05 = price is 5% below SMA
    """
    df = df.copy()

    grouped_close = df.groupby("ticker")["close"]

    sma_20 = grouped_close.transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    )

    sma_50 = grouped_close.transform(
        lambda x: x.rolling(50, min_periods=50).mean()
    )

    sma_200 = grouped_close.transform(
        lambda x: x.rolling(200, min_periods=200).mean()
    )

    df["sma_20_distance"] = df["close"] / sma_20 - 1.0
    df["sma_50_distance"] = df["close"] / sma_50 - 1.0
    df["sma_200_distance"] = df["close"] / sma_200 - 1.0

    return df


# ============================================================
# VOLATILITY FEATURES
# ============================================================

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add historical volatility and ATR features.

    Volatility is calculated from daily close-to-close returns.
    """
    df = df.copy()

    grouped_returns = df.groupby("ticker")["return_1d"]

    df["volatility_5d"] = grouped_returns.transform(
        lambda x: x.rolling(5, min_periods=5).std()
    )

    df["volatility_20d"] = grouped_returns.transform(
        lambda x: x.rolling(20, min_periods=20).std()
    )

    df["volatility_60d"] = grouped_returns.transform(
        lambda x: x.rolling(60, min_periods=60).std()
    )

    # ATR percentage.
    # Calculated independently for each ticker.
    atr_values = []

    for ticker, group in df.groupby("ticker", sort=False):
        group = group.copy()

        previous_close = group["close"].shift(1)

        true_range = pd.concat(
            [
                group["high"] - group["low"],
                (group["high"] - previous_close).abs(),
                (group["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr_14 = true_range.rolling(
            14,
            min_periods=14,
        ).mean()

        atr_pct = atr_14 / group["close"]

        temp = pd.DataFrame(
            {
                "index": group.index,
                "atr_14_pct": atr_pct,
            }
        )

        atr_values.append(temp)

    if atr_values:
        atr_df = pd.concat(atr_values).set_index("index")
        df["atr_14_pct"] = atr_df["atr_14_pct"]
    else:
        df["atr_14_pct"] = np.nan

    return df


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate RSI and MACD independently per ticker.
    """
    df = df.copy()

    rsi_values = pd.Series(index=df.index, dtype=float)
    macd_values = pd.Series(index=df.index, dtype=float)

    for ticker, group in df.groupby("ticker", sort=False):
        group = group.sort_values("date")

        close = group["close"]

        rsi_indicator = RSIIndicator(
            close=close,
            window=14,
        )

        macd_indicator = MACD(
            close=close,
            window_fast=12,
            window_slow=26,
            window_sign=9,
        )

        rsi_values.loc[group.index] = rsi_indicator.rsi()
        macd_values.loc[group.index] = macd_indicator.macd()

    df["rsi_14"] = rsi_values
    df["macd"] = macd_values

    return df


# ============================================================
# DRAWDOWN FEATURES
# ============================================================

def add_drawdown_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate drawdown from rolling historical highs.

    Example:
        close = 95
        rolling high = 100

        drawdown = -5%
    """
    df = df.copy()

    grouped_close = df.groupby("ticker")["close"]

    rolling_high_20 = grouped_close.transform(
        lambda x: x.rolling(
            20,
            min_periods=1,
        ).max()
    )

    rolling_high_60 = grouped_close.transform(
        lambda x: x.rolling(
            60,
            min_periods=1,
        ).max()
    )

    df["drawdown_20d"] = (
        df["close"] / rolling_high_20 - 1.0
    )

    df["drawdown_60d"] = (
        df["close"] / rolling_high_60 - 1.0
    )

    return df


# ============================================================
# VOLUME FEATURES
# ============================================================

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume momentum and volume z-score features.
    """
    df = df.copy()

    grouped_volume = df.groupby("ticker")["volume"]

    volume_mean_20 = grouped_volume.transform(
        lambda x: x.rolling(
            20,
            min_periods=20,
        ).mean()
    )

    volume_std_20 = grouped_volume.transform(
        lambda x: x.rolling(
            20,
            min_periods=20,
        ).std()
    )

    df["volume_change_5d"] = grouped_volume.transform(
        lambda x: x.pct_change(5)
    )

    df["volume_zscore_20d"] = (
        (df["volume"] - volume_mean_20)
        / volume_std_20.replace(0, np.nan)
    )

    return df


# ============================================================
# OHLC FEATURES
# ============================================================

def add_ohlc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price-action features.
    """
    df = df.copy()

    previous_close = df.groupby("ticker")["close"].shift(1)

    # Intraday high-low range relative to close.
    df["intraday_range"] = (
        (df["high"] - df["low"]) / df["close"]
    )

    # Where the close sits inside today's range.
    #
    # 0 = close at low
    # 1 = close at high
    #
    # Values outside [0,1] are theoretically impossible for valid OHLC.
    day_range = (df["high"] - df["low"]).replace(0, np.nan)

    df["close_position"] = (
        (df["close"] - df["low"]) / day_range
    )

    # Overnight gap relative to previous close.
    df["gap_return"] = (
        df["open"] / previous_close - 1.0
    )

    return df


# ============================================================
# MARKET-RELATIVE FEATURES
# ============================================================

def add_market_relative_features(
    df: pd.DataFrame,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """
    Add features describing each asset relative to SPY.

    Features:
        market_return_5d
        relative_return_5d
        beta_60d
        correlation_spy_60d

    The benchmark itself receives:
        relative_return_5d = 0
        beta_60d = 1
        correlation_spy_60d = 1
    """
    df = df.copy()

    benchmark_df = (
        df[df["ticker"] == benchmark]
        [["date", "return_1d", "return_5d"]]
        .copy()
    )

    if benchmark_df.empty:
        raise RuntimeError(
            f"Benchmark ticker '{benchmark}' not found in market_data."
        )

    benchmark_df = benchmark_df.rename(
        columns={
            "return_1d": "market_return_1d",
            "return_5d": "market_return_5d",
        }
    )

    df = df.merge(
        benchmark_df,
        on="date",
        how="left",
    )

    # Market-relative momentum.
    df["relative_return_5d"] = (
        df["return_5d"] - df["market_return_5d"]
    )

    # Calculate rolling beta and correlation independently.
    beta_values = pd.Series(index=df.index, dtype=float)
    correlation_values = pd.Series(index=df.index, dtype=float)

    for ticker, group in df.groupby("ticker", sort=False):
        group = group.sort_values("date")

        asset_returns = group["return_1d"]
        market_returns = group["market_return_1d"]

        rolling_cov = (
            asset_returns
            .rolling(60, min_periods=60)
            .cov(market_returns)
        )

        rolling_market_var = (
            market_returns
            .rolling(60, min_periods=60)
            .var()
        )

        beta = (
            rolling_cov
            / rolling_market_var.replace(0, np.nan)
        )

        correlation = (
            asset_returns
            .rolling(60, min_periods=60)
            .corr(market_returns)
        )

        beta_values.loc[group.index] = beta
        correlation_values.loc[group.index] = correlation

    df["beta_60d"] = beta_values
    df["correlation_spy_60d"] = correlation_values

    # For the benchmark itself, make the economic interpretation explicit.
    benchmark_mask = df["ticker"] == benchmark

    df.loc[benchmark_mask, "relative_return_5d"] = 0.0
    df.loc[benchmark_mask, "beta_60d"] = 1.0
    df.loc[benchmark_mask, "correlation_spy_60d"] = 1.0

    return df


# ============================================================
# POINT-IN-TIME REGIME
# ============================================================

def add_regime_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a simple point-in-time volatility regime.

    Regime:
        0 = normal / lower volatility
        1 = elevated volatility

    The threshold is calculated using only historical volatility
    observations available up to the current date.

    This avoids using future volatility information.
    """
    df = df.copy()

    regime_values = pd.Series(index=df.index, dtype=float)

    for ticker, group in df.groupby("ticker", sort=False):
        group = group.sort_values("date")

        volatility = group["volatility_20d"]

        # Expanding median uses only observations available so far.
        historical_median = (
            volatility
            .expanding(min_periods=20)
            .median()
        )

        regime = (
            volatility > historical_median
        ).astype(float)

        regime_values.loc[group.index] = regime

    df["regime"] = regime_values

    return df


# ============================================================
# SENTIMENT + GRAPH FEATURES
# ============================================================

def load_existing_daily_features(
    dates: List[date],
    tickers: List[str],
) -> pd.DataFrame:
    """
    Load existing sentiment and graph features from feature_store.

    This allows the new feature engine to preserve already-computed
    NLP and knowledge-graph features.

    Existing values are not fabricated here.

    Missing sentiment is filled with 0.0 because the current project
    convention uses zero as the neutral/default sentiment value.

    Graph values are left nullable when no graph value exists.
    """
    if not dates or not tickers:
        return pd.DataFrame(
            columns=[
                "ticker",
                "date",
                "sentiment_score",
                "degree_centrality",
                "pagerank",
            ]
        )

    query = text(
        """
        SELECT
            ticker,
            date,
            sentiment_score,
            degree_centrality,
            pagerank
        FROM feature_store
        WHERE ticker = ANY(:tickers)
          AND date = ANY(:dates)
        """
    )

    with engine.connect() as conn:
        existing = pd.read_sql(
            query,
            conn,
            params={
                "tickers": tickers,
                "dates": dates,
            },
        )

    if existing.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "date",
                "sentiment_score",
                "degree_centrality",
                "pagerank",
            ]
        )

    existing["date"] = pd.to_datetime(
        existing["date"]
    ).dt.date

    return existing


def merge_existing_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preserve NLP and graph features already present in feature_store.
    """
    df = df.copy()

    dates = sorted(df["date"].dropna().unique().tolist())
    tickers = sorted(df["ticker"].dropna().unique().tolist())

    existing = load_existing_daily_features(
        dates=dates,
        tickers=tickers,
    )

    if existing.empty:
        df["sentiment_score"] = 0.0
        df["degree_centrality"] = np.nan
        df["pagerank"] = np.nan
        return df

    # Remove feature columns from df if they happen to exist.
    for column in [
        "sentiment_score",
        "degree_centrality",
        "pagerank",
    ]:
        if column in df.columns:
            df = df.drop(columns=[column])

    df = df.merge(
        existing,
        on=["ticker", "date"],
        how="left",
    )

    # Current project convention: neutral sentiment = 0.
    df["sentiment_score"] = (
        df["sentiment_score"]
        .fillna(0.0)
    )

    return df


# ============================================================
# COMPLETE FEATURE CALCULATION
# ============================================================

def calculate_features(
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the complete feature set.

    Cross-sectional calculations are performed only after all tickers
    have been loaded, allowing SPY-relative features to be calculated
    correctly.
    """
    print()
    print("=" * 60)
    print("CALCULATING FEATURES")
    print("=" * 60)

    df = market_df.copy()

    df = clean_dataframe(df)

    print("[1/9] Momentum features...")
    df = add_momentum_features(df)

    print("[2/9] Trend features...")
    df = add_trend_features(df)

    print("[3/9] Volatility features...")
    df = add_volatility_features(df)

    print("[4/9] Technical indicators...")
    df = add_technical_features(df)

    print("[5/9] Drawdown features...")
    df = add_drawdown_features(df)

    print("[6/9] Volume features...")
    df = add_volume_features(df)

    print("[7/9] OHLC price-action features...")
    df = add_ohlc_features(df)

    print("[8/9] Market-relative features...")
    df = add_market_relative_features(df)

    print("[9/9] Point-in-time regime...")
    df = add_regime_feature(df)

    print("Loading existing NLP / graph features...")
    df = merge_existing_features(df)

    df = clean_dataframe(df)

    return df


# ============================================================
# FEATURE VALIDATION
# ============================================================

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "rsi_14",
    "macd",
    "volatility_5d",
    "volatility_20d",
    "volatility_60d",
    "atr_14_pct",
    "sma_20_distance",
    "sma_50_distance",
    "sma_200_distance",
    "drawdown_20d",
    "drawdown_60d",
    "volume_change_5d",
    "volume_zscore_20d",
    "market_return_5d",
    "relative_return_5d",
    "beta_60d",
    "correlation_spy_60d",
    "intraday_range",
    "close_position",
    "gap_return",
    "regime",
    "sentiment_score",
    "degree_centrality",
    "pagerank",
]


def validate_features(
    df: pd.DataFrame,
) -> None:
    """
    Validate feature values before database insertion.
    """
    print()
    print("=" * 60)
    print("VALIDATING FEATURES")
    print("=" * 60)

    if df.empty:
        raise RuntimeError("Feature dataframe is empty.")

    # Duplicate ticker/date check.
    duplicates = df.duplicated(
        subset=["ticker", "date"]
    ).sum()

    print(f"Duplicate ticker/date rows: {duplicates}")

    if duplicates:
        raise RuntimeError(
            f"Found {duplicates} duplicate ticker/date rows."
        )

    # Check infinite values.
    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    infinite_count = np.isinf(
        df[numeric_columns].to_numpy(
            dtype=float,
            na_value=np.nan,
        )
    ).sum()

    print(f"Infinite numeric values: {infinite_count}")

    if infinite_count:
        raise RuntimeError(
            "Infinite feature values detected."
        )

    # Report missing values.
    print()
    print("MISSING VALUES")
    print("-" * 60)

    missing = df[FEATURE_COLUMNS].isna().sum()

    for column, count in missing.items():
        if count > 0:
            percentage = (
                count / len(df) * 100
            )

            print(
                f"{column:<25} "
                f"{count:>6} "
                f"({percentage:>6.2f}%)"
            )

    print()
    print(
        f"Total rows: {len(df):,}"
    )

    print(
        f"Tickers: "
        f"{sorted(df['ticker'].unique().tolist())}"
    )

    print(
        f"Date range: "
        f"{df['date'].min()} -> {df['date'].max()}"
    )


# ============================================================
# DATABASE WRITE
# ============================================================

def save_features(
    df: pd.DataFrame,
) -> None:
    """
    Upsert calculated features into feature_store.

    NOTE:
        This function expects the database schema to contain the new
        feature columns.

        We intentionally do not create an Alembic migration here.
    """
    print()
    print("=" * 60)
    print("SAVING FEATURES TO POSTGRESQL")
    print("=" * 60)

    insert_sql = text(
        """
        INSERT INTO feature_store (
            ticker,
            date,

            return_1d,
            return_5d,
            return_10d,
            return_20d,
            return_60d,

            rsi_14,
            macd,

            volatility_5d,
            volatility_20d,
            volatility_60d,
            atr_14_pct,

            sma_20_distance,
            sma_50_distance,
            sma_200_distance,

            drawdown_20d,
            drawdown_60d,

            volume_change_5d,
            volume_zscore_20d,

            market_return_5d,
            relative_return_5d,
            beta_60d,
            correlation_spy_60d,

            intraday_range,
            close_position,
            gap_return,

            regime,
            sentiment_score,

            degree_centrality,
            pagerank
        )
        VALUES (
            :ticker,
            :date,

            :return_1d,
            :return_5d,
            :return_10d,
            :return_20d,
            :return_60d,

            :rsi_14,
            :macd,

            :volatility_5d,
            :volatility_20d,
            :volatility_60d,
            :atr_14_pct,

            :sma_20_distance,
            :sma_50_distance,
            :sma_200_distance,

            :drawdown_20d,
            :drawdown_60d,

            :volume_change_5d,
            :volume_zscore_20d,

            :market_return_5d,
            :relative_return_5d,
            :beta_60d,
            :correlation_spy_60d,

            :intraday_range,
            :close_position,
            :gap_return,

            :regime,
            :sentiment_score,

            :degree_centrality,
            :pagerank
        )
        ON CONFLICT (ticker, date)
        DO UPDATE SET
            return_1d = EXCLUDED.return_1d,
            return_5d = EXCLUDED.return_5d,
            return_10d = EXCLUDED.return_10d,
            return_20d = EXCLUDED.return_20d,
            return_60d = EXCLUDED.return_60d,

            rsi_14 = EXCLUDED.rsi_14,
            macd = EXCLUDED.macd,

            volatility_5d = EXCLUDED.volatility_5d,
            volatility_20d = EXCLUDED.volatility_20d,
            volatility_60d = EXCLUDED.volatility_60d,
            atr_14_pct = EXCLUDED.atr_14_pct,

            sma_20_distance = EXCLUDED.sma_20_distance,
            sma_50_distance = EXCLUDED.sma_50_distance,
            sma_200_distance = EXCLUDED.sma_200_distance,

            drawdown_20d = EXCLUDED.drawdown_20d,
            drawdown_60d = EXCLUDED.drawdown_60d,

            volume_change_5d = EXCLUDED.volume_change_5d,
            volume_zscore_20d = EXCLUDED.volume_zscore_20d,

            market_return_5d = EXCLUDED.market_return_5d,
            relative_return_5d = EXCLUDED.relative_return_5d,
            beta_60d = EXCLUDED.beta_60d,
            correlation_spy_60d = EXCLUDED.correlation_spy_60d,

            intraday_range = EXCLUDED.intraday_range,
            close_position = EXCLUDED.close_position,
            gap_return = EXCLUDED.gap_return,

            regime = EXCLUDED.regime,
            sentiment_score = EXCLUDED.sentiment_score,

            degree_centrality = EXCLUDED.degree_centrality,
            pagerank = EXCLUDED.pagerank
        """
    )

    records = []

    for _, row in df.iterrows():
        record = {
            "ticker": row["ticker"],
            "date": row["date"],
        }

        for column in FEATURE_COLUMNS:
            record[column] = safe_float(
                row.get(column)
            )

        records.append(record)

    batch_size = 500

    with engine.begin() as conn:
        for start in range(
            0,
            len(records),
            batch_size,
        ):
            batch = records[
                start:start + batch_size
            ]

            conn.execute(
                insert_sql,
                batch,
            )

            print(
                f"Saved rows: "
                f"{min(start + batch_size, len(records)):,}"
                f"/{len(records):,}"
            )

    print(
        f"Feature rows saved: {len(records):,}"
    )


# ============================================================
# REDIS CACHE
# ============================================================

def update_redis_cache(
    df: pd.DataFrame,
) -> None:
    """
    Update latest feature values in Redis.

    Redis is a cache only. PostgreSQL remains the source of truth.
    """
    if not REDIS_AVAILABLE:
        print("[WARN] Redis unavailable. Skipping cache update.")
        return

    print()
    print("=" * 60)
    print("UPDATING REDIS FEATURE CACHE")
    print("=" * 60)

    latest = (
        df.sort_values("date")
        .groupby("ticker")
        .tail(1)
    )

    updated = 0

    for _, row in latest.iterrows():
        ticker = row["ticker"]

        payload = {
            "ticker": ticker,
            "date": str(row["date"]),
        }

        for column in FEATURE_COLUMNS:
            value = safe_float(
                row.get(column)
            )

            payload[column] = value

        key = f"feature_store:{ticker}:latest"

        try:
            redis_client.set(
                key,
                json.dumps(payload),
            )

            updated += 1

            print(
                f"[OK] {ticker} -> "
                f"feature cache updated"
            )

        except Exception as exc:
            print(
                f"[WARN] Redis update failed for "
                f"{ticker}: {exc}"
            )

    print(
        f"Redis cache entries updated: {updated}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_feature_summary(
    df: pd.DataFrame,
) -> None:
    """
    Print a concise feature summary.
    """
    print()
    print("=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Tickers: "
        f"{', '.join(sorted(df['ticker'].unique()))}"
    )

    print(
        f"Date range: "
        f"{df['date'].min()} -> {df['date'].max()}"
    )

    print()
    print(
        f"Feature count: {len(FEATURE_COLUMNS)}"
    )

    print()
    print("LATEST OBSERVATIONS")
    print("-" * 60)

    display_columns = [
        "ticker",
        "date",
        "return_5d",
        "return_20d",
        "rsi_14",
        "volatility_20d",
        "sma_50_distance",
        "drawdown_20d",
        "relative_return_5d",
        "beta_60d",
        "correlation_spy_60d",
        "regime",
    ]

    latest = (
        df.sort_values("date")
        .groupby("ticker")
        .tail(1)
    )

    print(
        latest[display_columns]
        .to_string(index=False)
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline() -> None:
    print()
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    print()

    market_df = load_market_data()

    features_df = calculate_features(
        market_df
    )

    validate_features(
        features_df
    )

    print_feature_summary(
        features_df
    )

    save_features(
        features_df
    )

    update_redis_cache(
        features_df
    )

    print()
    print("=" * 60)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_pipeline()