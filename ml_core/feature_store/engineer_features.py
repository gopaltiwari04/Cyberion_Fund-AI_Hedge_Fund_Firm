import os
import json

import numpy as np
import pandas as pd
import redis

from sqlalchemy import create_engine, text
from ta.momentum import RSIIndicator
from ta.trend import MACD


# ============================================================
# CONFIGURATION
# ============================================================

DB_URL = os.getenv(
    "DB_URL",
    "postgresql://quant_user:quant_password@localhost:5432/quant_db",
)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

engine = create_engine(DB_URL)

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
)


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]


# ============================================================
# REDIS CONNECTION TEST
# ============================================================

def test_redis_connection():
    try:
        redis_client.ping()
        print("Redis connection: OK")
    except Exception as e:
        raise RuntimeError(
            f"Could not connect to Redis at "
            f"{REDIS_HOST}:{REDIS_PORT}: {e}"
        )


# ============================================================
# LOAD MARKET DATA
# ============================================================

def load_market_data(ticker):
    print(f"\nLoading market data for {ticker}...")

    query = text("""
        SELECT
            date,
            close,
            volume
        FROM market_data
        WHERE ticker = :ticker
        ORDER BY date ASC
    """)

    df = pd.read_sql(
        query,
        engine,
        params={"ticker": ticker},
    )

    if df.empty:
        print(f"No market data found for {ticker}")
        return df

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    print(f"Loaded {len(df)} rows for {ticker}")

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def calculate_features(df, ticker):
    print(f"Calculating features for {ticker}...")

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    df["return_1d"] = df["close"].pct_change(1)

    df["return_5d"] = df["close"].pct_change(5)

    # --------------------------------------------------------
    # Rolling volatility
    # --------------------------------------------------------

    df["volatility_20d"] = (
        df["return_1d"]
        .rolling(window=20)
        .std()
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi_indicator = RSIIndicator(
        close=df["close"],
        window=14,
    )

    df["rsi_14"] = rsi_indicator.rsi()

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd_indicator = MACD(
        close=df["close"],
        window_slow=26,
        window_fast=12,
        window_sign=9,
    )

    df["macd"] = macd_indicator.macd()

    # --------------------------------------------------------
    # Simple regime detection
    #
    # 0 = bearish
    # 1 = bullish
    #
    # This is deliberately NOT called an HMM yet.
    # We'll implement proper point-in-time regime detection
    # after validating the basic feature pipeline.
    # --------------------------------------------------------

    df["regime"] = np.where(
        df["return_1d"] >= 0,
        1,
        0,
    )

    # --------------------------------------------------------
    # Remove rows where rolling indicators aren't available
    # --------------------------------------------------------

    feature_columns = [
        "return_1d",
        "return_5d",
        "rsi_14",
        "macd",
        "volatility_20d",
        "regime",
    ]

    df = df.dropna(
        subset=feature_columns
    ).copy()

    df["ticker"] = ticker

    return df


# ============================================================
# SAVE FEATURES TO POSTGRESQL
# ============================================================

def save_features(df):
    if df.empty:
        return

    records = df[
        [
            "ticker",
            "date",
            "return_1d",
            "return_5d",
            "rsi_14",
            "macd",
            "volatility_20d",
            "regime",
        ]
    ].copy()

    records["date"] = records["date"].dt.date

    insert_sql = text("""
        INSERT INTO feature_store (
            ticker,
            date,
            return_1d,
            return_5d,
            rsi_14,
            macd,
            volatility_20d,
            regime
        )
        VALUES (
            :ticker,
            :date,
            :return_1d,
            :return_5d,
            :rsi_14,
            :macd,
            :volatility_20d,
            :regime
        )
        ON CONFLICT (ticker, date)
        DO UPDATE SET
            return_1d = EXCLUDED.return_1d,
            return_5d = EXCLUDED.return_5d,
            rsi_14 = EXCLUDED.rsi_14,
            macd = EXCLUDED.macd,
            volatility_20d = EXCLUDED.volatility_20d,
            regime = EXCLUDED.regime
    """)

    records = records.replace(
        {np.nan: None}
    )

    with engine.begin() as connection:
        connection.execute(
            insert_sql,
            records.to_dict(orient="records"),
        )

    print(
        f"Saved {len(records)} feature rows to PostgreSQL"
    )


# ============================================================
# CACHE LATEST FEATURES IN REDIS
# ============================================================

def cache_latest_features(df, ticker):
    if df.empty:
        return

    latest = df.iloc[-1]

    payload = {
        "ticker": ticker,
        "date": str(latest["date"].date()),
        "return_1d": float(latest["return_1d"]),
        "return_5d": float(latest["return_5d"]),
        "rsi_14": float(latest["rsi_14"]),
        "macd": float(latest["macd"]),
        "volatility_20d": float(
            latest["volatility_20d"]
        ),
        "regime": int(latest["regime"]),
    }

    redis_key = f"feature_store:{ticker}:latest"

    redis_client.set(
        redis_key,
        json.dumps(payload),
    )

    print(
        f"Cached latest features in Redis: {redis_key}"
    )


# ============================================================
# PROCESS ONE TICKER
# ============================================================

def process_ticker(ticker):
    df = load_market_data(ticker)

    if df.empty:
        return

    df = calculate_features(
        df,
        ticker,
    )

    print(
        f"Generated {len(df)} feature rows for {ticker}"
    )

    save_features(df)

    cache_latest_features(
        df,
        ticker,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    test_redis_connection()

    for ticker in TICKERS:
        process_ticker(ticker)

    print("\nFeature engineering completed successfully.")