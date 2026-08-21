import json
import os

import numpy as np
import pandas as pd
import redis
from sqlalchemy import create_engine, text
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
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

def calculate_point_in_time_regimes(returns, min_observations=500):
    """
    Calculate point-in-time market regimes.

    At date t, the model is fitted only using observations
    available up to t. No future returns are used.

    Returns:
        pandas.Series containing:
        0 = bearish/low-return regime
        1 = bullish/high-return regime
    """

    regimes = pd.Series(
        index=returns.index,
        dtype="Int64"
    )

    valid_returns = returns.dropna()

    if len(valid_returns) < min_observations:
        print(
            f"Not enough observations for regime model "
            f"({len(valid_returns)} < {min_observations})."
        )
        return regimes.fillna(0).astype(int)

    # Start assigning regimes once sufficient history exists.
    #
    # To keep runtime reasonable, we refit periodically rather
    # than fitting a new Markov model on every single day.
    last_regime = 0
    model_result = None

    for i in range(min_observations, len(valid_returns)):

        # Refit every 20 observations.
        # This prevents thousands of expensive model fits.
        if model_result is None or i % 20 == 0:

            training_data = valid_returns.iloc[:i]

            try:
                model = MarkovRegression(
                    training_data,
                    k_regimes=2,
                    trend="c",
                    switching_variance=True,
                )

                model_result = model.fit(
                    disp=False
                )

            except Exception as e:  # noqa: BLE001
                print(
                    f"Regime model fitting failed at "
                    f"{valid_returns.index[i].date()}: {e}"
                )
                model_result = None

        if model_result is None:
            regimes.loc[
                valid_returns.index[i]
            ] = last_regime
            continue

        try:
            # IMPORTANT:
            # Use only probabilities generated from the
            # information available through the training window.
            latest_probabilities = (
                model_result.filtered_marginal_probabilities
            )

            # Determine which model state has the higher estimated
            # mean return. Markov regime labels themselves are arbitrary.
            regime_means = model_result.params[
                [
                    "const[0]",
                    "const[1]",
                ]
            ]

            if regime_means.iloc[0] < regime_means.iloc[1]:
                bullish_regime = 1
            else:
                bullish_regime = 0

            raw_regime = int(
                latest_probabilities.iloc[-1].idxmax()
            )

            regime = (
                1
                if raw_regime == bullish_regime
                else 0
            )

            regimes.loc[
                valid_returns.index[i]
            ] = regime

            last_regime = regime

        except Exception:  # noqa: BLE001
            regimes.loc[
                valid_returns.index[i]
            ] = last_regime

    return regimes.fillna(0).astype(int)

def test_redis_connection():
    try:
        redis_client.ping()
        print("Redis connection: OK")
    except Exception as e:  # noqa: BLE001
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

    df["regime"] = calculate_point_in_time_regimes(
        df["return_1d"]
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