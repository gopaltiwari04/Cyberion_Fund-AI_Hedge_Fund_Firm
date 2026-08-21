import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql://quant_user:quant_password@localhost:5432/quant_db"
engine = create_engine(DB_URL)


FEATURES = [
    "rsi_14",
    "macd",
    "volatility_20d",
    "return_1d",
]


def load_modeling_dataset(ticker="AAPL"):
    """
    Load feature-store data and construct the true
    5-trading-day forward return target.

    The target at date t is:

        Close[t+5] / Close[t] - 1

    Only information available at date t is used by
    the features.
    """

    query = text("""
        SELECT
            f.ticker,
            f.date,
            f.return_1d,
            f.rsi_14,
            f.macd,
            f.volatility_20d,
            m.close
        FROM feature_store f
        JOIN market_data m
            ON f.ticker = m.ticker
            AND f.date = m.date
        WHERE f.ticker = :ticker
        ORDER BY f.date ASC
    """)

    df = pd.read_sql(
        query,
        engine,
        params={"ticker": ticker},
    )

    if df.empty:
        raise ValueError(
            f"No data found for ticker {ticker}"
        )

    df["date"] = pd.to_datetime(df["date"])

    # True 5-trading-day forward return
    df["target_5d"] = (
        df["close"].shift(-5) / df["close"] - 1.0
    )

    # The final 5 observations cannot have a known target.
    df = df.dropna(
        subset=FEATURES + ["target_5d"]
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":

    ticker = "AAPL"

    df = load_modeling_dataset(ticker)

    print("=" * 60)
    print("MODELING DATASET")
    print("=" * 60)

    print(f"Ticker       : {ticker}")
    print(f"Rows         : {len(df)}")
    print(
        f"Date range   : "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    print("\nFeatures:")
    for feature in FEATURES:
        print(f"  - {feature}")

    print("\nTarget:")
    print("  target_5d = Close[t+5] / Close[t] - 1")

    print("\nTarget statistics:")
    print(df["target_5d"].describe())

    print("\nLast 5 rows:")
    print(
        df[
            [
                "date",
                "close",
                "target_5d",
            ]
        ].tail()
    )