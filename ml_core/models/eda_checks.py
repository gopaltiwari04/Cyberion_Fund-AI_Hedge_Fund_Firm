import pandas as pd
from sqlalchemy import create_engine, text
from statsmodels.tsa.stattools import adfuller


DB_URL = "postgresql://quant_user:quant_password@localhost:5432/quant_db"

engine = create_engine(DB_URL)


def adf_test(series, name):
    series = series.dropna()

    result = adfuller(series)

    statistic = result[0]
    p_value = result[1]

    print(f"\n{name}")
    print("-" * 50)
    print(f"Observations : {len(series)}")
    print(f"ADF statistic: {statistic:.6f}")
    print(f"p-value      : {p_value:.6f}")

    if p_value < 0.05:
        print("Conclusion   : Stationary (reject H0)")
    else:
        print("Conclusion   : Non-stationary (fail to reject H0)")


def check_stationarity(ticker="AAPL"):

    query = text("""
        SELECT
            m.date,
            m.close,
            f.return_1d
        FROM market_data m
        INNER JOIN feature_store f
            ON m.ticker = f.ticker
            AND m.date = f.date
        WHERE m.ticker = :ticker
        ORDER BY m.date ASC
    """)

    df = pd.read_sql(
        query,
        engine,
        params={"ticker": ticker},
    )

    if df.empty:
        raise ValueError(
            f"No data found for {ticker}"
        )

    print("=" * 60)
    print(f"STATIONARITY ANALYSIS — {ticker}")
    print("=" * 60)

    adf_test(
        df["close"],
        "Close Price"
    )

    adf_test(
        df["return_1d"],
        "1-Day Return"
    )


if __name__ == "__main__":
    check_stationarity("AAPL")