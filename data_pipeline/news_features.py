"""
Latest news-sentiment snapshot per ticker (news_features table).

Point-in-time daily history used by the models lives in news_sentiment_daily
(see data_pipeline/sentiment_scoring.py); this table is the "right now" view
for the dashboard. FinBERT scores are preferred, falling back to Marketaux
entity sentiment for rows not yet scored.

Run: python -m data_pipeline.news_features
"""

import pandas as pd
from sqlalchemy import text

from common.db import get_engine
from common.runs import track_run
from common.universe import tickers

TICKERS = tickers()


def load_news():
    engine = get_engine()

    query = text(
        """
        SELECT
            uuid,
            ticker,
            title,
            published_at,
            COALESCE(finbert_score, entity_sentiment) AS entity_sentiment
        FROM financial_news
        WHERE ticker IS NOT NULL
        ORDER BY published_at
        """
    )

    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def build_features(news):
    if news.empty:
        return pd.DataFrame()

    news["published_at"] = pd.to_datetime(
        news["published_at"],
        utc=True,
    )

    news["entity_sentiment"] = pd.to_numeric(
        news["entity_sentiment"],
        errors="coerce",
    )

    news = news.dropna(
        subset=["ticker", "published_at"]
    )

    now = pd.Timestamp.now(tz="UTC")

    rows = []

    for ticker in TICKERS:
        ticker_news = news[
            news["ticker"] == ticker
        ].copy()

        last_24h = ticker_news[
            ticker_news["published_at"]
            >= now - pd.Timedelta(hours=24)
        ]

        last_7d = ticker_news[
            ticker_news["published_at"]
            >= now - pd.Timedelta(days=7)
        ]

        sentiment_24h = (
            last_24h["entity_sentiment"]
            .dropna()
        )

        sentiment_7d = (
            last_7d["entity_sentiment"]
            .dropna()
        )

        mean_24h = (
            sentiment_24h.mean()
            if not sentiment_24h.empty
            else 0.0
        )

        mean_7d = (
            sentiment_7d.mean()
            if not sentiment_7d.empty
            else 0.0
        )

        positive_ratio = (
            (sentiment_24h > 0).mean()
            if not sentiment_24h.empty
            else 0.0
        )

        negative_ratio = (
            (sentiment_24h < 0).mean()
            if not sentiment_24h.empty
            else 0.0
        )

        rows.append(
            {
                "ticker": ticker,
                "feature_timestamp": now,
                "news_count_24h": len(last_24h),
                "news_sentiment_mean_24h": mean_24h,
                "news_sentiment_mean_7d": mean_7d,
                "news_positive_ratio_24h": positive_ratio,
                "news_negative_ratio_24h": negative_ratio,
                "news_sentiment_momentum": (
                    mean_24h - mean_7d
                ),
            }
        )

    return pd.DataFrame(rows)


def save_features(features):
    if features.empty:
        return 0

    engine = get_engine()

    query = text(
        """
        INSERT INTO news_features (
            ticker,
            feature_timestamp,
            news_count_24h,
            news_sentiment_mean_24h,
            news_sentiment_mean_7d,
            news_positive_ratio_24h,
            news_negative_ratio_24h,
            news_sentiment_momentum
        )
        VALUES (
            :ticker,
            :feature_timestamp,
            :news_count_24h,
            :news_sentiment_mean_24h,
            :news_sentiment_mean_7d,
            :news_positive_ratio_24h,
            :news_negative_ratio_24h,
            :news_sentiment_momentum
        )
        ON CONFLICT (ticker)
        DO UPDATE SET
            feature_timestamp =
                EXCLUDED.feature_timestamp,
            news_count_24h =
                EXCLUDED.news_count_24h,
            news_sentiment_mean_24h =
                EXCLUDED.news_sentiment_mean_24h,
            news_sentiment_mean_7d =
                EXCLUDED.news_sentiment_mean_7d,
            news_positive_ratio_24h =
                EXCLUDED.news_positive_ratio_24h,
            news_negative_ratio_24h =
                EXCLUDED.news_negative_ratio_24h,
            news_sentiment_momentum =
                EXCLUDED.news_sentiment_momentum
        """
    )

    rows = features.to_dict(orient="records")

    with engine.begin() as connection:
        connection.execute(query, rows)

    return len(rows)


def main():
    with track_run("news_intelligence", "news_snapshot") as ctx:
        news = load_news()
        features = build_features(news)
        if features.empty:
            ctx.warn("No news available for snapshot features")
            return
        ctx.add_rows(save_features(features))


if __name__ == "__main__":
    main()