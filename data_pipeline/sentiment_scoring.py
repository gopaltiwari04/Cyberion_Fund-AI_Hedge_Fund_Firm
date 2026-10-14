"""
FinBERT sentiment scoring + point-in-time daily news aggregation.

Steps (each idempotent):
    1. backfill_legacy_news  - copy NewsAPI articles previously stored in
                               financial_text into financial_news (dedup by hash)
    2. score_news            - FinBERT on headline + description for unscored rows
    3. score_filings         - FinBERT on substantive 8-K text (press-release
                               exhibits preferred) for unscored filings
    4. aggregate_daily       - news_sentiment_daily keyed by the trading date
                               on which each article became usable (articles
                               after the 16:00 ET close roll to the next session)

Run: python -m data_pipeline.sentiment_scoring [--skip-filings]
"""

from __future__ import annotations

import argparse
import hashlib
import re

import numpy as np
import pandas as pd
from sqlalchemy import text

from common.db import get_engine
from common.logs import get_logger
from common.market_time import effective_trading_dates
from common.runs import RunContext, track_run
from data_pipeline.sec_ingestion import filing_text

logger = get_logger(__name__)

POSITIVE_THRESHOLD = 0.2


def mentions_company(ticker: str, *texts: str | None) -> bool:
    """Keyword-search news APIs return many unrelated articles; require the company to be named."""
    from ml_core.rag.retriever import COMPANY_ALIASES

    haystack = " ".join(value or "" for value in texts).lower()
    aliases = COMPANY_ALIASES.get(ticker, [ticker.lower()])
    return any(re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", haystack) for alias in aliases)


def remove_irrelevant_legacy_news() -> int:
    """Drop migrated NewsAPI copies that never mention the company (originals stay in financial_text)."""
    with get_engine().connect() as conn:
        rows = pd.read_sql(
            text("SELECT id, ticker, title, description FROM financial_news WHERE source = 'newsapi'"), conn
        )
    irrelevant = [int(row.id) for row in rows.itertuples() if not mentions_company(row.ticker, row.title, row.description)]
    if irrelevant:
        with get_engine().begin() as conn:
            conn.execute(text("DELETE FROM financial_news WHERE id = ANY(:ids)"), {"ids": irrelevant})
    return len(irrelevant)


def backfill_legacy_news() -> int:
    with get_engine().connect() as conn:
        legacy = pd.read_sql(
            text("SELECT ticker, published_at, title, raw_text FROM financial_text WHERE source = 'news'"),
            conn,
        )
    if legacy.empty:
        return 0
    legacy = legacy[[mentions_company(row.ticker, row.title, (row.raw_text or "")[:2000]) for row in legacy.itertuples()]]
    if legacy.empty:
        return 0
    legacy["uuid"] = [
        "newsapi:" + hashlib.sha1(f"{row.ticker}|{row.title}|{row.published_at}".encode()).hexdigest()
        for row in legacy.itertuples()
    ]
    records = [
        {
            "uuid": row.uuid,
            "ticker": row.ticker,
            "title": row.title,
            "description": (row.raw_text or "")[:2000],
            "source": "newsapi",
            "published_at": pd.Timestamp(row.published_at).tz_localize("UTC")
            if pd.Timestamp(row.published_at).tzinfo is None
            else pd.Timestamp(row.published_at),
        }
        for row in legacy.itertuples()
    ]
    with get_engine().begin() as conn:
        result = conn.execute(
            text(
                """
                INSERT INTO financial_news (uuid, ticker, title, description, source, published_at, language)
                VALUES (:uuid, :ticker, :title, :description, :source, :published_at, 'en')
                ON CONFLICT (uuid, ticker) DO NOTHING
                """
            ),
            records,
        )
    return result.rowcount or 0


def score_news(ctx: RunContext, limit: int = 5000) -> int:
    from ml_core.nlp.finbert import FinBert

    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, title, COALESCE(description, snippet, '') AS body
                FROM financial_news WHERE scored_at IS NULL ORDER BY published_at DESC LIMIT :limit
                """
            ),
            {"limit": limit},
        ).mappings().all()
    if not rows:
        return 0
    texts = [f"{row['title'] or ''}. {row['body'][:600]}".strip() for row in rows]
    scores = FinBert.get().score(texts, max_length=192)
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                UPDATE financial_news SET finbert_positive = :p, finbert_negative = :n, finbert_neutral = :u,
                    finbert_score = :s, scored_at = now()
                WHERE id = :id
                """
            ),
            [
                {"id": row["id"], "p": score.positive, "n": score.negative, "u": score.neutral, "s": score.score}
                for row, score in zip(rows, scores)
            ],
        )
    ctx.details["news_scored"] = len(rows)
    return len(rows)


def substantive_passages(document: str, max_passages: int = 6, size: int = 1500) -> list[str]:
    """Prefer press-release exhibits; skip cover-page boilerplate and short lines."""
    exhibit_start = document.find("EXHIBIT ")
    body = document[exhibit_start:] if exhibit_start >= 0 else document
    lines = [line for line in body.splitlines() if len(line) > 80 and not re.search(r"check the appropriate box", line, re.I)]
    passages, current = [], ""
    for line in lines:
        if len(current) + len(line) > size and current:
            passages.append(current)
            current = ""
            if len(passages) >= max_passages:
                break
        current += line + " "
    if current and len(passages) < max_passages:
        passages.append(current)
    return passages


FILING_PASSAGES = 3
FILING_TOKENS = 256
FILING_BATCH = 24


def _store_filing_scores(rows: list[dict]) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE sec_filings SET finbert_positive = :p, finbert_negative = :n, finbert_score = :s "
                "WHERE accession_no = :a"
            ),
            rows,
        )


def score_filings(ctx: RunContext, limit: int = 2000) -> int:
    """
    FinBERT over the first substantive passages of each 8-K (press-release exhibits
    preferred). Passages from many filings are batched together for CPU throughput
    and results are committed per batch, so an interrupted run resumes where it stopped.
    """
    from ml_core.nlp.finbert import FinBert

    with get_engine().connect() as conn:
        filings = conn.execute(
            text(
                """
                SELECT accession_no, ticker FROM sec_filings
                WHERE form_type = '8-K' AND status IN ('downloaded', 'indexed') AND finbert_score IS NULL
                ORDER BY filing_date DESC LIMIT :limit
                """
            ),
            {"limit": limit},
        ).mappings().all()
    model = FinBert.get()
    scored = 0
    for start in range(0, len(filings), FILING_BATCH):
        owners, texts = [], []
        for filing in filings[start:start + FILING_BATCH]:
            try:
                passages = substantive_passages(filing_text(filing["ticker"], filing["accession_no"]),
                                                max_passages=FILING_PASSAGES)
            except Exception as error:  # noqa: BLE001
                ctx.warn(f"{filing['accession_no']}: text unavailable ({error})")
                continue
            owners.extend([filing["accession_no"]] * len(passages))
            texts.extend(passages)
        if not texts:
            continue
        scores = model.score(texts, batch_size=16, max_length=FILING_TOKENS)
        frame = pd.DataFrame({"a": owners, "p": [x.positive for x in scores], "n": [x.negative for x in scores]})
        grouped = frame.groupby("a")[["p", "n"]].mean().reset_index()
        grouped["s"] = grouped["p"] - grouped["n"]
        _store_filing_scores(grouped.to_dict(orient="records"))
        scored += len(grouped)
    ctx.details["filings_scored"] = scored
    return scored


def trading_dates() -> list:
    with get_engine().connect() as conn:
        return [row[0] for row in conn.execute(text("SELECT DISTINCT date FROM market_data ORDER BY date"))]


def build_daily_sentiment(news: pd.DataFrame, dates: list) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame()
    news = news.copy()
    news["date"] = effective_trading_dates(news["published_at"], dates)
    news = news.dropna(subset=["date", "ticker"])
    if news.empty:
        return pd.DataFrame()
    grouped = news.groupby(["ticker", "date"])
    daily = grouped.agg(
        article_count=("uuid", "nunique"),
        finbert_sentiment_mean=("finbert_score", "mean"),
        marketaux_sentiment_mean=("entity_sentiment", "mean"),
    ).reset_index()
    daily["positive_ratio"] = grouped["finbert_score"].apply(lambda s: (s > POSITIVE_THRESHOLD).mean() if s.notna().any() else np.nan).values
    daily["negative_ratio"] = grouped["finbert_score"].apply(lambda s: (s < -POSITIVE_THRESHOLD).mean() if s.notna().any() else np.nan).values
    return daily


def aggregate_daily(ctx: RunContext) -> int:
    with get_engine().connect() as conn:
        news = pd.read_sql(
            text("SELECT uuid, ticker, published_at, finbert_score, entity_sentiment FROM financial_news WHERE ticker IS NOT NULL"),
            conn,
        )
    daily = build_daily_sentiment(news, trading_dates())
    if daily.empty:
        return 0
    records = daily.astype(object).where(daily.notna(), None).to_dict(orient="records")
    with get_engine().begin() as conn:
        # Fully derived from financial_news: rebuild atomically so removed articles disappear too.
        conn.execute(text("DELETE FROM news_sentiment_daily"))
        conn.execute(
            text(
                """
                INSERT INTO news_sentiment_daily (ticker, date, article_count, finbert_sentiment_mean,
                    marketaux_sentiment_mean, positive_ratio, negative_ratio, updated_at)
                VALUES (:ticker, :date, :article_count, :finbert_sentiment_mean, :marketaux_sentiment_mean,
                    :positive_ratio, :negative_ratio, now())
                ON CONFLICT (ticker, date) DO UPDATE SET
                    article_count = EXCLUDED.article_count,
                    finbert_sentiment_mean = EXCLUDED.finbert_sentiment_mean,
                    marketaux_sentiment_mean = EXCLUDED.marketaux_sentiment_mean,
                    positive_ratio = EXCLUDED.positive_ratio,
                    negative_ratio = EXCLUDED.negative_ratio,
                    updated_at = now()
                """
            ),
            records,
        )
    ctx.details["daily_rows"] = len(records)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-filings", action="store_true")
    args = parser.parse_args()
    pipeline, task = ("news_intelligence", "finbert_news") if args.skip_filings else ("sec_filings", "finbert_sentiment")
    with track_run(pipeline, task) as ctx:
        ctx.details["legacy_backfilled"] = backfill_legacy_news()
        ctx.details["legacy_irrelevant_removed"] = remove_irrelevant_legacy_news()
        ctx.add_rows(score_news(ctx))
        if not args.skip_filings:
            ctx.add_rows(score_filings(ctx))
        aggregate_daily(ctx)


if __name__ == "__main__":
    main()
