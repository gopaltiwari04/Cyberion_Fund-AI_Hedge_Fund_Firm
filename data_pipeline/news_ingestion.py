"""
Financial news: Marketaux -> MinIO raw -> PostgreSQL financial_news.

* One row per (article uuid, entity ticker); re-runs upsert on that key, so
  overlapping fetch windows never create duplicates.
* Raw API payloads are landed in MinIO for replay/audit.
* Plan limits: the Marketaux free tier returns 3 articles per request and
  100 requests per day. MARKETAUX_PAGES (default 2) controls requests per run;
  the hourly schedule therefore uses ~48 requests/day.

Run: python -m data_pipeline.news_ingestion
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from sqlalchemy import text

from common.config import get_settings
from common.db import get_engine
from common.logs import get_logger
from common.runs import track_run
from common.storage import get_object_store
from common.universe import sec_filers

logger = get_logger(__name__)

MARKETAUX_URL = "https://api.marketaux.com/v1/news/all"
PAGES = int(os.getenv("MARKETAUX_PAGES", "2"))
LOOKBACK_HOURS = int(os.getenv("MARKETAUX_LOOKBACK_HOURS", "48"))


def fetch_news(symbols: list[str], page: int = 1, limit: int = 3, published_after_hours: int = LOOKBACK_HOURS) -> dict:
    token = get_settings().marketaux_api_token
    if not token:
        raise RuntimeError("MARKETAUX_API_TOKEN is not configured.")
    published_after = (datetime.now(timezone.utc) - timedelta(hours=published_after_hours)).strftime("%Y-%m-%dT%H:%M:%S")
    params = {
        "api_token": token,
        "symbols": ",".join(symbols),
        "filter_entities": "true",
        "language": "en",
        "published_after": published_after,
        "limit": limit,
        "page": page,
    }
    for attempt in range(1, 4):
        response = requests.get(MARKETAUX_URL, params=params, timeout=30)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(10 * attempt)
            continue
        if response.status_code == 402:
            raise RuntimeError("Marketaux daily request limit reached (HTTP 402)")
        response.raise_for_status()
        return response.json()
    response.raise_for_status()
    return {}


def normalize_articles(articles: list[dict], universe: set[str] | None = None) -> pd.DataFrame:
    rows = []
    for article in articles:
        if not article.get("uuid") or not article.get("published_at"):
            continue
        for entity in article.get("entities") or [{}]:
            symbol = (entity.get("symbol") or "").upper() or None
            if universe is not None and symbol not in universe:
                continue
            rows.append(
                {
                    "uuid": article.get("uuid"),
                    "title": (article.get("title") or "")[:1000],
                    "description": article.get("description"),
                    "snippet": article.get("snippet"),
                    "url": article.get("url"),
                    "image_url": article.get("image_url"),
                    "source": article.get("source"),
                    "published_at": article.get("published_at"),
                    "language": article.get("language"),
                    "ticker": symbol,
                    "entity_name": entity.get("name"),
                    "entity_sentiment": entity.get("sentiment_score"),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(subset=["uuid", "ticker"], keep="last")
    return frame.astype(object).where(frame.notna(), None)


def save_to_database(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    query = text(
        """
        INSERT INTO financial_news (uuid, ticker, title, description, snippet, url, image_url, source,
            published_at, language, entity_name, entity_sentiment)
        VALUES (:uuid, :ticker, :title, :description, :snippet, :url, :image_url, :source,
            :published_at, :language, :entity_name, :entity_sentiment)
        ON CONFLICT (uuid, ticker) DO UPDATE SET
            entity_sentiment = EXCLUDED.entity_sentiment,
            title = EXCLUDED.title,
            description = EXCLUDED.description
        """
    )
    with get_engine().begin() as connection:
        connection.execute(query, df.to_dict(orient="records"))
    return len(df)


def main() -> None:
    symbols = sec_filers()
    universe = set(symbols) | {"SPY"}
    store = get_object_store()
    with track_run("news_intelligence", "ingest_news") as ctx:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
        total = 0
        for page in range(1, PAGES + 1):
            payload = fetch_news(symbols, page=page)
            store.put_bytes(
                f"news/marketaux/hour={stamp}/page={page}.json",
                json.dumps(payload).encode("utf-8"),
                "application/json",
            )
            articles = payload.get("data", [])
            total += save_to_database(normalize_articles(articles, universe))
            if len(articles) < 3:
                break
        ctx.add_rows(total)
        ctx.details["pages"] = page


if __name__ == "__main__":
    main()
