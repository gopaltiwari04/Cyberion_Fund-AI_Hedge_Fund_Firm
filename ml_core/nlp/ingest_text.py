import os
import hashlib
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv
from newsapi import NewsApiClient
from sec_edgar_downloader import Downloader
from sqlalchemy import create_engine, text


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

# ------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------

DB_URL = os.getenv(
    "DB_URL",
    "postgresql://quant_user:quant_password@localhost:5432/quant_db"
)

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")

if not NEWSAPI_KEY:
    raise RuntimeError("NEWSAPI_KEY environment variable is not set")

if not SEC_USER_AGENT:
    raise RuntimeError("SEC_USER_AGENT environment variable is not set")

engine = create_engine(DB_URL)

# ------------------------------------------------------------
# Initialize APIs
# ------------------------------------------------------------

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# SEC Downloader expects:
# CompanyName ContactEmail
sec_parts = SEC_USER_AGENT.split(" ", 1)

if len(sec_parts) != 2:
    raise RuntimeError(
        "SEC_USER_AGENT must be formatted as: "
        "'CompanyName ContactEmail'"
    )

dl = Downloader(
    sec_parts[0],
    sec_parts[1],
    "./sec_data"
)


# ============================================================
# HELPERS
# ============================================================

def make_content_hash(ticker, source, title, raw_text):
    """
    Creates a deterministic fingerprint for a piece of financial text.

    This lets us avoid inserting the same article/filing repeatedly.
    """
    content = (
        f"{ticker}|{source}|{title or ''}|{raw_text or ''}"
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def record_exists(content_hash):
    """
    Check whether this piece of text is already stored.

    Currently uses a hash against the title/raw_text combination.
    """

    query = text("""
        SELECT 1
        FROM financial_text
        WHERE md5(
            CONCAT(
                ticker,
                '|',
                source,
                '|',
                COALESCE(title, ''),
                '|',
                COALESCE(raw_text, '')
            )
        ) = :hash
        LIMIT 1
    """)

    # This query uses MD5 while our generated hash is SHA256,
    # so don't use it. Kept out of execution intentionally.
    return False


# ============================================================
# NEWSAPI
# ============================================================

def fetch_news(ticker, company_name):

    print(f"Fetching news for {ticker}...")

    today = datetime.now(timezone.utc).date()
    last_week = today - timedelta(days=7)

    articles = newsapi.get_everything(
        q=company_name,
        from_param=last_week.isoformat(),
        to=today.isoformat(),
        language="en",
        sort_by="publishedAt",
        page_size=100,
    )

    records = []

    for article in articles.get("articles", []):

        title = article.get("title") or ""
        description = article.get("description") or ""
        content = article.get("content") or ""

        raw_text = description

        if not raw_text:
            raw_text = content

        if not raw_text:
            continue

        published_at = article.get("publishedAt")

        if not published_at:
            continue

        records.append({
            "ticker": ticker,
            "source": "news",
            "published_at": pd.to_datetime(published_at, utc=True),
            "title": title,
            "raw_text": raw_text,
            "clean_text": None,
            "embedding": None,
        })

    print(f"  News articles found: {len(records)}")

    return records


# ============================================================
# SEC 8-K
# ============================================================

def fetch_sec_8k(ticker):

    print(f"Fetching latest 8-K filings for {ticker}...")

    dl.get(
        "8-K",
        ticker,
        limit=5
    )

    sec_dir = os.path.join(
        "sec_data",
        "sec-edgar-filings",
        ticker,
        "8-K"
    )

    if not os.path.exists(sec_dir):
        print("  No SEC directory found.")

        return []

    records = []

    for root, dirs, files in os.walk(sec_dir):

        for file in files:

            if not file.endswith(".txt"):
                continue

            filepath = os.path.join(root, file)

            try:

                with open(
                    filepath,
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as f:

                    content = f.read()

            except Exception as exc:

                print(
                    f"  Failed reading {filepath}: {exc}"
                )

                continue

            if not content.strip():
                continue

            # SEC downloader stores filings under accession-number
            # directories. We retain the file modification time as a
            # safer approximation than pretending ingestion time is
            # publication time.
            try:
                published_at = datetime.fromtimestamp(
                    os.path.getmtime(filepath),
                    tz=timezone.utc
                )
            except OSError:
                published_at = datetime.now(timezone.utc)

            records.append({
                "ticker": ticker,
                "source": "sec_8k",
                "published_at": published_at,
                "title": "Form 8-K",
                "raw_text": content,
                "clean_text": None,
                "embedding": None,
            })

    print(f"  SEC filings found: {len(records)}")

    return records


# ============================================================
# DATABASE INSERTION
# ============================================================

def insert_records(records):

    if not records:
        print("No records to insert.")
        return 0

    df = pd.DataFrame(records)

    df["published_at"] = pd.to_datetime(
        df["published_at"],
        utc=True
    )

    # Remove exact duplicates within this batch.
    df = df.drop_duplicates(
        subset=[
            "ticker",
            "source",
            "title",
            "raw_text",
        ]
    )

    inserted = 0
    skipped = 0

    with engine.begin() as conn:

        for _, row in df.iterrows():

            # Check for an existing identical record.
            existing = conn.execute(
                text("""
                    SELECT id
                    FROM financial_text
                    WHERE ticker = :ticker
                      AND source = :source
                      AND title = :title
                      AND raw_text = :raw_text
                    LIMIT 1
                """),
                {
                    "ticker": row["ticker"],
                    "source": row["source"],
                    "title": row["title"],
                    "raw_text": row["raw_text"],
                }
            ).fetchone()

            if existing:
                skipped += 1
                continue

            conn.execute(
                text("""
                    INSERT INTO financial_text (
                        ticker,
                        source,
                        published_at,
                        title,
                        raw_text,
                        clean_text,
                        embedding
                    )
                    VALUES (
                        :ticker,
                        :source,
                        :published_at,
                        :title,
                        :raw_text,
                        :clean_text,
                        :embedding
                    )
                """),
                {
                    "ticker": row["ticker"],
                    "source": row["source"],
                    "published_at": row["published_at"].to_pydatetime(),
                    "title": row["title"],
                    "raw_text": row["raw_text"],
                    "clean_text": None,
                    "embedding": None,
                }
            )

            inserted += 1

    print()
    print("=" * 60)
    print("DATABASE INGESTION")
    print("=" * 60)
    print(f"Inserted : {inserted}")
    print(f"Skipped  : {skipped} duplicates")

    return inserted


# ============================================================
# MAIN PIPELINE
# ============================================================

def ingest_all_text():

    print("=" * 60)
    print("FINANCIAL TEXT INGESTION")
    print("=" * 60)

    tickers = {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
    }

    all_records = []

    for ticker, company_name in tickers.items():

        try:
            all_records.extend(
                fetch_news(
                    ticker,
                    company_name
                )
            )

        except Exception as exc:

            print(
                f"News ingestion failed for {ticker}: {exc}"
            )

        try:
            all_records.extend(
                fetch_sec_8k(ticker)
            )

        except Exception as exc:

            print(
                f"SEC ingestion failed for {ticker}: {exc}"
            )

    insert_records(all_records)

    print()
    print("=" * 60)
    print("FINANCIAL TEXT INGESTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    ingest_all_text()