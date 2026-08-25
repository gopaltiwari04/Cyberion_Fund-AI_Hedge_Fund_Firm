import os
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import torch
import torch.nn.functional as F
import redis

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sqlalchemy import create_engine, text


# ============================================================
# CONFIGURATION
# ============================================================

DB_URL = os.getenv(
    "DB_URL",
    "postgresql://quant_user:quant_password@localhost:5432/quant_db",
)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

MODEL_NAME = "ProsusAI/finbert"

# Conservative batch size for CPU / Docker environments.
BATCH_SIZE = 8


# ============================================================
# DATABASE
# ============================================================

engine = create_engine(DB_URL)


# ============================================================
# REDIS
# ============================================================

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
)


# ============================================================
# FINBERT
# ============================================================

print("=" * 60)
print("FINBERT SENTIMENT ANALYSIS")
print("=" * 60)

print()
print("Loading FinBERT...")
print(f"Model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print(f"Device: {device}")

print("Model labels:")

# FinBERT normally exposes:
# 0 -> positive
# 1 -> negative
# 2 -> neutral
#
# We inspect the actual model configuration rather than
# blindly relying on the ordering.

label_map = {}

for index, label in model.config.id2label.items():
    label_map[int(index)] = label.lower()

print(label_map)


def find_label_index(name):
    """Find the FinBERT class index by label name."""

    name = name.lower()

    for index, label in label_map.items():
        if name in label:
            return index

    raise RuntimeError(
        f"Could not find '{name}' label in FinBERT configuration: "
        f"{label_map}"
    )


POSITIVE_INDEX = find_label_index("positive")
NEGATIVE_INDEX = find_label_index("negative")

print(f"Positive index: {POSITIVE_INDEX}")
print(f"Negative index: {NEGATIVE_INDEX}")


# ============================================================
# SENTIMENT INFERENCE
# ============================================================

def score_text_batch(texts):
    """
    Score a batch of financial text.

    Polarity:
        P(positive) - P(negative)

    Range:
        -1.0 -> strongly negative
         0.0 -> neutral
        +1.0 -> strongly positive
    """

    if not texts:
        return []

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(
        outputs.logits,
        dim=1,
    ).cpu()

    scores = []

    for probability in probabilities:
        positive_probability = float(
            probability[POSITIVE_INDEX]
        )

        negative_probability = float(
            probability[NEGATIVE_INDEX]
        )

        polarity = (
            positive_probability
            - negative_probability
        )

        scores.append(polarity)

    return scores


# ============================================================
# LOAD UNSCORED TEXT
# ============================================================

def load_unprocessed_text():
    """
    Load financial text from the previous 24 hours.

    We use the embedding column as the processing marker:
    text that already has an embedding is considered processed
    by the NLP pipeline.

    Sentiment is calculated independently and stored in the
    financial_text table.
    """

    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)

    query = text(
        """
        SELECT
            id,
            ticker,
            published_at,
            clean_text
        FROM financial_text
        WHERE published_at >= :start_time
          AND published_at <= :end_time
          AND clean_text IS NOT NULL
          AND LENGTH(TRIM(clean_text)) > 0
        ORDER BY published_at
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={
                "start_time": yesterday,
                "end_time": now,
            },
        )

    return df


# ============================================================
# STORE INDIVIDUAL SENTIMENT
# ============================================================

def store_individual_scores(df):
    """
    Store individual sentiment scores.

    The current financial_text schema does not yet contain a
    sentiment_score column, so we do not modify that table here.

    Individual scores are instead returned to the caller and
    aggregated into FeatureStore.

    This function exists as a clear extension point for a future
    financial_text.sentiment_score migration.
    """

    return df


# ============================================================
# UPDATE FEATURE STORE
# ============================================================

def update_feature_store(daily_sentiment):
    """
    Update today's FeatureStore sentiment values.

    If today's row does not exist, create it.

    Existing quantitative features are preserved.
    """

    today = datetime.now(timezone.utc).date()

    with engine.begin() as conn:

        for _, row in daily_sentiment.iterrows():

            ticker = row["ticker"]
            sentiment = float(row["polarity_score"])

            # Check whether today's feature row exists.
            existing = conn.execute(
                text(
                    """
                    SELECT id
                    FROM feature_store
                    WHERE ticker = :ticker
                      AND date = :date
                    """
                ),
                {
                    "ticker": ticker,
                    "date": today,
                },
            ).fetchone()

            if existing:

                conn.execute(
                    text(
                        """
                        UPDATE feature_store
                        SET sentiment_score = :sentiment
                        WHERE ticker = :ticker
                          AND date = :date
                        """
                    ),
                    {
                        "sentiment": sentiment,
                        "ticker": ticker,
                        "date": today,
                    },
                )

            else:

                conn.execute(
                    text(
                        """
                        INSERT INTO feature_store
                            (
                                ticker,
                                date,
                                sentiment_score
                            )
                        VALUES
                            (
                                :ticker,
                                :date,
                                :sentiment
                            )
                        """
                    ),
                    {
                        "ticker": ticker,
                        "date": today,
                        "sentiment": sentiment,
                    },
                )


# ============================================================
# REDIS CACHE
# ============================================================

def update_redis_cache(daily_sentiment):
    """
    Merge sentiment into the existing Redis feature cache.

    Key:
        feature_store:{TICKER}:latest
    """

    print()
    print("=" * 60)
    print("REDIS CACHE UPDATE")
    print("=" * 60)

    for _, row in daily_sentiment.iterrows():

        ticker = row["ticker"]

        sentiment = float(
            row["polarity_score"]
        )

        cache_key = (
            f"feature_store:{ticker}:latest"
        )

        try:

            cached_data = redis_client.get(
                cache_key
            )

            if cached_data:

                feature_dict = json.loads(
                    cached_data
                )

                feature_dict[
                    "sentiment_score"
                ] = sentiment

                redis_client.set(
                    cache_key,
                    json.dumps(feature_dict),
                )

                print(
                    f"[OK] {ticker} -> "
                    f"sentiment={sentiment:.4f}"
                )

            else:

                # Don't create a fake feature record if the
                # quantitative pipeline hasn't created one yet.
                print(
                    f"[SKIP] {ticker} -> "
                    "Redis feature cache does not exist"
                )

        except redis.RedisError as exc:

            print(
                f"[WARN] Redis unavailable: {exc}"
            )

        except json.JSONDecodeError as exc:

            print(
                f"[WARN] Invalid Redis JSON for "
                f"{ticker}: {exc}"
            )

    print()
    print("Redis cache update completed.")


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_daily_sentiment_scoring():

    print()
    print("=" * 60)
    print("LOADING FINANCIAL TEXT")
    print("=" * 60)

    df = load_unprocessed_text()

    print(
        f"Records loaded: {len(df)}"
    )

    if df.empty:

        print(
            "No financial text found in the "
            "previous 24 hours."
        )

        return pd.DataFrame(
            columns=[
                "ticker",
                "polarity_score",
            ]
        )

    # --------------------------------------------------------
    # SCORE IN BATCHES
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RUNNING FINBERT")
    print("=" * 60)

    all_scores = []

    for start in range(
        0,
        len(df),
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            len(df),
        )

        batch_texts = (
            df["clean_text"]
            .iloc[start:end]
            .tolist()
        )

        print(
            f"Scoring records "
            f"{start + 1}-{end} "
            f"of {len(df)}..."
        )

        batch_scores = score_text_batch(
            batch_texts
        )

        all_scores.extend(
            batch_scores
        )

    df["polarity_score"] = all_scores

    # --------------------------------------------------------
    # DAILY AGGREGATION
    # --------------------------------------------------------

    daily_sentiment = (
        df.groupby("ticker")[
            "polarity_score"
        ]
        .mean()
        .reset_index()
    )

    print()
    print("=" * 60)
    print("DAILY SENTIMENT")
    print("=" * 60)

    for _, row in daily_sentiment.iterrows():

        print(
            f"{row['ticker']}: "
            f"{row['polarity_score']:.4f}"
        )

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    print()
    print("Updating PostgreSQL FeatureStore...")

    update_feature_store(
        daily_sentiment
    )

    print(
        "PostgreSQL FeatureStore updated."
    )

    # --------------------------------------------------------
    # REDIS
    # --------------------------------------------------------

    update_redis_cache(
        daily_sentiment
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SENTIMENT ANALYSIS COMPLETED")
    print("=" * 60)

    print(
        f"Text records scored : {len(df)}"
    )

    print(
        f"Tickers aggregated   : "
        f"{len(daily_sentiment)}"
    )

    return daily_sentiment


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_daily_sentiment_scoring()