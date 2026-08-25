import re
import json

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text


# ============================================================
# CONFIG
# ============================================================

DB_URL = "postgresql://quant_user:quant_password@localhost:5432/quant_db"
engine = create_engine(DB_URL)

MODEL_NAME = "all-MiniLM-L6-v2"

# MiniLM works with relatively short text inputs.
# We deliberately stay below its maximum token capacity.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

BATCH_SIZE = 16
LIMIT = 100


# ============================================================
# MODEL
# ============================================================

print("=" * 60)
print("FINANCIAL TEXT PROCESSING")
print("=" * 60)
print()
print("Loading Sentence Transformer Model...")
print(f"Model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Embedding dimension:", model.get_sentence_embedding_dimension())


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_html(raw_html):
    """
    Convert HTML/XBRL-heavy filing text into readable plain text.
    """

    if not raw_html:
        return ""

    soup = BeautifulSoup(
        raw_html,
        "html.parser"
    )

    # Remove elements that contain mostly technical noise.
    for element in soup([
        "script",
        "style",
        "noscript"
    ]):
        element.decompose()

    text_content = soup.get_text(
        separator=" "
    )

    # Normalize whitespace.
    text_content = re.sub(
        r"\s+",
        " ",
        text_content
    ).strip()

    return text_content


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(
    text_content,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """
    Split long documents into overlapping character chunks.

    Overlap prevents important information sitting exactly
    at a chunk boundary from being completely separated.
    """

    if not text_content:
        return []

    if len(text_content) <= chunk_size:
        return [text_content]

    chunks = []

    start = 0
    text_length = len(text_content)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        chunk = text_content[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# ============================================================
# EMBEDDING
# ============================================================

def generate_document_embedding(text_content):
    """
    Generate one document-level embedding.

    Short documents:
        one embedding.

    Long SEC filings:
        multiple chunk embeddings -> mean pooled vector.

    Finally L2-normalize the document vector.
    """

    chunks = chunk_text(text_content)

    if not chunks:
        return None, 0

    embeddings = model.encode(
        chunks,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False
    )

    # Mean-pool all chunk vectors into one document vector.
    document_embedding = np.mean(
        embeddings,
        axis=0
    )

    # L2 normalization.
    norm = np.linalg.norm(
        document_embedding
    )

    if norm > 0:
        document_embedding = (
            document_embedding / norm
        )

    return (
        document_embedding.astype(float).tolist(),
        len(chunks)
    )


# ============================================================
# LOAD UNPROCESSED TEXT
# ============================================================

def load_unprocessed_text():

    query = text("""
        SELECT
            id,
            ticker,
            source,
            title,
            raw_text
        FROM financial_text
        WHERE clean_text IS NULL
        ORDER BY id
        LIMIT :limit
    """)

    with engine.connect() as conn:

        df = pd.read_sql(
            query,
            conn,
            params={"limit": LIMIT}
        )

    return df


# ============================================================
# DATABASE UPDATE
# ============================================================

def update_record(
    record_id,
    clean_text,
    embedding
):

    query = text("""
        UPDATE financial_text
        SET
            clean_text = :clean_text,
            embedding = :embedding
        WHERE id = :id
    """)

    with engine.begin() as conn:

        conn.execute(
            query,
            {
                "id": int(record_id),
                "clean_text": clean_text,
                "embedding": json.dumps(embedding),
            }
        )


# ============================================================
# MAIN PROCESSING PIPELINE
# ============================================================

def process_and_embed():

    df = load_unprocessed_text()

    if df.empty:

        print()
        print("No new text to process.")
        return

    print()
    print("=" * 60)
    print("PROCESSING TEXT")
    print("=" * 60)
    print(f"Records loaded: {len(df)}")
    print()

    processed = 0
    failed = 0
    total_chunks = 0

    for _, row in df.iterrows():

        record_id = row["id"]

        try:

            clean_text = clean_html(
                row["raw_text"]
            )

            if not clean_text:

                print(
                    f"[SKIP] ID {record_id}: empty after cleaning"
                )

                continue

            embedding, chunk_count = (
                generate_document_embedding(
                    clean_text
                )
            )

            if embedding is None:

                print(
                    f"[SKIP] ID {record_id}: no embedding"
                )

                continue

            update_record(
                record_id,
                clean_text,
                embedding
            )

            processed += 1
            total_chunks += chunk_count

            print(
                f"[OK] ID {record_id} | "
                f"{row['ticker']} | "
                f"{row['source']} | "
                f"chunks={chunk_count} | "
                f"chars={len(clean_text):,}"
            )

        except Exception as exc:

            failed += 1

            print(
                f"[ERROR] ID {record_id}: {exc}"
            )

    print()
    print("=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Processed records : {processed}")
    print(f"Failed records    : {failed}")
    print(f"Total chunks      : {total_chunks}")

    if processed:
        print(
            f"Average chunks/document: "
            f"{total_chunks / processed:.2f}"
        )

    print()
    print("Text cleaning and embedding completed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    process_and_embed()