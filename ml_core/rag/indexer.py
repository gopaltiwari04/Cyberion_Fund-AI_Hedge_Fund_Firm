"""
SEC RAG indexer: downloaded filing text -> section chunks -> embeddings -> sec_filing_chunks.

Only filings inside the RAG window (SEC_RAG_YEARS, default 3) are indexed.
Re-indexing a filing replaces its chunks inside one transaction, so the task
is idempotent and safe to re-run.

Run: python -m ml_core.rag.indexer [--limit N]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from common.db import get_engine
from common.logs import get_logger
from common.runs import track_run
from data_pipeline.sec_ingestion import RAG_YEARS, filing_text
from ml_core.rag import embeddings
from ml_core.rag.chunking import chunk_filing

logger = get_logger(__name__)


def has_embedding_column() -> bool:
    with get_engine().connect() as conn:
        return bool(
            conn.execute(
                text(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = 'sec_filing_chunks' AND column_name = 'embedding'"
                )
            ).scalar()
        )


def index_filing(filing: dict, use_embeddings: bool) -> int:
    chunks = chunk_filing(filing_text(filing["ticker"], filing["accession_no"]), filing["form_type"])
    if not chunks:
        return 0
    vectors = None
    if use_embeddings:
        header = f"{filing['ticker']} {filing['form_type']} filed {filing['filing_date']}"
        vectors = embeddings.embed_passages([f"{header} | {chunk.section}\n{chunk.content}" for chunk in chunks])

    rows = []
    for position, chunk in enumerate(chunks):
        row = {
            "accession_no": filing["accession_no"],
            "ticker": filing["ticker"],
            "form_type": filing["form_type"],
            "filing_date": filing["filing_date"],
            "section": chunk.section,
            "chunk_index": chunk.index,
            "content": chunk.content,
            "content_hash": chunk.content_hash,
            "embedding_model": embeddings.model_name() if vectors else None,
        }
        if vectors:
            row["embedding"] = embeddings.to_pgvector(vectors[position])
        rows.append(row)

    columns = "accession_no, ticker, form_type, filing_date, section, chunk_index, content, content_hash, embedding_model"
    values = ":accession_no, :ticker, :form_type, :filing_date, :section, :chunk_index, :content, :content_hash, :embedding_model"
    if vectors:
        columns += ", embedding"
        values += ", CAST(:embedding AS vector)"

    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM sec_filing_chunks WHERE accession_no = :a"), {"a": filing["accession_no"]})
        conn.execute(text(f"INSERT INTO sec_filing_chunks ({columns}) VALUES ({values})"), rows)
        conn.execute(
            text("UPDATE sec_filings SET status = 'indexed', chunk_count = :n, processed_at = now() WHERE accession_no = :a"),
            {"n": len(rows), "a": filing["accession_no"]},
        )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=365 * RAG_YEARS)
    with track_run("sec_filings", "index_filings") as ctx:
        use_embeddings = has_embedding_column() and embeddings.available()
        if not use_embeddings:
            ctx.warn("Embeddings unavailable; chunks indexed for full-text search only")
        with get_engine().connect() as conn:
            filings = conn.execute(
                text(
                    """
                    SELECT accession_no, ticker, form_type, filing_date
                    FROM sec_filings f
                    WHERE filing_date >= :cutoff AND (
                        status = 'downloaded'
                        -- self-healing: re-embed filings indexed while embeddings were unavailable
                        OR (:embed AND status = 'indexed' AND EXISTS (
                            SELECT 1 FROM sec_filing_chunks c
                            WHERE c.accession_no = f.accession_no AND c.embedding_model IS NULL))
                    )
                    ORDER BY filing_date DESC
                    LIMIT :limit
                    """
                ),
                {"cutoff": cutoff, "limit": args.limit, "embed": use_embeddings},
            ).mappings().all()
        indexed = 0
        for filing in filings:
            try:
                count = index_filing(dict(filing), use_embeddings)
                ctx.add_rows(count)
                indexed += 1
            except Exception as error:  # noqa: BLE001 - one bad document must not stop the batch
                ctx.warn(f"{filing['accession_no']}: {error}")
        ctx.details.update({"filings_indexed": indexed, "embeddings": use_embeddings, "cutoff": str(cutoff)})


if __name__ == "__main__":
    main()
