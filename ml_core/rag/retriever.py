"""
Hybrid retrieval over SEC filing chunks.

* Semantic: pgvector cosine distance on bge-small embeddings (when available)
* Lexical:  PostgreSQL full-text search (OR of salient terms, ts_rank_cd)
* Fusion:   reciprocal rank fusion (k=60)

Filters: ticker(s), form types and filing-date range. When the question names
a covered company and no ticker filter is given, the filter is inferred.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date

from sqlalchemy import text

from common.db import get_engine
from common.logs import get_logger
from ml_core.rag import embeddings

logger = get_logger(__name__)

RRF_K = 60
CANDIDATES = 40
STOPWORDS = set(
    """a an and are as at be been by can could did do does for from had has have how i if in into is it its
    of on or our over should so than that the their them then there these they this to was we were what when
    where which while who why will with would you your about any all also between both each more most other
    such recent recently latest company companies disclosed disclose report reported tell me please""".split()
)
COMPANY_ALIASES = {
    "AAPL": ["apple", "aapl", "iphone"],
    "MSFT": ["microsoft", "msft", "azure"],
    "NVDA": ["nvidia", "nvda"],
    "GOOGL": ["alphabet", "google", "googl"],
    "META": ["meta platforms", "meta", "facebook"],
    "AMZN": ["amazon", "amzn", "aws"],
    "JPM": ["jpmorgan", "jp morgan", "jpm", "chase"],
    "XOM": ["exxon", "exxonmobil", "xom"],
    "JNJ": ["johnson & johnson", "johnson and johnson", "j&j", "jnj"],
}


@dataclass
class Passage:
    chunk_id: int
    accession_no: str
    ticker: str
    form_type: str
    filing_date: str
    section: str | None
    content: str
    url: str
    score: float
    retrieval: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def infer_tickers(question: str) -> list[str]:
    lowered = question.lower()
    found = []
    for ticker, aliases in COMPANY_ALIASES.items():
        if any(re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", lowered) for alias in aliases):
            found.append(ticker)
    return found


def lexical_query(question: str) -> str | None:
    terms = [term for term in re.findall(r"[a-z0-9]+", question.lower()) if len(term) > 2 and term not in STOPWORDS]
    terms = list(dict.fromkeys(terms))[:12]
    return " | ".join(terms) if terms else None


def _filters(tickers: list[str] | None, form_types: list[str] | None, date_from: date | None, date_to: date | None):
    clauses, params = [], {}
    if tickers:
        clauses.append("c.ticker = ANY(:tickers)")
        params["tickers"] = tickers
    if form_types:
        clauses.append("c.form_type = ANY(:forms)")
        params["forms"] = form_types
    if date_from:
        clauses.append("c.filing_date >= :date_from")
        params["date_from"] = date_from
    if date_to:
        clauses.append("c.filing_date <= :date_to")
        params["date_to"] = date_to
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


SELECT_COLUMNS = """
    c.id, c.accession_no, c.ticker, c.form_type, c.filing_date, c.section, c.content, f.document_url
"""


def semantic_candidates(question: str, where: str, params: dict) -> list[dict]:
    vector = embeddings.embed_query(question)
    if vector is None:
        return []
    with get_engine().connect() as conn:
        has_column = conn.execute(
            text("SELECT 1 FROM information_schema.columns WHERE table_name='sec_filing_chunks' AND column_name='embedding'")
        ).scalar()
        if not has_column:
            return []
        if where:
            # Filters are applied after the HNSW scan; iterative scans (pgvector >= 0.8) keep
            # searching until enough filtered rows are found instead of returning too few.
            try:
                with conn.begin_nested():
                    conn.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            except Exception:  # noqa: BLE001 - older pgvector: fall back to the default scan
                pass
        rows = conn.execute(
            text(
                f"""
                SELECT {SELECT_COLUMNS}
                FROM sec_filing_chunks c JOIN sec_filings f USING (accession_no)
                WHERE c.embedding IS NOT NULL {where}
                ORDER BY c.embedding <=> CAST(:vector AS vector)
                LIMIT :limit
                """
            ),
            {**params, "vector": embeddings.to_pgvector(vector), "limit": CANDIDATES},
        ).mappings().all()
    return [dict(row) for row in rows]


def lexical_candidates(question: str, where: str, params: dict) -> list[dict]:
    query = lexical_query(question)
    if not query:
        return []
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT {SELECT_COLUMNS}, ts_rank_cd(c.search_vector, to_tsquery('english', :q)) AS rank
                FROM sec_filing_chunks c JOIN sec_filings f USING (accession_no)
                WHERE c.search_vector @@ to_tsquery('english', :q) {where}
                ORDER BY rank DESC, c.filing_date DESC
                LIMIT :limit
                """
            ),
            {**params, "q": query, "limit": CANDIDATES},
        ).mappings().all()
    return [dict(row) for row in rows]


def search(question: str, tickers: list[str] | None = None, form_types: list[str] | None = None,
           date_from: date | None = None, date_to: date | None = None, k: int = 8) -> tuple[list[Passage], dict]:
    inferred = False
    if not tickers:
        tickers = infer_tickers(question) or None
        inferred = bool(tickers)
    where, params = _filters(tickers, form_types, date_from, date_to)

    semantic = semantic_candidates(question, where, params)
    lexical = lexical_candidates(question, where, params)
    fused: dict[int, dict] = {}
    for source, rows in (("semantic", semantic), ("lexical", lexical)):
        for rank, row in enumerate(rows, start=1):
            entry = fused.setdefault(row["id"], {"row": row, "score": 0.0, "retrieval": []})
            entry["score"] += 1.0 / (RRF_K + rank)
            entry["retrieval"].append(source)

    ranked = sorted(fused.values(), key=lambda item: -item["score"])[:k]
    passages = [
        Passage(
            chunk_id=item["row"]["id"],
            accession_no=item["row"]["accession_no"],
            ticker=item["row"]["ticker"],
            form_type=item["row"]["form_type"],
            filing_date=str(item["row"]["filing_date"]),
            section=item["row"]["section"],
            content=item["row"]["content"],
            url=item["row"]["document_url"],
            score=round(item["score"], 5),
            retrieval=item["retrieval"],
        )
        for item in ranked
    ]
    mode = "hybrid" if semantic and lexical else "semantic" if semantic else "lexical" if lexical else "none"
    return passages, {"mode": mode, "tickers": tickers, "tickers_inferred": inferred,
                      "semantic_candidates": len(semantic), "lexical_candidates": len(lexical)}
