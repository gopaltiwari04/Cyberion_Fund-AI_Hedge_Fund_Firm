"""SEC research analyst (RAG with verified citations)."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.deps import get_db, jsonable
from api.ratelimit import limiter
from api.schemas import AnalystQuestion
from api.security import User, optional_user, require_user
from common.logs import get_logger
from ml_core.rag.analyst import answer

logger = get_logger(__name__)
router = APIRouter(prefix="/analyst", tags=["analyst"])


@router.post("/ask")
def ask(payload: AnalystQuestion, user: User = Depends(require_user),
        _: None = Depends(limiter("analyst", "analyst_daily")), db: Session = Depends(get_db)):
    result = jsonable(answer(payload.question, payload.tickers, payload.form_types, payload.date_from, payload.date_to))
    try:
        db.execute(
            text(
                "INSERT INTO research_queries (email, ticker, question, answer, latency_ms) "
                "VALUES (:e, :t, :q, CAST(:a AS jsonb), :l)"
            ),
            {"e": user.email, "t": ",".join(payload.tickers or []) or None, "q": payload.question,
             "a": json.dumps({"status": result["status"], "summary": result.get("summary"),
                              "statements": len(result.get("statements") or []),
                              "rejected": len(result.get("rejected_statements") or [])}),
             "l": result.get("latency_ms")},
        )
        db.commit()
    except Exception as error:  # noqa: BLE001 - auditing must not break the answer
        logger.warning("Could not log research query: %s", error)
        db.rollback()
    return result


@router.get("/coverage", dependencies=[Depends(optional_user)])
def coverage(db: Session = Depends(get_db)):
    rows = db.execute(
        text(
            """
            SELECT f.ticker, f.form_type, COUNT(DISTINCT f.accession_no) AS filings, MIN(f.filing_date) AS first,
                   MAX(f.filing_date) AS last, COALESCE(SUM(f.chunk_count), 0) AS chunks
            FROM sec_filings f WHERE f.status = 'indexed'
            GROUP BY f.ticker, f.form_type ORDER BY f.ticker, f.form_type
            """
        )
    ).mappings().all()
    return {
        "coverage": [jsonable(dict(row)) for row in rows],
        "scope_note": "Indexed: narrative sections of 10-K/10-Q (business, risk factors, MD&A, market risk, legal, "
                      "controls) and full 8-K text incl. press-release exhibits, for the last 3 years. Financial "
                      "statement tables are not indexed.",
    }
