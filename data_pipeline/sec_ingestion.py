"""
SEC EDGAR filings: submissions API -> sec_filings, primary documents -> MinIO.

* Uses the official EDGAR JSON APIs with a declared User-Agent
  (SEC_USER_AGENT="Company contact@email") and stays under the SEC fair-access
  limit of 10 requests/second.
* Filing timestamps come from EDGAR's ``acceptanceDateTime`` (when the filing
  became public), not from when we downloaded it. Feature builders use this to
  avoid look-ahead.
* Registration and download are idempotent (accession number is the key).
  Downloads are resumable: failed documents keep status 'failed' with the
  error and are retried on the next run.

Run: python -m data_pipeline.sec_ingestion [--max-downloads N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import warnings
from datetime import date, datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from sqlalchemy import text

from common.config import get_settings
from common.db import get_engine
from common.logs import get_logger
from common.runs import RunContext, track_run
from common.storage import get_object_store
from common.universe import sec_filers

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

FORMS = {"10-K", "10-Q", "8-K"}
HISTORY_START = date.fromisoformat(os.getenv("SEC_HISTORY_START", "2016-01-01"))
RAG_YEARS = int(os.getenv("SEC_RAG_YEARS", "3"))
MAX_DOWNLOADS = int(os.getenv("SEC_MAX_DOWNLOADS_PER_RUN", "400"))
EXHIBIT_PATTERN = re.compile(r"ex[-_]?99", re.IGNORECASE)

# Registrants whose history sits under an additional CIK (verified on EDGAR):
# Exxon Mobil Corp (0000034088) reorganised under ExxonMobil Holdings Corp in 2026.
ADDITIONAL_CIKS = {"XOM": ["0000034088"]}


class SecClient:
    """Small EDGAR client with a global rate limit and retry/backoff."""

    min_interval = 0.15  # ~6.6 requests per second

    def __init__(self) -> None:
        agent = get_settings().sec_user_agent
        if not agent or "@" not in agent:
            raise RuntimeError("SEC_USER_AGENT must be set to 'Company contact@email' (SEC fair-access policy)")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": agent, "Accept-Encoding": "gzip, deflate"})
        self._lock = threading.Lock()
        self._last = 0.0

    def get(self, url: str) -> requests.Response:
        for attempt in range(1, 6):
            with self._lock:
                wait = self.min_interval - (time.monotonic() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
            try:
                response = self.session.get(url, timeout=60)
            except requests.RequestException as error:
                logger.warning("SEC request error %s (attempt %s): %s", url, attempt, error)
                time.sleep(2 ** attempt)
                continue
            if response.status_code == 200:
                return response
            if response.status_code in (403, 429) or response.status_code >= 500:
                logger.warning("SEC %s for %s (attempt %s)", response.status_code, url, attempt)
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            response.raise_for_status()
        raise RuntimeError(f"SEC request failed after retries: {url}")

    def json(self, url: str) -> dict:
        return self.get(url).json()


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------

def resolve_ciks(client: SecClient, wanted: list[str]) -> dict[str, str]:
    mapping = client.json("https://www.sec.gov/files/company_tickers.json")
    by_ticker = {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in mapping.values()}
    missing = [ticker for ticker in wanted if ticker not in by_ticker]
    if missing:
        logger.warning("No CIK found for %s", missing)
    return {ticker: by_ticker[ticker] for ticker in wanted if ticker in by_ticker}


def _rows_from_block(block: dict) -> list[dict]:
    keys = ["accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "form", "primaryDocument", "items"]
    columns = {key: block.get(key, []) for key in keys}
    count = len(columns["accessionNumber"])
    return [{key: (columns[key][i] if i < len(columns[key]) else None) for key in keys} for i in range(count)]


def list_filings(client: SecClient, cik: str) -> tuple[dict, list[dict]]:
    submissions = client.json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    rows = _rows_from_block(submissions["filings"]["recent"])
    for extra in submissions["filings"].get("files", []):
        if extra.get("filingTo") and date.fromisoformat(extra["filingTo"]) < HISTORY_START:
            continue
        rows.extend(_rows_from_block(client.json(f"https://data.sec.gov/submissions/{extra['name']}")))
    profile = {"sic": submissions.get("sic"), "sic_description": submissions.get("sicDescription"),
               "name": submissions.get("name")}
    return profile, rows


def document_url(cik: str, accession: str, document: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{document}"


def should_download(form: str, filing_date: date, today: date) -> bool:
    if form == "8-K":
        return True  # 8-K text feeds filing-sentiment features back to HISTORY_START
    return filing_date >= today - timedelta(days=365 * RAG_YEARS)


REGISTER_SQL = text(
    """
    INSERT INTO sec_filings (accession_no, cik, ticker, form_type, filing_date, accepted_at,
        report_date, items, primary_document, document_url, status)
    VALUES (:accession, :cik, :ticker, :form, :filing_date, :accepted_at, :report_date, :items,
        :primary_document, :url, :status)
    ON CONFLICT (accession_no) DO UPDATE SET
        accepted_at = EXCLUDED.accepted_at,
        items = EXCLUDED.items,
        status = CASE WHEN sec_filings.status = 'metadata' AND EXCLUDED.status = 'pending'
                      THEN 'pending' ELSE sec_filings.status END
    """
)


def filing_record(row: dict, ticker: str, cik: str, today: date) -> dict | None:
    form = row["form"]
    if form not in FORMS or not row["filingDate"]:
        return None
    filing_date = date.fromisoformat(row["filingDate"])
    if filing_date < HISTORY_START:
        return None
    accepted = row["acceptanceDateTime"]
    return {
        "accession": row["accessionNumber"],
        "cik": cik,
        "ticker": ticker,
        "form": form,
        "filing_date": filing_date,
        "accepted_at": datetime.fromisoformat(accepted.replace("Z", "+00:00")) if accepted else None,
        "report_date": date.fromisoformat(row["reportDate"]) if row.get("reportDate") else None,
        "items": row.get("items") or None,
        "primary_document": row["primaryDocument"],
        "url": document_url(cik, row["accessionNumber"], row["primaryDocument"]),
        "status": "pending" if should_download(form, filing_date, today) else "metadata",
    }


def register_filings(client: SecClient, ctx: RunContext) -> int:
    ciks = resolve_ciks(client, sec_filers())
    today = datetime.now(timezone.utc).date()
    registered = 0
    with get_engine().begin() as conn:
        for ticker, primary_cik in ciks.items():
            for cik in [primary_cik, *ADDITIONAL_CIKS.get(ticker, [])]:
                profile, rows = list_filings(client, cik)
                if cik == primary_cik:
                    conn.execute(
                        text(
                            """
                            UPDATE assets SET cik = :cik, sic_code = :sic, sic_description = :sic_description,
                                updated_at = now()
                            WHERE ticker = :ticker
                            """
                        ),
                        {"cik": cik, "ticker": ticker, "sic": profile["sic"], "sic_description": profile["sic_description"]},
                    )
                records = [record for row in rows if (record := filing_record(row, ticker, cik, today))]
                if records:
                    registered += conn.execute(REGISTER_SQL, records).rowcount or 0
                ctx.details.setdefault("filers", {})[f"{ticker}:{cik}"] = len(records)
    return registered


# ----------------------------------------------------------------------
# Download + text extraction
# ----------------------------------------------------------------------

HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.IGNORECASE)


def html_to_text(html: bytes | str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "title"]):
        tag.decompose()
    for tag in soup.find_all(re.compile(r"^ix:header$", re.IGNORECASE)):
        tag.decompose()
    for tag in soup.find_all(style=HIDDEN_STYLE):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(["p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"]):
        block.append("\n")
    raw = soup.get_text(" ")
    raw = raw.replace("\xa0", " ").replace("​", "")
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line or re.fullmatch(r"(page\s*)?\d{1,3}|table of contents", line, re.IGNORECASE):
            continue
        lines.append(line)
    text_value = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text_value)


def exhibit_documents(client: SecClient, cik: str, accession: str) -> list[str]:
    index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/index.json"
    try:
        items = client.json(index_url).get("directory", {}).get("item", [])
    except (RuntimeError, ValueError, requests.RequestException) as error:
        logger.warning("Filing index unavailable for %s: %s", accession, error)
        return []
    names = [item["name"] for item in items if item.get("name", "").lower().endswith((".htm", ".html"))]
    return [name for name in names if EXHIBIT_PATTERN.search(name)][:2]


def download_filing(client: SecClient, filing: dict) -> tuple[str, int]:
    store = get_object_store()
    base = f"sec/{filing['ticker']}/{filing['accession_no']}"
    primary = client.get(filing["document_url"]).content
    raw_key = store.put_bytes(f"{base}/{filing['primary_document']}", primary, "text/html")
    parts = [html_to_text(primary)]
    if filing["form_type"] == "8-K":
        for name in exhibit_documents(client, filing["cik"], filing["accession_no"]):
            exhibit = client.get(document_url(filing["cik"], filing["accession_no"], name)).content
            store.put_bytes(f"{base}/{name}", exhibit, "text/html")
            parts.append(f"EXHIBIT {name}\n" + html_to_text(exhibit))
    full_text = "\n\n".join(part for part in parts if part)
    store.put_bytes(f"{base}/text.txt", full_text.encode("utf-8"), "text/plain; charset=utf-8")
    return raw_key, len(full_text)


def filing_text(ticker: str, accession: str) -> str:
    return get_object_store().get_bytes(f"sec/{ticker}/{accession}/text.txt").decode("utf-8")


def download_pending(client: SecClient, ctx: RunContext, limit: int) -> int:
    with get_engine().connect() as conn:
        pending = conn.execute(
            text(
                """
                SELECT accession_no, cik, ticker, form_type, primary_document, document_url
                FROM sec_filings
                WHERE status IN ('pending', 'failed')
                ORDER BY (form_type <> '8-K') DESC, filing_date DESC
                LIMIT :limit
                """
            ),
            {"limit": limit},
        ).mappings().all()

    downloaded = 0
    for filing in pending:
        try:
            raw_key, chars = download_filing(client, dict(filing))
            status, error = ("downloaded", None) if chars > 200 else ("failed", "extracted text too short")
        except Exception as exc:  # noqa: BLE001 - keep going, record the failure
            raw_key, chars, status, error = None, None, "failed", str(exc)[:1000]
            ctx.warn(f"{filing['accession_no']}: {error}")
        with get_engine().begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE sec_filings SET status = :status, error = :error, raw_object_key = COALESCE(:raw_key, raw_object_key),
                        text_chars = COALESCE(:chars, text_chars), processed_at = now()
                    WHERE accession_no = :accession
                    """
                ),
                {"status": status, "error": error, "raw_key": raw_key, "chars": chars, "accession": filing["accession_no"]},
            )
        downloaded += status == "downloaded"
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-downloads", type=int, default=MAX_DOWNLOADS)
    args = parser.parse_args()
    with track_run("sec_filings", "ingest_filings") as ctx:
        client = SecClient()
        ctx.details["registered_or_updated"] = register_filings(client, ctx)
        count = download_pending(client, ctx, args.max_downloads)
        ctx.add_rows(count)
        with get_engine().connect() as conn:
            ctx.details["status_counts"] = json.loads(
                json.dumps(dict(conn.execute(text("SELECT status, COUNT(*) FROM sec_filings GROUP BY status")).all()))
            )


if __name__ == "__main__":
    main()
