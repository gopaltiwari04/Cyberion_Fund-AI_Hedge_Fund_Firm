from ml_core.rag.analyst import verify
from ml_core.rag.chunking import chunk_filing, chunk_text, split_sections
from ml_core.rag.retriever import Passage, infer_tickers, lexical_query

PARAGRAPH = "The Company faces intense competition in all of its markets and product categories. " * 8

TEN_K = "\n".join([
    "Table of Contents", "Item 1. Business 3", "Item 1A. Risk Factors 10", "Item 7. MD&A 30",
    "PART I", "Item 1. Business", PARAGRAPH, PARAGRAPH,
    "Item 1A. Risk Factors", PARAGRAPH, PARAGRAPH, PARAGRAPH,
    "Item 7. Management's Discussion and Analysis", "Net sales were $94,930 million, up 6%.", PARAGRAPH,
    "Item 8. Financial Statements", "Revenue 94,930 85,777 " * 50,
])


def test_ten_k_sections_skip_toc_and_financial_statements():
    sections = dict(split_sections(TEN_K, "10-K"))
    assert set(sections) == {"Item 1. Business", "Item 1A. Risk Factors", "Item 7. Management's Discussion and Analysis"}
    assert "94,930 million" in sections["Item 7. Management's Discussion and Analysis"]


def test_chunks_respect_size_and_overlap():
    body = "\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(30))
    chunks = chunk_text("Item 1A. Risk Factors", body, size=1200, overlap=200)
    assert len(chunks) > 3
    assert all(len(chunk.content) <= 1600 for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    carried = chunks[1].content.split("\n")[0]
    assert len(carried) <= 200 and chunks[0].content.endswith(carried)  # overlap carries the end of the previous chunk


def test_fallback_to_headings_when_items_cover_little():
    body = "\n".join(["Executive Overview", "Revenue increased due to higher net interest income. " * 6] * 20)
    # Bank-style layout: Item headings only in a cross-reference index, narrative under its own headings.
    text_value = "Item 1. Financial Statements\nItem 2. Management's Discussion and Analysis\nItem 6. Exhibits\n" + body
    sections = split_sections(text_value, "10-Q")
    assert sections and all(label.startswith("10-Q") for label, _ in sections)


def test_eight_k_kept_whole():
    chunks = chunk_filing("Item 2.02 Results of Operations\n" + PARAGRAPH * 3, "8-K")
    assert chunks and chunks[0].section == "Current report (8-K)"


def passage(content, index=1):
    return Passage(chunk_id=index, accession_no="0000-1", ticker="AAPL", form_type="10-K", filing_date="2025-10-31",
                   section="Item 7", content=content, url="https://www.sec.gov/x", score=0.1, retrieval=["lexical"])


def test_verify_rejects_missing_and_invented_citations_and_flags_numbers():
    passages = [passage("Net sales were $94,930 million, an increase of 6%."), passage("Risks include tariffs.", 2)]
    output = {"statements": [
        {"text": "Net sales were $94,930 million.", "type": "fact", "citations": ["S1"]},
        {"text": "Net sales were $99,999 million.", "type": "fact", "citations": ["S1"]},
        {"text": "Tariffs may pressure margins.", "type": "inference", "citations": ["S2", "S9"]},
        {"text": "The stock will rise.", "type": "inference", "citations": []},
        {"text": "Made up", "type": "fact", "citations": ["S7"]},
    ]}
    accepted, rejected = verify(output, passages)
    assert [s["verification"] for s in accepted] == ["cited", "unverified_numbers", "cited"]
    assert accepted[1]["unverified_numbers"] == ["99,999"]
    assert accepted[2]["invalid_citations"] == ["S9"]
    assert {r["reason"] for r in rejected} == {"no valid citation"}
    assert len(rejected) == 2


def test_query_helpers():
    assert infer_tickers("What risks did Apple and JPMorgan disclose?") == ["AAPL", "JPM"]
    assert infer_tickers("metadata pipelines") == []  # 'meta' inside a word must not match
    query = lexical_query("What are the main risk factors for the company's supply chain?")
    assert "risk" in query and "the" not in query.split(" | ")
