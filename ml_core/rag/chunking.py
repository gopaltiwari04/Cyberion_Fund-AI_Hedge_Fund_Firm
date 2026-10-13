"""
Section-aware chunking of SEC filings.

10-K / 10-Q documents are split on their "Item" headings so every chunk
carries a citation-friendly section label (e.g. "Item 1A. Risk Factors").
Financial-statement sections and governance items incorporated by reference
are excluded to keep the index focused on narrative disclosures and bounded
in size; this limitation is surfaced to users in the analyst UI.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

ITEM_HEADING = re.compile(r"^\s*item\s*(\d{1,2}[a-c]?)\s*[\.:\-–—]?\s*(.{0,160})$", re.IGNORECASE)
PART_HEADING = re.compile(r"^\s*part\s+(iv|i{1,3})\b", re.IGNORECASE)

TEN_K_SECTIONS = {
    "1": "Business",
    "1A": "Risk Factors",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "5": "Market for Registrant's Common Equity",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "9A": "Controls and Procedures",
}
TEN_Q_SECTIONS = {
    ("I", "2"): "Management's Discussion and Analysis",
    ("I", "3"): "Quantitative and Qualitative Disclosures About Market Risk",
    ("I", "4"): "Controls and Procedures",
    ("II", "1"): "Legal Proceedings",
    ("II", "1A"): "Risk Factors",
    ("II", "2"): "Unregistered Sales of Equity Securities and Use of Proceeds",
    ("II", "5"): "Other Information",
}

MIN_SECTION_CHARS = 400


@dataclass(frozen=True)
class Chunk:
    section: str
    index: int
    content: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


def split_sections(text: str, form_type: str) -> list[tuple[str, str]]:
    """Return (section_label, section_text) pairs in document order."""
    if form_type == "8-K":
        return [("Current report (8-K)", text)]

    sections: list[tuple[str, list[str]]] = [("Cover", [])]
    part = "I"
    for line in text.splitlines():
        part_match = PART_HEADING.match(line)
        if part_match and len(line) < 120:
            part = part_match.group(1).upper()
        item_match = ITEM_HEADING.match(line)
        if item_match and len(line) < 200:
            number = item_match.group(1).upper()
            if form_type == "10-K":
                label = TEN_K_SECTIONS.get(number)
                sections.append((f"Item {number}. {label}" if label else "", []))
            else:
                label = TEN_Q_SECTIONS.get((part, number))
                sections.append((f"Part {part}, Item {number}. {label}" if label else "", []))
            continue
        sections[-1][1].append(line)

    result = []
    for label, lines in sections:
        body = "\n".join(lines).strip()
        if label and label != "Cover" and len(body) >= MIN_SECTION_CHARS:
            result.append((label, body))

    # Some registrants (e.g. large banks) present MD&A under their own headings
    # and list "Item" numbers only in a cross-reference index. When the Item
    # structure covers too little of the document, fall back to headings.
    if sum(len(body) for _, body in result) < 0.25 * len(text):
        return split_on_headings(text, form_type)
    return result


def _looks_like_heading(line: str) -> bool:
    words = line.split()
    return (
        3 <= len(line) <= 90
        and 1 <= len(words) <= 12
        and line[0].isupper()
        and not line.rstrip().endswith((".", ",", ";", ":"))
        and sum(character.isdigit() for character in line) < len(line) * 0.2
    )


def split_on_headings(text: str, form_type: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = [(f"{form_type} (general)", [])]
    lines = text.splitlines()
    skip = int(len(lines) * 0.02)  # cover page boilerplate
    for position, line in enumerate(lines[skip:], start=skip):
        following = lines[position + 1] if position + 1 < len(lines) else ""
        current_size = sum(len(existing) for existing in sections[-1][1])
        if _looks_like_heading(line) and len(following) > 150 and current_size >= MIN_SECTION_CHARS:
            sections.append((f"{form_type}: {line.strip()}", []))
            continue
        sections[-1][1].append(line)
    return [
        (label, "\n".join(body).strip())
        for label, body in sections
        if len("\n".join(body)) >= MIN_SECTION_CHARS
    ]


def _split_long(paragraph: str, size: int) -> list[str]:
    if len(paragraph) <= size:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?;])\s+", paragraph)
    pieces, current = [], ""
    for sentence in sentences:
        while len(sentence) > size:
            pieces.append(sentence[:size])
            sentence = sentence[size:]
        if len(current) + len(sentence) + 1 > size and current:
            pieces.append(current.strip())
            current = ""
        current += sentence + " "
    if current.strip():
        pieces.append(current.strip())
    return pieces


def chunk_text(section: str, body: str, start_index: int = 0, size: int = 1200, overlap: int = 200) -> list[Chunk]:
    paragraphs = [piece for paragraph in body.split("\n") if paragraph.strip() for piece in _split_long(paragraph.strip(), size)]
    chunks: list[Chunk] = []
    current: list[str] = []
    length = 0
    for paragraph in paragraphs:
        if length + len(paragraph) > size and current:
            chunks.append(Chunk(section, start_index + len(chunks), "\n".join(current)))
            tail: list[str] = []
            tail_length = 0
            for previous in reversed(current):
                if tail_length + len(previous) > overlap:
                    if not tail:  # carry the end of a long paragraph, starting at a word boundary
                        piece = previous[-overlap:]
                        piece = piece[piece.find(" ") + 1:] if " " in piece else piece
                        tail, tail_length = [piece], len(piece)
                    break
                tail.insert(0, previous)
                tail_length += len(previous)
            current, length = tail, tail_length
        current.append(paragraph)
        length += len(paragraph)
    if current and length > 80:
        chunks.append(Chunk(section, start_index + len(chunks), "\n".join(current)))
    return chunks


def chunk_filing(text: str, form_type: str, max_chunks: int = 400) -> list[Chunk]:
    chunks: list[Chunk] = []
    for label, body in split_sections(text, form_type):
        chunks.extend(chunk_text(label, body, start_index=len(chunks)))
        if len(chunks) >= max_chunks:
            break
    return chunks[:max_chunks]
