"""Turns a knowledge-base file's bytes into plain text, then overlapping chunks to embed. Pure text
processing, no DB or engine (the next step: document_ingest.py).
"""

import csv
import hashlib
import json
from html.parser import HTMLParser
from io import BytesIO, StringIO

import numpy as np
from docx import Document as DocxFile
from odf import teletype
from odf.opendocument import load as load_odf
from odf.text import P as OdfParagraph
from pypdf import PdfReader
from striprtf.striprtf import rtf_to_text

from app.config import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE

# File types the folder scanner can read. Anything else is ignored, not an error: these are plain folders
# an admin may also use for other files (e.g. a README).
SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".rtf",
    ".html",
    ".htm",
    ".csv",
    ".json",
    ".odt",
}


class _HTMLTextExtractor(HTMLParser):
    """Minimal stdlib HTML-to-text: strips tags/scripts/styles, keeps visible text (formatting is irrelevant once
    embedded)."""

    def __init__(self):
        super().__init__()
        self._chunks = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._chunks.append(data)

    def get_text(self) -> str:
        return " ".join(self._chunks)


def extract_text(raw: bytes, suffix: str) -> str:
    """Turns a file's bytes into plain text to chunk and embed; each format needs its own parsing (only.txt/.md are
    plain UTF-8)."""
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(raw))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        doc = DocxFile(BytesIO(raw))
        return "\n\n".join(paragraph.text for paragraph in doc.paragraphs)

    if suffix == ".odt":
        odf_doc = load_odf(BytesIO(raw))
        paragraphs = odf_doc.getElementsByType(OdfParagraph)
        return "\n\n".join(teletype.extractText(p) for p in paragraphs)

    if suffix == ".rtf":
        return rtf_to_text(raw.decode("utf-8", errors="ignore"))

    if suffix in (".html", ".htm"):
        parser = _HTMLTextExtractor()
        parser.feed(raw.decode("utf-8", errors="ignore"))
        return parser.get_text()

    if suffix == ".csv":
        text = raw.decode("utf-8", errors="ignore")
        # Space-joined rows rather than raw CSV, so cells read naturally once chunked.
        return "\n".join(" ".join(row) for row in csv.reader(StringIO(text)))

    if suffix == ".json":
        data = json.loads(raw.decode("utf-8", errors="ignore"))
        return json.dumps(data, indent=2, ensure_ascii=False)

    return raw.decode("utf-8", errors="ignore")


def hash_bytes(raw: bytes) -> str:
    """Content fingerprint to skip unchanged files instead of re-embedding on every sync."""
    return hashlib.sha256(raw).hexdigest()


def chunk_text(text: str, size: int = RAG_CHUNK_SIZE, overlap: int = RAG_CHUNK_OVERLAP) -> list[str]:
    """Splits `text` into overlapping windows of ~`size` characters; the overlap keeps a sentence on a boundary whole in
    at least one chunk."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def vector_to_bytes(vector: list[float]) -> bytes:
    """Serializes an embedding vector to raw bytes for SQLite storage."""
    return np.asarray(vector, dtype=np.float32).tobytes()
