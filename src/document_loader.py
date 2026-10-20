"""Local document loading and text extraction (zero-cost path).

Mirrors the Databricks extraction notebook (Phase 2) for a local machine:
discovers files in a directory, extracts text per file type, and produces
cleaned records with stable source metadata. Pure-Python; PDF/DOCX parsers are
imported lazily so plain .txt/.md works with no extra dependency.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

from src.text_cleaning import clean_text

SUPPORTED_EXTENSIONS = ("txt", "md", "pdf", "docx")


def _iter_docx_blocks(document):
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document).text
        elif isinstance(child, CT_Tbl):
            for row in Table(child, document).rows:
                yield "\t".join(cell.text for cell in row.cells)


def extract_text(path: Path, ext: str) -> str:
    """Extract raw text from one file. Raises on parser failure."""
    if ext in ("txt", "md"):
        return path.read_text(encoding="utf-8", errors="replace")
    if ext == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == "docx":
        from docx import Document

        return "\n".join(_iter_docx_blocks(Document(str(path))))
    raise ValueError(f"No parser for extension '{ext}'")


def load_documents(
    directory: str,
    supported_extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS,
    min_chars: int = 20,
) -> list[dict]:
    """Load, extract, and clean every supported document under ``directory``.

    Returns a list of records: document_id, file_name, text, file_type,
    source_path, content_hash, ingested_at. Files that fail to parse or are
    shorter than ``min_chars`` are skipped rather than raising, so one bad file
    does not abort the batch; callers can compare input vs output counts.
    """
    base = Path(directory)
    if not base.is_dir():
        raise FileNotFoundError(f"Not a directory: {directory}")

    records: list[dict] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lstrip(".").lower()
        if ext not in supported_extensions:
            continue
        try:
            raw = extract_text(path, ext)
        except Exception:
            continue
        cleaned = clean_text(raw)
        if len(cleaned) < min_chars:
            continue
        content = path.read_bytes()
        records.append(
            {
                "document_id": path.stem,
                "file_name": path.name,
                "text": cleaned,
                "file_type": ext,
                "source_path": str(path),
                "content_hash": hashlib.sha256(content).hexdigest(),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return records
