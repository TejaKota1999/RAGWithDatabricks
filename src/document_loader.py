# Databricks notebook source
"""Local document loading and text extraction (zero-cost path).

Mirrors the Databricks extraction notebook (Phase 2) for a local machine:
discovers files in a directory, extracts text per file type, and produces
cleaned records with stable source metadata. Pure-Python; PDF/DOCX parsers are
imported lazily so plain .txt/.md works with no extra dependency.
"""

from __future__ import annotations

# hashlib: content fingerprint; os: unused-but-handy path helpers; datetime: timestamp;
# Path: a friendly way to work with file paths.
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

# Reuse the exact same cleaning function as the notebook.
from src.text_cleaning import clean_text

# File types we know how to read.
SUPPORTED_EXTENSIONS = ("txt", "md", "pdf", "docx")


# COMMAND ----------
# Read a Word document's paragraphs and tables in order (keeps tables inline).
def _iter_docx_blocks(document):
    # Low-level Word building blocks.
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in document.element.body.iterchildren():   # walk the document body in order
        if isinstance(child, CT_P):                       # a paragraph...
            yield Paragraph(child, document).text          # ...yield its text
        elif isinstance(child, CT_Tbl):                   # a table...
            for row in Table(child, document).rows:        # ...go row by row
                yield "\t".join(cell.text for cell in row.cells)  # join cells with tabs


# COMMAND ----------
# Choose the right parser for a file and return its raw text.
def extract_text(path: Path, ext: str) -> str:
    """Extract raw text from one file. Raises on parser failure."""
    if ext in ("txt", "md"):                               # plain text / Markdown
        return path.read_text(encoding="utf-8", errors="replace")
    if ext == "pdf":                                       # PDF -> join all pages
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == "docx":                                     # Word -> join all blocks
        from docx import Document

        return "\n".join(_iter_docx_blocks(Document(str(path))))
    raise ValueError(f"No parser for extension '{ext}'")  # unknown type


# COMMAND ----------
# Load every supported document under a folder and return cleaned records.
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
    if not base.is_dir():                                  # the folder must exist
        raise FileNotFoundError(f"Not a directory: {directory}")

    records: list[dict] = []
    for path in sorted(base.rglob("*")):                   # every file under the folder (recursively)
        if not path.is_file():                             # skip sub-folders
            continue
        ext = path.suffix.lstrip(".").lower()              # the extension, e.g. "docx"
        if ext not in supported_extensions:                # skip types we do not handle
            continue
        try:
            raw = extract_text(path, ext)                  # read the text
        except Exception:                                  # log-and-skip on parser errors
            continue
        cleaned = clean_text(raw)                          # tidy the text
        if len(cleaned) < min_chars:                       # skip near-empty files
            continue
        content = path.read_bytes()                        # raw bytes for the fingerprint
        records.append(
            {
                "document_id": path.stem,                  # file name without extension
                "file_name": path.name,
                "text": cleaned,
                "file_type": ext,
                "source_path": str(path),                  # where it came from
                "content_hash": hashlib.sha256(content).hexdigest(),  # change-detection fingerprint
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return records
