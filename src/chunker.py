"""Chunking logic shared by the chunking notebook (Phase 3) and tests.

Paragraph-aware packing with a configurable size and overlap, stable chunk IDs,
heading-based section tagging, and preservation of source metadata. Pure Python
with no Spark/Databricks dependency so it is unit-testable.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Iterable

_HEADING_RE = re.compile(r"^\s*(#{1,6}\s+.+|\d+(?:\.\d+)*\.?\s+\S.+)\s*$")


def chunk_document(text: str, size: int, overlap: int) -> list[str]:
    """Split one document's text into overlapping, paragraph-aware chunks.

    Whole paragraphs are packed up to ``size`` characters, carrying ``overlap``
    characters of context between chunks. A paragraph longer than ``size`` is
    hard-windowed.
    """
    if size <= 0:
        raise ValueError("size must be > 0")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and < size")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > size:
            if current:
                chunks.append(current)
                current = ""
            step = max(1, size - overlap)
            for i in range(0, len(para), step):
                chunks.append(para[i:i + size])
            continue
        if current and len(current) + 2 + len(para) > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}".strip() if tail else para
        else:
            current = f"{current}\n\n{para}".strip() if current else para
    if current:
        chunks.append(current)
    return chunks


def detect_section(chunk: str, carried: str | None) -> str | None:
    """Return the first heading inside the chunk, else the carried section."""
    for line in chunk.splitlines():
        if _HEADING_RE.match(line):
            return line.strip().lstrip("#").strip()
    return carried


def build_chunks(
    records: Iterable[dict],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
    now_iso: str | None = None,
) -> list[dict]:
    """Turn cleaned document records into chunk rows.

    Each record must provide: document_id, text, and (optionally) page_number,
    section, file_type, source_path, content_hash. Returns chunk dicts with a
    stable ``chunk_id`` and preserved source metadata. Chunks shorter than
    ``min_chunk_chars`` are dropped.
    """
    created_at = now_iso or datetime.now(timezone.utc).isoformat()
    out: list[dict] = []
    for rec in records:
        page = rec.get("page_number") if rec.get("page_number") is not None else 0
        carried_section = rec.get("section")
        idx = 0
        for piece in chunk_document(rec["text"], chunk_size, chunk_overlap):
            piece = piece.strip()
            if len(piece) < min_chunk_chars:
                continue
            carried_section = detect_section(piece, carried_section)
            out.append(
                {
                    "chunk_id": f"{rec['document_id']}:{page}:{idx:04d}",
                    "document_id": rec["document_id"],
                    "chunk_index": idx,
                    "chunk_text": piece,
                    "char_count": len(piece),
                    "section": carried_section,
                    "page_number": rec.get("page_number"),
                    "file_type": rec.get("file_type"),
                    "source_path": rec.get("source_path"),
                    "content_hash": rec.get("content_hash"),
                    "chunk_hash": hashlib.sha256(piece.encode("utf-8")).hexdigest(),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "created_at": created_at,
                }
            )
            idx += 1
    return out
