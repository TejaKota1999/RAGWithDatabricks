"""Unit tests for src/chunker.py.

Checks the chunking logic: splitting text into size-limited, overlapping pieces,
detecting section headings, and building chunk rows with stable ids + metadata.
All inputs are invented here; no Databricks or files are read.
"""

# The three functions under test.
from src.chunker import build_chunks, chunk_document, detect_section


# A short text that fits in one chunk should come back as a single chunk.
def test_short_text_is_single_chunk():
    chunks = chunk_document("A short paragraph.", size=1000, overlap=100)
    assert chunks == ["A short paragraph."]   # exactly one chunk, unchanged


# When text is long, it must be split, and each packed chunk stays within `size`.
def test_respects_size_across_paragraphs():
    # Build 20 short paragraphs joined by blank lines.
    paras = "\n\n".join(f"Paragraph number {i} with some filler text." for i in range(20))
    chunks = chunk_document(paras, size=120, overlap=20)
    assert len(chunks) > 1                     # it actually split into several
    # Every packed chunk should be at most `size` characters long.
    assert all(len(c) <= 120 for c in chunks)


# Overlap repeats a bit of context between chunks, so it never REDUCES the count.
def test_overlap_carries_context():
    paras = "\n\n".join(f"Sentence {i} aaaa bbbb cccc." for i in range(10))
    no_overlap = chunk_document(paras, size=80, overlap=0)    # no repeated context
    with_overlap = chunk_document(paras, size=80, overlap=30)  # repeats 30 chars
    # With overlap you get the same number of chunks or more (never fewer).
    assert len(with_overlap) >= len(no_overlap)


# A single paragraph bigger than `size` must be cut into fixed windows.
def test_oversized_paragraph_is_windowed():
    big = "x" * 250                             # one 250-char "paragraph"
    chunks = chunk_document(big, size=100, overlap=10)
    assert len(chunks) >= 3                     # 250 chars -> at least 3 windows
    assert all(len(c) <= 100 for c in chunks)   # none exceeds the window size


# Bad settings must raise an error instead of producing nonsense.
def test_invalid_params_raise():
    import pytest

    with pytest.raises(ValueError):             # size must be > 0
        chunk_document("abc", size=0, overlap=0)
    with pytest.raises(ValueError):             # overlap must be smaller than size
        chunk_document("abc", size=100, overlap=100)


# detect_section should find a heading, or keep the previous one if there's none.
def test_detect_section_markdown_and_numbered():
    assert detect_section("## Returns\nbody", None) == "Returns"          # markdown heading
    assert detect_section("4. Definitions\ntext", None) == "4. Definitions"  # numbered heading
    assert detect_section("no heading here", "Prev") == "Prev"           # no heading -> carry previous


# build_chunks should produce stable, ordered ids and keep all source metadata.
def test_build_chunks_stable_ids_and_metadata():
    # One cleaned-document record with 8 paragraphs.
    records = [
        {
            "document_id": "policy",
            "text": "\n\n".join(f"Para {i} with enough words to count." for i in range(8)),
            "page_number": None,
            "section": None,
            "file_type": "md",
            "source_path": "/Volumes/x/y/policy.md",
            "content_hash": "deadbeef",
        }
    ]
    rows = build_chunks(records, chunk_size=100, chunk_overlap=20, min_chunk_chars=10)
    assert rows, "expected at least one chunk"
    # Ids are stable and zero-padded, in order: <doc>:<page>:<index>.
    assert rows[0]["chunk_id"] == "policy:0:0000"
    assert rows[1]["chunk_id"] == "policy:0:0001"
    # Every chunk keeps the document's source metadata (traceability).
    for r in rows:
        assert r["document_id"] == "policy"
        assert r["source_path"] == "/Volumes/x/y/policy.md"
        assert r["content_hash"] == "deadbeef"
        assert len(r["chunk_hash"]) == 64       # SHA-256 hex is 64 chars


# Chunks from two different documents must never be mixed together.
def test_build_chunks_does_not_mix_documents():
    records = [
        {"document_id": "a", "text": "Alpha paragraph one. " * 10},
        {"document_id": "b", "text": "Bravo paragraph two. " * 10},
    ]
    rows = build_chunks(records, chunk_size=120, chunk_overlap=20, min_chunk_chars=10)
    # A chunk from document "a" must not contain document "b"'s words, and vice versa.
    for r in rows:
        if r["document_id"] == "a":
            assert "Bravo" not in r["chunk_text"]
        else:
            assert "Alpha" not in r["chunk_text"]


# Chunks shorter than min_chunk_chars (e.g. headers/footers) should be dropped.
def test_build_chunks_drops_tiny_fragments():
    records = [{"document_id": "d", "text": "ok"}]   # only 2 characters of text
    rows = build_chunks(records, chunk_size=100, chunk_overlap=0, min_chunk_chars=50)
    assert rows == []                                # too short -> no chunks kept
