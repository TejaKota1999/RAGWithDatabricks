"""Unit tests for src/chunker.py.

Checks the chunking logic: splitting text into size-limited, overlapping pieces,
detecting section headings, and building chunk rows with stable ids + metadata.
All inputs are invented here; no Databricks or files are read.
"""

# The three functions under test.
from src.offline.chunker import build_chunks, chunk_document, detect_section


# A short text that fits in one chunk should come back as a single chunk.
def test_short_text_is_single_chunk():
    # Example: a one-line note "A short paragraph." with size=1000
    #   -> ["A short paragraph."]   (small enough to stay as one chunk)
    chunks = chunk_document("A short paragraph.", size=1000, overlap=100)
    assert chunks == ["A short paragraph."]


# When text is long, it must be split, and each packed chunk stays within `size`.
def test_respects_size_across_paragraphs():
    # Example: a 20-paragraph policy packed into size=120 chunks
    #   -> several chunks, each <= 120 chars (whole paragraphs packed together)
    paras = "\n\n".join(f"Paragraph number {i} with some filler text." for i in range(20))
    chunks = chunk_document(paras, size=120, overlap=20)
    assert len(chunks) > 1                     # it actually split into several
    assert all(len(c) <= 120 for c in chunks)  # every packed chunk <= size


# Overlap repeats a bit of context between chunks, so it never REDUCES the count.
def test_overlap_carries_context():
    # Example: same text chunked with overlap=0 gives e.g. 4 chunks; with
    #   overlap=30 it repeats 30 chars at each seam, giving 4 or more chunks.
    paras = "\n\n".join(f"Sentence {i} aaaa bbbb cccc." for i in range(10))
    no_overlap = chunk_document(paras, size=80, overlap=0)
    with_overlap = chunk_document(paras, size=80, overlap=30)
    assert len(with_overlap) >= len(no_overlap)


# A single paragraph bigger than `size` must be cut into fixed windows.
def test_oversized_paragraph_is_windowed():
    # Example: one 250-char blob with size=100 -> at least 3 windows
    #   "xxxx...(100)", "xxxx...(100)", "xxxx..."  (each <= 100 chars)
    big = "x" * 250
    chunks = chunk_document(big, size=100, overlap=10)
    assert len(chunks) >= 3
    assert all(len(c) <= 100 for c in chunks)


# Bad settings must raise an error instead of producing nonsense.
def test_invalid_params_raise():
    # Example: size=0 (a chunk of zero length makes no sense) -> ValueError;
    #   overlap=100 with size=100 (overlap must be smaller) -> ValueError.
    import pytest

    with pytest.raises(ValueError):
        chunk_document("abc", size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_document("abc", size=100, overlap=100)


# detect_section should find a heading, or keep the previous one if there's none.
def test_detect_section_markdown_and_numbered():
    # Example: "## Returns\nbody"      -> "Returns"         (markdown heading)
    #          "4. Definitions\ntext"  -> "4. Definitions"  (numbered heading)
    #          "no heading here" (prev="Prev") -> "Prev"    (carry previous)
    assert detect_section("## Returns\nbody", None) == "Returns"
    assert detect_section("4. Definitions\ntext", None) == "4. Definitions"
    assert detect_section("no heading here", "Prev") == "Prev"


# build_chunks should produce stable, ordered ids and keep all source metadata.
def test_build_chunks_stable_ids_and_metadata():
    # Example: a doc "policy.md" with 8 paragraphs ->
    #   chunk ids "policy:0:0000", "policy:0:0001", ...  (document:page:index)
    #   and each chunk still carries source_path + content_hash of the file.
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
    assert rows[0]["chunk_id"] == "policy:0:0000"   # stable, zero-padded, in order
    assert rows[1]["chunk_id"] == "policy:0:0001"
    for r in rows:
        assert r["document_id"] == "policy"
        assert r["source_path"] == "/Volumes/x/y/policy.md"
        assert r["content_hash"] == "deadbeef"
        assert len(r["chunk_hash"]) == 64           # SHA-256 hex is 64 chars


# Chunks from two different documents must never be mixed together.
def test_build_chunks_does_not_mix_documents():
    # Example: doc "a" (all "Alpha...") and doc "b" (all "Bravo...") ->
    #   every chunk belongs to exactly one doc; an "a" chunk never contains "Bravo".
    records = [
        {"document_id": "a", "text": "Alpha paragraph one. " * 10},
        {"document_id": "b", "text": "Bravo paragraph two. " * 10},
    ]
    rows = build_chunks(records, chunk_size=120, chunk_overlap=20, min_chunk_chars=10)
    for r in rows:
        if r["document_id"] == "a":
            assert "Bravo" not in r["chunk_text"]
        else:
            assert "Alpha" not in r["chunk_text"]


# Chunks shorter than min_chunk_chars (e.g. headers/footers) should be dropped.
def test_build_chunks_drops_tiny_fragments():
    # Example: a doc whose only text is "ok" (2 chars) with min_chunk_chars=50
    #   -> []  (too small to be a useful chunk, so nothing is kept)
    records = [{"document_id": "d", "text": "ok"}]
    rows = build_chunks(records, chunk_size=100, chunk_overlap=0, min_chunk_chars=50)
    assert rows == []
