"""Unit tests for src/chunker.py."""

from src.chunker import build_chunks, chunk_document, detect_section


def test_short_text_is_single_chunk():
    chunks = chunk_document("A short paragraph.", size=1000, overlap=100)
    assert chunks == ["A short paragraph."]


def test_respects_size_across_paragraphs():
    paras = "\n\n".join(f"Paragraph number {i} with some filler text." for i in range(20))
    chunks = chunk_document(paras, size=120, overlap=20)
    assert len(chunks) > 1
    # Packed chunks (not the hard-windowed case) stay within size.
    assert all(len(c) <= 120 for c in chunks)


def test_overlap_carries_context():
    paras = "\n\n".join(f"Sentence {i} aaaa bbbb cccc." for i in range(10))
    no_overlap = chunk_document(paras, size=80, overlap=0)
    with_overlap = chunk_document(paras, size=80, overlap=30)
    # Overlap duplicates context, so it never produces fewer chunks.
    assert len(with_overlap) >= len(no_overlap)


def test_oversized_paragraph_is_windowed():
    big = "x" * 250
    chunks = chunk_document(big, size=100, overlap=10)
    assert len(chunks) >= 3
    assert all(len(c) <= 100 for c in chunks)


def test_invalid_params_raise():
    import pytest

    with pytest.raises(ValueError):
        chunk_document("abc", size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_document("abc", size=100, overlap=100)


def test_detect_section_markdown_and_numbered():
    assert detect_section("## Returns\nbody", None) == "Returns"
    assert detect_section("4. Definitions\ntext", None) == "4. Definitions"
    assert detect_section("no heading here", "Prev") == "Prev"


def test_build_chunks_stable_ids_and_metadata():
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
    # Stable, zero-padded ids in order.
    assert rows[0]["chunk_id"] == "policy:0:0000"
    assert rows[1]["chunk_id"] == "policy:0:0001"
    # Metadata preserved on every chunk.
    for r in rows:
        assert r["document_id"] == "policy"
        assert r["source_path"] == "/Volumes/x/y/policy.md"
        assert r["content_hash"] == "deadbeef"
        assert len(r["chunk_hash"]) == 64


def test_build_chunks_does_not_mix_documents():
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


def test_build_chunks_drops_tiny_fragments():
    records = [{"document_id": "d", "text": "ok"}]  # 2 chars
    rows = build_chunks(records, chunk_size=100, chunk_overlap=0, min_chunk_chars=50)
    assert rows == []
