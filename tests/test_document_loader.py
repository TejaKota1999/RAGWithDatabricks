"""Unit tests for src/document_loader.py (local txt/md loading)."""

from src.document_loader import extract_text, load_documents


def test_load_txt_and_md(tmp_path):
    (tmp_path / "returns.md").write_text("# Returns\n\nReturns within 30 days.", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Some plain notes here with enough text.", encoding="utf-8")
    records = load_documents(str(tmp_path))
    ids = {r["document_id"] for r in records}
    assert ids == {"returns", "notes"}
    for r in records:
        assert r["text"]
        assert len(r["content_hash"]) == 64
        assert r["source_path"].endswith((".md", ".txt"))


def test_skips_unsupported_and_short(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "tiny.md").write_text("hi", encoding="utf-8")  # below min_chars
    records = load_documents(str(tmp_path), min_chars=20)
    assert records == []


def test_extract_text_unknown_extension_raises(tmp_path):
    p = tmp_path / "x.xyz"
    p.write_text("data", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError):
        extract_text(p, "xyz")


def test_missing_directory_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        load_documents("does/not/exist")
