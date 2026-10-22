"""Unit tests for src/document_loader.py (local txt/md loading).

These tests create tiny real files in a temporary folder (pytest's `tmp_path`),
then check that load_documents reads them, skips unsupported/too-short files,
and that extract_text rejects unknown file types. No Databricks is involved.
"""

# The two functions under test.
from src.document_loader import extract_text, load_documents


# .txt and .md files should be loaded with text + metadata (content_hash, path).
def test_load_txt_and_md(tmp_path):
    # Example: a folder with "returns.md" and "notes.txt" ->
    #   load_documents returns 2 records with document_ids {"returns","notes"},
    #   each having extracted text, a 64-char content_hash, and a source_path.
    (tmp_path / "returns.md").write_text("# Returns\n\nReturns within 30 days.", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Some plain notes here with enough text.", encoding="utf-8")
    records = load_documents(str(tmp_path))
    ids = {r["document_id"] for r in records}
    assert ids == {"returns", "notes"}
    for r in records:
        assert r["text"]
        assert len(r["content_hash"]) == 64
        assert r["source_path"].endswith((".md", ".txt"))


# Unsupported types (e.g. .png) and too-short files should be skipped, not loaded.
def test_skips_unsupported_and_short(tmp_path):
    # Example: a folder with "image.png" (not text) and "tiny.md" ("hi", too short
    #   for min_chars=20) -> load_documents returns []  (both are skipped).
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "tiny.md").write_text("hi", encoding="utf-8")
    records = load_documents(str(tmp_path), min_chars=20)
    assert records == []


# extract_text should raise for a file type it has no parser for.
def test_extract_text_unknown_extension_raises(tmp_path):
    # Example: a file "x.xyz" -> extract_text(..., "xyz") -> ValueError
    #   (there is no parser for the ".xyz" extension).
    p = tmp_path / "x.xyz"
    p.write_text("data", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError):
        extract_text(p, "xyz")


# Pointing load_documents at a folder that doesn't exist should raise clearly.
def test_missing_directory_raises():
    # Example: load_documents("does/not/exist") -> FileNotFoundError
    #   (a clear error instead of silently returning nothing).
    import pytest

    with pytest.raises(FileNotFoundError):
        load_documents("does/not/exist")
