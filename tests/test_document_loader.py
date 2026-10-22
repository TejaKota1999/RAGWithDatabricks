"""Unit tests for src/document_loader.py (local txt/md loading).

These tests create tiny real files in a temporary folder (pytest's `tmp_path`),
then check that load_documents reads them, skips unsupported/too-short files,
and that extract_text rejects unknown file types. No Databricks is involved.
"""

# The two functions under test.
from src.document_loader import extract_text, load_documents


# .txt and .md files should be loaded with text + metadata (content_hash, path).
def test_load_txt_and_md(tmp_path):
    # tmp_path is a fresh temporary directory pytest gives us; write two files in it.
    (tmp_path / "returns.md").write_text("# Returns\n\nReturns within 30 days.", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Some plain notes here with enough text.", encoding="utf-8")
    records = load_documents(str(tmp_path))          # load everything in the folder
    ids = {r["document_id"] for r in records}        # the document ids found
    assert ids == {"returns", "notes"}               # both files were loaded
    for r in records:
        assert r["text"]                             # non-empty text extracted
        assert len(r["content_hash"]) == 64          # SHA-256 fingerprint present
        assert r["source_path"].endswith((".md", ".txt"))  # source path recorded


# Unsupported types (e.g. .png) and too-short files should be skipped, not loaded.
def test_skips_unsupported_and_short(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")        # not a supported text type
    (tmp_path / "tiny.md").write_text("hi", encoding="utf-8")   # shorter than min_chars
    records = load_documents(str(tmp_path), min_chars=20)
    assert records == []                                         # nothing qualifies


# extract_text should raise for a file type it has no parser for.
def test_extract_text_unknown_extension_raises(tmp_path):
    p = tmp_path / "x.xyz"
    p.write_text("data", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError):                  # "xyz" is not a known extension
        extract_text(p, "xyz")


# Pointing load_documents at a folder that doesn't exist should raise clearly.
def test_missing_directory_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        load_documents("does/not/exist")
