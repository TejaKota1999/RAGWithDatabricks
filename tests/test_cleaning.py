"""Unit tests for src/text_cleaning.py."""

from src.text_cleaning import clean_text


def test_normalizes_line_endings_and_trailing_space():
    assert clean_text("a  \r\nb   \r\n") == "a\nb"


def test_collapses_excess_blank_lines():
    assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_preserves_headings_punctuation_and_tables():
    raw = "# Title\n\nReturns allowed within 30 days.\n\nCol1\tCol2\tCol3"
    cleaned = clean_text(raw)
    assert "# Title" in cleaned
    assert "within 30 days." in cleaned
    assert "Col1\tCol2\tCol3" in cleaned


def test_removes_null_bytes():
    assert "\x00" not in clean_text("a\x00b")


def test_empty_input():
    assert clean_text("") == ""
    assert clean_text("   \n\n  ") == ""
