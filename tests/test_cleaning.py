"""Unit tests for src/text_cleaning.py.

Each test feeds a known messy string to clean_text() and checks the tidied
result. clean_text only normalizes whitespace; it must NOT delete real content.
No Databricks is involved - the inputs are written right here in the tests.
"""

# Import the one function we are testing.
from src.offline.text_cleaning import clean_text


# Trailing spaces should be removed and Windows line endings (\r\n) become \n.
def test_normalizes_line_endings_and_trailing_space():
    # Example (real-time): a file saved on Windows gives "Hello  \r\nWorld   \r\n".
    #   clean_text ->  "Hello\nWorld"   (trailing spaces gone, \r\n -> \n)
    assert clean_text("a  \r\nb   \r\n") == "a\nb"


# Many blank lines in a row should collapse to a single blank line.
def test_collapses_excess_blank_lines():
    # Example: a PDF extract with big gaps "Section 1\n\n\n\n\nSection 2"
    #   clean_text ->  "Section 1\n\nSection 2"   (one blank line only)
    assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


# Cleaning must KEEP meaningful content: headings, punctuation, and table tabs.
def test_preserves_headings_punctuation_and_tables():
    # Example: a policy doc "# Returns\n\nReturns within 30 days.\n\nCol1\tCol2"
    #   clean_text keeps the heading, the sentence, and the tab-separated table.
    raw = "# Title\n\nReturns allowed within 30 days.\n\nCol1\tCol2\tCol3"
    cleaned = clean_text(raw)
    assert "# Title" in cleaned                 # heading survived
    assert "within 30 days." in cleaned         # policy text survived
    assert "Col1\tCol2\tCol3" in cleaned         # table (tabs) survived


# Null bytes (\x00) that some files contain should be stripped out.
def test_removes_null_bytes():
    # Example: a corrupt export "a\x00b" (hidden null byte between a and b)
    #   clean_text ->  "ab"   (the null byte is removed)
    assert "\x00" not in clean_text("a\x00b")


# Empty or whitespace-only input should return an empty string (not crash).
def test_empty_input():
    # Example: a blank file "" or a spaces-only file "   \n\n  "
    #   clean_text ->  ""   (both become an empty string, no error)
    assert clean_text("") == ""
    assert clean_text("   \n\n  ") == ""
