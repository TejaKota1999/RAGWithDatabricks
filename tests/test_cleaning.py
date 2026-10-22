"""Unit tests for src/text_cleaning.py.

Each test feeds a known messy string to clean_text() and checks the tidied
result. clean_text only normalizes whitespace; it must NOT delete real content.
No Databricks is involved - the inputs are written right here in the tests.
"""

# Import the one function we are testing.
from src.text_cleaning import clean_text


# Trailing spaces should be removed and Windows line endings (\r\n) become \n.
def test_normalizes_line_endings_and_trailing_space():
    # Input has trailing spaces and \r\n; expected output is the clean "a\nb".
    assert clean_text("a  \r\nb   \r\n") == "a\nb"


# Many blank lines in a row should collapse to a single blank line.
def test_collapses_excess_blank_lines():
    # 5 newlines between a and b -> just one blank line ("\n\n").
    assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


# Cleaning must KEEP meaningful content: headings, punctuation, and table tabs.
def test_preserves_headings_punctuation_and_tables():
    # A sample with a markdown heading, a sentence, and a tab-separated table row.
    raw = "# Title\n\nReturns allowed within 30 days.\n\nCol1\tCol2\tCol3"
    cleaned = clean_text(raw)
    # All three must still be present after cleaning.
    assert "# Title" in cleaned                 # heading survived
    assert "within 30 days." in cleaned         # policy text survived
    assert "Col1\tCol2\tCol3" in cleaned         # table (tabs) survived


# Null bytes (\x00) that some files contain should be stripped out.
def test_removes_null_bytes():
    # After cleaning, there must be no null character left.
    assert "\x00" not in clean_text("a\x00b")


# Empty or whitespace-only input should return an empty string (not crash).
def test_empty_input():
    assert clean_text("") == ""              # empty in -> empty out
    assert clean_text("   \n\n  ") == ""     # only spaces/newlines -> empty
