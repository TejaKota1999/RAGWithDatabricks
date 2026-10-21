"""Text cleaning shared by the extraction notebook (Phase 2) and tests.

Normalizes whitespace only. It must NOT remove meaningful content: headings,
punctuation, list markers, table characters, or policy conditions.
"""

from __future__ import annotations

# Regular-expression module, used to collapse repeated blank lines.
import re


# COMMAND ----------
# Tidy up raw extracted text: fix whitespace only, keep all real content.
def clean_text(raw: str) -> str:
    """Normalize whitespace without removing meaningful content."""
    # Remove null characters that break text processing.
    text = raw.replace("\x00", "")
    # Standardize line endings: Windows "\r\n" and old Mac "\r" both become "\n".
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing spaces on each line (keeps leading indentation / list markers).
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Collapse 3+ blank lines down to a single blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trim blank space at the very start/end and return the cleaned text.
    return text.strip()
