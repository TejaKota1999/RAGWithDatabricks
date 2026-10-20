# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Extract and clean text
# MAGIC
# MAGIC Phase 2 of the RAG POC. Run **cell by cell** on Databricks Free Edition.
# MAGIC
# MAGIC Reads the documents in the volume (or a parameterized source), extracts
# MAGIC text with a parser chosen per file type, normalizes whitespace **without**
# MAGIC deleting headings, punctuation, tables, or policy conditions, and writes a
# MAGIC cleaned Delta table. Parser failures and empty files are **logged**, not
# MAGIC silently dropped.
# MAGIC
# MAGIC Every configurable value is a widget / job parameter. Nothing is hardcoded.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 - Declare parameters (widgets)

# COMMAND ----------

dbutils.widgets.text("catalog", "databricks_rag_dev", "1. Catalog")
dbutils.widgets.text("schema", "rag", "2. Schema")
dbutils.widgets.text("volume", "source_docs", "3. Volume")
dbutils.widgets.text("source_path", "", "4. Source path (blank = the volume)")
dbutils.widgets.text("supported_extensions", "pdf,txt,md,docx", "5. Supported extensions")
dbutils.widgets.text("min_chars", "20", "6. Min chars to keep a document")
dbutils.widgets.text("cleaned_table", "documents_cleaned", "7. Cleaned table name")
dbutils.widgets.dropdown("write_mode", "overwrite", ["overwrite", "append"], "8. Cleaned write mode")
dbutils.widgets.dropdown("save_cleaned", "true", ["true", "false"], "9. Save cleaned to Delta")

print("Widgets declared.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 - Read parameters into variables

# COMMAND ----------

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
VOLUME = dbutils.widgets.get("volume").strip()
SOURCE_PATH_PARAM = dbutils.widgets.get("source_path").strip()
SUPPORTED_EXTENSIONS = [
    e.strip().lower().lstrip(".")
    for e in dbutils.widgets.get("supported_extensions").split(",")
    if e.strip()
]
MIN_CHARS = int(dbutils.widgets.get("min_chars"))
CLEANED_TABLE = dbutils.widgets.get("cleaned_table").strip()
WRITE_MODE = dbutils.widgets.get("write_mode")
SAVE_CLEANED = dbutils.widgets.get("save_cleaned") == "true"

for required_name, required_value in {"catalog": CATALOG, "schema": SCHEMA, "volume": VOLUME}.items():
    assert required_value, f"Parameter '{required_name}' must not be empty."

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
SOURCE_PATH = SOURCE_PATH_PARAM or VOLUME_PATH
CLEANED_FQN = f"{CATALOG}.{SCHEMA}.{CLEANED_TABLE}"

print("SOURCE_PATH         :", SOURCE_PATH)
print("SUPPORTED_EXTENSIONS:", SUPPORTED_EXTENSIONS)
print("MIN_CHARS           :", MIN_CHARS)
print("SAVE_CLEANED        :", SAVE_CLEANED, "->", CLEANED_FQN, f"({WRITE_MODE})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 - Ensure the needed parsers are installed
# MAGIC Dynamic: installs only the libraries required by `supported_extensions`
# MAGIC (`pdf` -> pypdf, `docx` -> python-docx). Uses in-process pip, so **no
# MAGIC Python restart** and no need to re-run earlier cells. Safe to re-run.

# COMMAND ----------

import importlib
import subprocess
import sys

# Map a file extension to (pip package spec, importable module name).
PARSER_DEPS = {
    "pdf": ("pypdf==5.1.0", "pypdf"),
    "docx": ("python-docx==1.1.2", "docx"),
}


def _ensure(pip_spec: str, module_name: str) -> None:
    try:
        importlib.import_module(module_name)
        print(f"ready: {module_name} (already installed)")
    except ModuleNotFoundError:
        print(f"installing {pip_spec} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pip_spec])
        importlib.invalidate_caches()
        importlib.import_module(module_name)
        print(f"ready: {module_name}")


for _ext in SUPPORTED_EXTENSIONS:
    if _ext in PARSER_DEPS:
        _ensure(*PARSER_DEPS[_ext])
print("Parser dependencies satisfied.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 - Parser and cleaning helpers
# MAGIC `clean_text` only normalizes whitespace: it preserves headings,
# MAGIC punctuation, lists, and table characters.

# COMMAND ----------

import re


def clean_text(raw: str) -> str:
    """Normalize whitespace without removing meaningful content."""
    text = raw.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing spaces on each line (keeps leading indentation / list markers).
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Collapse 3+ blank lines down to a single blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _iter_docx_blocks(document):
    """Yield paragraph and table text in document order (keeps tables inline)."""
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document).text
        elif isinstance(child, CT_Tbl):
            for row in Table(child, document).rows:
                yield "\t".join(cell.text for cell in row.cells)


def extract_pages(local_path: str, ext: str):
    """Yield (page_number, raw_text) tuples for one file.

    page_number is 1-based for PDFs and None for single-body formats.
    Raises on parser failure so the caller can log it.
    """
    if ext in ("txt", "md"):
        with open(local_path, "r", encoding="utf-8", errors="replace") as f:
            yield (None, f.read())
    elif ext == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(local_path)
        for i, page in enumerate(reader.pages):
            yield (i + 1, page.extract_text() or "")
    elif ext == "docx":
        from docx import Document

        document = Document(local_path)
        yield (None, "\n".join(_iter_docx_blocks(document)))
    else:
        raise ValueError(f"No parser for extension '{ext}'")


print("Helpers defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 - List the source files

# COMMAND ----------

source_files = [e for e in dbutils.fs.ls(SOURCE_PATH) if not e.isDir()]
print(f"{len(source_files)} file(s) under {SOURCE_PATH}")
for e in source_files:
    print(" ", e.name, "-", e.size, "bytes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 - Extract and clean each file
# MAGIC Builds cleaned rows; records skips and failures separately.

# COMMAND ----------

import hashlib
import os
from datetime import datetime, timezone

rows = []
skipped_unsupported = []
empty_or_short = []
failures = []

for entry in source_files:
    file_name = entry.name
    ext = os.path.splitext(file_name)[1].lstrip(".").lower()
    if ext not in SUPPORTED_EXTENSIONS:
        skipped_unsupported.append(file_name)
        continue

    local_path = entry.path.replace("dbfs:", "")
    try:
        with open(local_path, "rb") as f:
            content_hash = hashlib.sha256(f.read()).hexdigest()

        for page_number, raw in extract_pages(local_path, ext):
            cleaned = clean_text(raw)
            if len(cleaned) < MIN_CHARS:
                empty_or_short.append((file_name, page_number, len(cleaned)))
                continue
            rows.append(
                {
                    "document_id": os.path.splitext(file_name)[0],
                    "file_name": file_name,
                    "page_number": page_number,
                    "section": None,  # heading-aware sectioning happens in Phase 3
                    "text": cleaned,
                    "char_count": len(cleaned),
                    "file_type": ext,
                    "source_path": f"{SOURCE_PATH.rstrip('/')}/{file_name}",
                    "content_hash": content_hash,
                    "cleaned_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    except Exception as exc:  # log parser failures; do not abort the run
        failures.append((file_name, repr(exc)))

print(f"cleaned rows        : {len(rows)}")
print(f"skipped unsupported : {skipped_unsupported}")
print(f"empty/short         : {empty_or_short}")
print(f"failures            : {failures}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 - Build the cleaned DataFrame

# COMMAND ----------

from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Explicit schema so page_number stays typed even when every value is null.
cleaned_schema = StructType(
    [
        StructField("document_id", StringType(), False),
        StructField("file_name", StringType(), False),
        StructField("page_number", IntegerType(), True),
        StructField("section", StringType(), True),
        StructField("text", StringType(), False),
        StructField("char_count", IntegerType(), False),
        StructField("file_type", StringType(), False),
        StructField("source_path", StringType(), False),
        StructField("content_hash", StringType(), False),
        StructField("cleaned_at", StringType(), False),
    ]
)

assert rows, (
    f"No text extracted from {SOURCE_PATH}. "
    f"Check failures above, supported_extensions, and min_chars."
)
cleaned_df = spark.createDataFrame(rows, schema=cleaned_schema)
display(cleaned_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8 - (Optional) Save the cleaned text as a Delta table

# COMMAND ----------

if SAVE_CLEANED:
    (cleaned_df.write.mode(WRITE_MODE).saveAsTable(CLEANED_FQN))
    print(f"Saved cleaned text to {CLEANED_FQN} (mode={WRITE_MODE}), rows={cleaned_df.count()}")
else:
    print("save_cleaned=false -> cleaned text not persisted")
