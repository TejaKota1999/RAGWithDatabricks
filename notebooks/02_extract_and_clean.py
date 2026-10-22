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
# MAGIC Widgets are the little input boxes that appear at the top of the notebook.
# MAGIC They let you change settings without editing code.

# COMMAND ----------

# dbutils.widgets.text(name, default_value, label) creates a text input box.
# "catalog" is the Unity Catalog catalog (top-level container) to read/write in.
dbutils.widgets.text("catalog", "databricks_rag_dev", "1. Catalog")
# "schema" is the schema (like a folder) inside the catalog.
dbutils.widgets.text("schema", "rag", "2. Schema")
# "volume" is the Unity Catalog volume (a file storage area) holding the documents.
dbutils.widgets.text("volume", "source_docs", "3. Volume")
# "source_path" lets you point at a specific folder; blank means "use the volume".
dbutils.widgets.text("source_path", "", "4. Source path (blank = the volume)")
# Which file types we will try to read, as a comma-separated list.
dbutils.widgets.text("supported_extensions", "pdf,txt,md,docx", "5. Supported extensions")
# Documents whose cleaned text is shorter than this many characters are skipped.
dbutils.widgets.text("min_chars", "20", "6. Min chars to keep a document")
# Name of the output Delta table that will hold the cleaned text.
dbutils.widgets.text("cleaned_table", "documents_cleaned", "7. Cleaned table name")
# dropdown(name, default, choices, label): "overwrite" replaces the table, "append" adds to it.
dbutils.widgets.dropdown("write_mode", "overwrite", ["overwrite", "append"], "8. Cleaned write mode")
# Whether to actually save the cleaned text to Delta, or just preview it.
dbutils.widgets.dropdown("save_cleaned", "true", ["true", "false"], "9. Save cleaned to Delta")
# Incremental: only process NEW or CHANGED files (skip already-processed ones).
dbutils.widgets.dropdown("incremental", "true", ["true", "false"], "10. Incremental (only new/changed)")

# Print a confirmation so you know the widgets were created.
print("Widgets declared.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 - Read parameters into variables
# MAGIC Copies each widget's current value into a normal Python variable so the
# MAGIC rest of the notebook can use them.

# COMMAND ----------

# dbutils.widgets.get(name) reads the current value of a widget as text.
# .strip() removes any accidental spaces around it.
CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
VOLUME = dbutils.widgets.get("volume").strip()
SOURCE_PATH_PARAM = dbutils.widgets.get("source_path").strip()
# Turn the comma-separated extensions text into a clean Python list, e.g. ["pdf","txt"].
SUPPORTED_EXTENSIONS = [
    e.strip().lower().lstrip(".")              # tidy each item: trim spaces, lowercase, drop a leading dot
    for e in dbutils.widgets.get("supported_extensions").split(",")  # split "pdf,txt" into ["pdf"," txt"]
    if e.strip()                               # ignore empty pieces (e.g. a trailing comma)
]
# Convert the min-chars text into a whole number (integer).
MIN_CHARS = int(dbutils.widgets.get("min_chars"))
CLEANED_TABLE = dbutils.widgets.get("cleaned_table").strip()
WRITE_MODE = dbutils.widgets.get("write_mode")
# Compare the dropdown text to "true" so SAVE_CLEANED becomes a real True/False value.
SAVE_CLEANED = dbutils.widgets.get("save_cleaned") == "true"
# When true, skip files whose content was already processed (cost-saving).
INCREMENTAL = dbutils.widgets.get("incremental") == "true"

# Safety check: catalog, schema, and volume must not be blank. If one is, stop with a clear message.
for required_name, required_value in {"catalog": CATALOG, "schema": SCHEMA, "volume": VOLUME}.items():
    assert required_value, f"Parameter '{required_name}' must not be empty."

# Build the full path to the volume where files live, e.g. /Volumes/cat/schema/vol.
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
# If a specific source_path was given use it, otherwise fall back to the volume path.
SOURCE_PATH = SOURCE_PATH_PARAM or VOLUME_PATH
# The fully-qualified table name: catalog.schema.table.
CLEANED_FQN = f"{CATALOG}.{SCHEMA}.{CLEANED_TABLE}"

# Print the resolved settings so you can confirm them before running further.
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

# Standard-library helpers: importlib to import by name, subprocess to run pip, sys for the Python path.
import importlib
import subprocess
import sys

# For each file type that needs an extra library, record (pip package, import name).
PARSER_DEPS = {
    "pdf": ("pypdf==5.1.0", "pypdf"),            # PDFs need the pypdf package
    "docx": ("python-docx==1.1.2", "docx"),      # Word files need python-docx (imported as "docx")
}


# Helper that makes sure one library is importable, installing it only if missing.
def _ensure(pip_spec: str, module_name: str) -> None:
    try:
        # Try to import it first; if this works, it is already installed.
        importlib.import_module(module_name)
        print(f"ready: {module_name} (already installed)")
    except ModuleNotFoundError:
        # Not installed yet -> install it quietly with pip into the current Python.
        print(f"installing {pip_spec} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pip_spec])
        # Tell Python to notice the newly installed files, then import it.
        importlib.invalidate_caches()
        importlib.import_module(module_name)
        print(f"ready: {module_name}")


# Loop over the file types you enabled and install only the parsers you actually need.
for _ext in SUPPORTED_EXTENSIONS:
    if _ext in PARSER_DEPS:              # txt/md need nothing extra, so they are skipped here
        _ensure(*PARSER_DEPS[_ext])      # the * unpacks the tuple into (pip_spec, module_name)
print("Parser dependencies satisfied.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 - Parser and cleaning helpers
# MAGIC `clean_text` only normalizes whitespace: it preserves headings,
# MAGIC punctuation, lists, and table characters.

# COMMAND ----------

# "re" is Python's regular-expression module, used below to collapse blank lines.
import re


# Takes raw extracted text and returns a tidied version (whitespace only; meaning preserved).
def clean_text(raw: str) -> str:
    """Normalize whitespace without removing meaningful content."""
    # Remove null characters that some files contain (they break text processing).
    text = raw.replace("\x00", "")
    # Standardize line endings: Windows "\r\n" and old Mac "\r" both become "\n".
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing spaces on each line (keeps leading indentation / list markers).
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Collapse 3+ blank lines down to a single blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove leading/trailing blank space for the whole document and return it.
    return text.strip()


# Reads a Word (.docx) document and yields its text in order, keeping tables inline.
def _iter_docx_blocks(document):
    """Yield paragraph and table text in document order (keeps tables inline)."""
    # These imports describe the low-level Word building blocks (paragraphs and tables).
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    # Walk through every child element of the document body, in reading order.
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):                     # if it is a paragraph...
            yield Paragraph(child, document).text        # ...give back its text
        elif isinstance(child, CT_Tbl):                 # if it is a table...
            for row in Table(child, document).rows:      # ...go row by row
                yield "\t".join(cell.text for cell in row.cells)  # join cells with tabs

# COMMAND ----------

# Chooses the right parser for a file and yields (page_number, raw_text) pieces.
def extract_pages(local_path: str, ext: str):
    """Yield (page_number, raw_text) tuples for one file.

    page_number is 1-based for PDFs and None for single-body formats.
    Raises on parser failure so the caller can log it.
    """
    # Plain text and Markdown: just read the whole file as text (no page numbers).
    if ext in ("txt", "md"):
        with open(local_path, "r", encoding="utf-8", errors="replace") as f:
            yield (None, f.read())
    # PDF: read page by page so we can keep a page number with each piece of text.
    elif ext == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(local_path)                   # open the PDF
        for i, page in enumerate(reader.pages):          # i is 0,1,2...; page is each page
            yield (i + 1, page.extract_text() or "")     # i+1 makes it 1-based; "" if no text
    # Word document: extract all paragraphs/tables as one body of text.
    elif ext == "docx":
        from docx import Document

        document = Document(local_path)                  # open the Word file
        yield (None, "\n".join(_iter_docx_blocks(document)))  # join all blocks into one string
    # Any other extension has no parser here, so fail loudly.
    else:
        raise ValueError(f"No parser for extension '{ext}'")


# Confirm the helper functions are defined and ready to use.
print("Helpers defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 - List the source files

# COMMAND ----------

# datetime helps turn the file's modification time (epoch ms) into a readable date.
from datetime import datetime, timezone

# dbutils.fs.ls lists everything at the path; keep only real files (drop sub-folders).
source_files = [e for e in dbutils.fs.ls(SOURCE_PATH) if not e.isDir()]
# Show how many files were found.
print(f"{len(source_files)} file(s) under {SOURCE_PATH}")
# Print each file's name, size, and last-modified time (the upload timestamp).
for e in source_files:
    # modificationTime is milliseconds since 1970; convert to a UTC date string.
    mod = datetime.fromtimestamp(e.modificationTime / 1000, tz=timezone.utc).isoformat()
    print(" ", e.name, "-", e.size, "bytes - modified", mod)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 - Extract and clean each file
# MAGIC Builds cleaned rows; records skips and failures separately.

# COMMAND ----------

# hashlib makes a content fingerprint; os handles file-name parts; datetime stamps the time.
import hashlib
import os
from datetime import datetime, timezone

# These lists collect the results and anything that was skipped or failed.
rows = []                 # successful cleaned rows
skipped_unsupported = []  # files whose type we do not handle
empty_or_short = []       # files/pages that were empty or too short
failures = []             # files that raised an error while parsing
skipped_existing = []     # files already processed (incremental mode)

# INCREMENTAL: load the fingerprints of files already processed, so we can skip
# them. A file is "already processed" if its exact content_hash is already in the
# cleaned table. New files (new hash) and changed files (different hash) are NOT
# skipped, so they still get processed.
existing_hashes = set()
if INCREMENTAL and spark.catalog.tableExists(CLEANED_FQN):
    existing_hashes = {
        r["content_hash"]
        for r in spark.table(CLEANED_FQN).select("content_hash").distinct().collect()
    }
    print(f"incremental: {len(existing_hashes)} already-processed file fingerprint(s) found")

# Go through each file we found in Cell 5.
for entry in source_files:
    file_name = entry.name                                   # just the file name, e.g. policy.docx
    ext = os.path.splitext(file_name)[1].lstrip(".").lower() # the extension, e.g. "docx"
    if ext not in SUPPORTED_EXTENSIONS:                      # skip types we were not told to handle
        skipped_unsupported.append(file_name)
        continue

    # Volume paths come back prefixed with "dbfs:"; remove it to open the file normally.
    local_path = entry.path.replace("dbfs:", "")
    try:
        # Read the raw bytes once to compute a SHA-256 fingerprint (detects future changes).
        with open(local_path, "rb") as f:
            content_hash = hashlib.sha256(f.read()).hexdigest()

        # INCREMENTAL: if this exact content was already processed, skip it (no cost).
        if content_hash in existing_hashes:
            skipped_existing.append(file_name)
            continue

        # Extract the file's text (one loop pass per page for PDFs, one pass otherwise).
        for page_number, raw in extract_pages(local_path, ext):
            cleaned = clean_text(raw)                        # tidy the text
            if len(cleaned) < MIN_CHARS:                     # too little text -> record and skip
                empty_or_short.append((file_name, page_number, len(cleaned)))
                continue
            # Build one output row (a dictionary of column name -> value).
            rows.append(
                {
                    "document_id": os.path.splitext(file_name)[0],  # file name without extension
                    "file_name": file_name,
                    "page_number": page_number,                     # page for PDFs, None otherwise
                    "section": None,  # heading-aware sectioning happens in Phase 3
                    "text": cleaned,
                    "char_count": len(cleaned),                     # how many characters of text
                    "file_type": ext,
                    "source_path": f"{SOURCE_PATH.rstrip('/')}/{file_name}",  # where it came from
                    "content_hash": content_hash,
                    "cleaned_at": datetime.now(timezone.utc).isoformat(),     # timestamp (UTC)
                }
            )
    except Exception as exc:  # log parser failures; do not abort the run
        failures.append((file_name, repr(exc)))

# Summaries so you can see exactly what happened.
print(f"cleaned rows        : {len(rows)}")
print(f"skipped (already done): {skipped_existing}")
print(f"skipped unsupported : {skipped_unsupported}")
print(f"empty/short         : {empty_or_short}")
print(f"failures            : {failures}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 - Build the cleaned DataFrame

# COMMAND ----------

# Import the Spark column types used to describe the table's shape.
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Explicit schema so page_number stays typed even when every value is null.
# StructField(name, type, nullable): nullable=False means the column must have a value.
cleaned_schema = StructType(
    [
        StructField("document_id", StringType(), False),
        StructField("file_name", StringType(), False),
        StructField("page_number", IntegerType(), True),   # can be null (non-PDF files)
        StructField("section", StringType(), True),        # filled in later (Phase 3)
        StructField("text", StringType(), False),
        StructField("char_count", IntegerType(), False),
        StructField("file_type", StringType(), False),
        StructField("source_path", StringType(), False),
        StructField("content_hash", StringType(), False),
        StructField("cleaned_at", StringType(), False),
    ]
)

# In incremental mode, "no new rows" is a normal, successful outcome (every file was
# already processed), so we do not error - we just skip writing. In full mode, an empty
# result means something is wrong, so we stop with a helpful message.
if not rows:
    if INCREMENTAL:
        print("No new or changed files -> nothing to write (all files already processed).")
        cleaned_df = None
    else:
        raise AssertionError(
            f"No text extracted from {SOURCE_PATH}. "
            f"Check failures above, supported_extensions, and min_chars."
        )
else:
    # Turn the list of row-dictionaries into a Spark DataFrame (a table in memory).
    cleaned_df = spark.createDataFrame(rows, schema=cleaned_schema)
    # display() shows the table nicely in the notebook UI.
    display(cleaned_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8 - (Optional) Save the cleaned text as a Delta table

# COMMAND ----------

# Only write to Delta if the save_cleaned widget was "true" and we have new rows.
if SAVE_CLEANED and cleaned_df is not None:
    if INCREMENTAL and spark.catalog.tableExists(CLEANED_FQN):
        # Upsert: remove any OLD version of the documents we just processed (handles a
        # changed file), then APPEND the new rows. Untouched documents stay as they are.
        doc_ids = {r["document_id"] for r in rows}
        id_list = ", ".join("'" + d.replace("'", "''") + "'" for d in doc_ids)
        spark.sql(f"DELETE FROM {CLEANED_FQN} WHERE document_id IN ({id_list})")
        cleaned_df.write.mode("append").saveAsTable(CLEANED_FQN)
        print(f"Incremental: appended {cleaned_df.count()} new/changed row(s) to {CLEANED_FQN}")
    else:
        # First run (or incremental off): write the whole table using the chosen mode.
        (cleaned_df.write.mode(WRITE_MODE).saveAsTable(CLEANED_FQN))
        print(f"Saved cleaned text to {CLEANED_FQN} (mode={WRITE_MODE}), rows={cleaned_df.count()}")
elif cleaned_df is None:
    # Incremental run with nothing new -> nothing to write (expected, not an error).
    print("Nothing new to save (incremental).")
else:
    # Otherwise just tell the user nothing was written.
    print("save_cleaned=false -> cleaned text not persisted")
