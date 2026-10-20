# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Chunk documents
# MAGIC
# MAGIC Phase 3 of the RAG POC. Run **cell by cell** on Databricks Free Edition.
# MAGIC
# MAGIC Reads the cleaned text table from Phase 2, splits each document into
# MAGIC retrieval-sized chunks, and writes a chunk table. Chunking is
# MAGIC **paragraph-aware** with a configurable size and overlap, never mixes two
# MAGIC documents in one chunk, carries source metadata into every chunk, and
# MAGIC drops header/footer-only fragments.
# MAGIC
# MAGIC Why chunk? A whole document is too big to feed to a search/LLM at once.
# MAGIC We split it into smaller passages so we can retrieve just the relevant bits.
# MAGIC
# MAGIC Every configurable value is a widget / job parameter. Nothing is hardcoded.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 - Declare parameters (widgets)

# COMMAND ----------

# Catalog / schema that hold the input and output tables.
dbutils.widgets.text("catalog", "databricks_rag_dev", "1. Catalog")
dbutils.widgets.text("schema", "rag", "2. Schema")
# The input table produced by notebook 02 (the cleaned text).
dbutils.widgets.text("source_table", "documents_cleaned", "3. Source (cleaned) table")
# The output table this notebook will create (one row per chunk).
dbutils.widgets.text("chunks_table", "document_chunks", "4. Output chunks table")
# Target size of each chunk, in characters.
dbutils.widgets.text("chunk_size", "1000", "5. Chunk size (chars)")
# How many characters of the previous chunk to repeat at the start of the next,
# so context is not lost at the boundary.
dbutils.widgets.text("chunk_overlap", "150", "6. Chunk overlap (chars)")
# Chunks shorter than this are dropped (usually page headers/footers).
dbutils.widgets.text("min_chunk_chars", "60", "7. Min chars to keep a chunk")
# Overwrite replaces the chunks table; append adds to it.
dbutils.widgets.dropdown("write_mode", "overwrite", ["overwrite", "append"], "8. Write mode")
# Whether to actually save the chunks to Delta, or just preview them.
dbutils.widgets.dropdown("save_chunks", "true", ["true", "false"], "9. Save chunks to Delta")

print("Widgets declared.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 - Read parameters into variables

# COMMAND ----------

# Copy each widget value into a variable (text values are trimmed of spaces).
CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
SOURCE_TABLE = dbutils.widgets.get("source_table").strip()
CHUNKS_TABLE = dbutils.widgets.get("chunks_table").strip()
# Numeric settings must be converted from text to integers with int(...).
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))
MIN_CHUNK_CHARS = int(dbutils.widgets.get("min_chunk_chars"))
WRITE_MODE = dbutils.widgets.get("write_mode")
# True only when the dropdown says "true".
SAVE_CHUNKS = dbutils.widgets.get("save_chunks") == "true"

# Make sure the key names are not blank.
for name, value in {"catalog": CATALOG, "schema": SCHEMA, "source_table": SOURCE_TABLE}.items():
    assert value, f"Parameter '{name}' must not be empty."
# Sanity checks on the numbers: size must be positive; overlap must be smaller than size.
assert CHUNK_SIZE > 0, "chunk_size must be > 0"
assert 0 <= CHUNK_OVERLAP < CHUNK_SIZE, "chunk_overlap must be >= 0 and < chunk_size"

# Full table names: catalog.schema.table.
SOURCE_FQN = f"{CATALOG}.{SCHEMA}.{SOURCE_TABLE}"
CHUNKS_FQN = f"{CATALOG}.{SCHEMA}.{CHUNKS_TABLE}"

# Echo the resolved settings.
print("SOURCE_FQN     :", SOURCE_FQN)
print("CHUNKS_FQN     :", CHUNKS_FQN)
print("CHUNK_SIZE     :", CHUNK_SIZE)
print("CHUNK_OVERLAP  :", CHUNK_OVERLAP)
print("MIN_CHUNK_CHARS:", MIN_CHUNK_CHARS)
print("SAVE_CHUNKS    :", SAVE_CHUNKS, f"({WRITE_MODE})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 - Load the cleaned documents
# MAGIC The POC corpus is small, so collecting to the driver is fine. For a large
# MAGIC corpus, replace this with a pandas UDF that chunks partition-side.

# COMMAND ----------

# The columns we need from the cleaned table.
source_cols = ["document_id", "file_name", "page_number", "section",
               "text", "file_type", "source_path", "content_hash"]
# spark.table(...) opens the Delta table; .select(...) keeps only those columns;
# .collect() pulls the rows into a normal Python list (OK because the data is small).
src_rows = spark.table(SOURCE_FQN).select(*source_cols).collect()
# Report how many cleaned rows we loaded.
print(f"{len(src_rows)} cleaned row(s) loaded from {SOURCE_FQN}")
# Stop early with a clear message if Phase 2 has not been run.
assert src_rows, f"{SOURCE_FQN} is empty. Run Phase 2 first."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 - Chunking helpers
# MAGIC `chunk_document` packs whole paragraphs up to `size`, carrying `overlap`
# MAGIC characters of context between chunks. Paragraphs longer than `size` are
# MAGIC windowed. `detect_section` tags a chunk with the nearest heading.

# COMMAND ----------

# Regular-expression module, used to split paragraphs and recognize headings.
import re

# A pattern that matches a heading line: either Markdown ("# Title", "## Sub")
# or a numbered heading ("1. Purpose", "4.2 Scope").
_HEADING_RE = re.compile(r"^\s*(#{1,6}\s+.+|\d+(?:\.\d+)*\.?\s+\S.+)\s*$")


# Splits one document's text into a list of overlapping chunks.
def chunk_document(text: str, size: int, overlap: int) -> list[str]:
    """Split one document's text into overlapping, paragraph-aware chunks."""
    # Split on blank lines into paragraphs, trim each, and drop empty ones.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []   # finished chunks go here
    current = ""             # the chunk we are currently building up
    for para in paragraphs:
        # Case 1: a single paragraph is bigger than one chunk.
        if len(para) > size:
            # First save whatever we were building.
            if current:
                chunks.append(current)
                current = ""
            # Then cut the long paragraph into fixed windows that overlap.
            step = max(1, size - overlap)                # how far to move the window each time
            for i in range(0, len(para), step):
                chunks.append(para[i:i + size])          # take a slice of length "size"
            continue
        # Case 2: adding this paragraph would overflow the current chunk.
        if current and len(current) + 2 + len(para) > size:
            chunks.append(current)                       # save the full chunk
            tail = current[-overlap:] if overlap else "" # keep the last "overlap" chars for context
            current = f"{tail}\n\n{para}".strip() if tail else para  # start the next chunk
        # Case 3: it still fits, so append the paragraph to the current chunk.
        else:
            current = f"{current}\n\n{para}".strip() if current else para
    # Save the last chunk if anything is left.
    if current:
        chunks.append(current)
    return chunks


# Finds the nearest heading for a chunk so we can label which section it came from.
def detect_section(chunk: str, carried: str | None) -> str | None:
    """Return the first heading inside the chunk, else the carried section."""
    for line in chunk.splitlines():          # look at each line of the chunk
        if _HEADING_RE.match(line):           # if it looks like a heading...
            return line.strip().lstrip("#").strip()  # ...return it (without leading "#")
    return carried                            # otherwise keep the previous section label


print("Helpers defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 - Build chunk rows
# MAGIC Stable `chunk_id = <document_id>:<page>:<index>`. Chunks shorter than
# MAGIC `min_chunk_chars` (likely headers/footers) are dropped.

# COMMAND ----------

# hashlib for a per-chunk fingerprint; datetime for a timestamp.
import hashlib
from datetime import datetime, timezone

chunk_rows = []      # all the chunk rows we will save
dropped_small = 0    # counter for tiny fragments we skip

# Process each cleaned document row.
for r in src_rows:
    # Use the page number if present, else 0 (so the id is still well-formed).
    page = r["page_number"] if r["page_number"] is not None else 0
    carried_section = r["section"]   # running "current section" label
    idx = 0                          # chunk counter within this document
    # Split this document's text into chunks and loop over them.
    for piece in chunk_document(r["text"], CHUNK_SIZE, CHUNK_OVERLAP):
        piece = piece.strip()
        if len(piece) < MIN_CHUNK_CHARS:    # skip tiny fragments (headers/footers)
            dropped_small += 1
            continue
        carried_section = detect_section(piece, carried_section)  # update the section label
        # Build one chunk row with the text plus all its source metadata.
        chunk_rows.append(
            {
                # Stable id, e.g. "policy:0:0003" (document:page:index), zero-padded.
                "chunk_id": f"{r['document_id']}:{page}:{idx:04d}",
                "document_id": r["document_id"],
                "chunk_index": idx,                    # position of this chunk in the doc
                "chunk_text": piece,                   # the actual chunk text
                "char_count": len(piece),              # its length
                "section": carried_section,            # which heading/section it belongs to
                "page_number": r["page_number"],       # original page (may be null)
                "file_type": r["file_type"],
                "source_path": r["source_path"],       # where the document came from
                "content_hash": r["content_hash"],     # fingerprint of the whole source doc
                "chunk_hash": hashlib.sha256(piece.encode("utf-8")).hexdigest(),  # fingerprint of this chunk
                "chunk_size": CHUNK_SIZE,              # settings recorded for traceability
                "chunk_overlap": CHUNK_OVERLAP,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        idx += 1    # move to the next chunk index

# Report how many chunks we built and how many tiny fragments were dropped.
print(f"chunks built : {len(chunk_rows)}")
print(f"dropped small: {dropped_small}")
# Stop if nothing was produced.
assert chunk_rows, "No chunks produced. Check chunk_size / min_chunk_chars and the source text."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 - Build the chunks DataFrame

# COMMAND ----------

# Column types for the chunks table.
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

# Describe each column and whether it may be null (True) or must have a value (False).
chunks_schema = StructType(
    [
        StructField("chunk_id", StringType(), False),
        StructField("document_id", StringType(), False),
        StructField("chunk_index", IntegerType(), False),
        StructField("chunk_text", StringType(), False),
        StructField("char_count", IntegerType(), False),
        StructField("section", StringType(), True),
        StructField("page_number", IntegerType(), True),
        StructField("file_type", StringType(), False),
        StructField("source_path", StringType(), False),
        StructField("content_hash", StringType(), False),
        StructField("chunk_hash", StringType(), False),
        StructField("chunk_size", IntegerType(), False),
        StructField("chunk_overlap", IntegerType(), False),
        StructField("created_at", StringType(), False),
    ]
)

# Turn the list of chunk dictionaries into a Spark DataFrame.
chunks_df = spark.createDataFrame(chunk_rows, schema=chunks_schema)
# Show how many chunks each document produced.
print("per-document chunk counts:")
chunks_df.groupBy("document_id").count().orderBy("document_id").show(truncate=False)
# Display the chunks in reading order (by document, then chunk index).
display(chunks_df.orderBy("document_id", "chunk_index"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 - (Optional) Save the chunks as a Delta table

# COMMAND ----------

# Only write if the save_chunks widget is "true".
if SAVE_CHUNKS:
    # Save the DataFrame as a Delta table.
    (chunks_df.write.mode(WRITE_MODE).saveAsTable(CHUNKS_FQN))
    print(f"Saved chunks to {CHUNKS_FQN} (mode={WRITE_MODE}), rows={chunks_df.count()}")
else:
    print("save_chunks=false -> chunks not persisted")
