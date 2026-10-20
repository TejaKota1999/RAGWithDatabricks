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
# MAGIC Every configurable value is a widget / job parameter. Nothing is hardcoded.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 - Declare parameters (widgets)

# COMMAND ----------

dbutils.widgets.text("catalog", "databricks_rag_dev", "1. Catalog")
dbutils.widgets.text("schema", "rag", "2. Schema")
dbutils.widgets.text("source_table", "documents_cleaned", "3. Source (cleaned) table")
dbutils.widgets.text("chunks_table", "document_chunks", "4. Output chunks table")
dbutils.widgets.text("chunk_size", "1000", "5. Chunk size (chars)")
dbutils.widgets.text("chunk_overlap", "150", "6. Chunk overlap (chars)")
dbutils.widgets.text("min_chunk_chars", "60", "7. Min chars to keep a chunk")
dbutils.widgets.dropdown("write_mode", "overwrite", ["overwrite", "append"], "8. Write mode")
dbutils.widgets.dropdown("save_chunks", "true", ["true", "false"], "9. Save chunks to Delta")

print("Widgets declared.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 - Read parameters into variables

# COMMAND ----------

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
SOURCE_TABLE = dbutils.widgets.get("source_table").strip()
CHUNKS_TABLE = dbutils.widgets.get("chunks_table").strip()
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))
MIN_CHUNK_CHARS = int(dbutils.widgets.get("min_chunk_chars"))
WRITE_MODE = dbutils.widgets.get("write_mode")
SAVE_CHUNKS = dbutils.widgets.get("save_chunks") == "true"

for name, value in {"catalog": CATALOG, "schema": SCHEMA, "source_table": SOURCE_TABLE}.items():
    assert value, f"Parameter '{name}' must not be empty."
assert CHUNK_SIZE > 0, "chunk_size must be > 0"
assert 0 <= CHUNK_OVERLAP < CHUNK_SIZE, "chunk_overlap must be >= 0 and < chunk_size"

SOURCE_FQN = f"{CATALOG}.{SCHEMA}.{SOURCE_TABLE}"
CHUNKS_FQN = f"{CATALOG}.{SCHEMA}.{CHUNKS_TABLE}"

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

source_cols = ["document_id", "file_name", "page_number", "section",
               "text", "file_type", "source_path", "content_hash"]
src_rows = spark.table(SOURCE_FQN).select(*source_cols).collect()
print(f"{len(src_rows)} cleaned row(s) loaded from {SOURCE_FQN}")
assert src_rows, f"{SOURCE_FQN} is empty. Run Phase 2 first."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 - Chunking helpers
# MAGIC `chunk_document` packs whole paragraphs up to `size`, carrying `overlap`
# MAGIC characters of context between chunks. Paragraphs longer than `size` are
# MAGIC windowed. `detect_section` tags a chunk with the nearest heading.

# COMMAND ----------

import re

_HEADING_RE = re.compile(r"^\s*(#{1,6}\s+.+|\d+(?:\.\d+)*\.?\s+\S.+)\s*$")


def chunk_document(text: str, size: int, overlap: int) -> list[str]:
    """Split one document's text into overlapping, paragraph-aware chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > size:
            # Flush, then hard-window the oversized paragraph.
            if current:
                chunks.append(current)
                current = ""
            step = max(1, size - overlap)
            for i in range(0, len(para), step):
                chunks.append(para[i:i + size])
            continue
        if current and len(current) + 2 + len(para) > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}".strip() if tail else para
        else:
            current = f"{current}\n\n{para}".strip() if current else para
    if current:
        chunks.append(current)
    return chunks


def detect_section(chunk: str, carried: str | None) -> str | None:
    """Return the first heading inside the chunk, else the carried section."""
    for line in chunk.splitlines():
        if _HEADING_RE.match(line):
            return line.strip().lstrip("#").strip()
    return carried


print("Helpers defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 - Build chunk rows
# MAGIC Stable `chunk_id = <document_id>:<page>:<index>`. Chunks shorter than
# MAGIC `min_chunk_chars` (likely headers/footers) are dropped.

# COMMAND ----------

import hashlib
from datetime import datetime, timezone

chunk_rows = []
dropped_small = 0

for r in src_rows:
    page = r["page_number"] if r["page_number"] is not None else 0
    carried_section = r["section"]
    idx = 0
    for piece in chunk_document(r["text"], CHUNK_SIZE, CHUNK_OVERLAP):
        piece = piece.strip()
        if len(piece) < MIN_CHUNK_CHARS:
            dropped_small += 1
            continue
        carried_section = detect_section(piece, carried_section)
        chunk_rows.append(
            {
                "chunk_id": f"{r['document_id']}:{page}:{idx:04d}",
                "document_id": r["document_id"],
                "chunk_index": idx,
                "chunk_text": piece,
                "char_count": len(piece),
                "section": carried_section,
                "page_number": r["page_number"],
                "file_type": r["file_type"],
                "source_path": r["source_path"],
                "content_hash": r["content_hash"],
                "chunk_hash": hashlib.sha256(piece.encode("utf-8")).hexdigest(),
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        idx += 1

print(f"chunks built : {len(chunk_rows)}")
print(f"dropped small: {dropped_small}")
assert chunk_rows, "No chunks produced. Check chunk_size / min_chunk_chars and the source text."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 - Build the chunks DataFrame

# COMMAND ----------

from pyspark.sql.types import IntegerType, StringType, StructField, StructType

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

chunks_df = spark.createDataFrame(chunk_rows, schema=chunks_schema)
print("per-document chunk counts:")
chunks_df.groupBy("document_id").count().orderBy("document_id").show(truncate=False)
display(chunks_df.orderBy("document_id", "chunk_index"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 - (Optional) Save the chunks as a Delta table

# COMMAND ----------

if SAVE_CHUNKS:
    (chunks_df.write.mode(WRITE_MODE).saveAsTable(CHUNKS_FQN))
    print(f"Saved chunks to {CHUNKS_FQN} (mode={WRITE_MODE}), rows={chunks_df.count()}")
else:
    print("save_chunks=false -> chunks not persisted")
