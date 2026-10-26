# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Generate embeddings (Databricks endpoint)
# MAGIC
# MAGIC Phase 5 of the RAG POC. Run **cell by cell** on Databricks Free Edition.
# MAGIC
# MAGIC Reads the chunk table from Phase 3, embeds each chunk by calling the
# MAGIC **Databricks Foundation Model embedding endpoint**
# MAGIC (`databricks-qwen3-embedding-0-6b`, 1024-dim), and writes a
# MAGIC `document_embeddings` Delta table mapping each `chunk_id` to its vector.
# MAGIC
# MAGIC No local model is downloaded and no `sentence-transformers`/torch install is
# MAGIC needed - the model runs on Databricks. The same endpoint must embed
# MAGIC questions at query time.
# MAGIC
# MAGIC Cost note: the embedding endpoint is included in Free Edition but metered
# MAGIC against the serverless quota. This POC corpus is tiny, so usage is small.
# MAGIC
# MAGIC Every configurable value is a widget / job parameter. Nothing is hardcoded.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 - Declare parameters (widgets)

# COMMAND ----------

# Catalog / schema holding the input and output tables.
dbutils.widgets.text("catalog", "databricks_rag_dev", "1. Catalog")
dbutils.widgets.text("schema", "rag", "2. Schema")
# Input table of chunks (from notebook 03).
dbutils.widgets.text("chunks_table", "document_chunks", "3. Source chunks table")
# Output table that will store one vector per chunk.
dbutils.widgets.text("embeddings_table", "document_embeddings", "4. Output embeddings table")
# Which Databricks embedding endpoint to call (1024-dim vectors).
dbutils.widgets.text("embedding_endpoint", "databricks-qwen3-embedding-0-6b", "5. Embedding endpoint")
# How many chunks to send per request (endpoints cap request size).
dbutils.widgets.text("batch_size", "16", "6. Encode batch size")
# Normalizing vectors makes "cosine similarity" (the usual similarity measure) simple and correct.
dbutils.widgets.dropdown("normalize", "true", ["true", "false"], "7. Normalize vectors (cosine)")
# Overwrite replaces the table; append adds to it.
dbutils.widgets.dropdown("write_mode", "overwrite", ["overwrite", "append"], "8. Write mode")
# Whether to actually save the embeddings, or just preview them.
dbutils.widgets.dropdown("save_embeddings", "true", ["true", "false"], "9. Save embeddings to Delta")
# Incremental: only embed new/changed documents (skip already-embedded ones).
# This is the main cost-saver: the embedding endpoint is only called for new text.
dbutils.widgets.dropdown("incremental", "true", ["true", "false"], "10. Incremental (only new/changed)")

print("Widgets declared.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 - Read parameters into variables

# COMMAND ----------

# Copy widget values into variables.
CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
CHUNKS_TABLE = dbutils.widgets.get("chunks_table").strip()
EMBEDDINGS_TABLE = dbutils.widgets.get("embeddings_table").strip()
EMBEDDING_ENDPOINT = dbutils.widgets.get("embedding_endpoint").strip()
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))      # text -> integer
NORMALIZE = dbutils.widgets.get("normalize") == "true"   # text -> True/False
WRITE_MODE = dbutils.widgets.get("write_mode")
SAVE_EMBEDDINGS = dbutils.widgets.get("save_embeddings") == "true"
# When true, only embed new/changed documents (skip ones already embedded).
INCREMENTAL = dbutils.widgets.get("incremental") == "true"

# Make sure the important names are not blank.
for name, value in {"catalog": CATALOG, "schema": SCHEMA, "chunks_table": CHUNKS_TABLE,
                    "embedding_endpoint": EMBEDDING_ENDPOINT}.items():
    assert value, f"Parameter '{name}' must not be empty."
# Batch size must be positive.
assert BATCH_SIZE > 0, "batch_size must be > 0"

# Full table names.
CHUNKS_FQN = f"{CATALOG}.{SCHEMA}.{CHUNKS_TABLE}"
EMBEDDINGS_FQN = f"{CATALOG}.{SCHEMA}.{EMBEDDINGS_TABLE}"

# Echo the resolved settings.
print("CHUNKS_FQN        :", CHUNKS_FQN)
print("EMBEDDINGS_FQN    :", EMBEDDINGS_FQN)
print("EMBEDDING_ENDPOINT:", EMBEDDING_ENDPOINT)
print("BATCH_SIZE        :", BATCH_SIZE)
print("NORMALIZE         :", NORMALIZE)
print("SAVE_EMBEDDINGS   :", SAVE_EMBEDDINGS, f"({WRITE_MODE})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 - Load the chunks
# MAGIC Small POC corpus, so collecting to the driver is fine. For a large corpus,
# MAGIC embed partition-side instead of collecting.

# COMMAND ----------

# The chunk columns we need (text to embed, plus metadata to keep beside the vector).
chunk_cols = ["chunk_id", "document_id", "section", "page_number",
              "source_path", "content_hash", "chunk_text"]
# Read the chunks table, keep those columns, sort by id, and bring rows to Python.
chunk_rows = spark.table(CHUNKS_FQN).select(*chunk_cols).orderBy("chunk_id").collect()
print(f"{len(chunk_rows)} chunk(s) loaded from {CHUNKS_FQN}")
# Stop if Phase 3 has not been run.
assert chunk_rows, f"{CHUNKS_FQN} is empty. Run Phase 3 first."

# INCREMENTAL: keep only chunks of documents that are NOT already embedded. A
# document is "already embedded" if its content_hash is in the embeddings table.
# This means the (costly) embedding endpoint is only called for new/changed text.
if INCREMENTAL and spark.catalog.tableExists(EMBEDDINGS_FQN):
    existing_hashes = {
        r["content_hash"]
        for r in spark.table(EMBEDDINGS_FQN).select("content_hash").distinct().collect()
    }
    before = len(chunk_rows)
    chunk_rows = [r for r in chunk_rows if r["content_hash"] not in existing_hashes]
    print(f"incremental: {before - len(chunk_rows)} chunk(s) already embedded; "
          f"{len(chunk_rows)} new/changed to embed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 - Embed the chunks via the Databricks endpoint
# MAGIC Calls the serving endpoint in small batches. The **same** endpoint must
# MAGIC later embed questions at query time.

# COMMAND ----------

# The Databricks SDK is pre-installed in the workspace; WorkspaceClient() auto-authenticates.
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# If incremental found nothing new, skip the endpoint calls entirely (zero cost).
if not chunk_rows:
    print("Nothing new to embed -> skipping the embedding endpoint (no cost).")
    vectors = []
    EMBEDDING_DIM = 0
else:
    # Pull out just the chunk texts into a list.
    texts = [r["chunk_text"] for r in chunk_rows]

    # Call the embedding endpoint batch by batch and collect one vector per chunk.
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        resp = w.serving_endpoints.query(name=EMBEDDING_ENDPOINT, input=batch)   # the embedding call
        for item in resp.data:                       # one result per input text
            vectors.append([float(x) for x in item.embedding])   # the vector as plain floats
        print(f"  embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")

    # The length of a vector is the embedding dimension (1024 for this endpoint).
    EMBEDDING_DIM = len(vectors[0])
    print("embedded:", len(vectors), "vectors | dim:", EMBEDDING_DIM)
    assert len(vectors) == len(chunk_rows), "one vector per chunk expected"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 - Optionally L2-normalize the vectors (for cosine similarity)

# COMMAND ----------

# numpy makes normalizing the whole batch easy.
import numpy as np

if chunk_rows:
    arr = np.asarray(vectors, dtype="float32")
    if NORMALIZE:
        # Divide each vector by its length so inner product == cosine similarity.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0        # avoid dividing by zero
        arr = arr / norms
        print("vectors L2-normalized")
    else:
        print("vectors left un-normalized")
    vectors = arr  # keep working with the (possibly normalized) array
else:
    print("no vectors to normalize (nothing new)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 - Build the embeddings DataFrame

# COMMAND ----------

# datetime for a timestamp; the pyspark types describe the output table.
from datetime import datetime, timezone

from pyspark.sql.types import (
    ArrayType,       # for the embedding (a list/array of floats)
    BooleanType,     # for the normalized True/False flag
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# One timestamp reused for every row in this run.
now_iso = datetime.now(timezone.utc).isoformat()
emb_rows = []
# zip pairs each chunk row with its matching vector, in order.
for r, vec in zip(chunk_rows, vectors):
    emb_rows.append(
        {
            "chunk_id": r["chunk_id"],                       # links the vector back to its chunk
            "document_id": r["document_id"],
            "section": r["section"],
            "page_number": r["page_number"],
            "source_path": r["source_path"],
            "content_hash": r["content_hash"],
            "chunk_text": r["chunk_text"],                   # keep the text beside the vector
            "embedding": [float(x) for x in vec.tolist()],   # the vector as a plain list of floats
            "embedding_model": EMBEDDING_ENDPOINT,           # record which endpoint made it
            "embedding_dim": int(EMBEDDING_DIM),             # and its length (1024)
            "normalized": NORMALIZE,                         # whether it was unit-normalized
            "created_at": now_iso,
        }
    )

# Describe the output table's columns and types.
embeddings_schema = StructType(
    [
        StructField("chunk_id", StringType(), False),
        StructField("document_id", StringType(), False),
        StructField("section", StringType(), True),
        StructField("page_number", IntegerType(), True),
        StructField("source_path", StringType(), False),
        StructField("content_hash", StringType(), False),
        StructField("chunk_text", StringType(), False),
        StructField("embedding", ArrayType(FloatType()), False),   # array<float> column
        StructField("embedding_model", StringType(), False),
        StructField("embedding_dim", IntegerType(), False),
        StructField("normalized", BooleanType(), False),           # True/False column
        StructField("created_at", StringType(), False),
    ]
)

# Build the DataFrame and preview a few columns (skip if nothing new).
if emb_rows:
    embeddings_df = spark.createDataFrame(emb_rows, schema=embeddings_schema)
    display(embeddings_df.select("chunk_id", "document_id", "section", "embedding_model", "embedding_dim"))
else:
    embeddings_df = None
    print("No embeddings to build (incremental: nothing new).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 - (Optional) Save the embeddings as a Delta table (with Change Data Feed)
# MAGIC Change Data Feed is required so a Vector Search **Delta Sync** index can
# MAGIC keep itself up to date (Phase 6).

# COMMAND ----------

# Only write if requested and there are new/changed embeddings.
if SAVE_EMBEDDINGS and embeddings_df is not None:
    if INCREMENTAL and spark.catalog.tableExists(EMBEDDINGS_FQN):
        # Upsert: delete any OLD embeddings for the documents we just embedded (handles a
        # changed document), then APPEND the new ones. Other documents are left untouched.
        doc_ids = {r["document_id"] for r in emb_rows}
        id_list = ", ".join("'" + d.replace("'", "''") + "'" for d in doc_ids)
        spark.sql(f"DELETE FROM {EMBEDDINGS_FQN} WHERE document_id IN ({id_list})")
        embeddings_df.write.mode("append").saveAsTable(EMBEDDINGS_FQN)
        print(f"Incremental: appended {embeddings_df.count()} embedding(s) for {len(doc_ids)} document(s)")
    else:
        # First run (or incremental off): write the whole table using the chosen mode.
        (embeddings_df.write.mode(WRITE_MODE).saveAsTable(EMBEDDINGS_FQN))
        print(f"Saved embeddings to {EMBEDDINGS_FQN} (mode={WRITE_MODE}), rows={embeddings_df.count()}")
    # Enable Change Data Feed so Vector Search Delta Sync can track changes (idempotent).
    spark.sql(f"ALTER TABLE {EMBEDDINGS_FQN} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    print("Change Data Feed enabled (required for Delta Sync index).")
elif embeddings_df is None:
    print("Nothing new to save (incremental).")
else:
    print("save_embeddings=false -> embeddings not persisted")
