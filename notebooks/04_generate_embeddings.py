# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Generate embeddings
# MAGIC
# MAGIC Phase 5 of the RAG POC. Run **cell by cell** on Databricks Free Edition.
# MAGIC
# MAGIC Reads the chunk table from Phase 3, embeds each chunk with a small
# MAGIC **open-source** `sentence-transformers` model (no paid API / endpoint),
# MAGIC and writes a `document_embeddings` Delta table that maps each `chunk_id`
# MAGIC to its vector, recording the model name and dimension.
# MAGIC
# MAGIC What is an embedding? A list of numbers (a "vector") that captures the
# MAGIC meaning of a piece of text. Similar meanings produce similar vectors, which
# MAGIC is how semantic search finds relevant chunks later.
# MAGIC
# MAGIC Embeddings are numeric representations, not answers. The original chunk
# MAGIC text and metadata are kept alongside each vector for retrieval (Phase 6).
# MAGIC
# MAGIC Every configurable value is a widget / job parameter. Nothing is hardcoded.
# MAGIC
# MAGIC Note: this cell set installs `sentence-transformers` (pulls torch) and
# MAGIC downloads the model from the internet. That needs outbound network from
# MAGIC the workspace; if Free Edition blocks it, use the local embedder adapter
# MAGIC in `src/embedder.py` instead.

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
# Which embedding model to use. all-MiniLM-L6-v2 is small, fast, and free.
dbutils.widgets.text("model_name", "sentence-transformers/all-MiniLM-L6-v2", "5. Embedding model")
# How many chunks to encode at once (bigger = faster but more memory).
dbutils.widgets.text("batch_size", "32", "6. Encode batch size")
# Normalizing vectors makes "cosine similarity" (the usual similarity measure) simple and correct.
dbutils.widgets.dropdown("normalize", "true", ["true", "false"], "7. Normalize vectors (cosine)")
# Overwrite replaces the table; append adds to it.
dbutils.widgets.dropdown("write_mode", "overwrite", ["overwrite", "append"], "8. Write mode")
# Whether to actually save the embeddings, or just preview them.
dbutils.widgets.dropdown("save_embeddings", "true", ["true", "false"], "9. Save embeddings to Delta")

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
MODEL_NAME = dbutils.widgets.get("model_name").strip()
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))      # text -> integer
NORMALIZE = dbutils.widgets.get("normalize") == "true"   # text -> True/False
WRITE_MODE = dbutils.widgets.get("write_mode")
SAVE_EMBEDDINGS = dbutils.widgets.get("save_embeddings") == "true"

# Make sure the important names are not blank.
for name, value in {"catalog": CATALOG, "schema": SCHEMA, "chunks_table": CHUNKS_TABLE,
                    "model_name": MODEL_NAME}.items():
    assert value, f"Parameter '{name}' must not be empty."
# Batch size must be positive.
assert BATCH_SIZE > 0, "batch_size must be > 0"

# Full table names.
CHUNKS_FQN = f"{CATALOG}.{SCHEMA}.{CHUNKS_TABLE}"
EMBEDDINGS_FQN = f"{CATALOG}.{SCHEMA}.{EMBEDDINGS_TABLE}"

# Echo the resolved settings.
print("CHUNKS_FQN     :", CHUNKS_FQN)
print("EMBEDDINGS_FQN :", EMBEDDINGS_FQN)
print("MODEL_NAME     :", MODEL_NAME)
print("BATCH_SIZE     :", BATCH_SIZE)
print("NORMALIZE      :", NORMALIZE)
print("SAVE_EMBEDDINGS:", SAVE_EMBEDDINGS, f"({WRITE_MODE})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 - Ensure sentence-transformers is installed
# MAGIC In-process pip, so **no Python restart**. Pulls torch the first time, which
# MAGIC can take a few minutes. Safe to re-run.

# COMMAND ----------

# Standard-library helpers for importing and installing.
import importlib
import subprocess
import sys


# Install a library only if it is not already importable (same helper as notebook 02).
def _ensure(pip_spec: str, module_name: str) -> None:
    try:
        importlib.import_module(module_name)            # already installed?
        print(f"ready: {module_name} (already installed)")
    except ModuleNotFoundError:
        print(f"installing {pip_spec} ... (this can take a few minutes)")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pip_spec])
        importlib.invalidate_caches()                   # notice new files
        importlib.import_module(module_name)            # import it now
        print(f"ready: {module_name}")


# sentence-transformers provides the embedding model (and pulls PyTorch).
_ensure("sentence-transformers==3.3.1", "sentence_transformers")
print("Embedding dependency satisfied.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 - Load the chunks
# MAGIC Small POC corpus, so collecting to the driver is fine. For a large corpus,
# MAGIC embed partition-side with a pandas UDF instead of collecting.

# COMMAND ----------

# The chunk columns we need (text to embed, plus metadata to keep beside the vector).
chunk_cols = ["chunk_id", "document_id", "section", "page_number",
              "source_path", "content_hash", "chunk_text"]
# Read the chunks table, keep those columns, sort by id, and bring rows to Python.
chunk_rows = spark.table(CHUNKS_FQN).select(*chunk_cols).orderBy("chunk_id").collect()
print(f"{len(chunk_rows)} chunk(s) loaded from {CHUNKS_FQN}")
# Stop if Phase 3 has not been run.
assert chunk_rows, f"{CHUNKS_FQN} is empty. Run Phase 3 first."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 - Encode the chunks
# MAGIC The **same** model must later embed questions at query time.

# COMMAND ----------

# Import the model class.
from sentence_transformers import SentenceTransformer

# Load the model by name (downloads it the first time).
model = SentenceTransformer(MODEL_NAME)
print(f"loaded {MODEL_NAME}")

# Pull out just the chunk texts into a list.
texts = [r["chunk_text"] for r in chunk_rows]
# Turn every chunk's text into a vector. This is the core embedding step.
vectors = model.encode(
    texts,
    batch_size=BATCH_SIZE,            # how many to process at a time
    normalize_embeddings=NORMALIZE,   # make them unit-length for cosine similarity
    show_progress_bar=False,
    convert_to_numpy=True,            # return a numpy array of shape (num_chunks, dim)
)
# The number of columns in the array is the embedding dimension (e.g. 384).
EMBEDDING_DIM = int(vectors.shape[1])
print("encoded:", vectors.shape, "| dim:", EMBEDDING_DIM)
# Safety: one vector per chunk.
assert vectors.shape[0] == len(chunk_rows)

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
            "embedding_model": MODEL_NAME,                   # record which model made it
            "embedding_dim": int(EMBEDDING_DIM),             # and its length
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

# Build the DataFrame and preview a few columns (not the huge vectors).
embeddings_df = spark.createDataFrame(emb_rows, schema=embeddings_schema)
display(embeddings_df.select("chunk_id", "document_id", "section", "embedding_model", "embedding_dim"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 - (Optional) Save the embeddings as a Delta table

# COMMAND ----------

# Only write if requested.
if SAVE_EMBEDDINGS:
    (embeddings_df.write.mode(WRITE_MODE).saveAsTable(EMBEDDINGS_FQN))
    print(f"Saved embeddings to {EMBEDDINGS_FQN} (mode={WRITE_MODE}), rows={embeddings_df.count()}")
else:
    print("save_embeddings=false -> embeddings not persisted")
