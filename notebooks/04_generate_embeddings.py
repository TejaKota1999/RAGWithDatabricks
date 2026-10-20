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

dbutils.widgets.text("catalog", "databricks_rag_dev", "1. Catalog")
dbutils.widgets.text("schema", "rag", "2. Schema")
dbutils.widgets.text("chunks_table", "document_chunks", "3. Source chunks table")
dbutils.widgets.text("embeddings_table", "document_embeddings", "4. Output embeddings table")
dbutils.widgets.text("model_name", "sentence-transformers/all-MiniLM-L6-v2", "5. Embedding model")
dbutils.widgets.text("batch_size", "32", "6. Encode batch size")
dbutils.widgets.dropdown("normalize", "true", ["true", "false"], "7. Normalize vectors (cosine)")
dbutils.widgets.dropdown("write_mode", "overwrite", ["overwrite", "append"], "8. Write mode")
dbutils.widgets.dropdown("save_embeddings", "true", ["true", "false"], "9. Save embeddings to Delta")

print("Widgets declared.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 - Read parameters into variables

# COMMAND ----------

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
CHUNKS_TABLE = dbutils.widgets.get("chunks_table").strip()
EMBEDDINGS_TABLE = dbutils.widgets.get("embeddings_table").strip()
MODEL_NAME = dbutils.widgets.get("model_name").strip()
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))
NORMALIZE = dbutils.widgets.get("normalize") == "true"
WRITE_MODE = dbutils.widgets.get("write_mode")
SAVE_EMBEDDINGS = dbutils.widgets.get("save_embeddings") == "true"

for name, value in {"catalog": CATALOG, "schema": SCHEMA, "chunks_table": CHUNKS_TABLE,
                    "model_name": MODEL_NAME}.items():
    assert value, f"Parameter '{name}' must not be empty."
assert BATCH_SIZE > 0, "batch_size must be > 0"

CHUNKS_FQN = f"{CATALOG}.{SCHEMA}.{CHUNKS_TABLE}"
EMBEDDINGS_FQN = f"{CATALOG}.{SCHEMA}.{EMBEDDINGS_TABLE}"

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

import importlib
import subprocess
import sys


def _ensure(pip_spec: str, module_name: str) -> None:
    try:
        importlib.import_module(module_name)
        print(f"ready: {module_name} (already installed)")
    except ModuleNotFoundError:
        print(f"installing {pip_spec} ... (this can take a few minutes)")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pip_spec])
        importlib.invalidate_caches()
        importlib.import_module(module_name)
        print(f"ready: {module_name}")


_ensure("sentence-transformers==3.3.1", "sentence_transformers")
print("Embedding dependency satisfied.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 - Load the chunks
# MAGIC Small POC corpus, so collecting to the driver is fine. For a large corpus,
# MAGIC embed partition-side with a pandas UDF instead of collecting.

# COMMAND ----------

chunk_cols = ["chunk_id", "document_id", "section", "page_number",
              "source_path", "content_hash", "chunk_text"]
chunk_rows = spark.table(CHUNKS_FQN).select(*chunk_cols).orderBy("chunk_id").collect()
print(f"{len(chunk_rows)} chunk(s) loaded from {CHUNKS_FQN}")
assert chunk_rows, f"{CHUNKS_FQN} is empty. Run Phase 3 first."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 - Encode the chunks
# MAGIC The **same** model must later embed questions at query time.

# COMMAND ----------

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(MODEL_NAME)
print(f"loaded {MODEL_NAME}")

texts = [r["chunk_text"] for r in chunk_rows]
vectors = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    normalize_embeddings=NORMALIZE,
    show_progress_bar=False,
    convert_to_numpy=True,
)
# Derive the dimension from the output (version-independent).
EMBEDDING_DIM = int(vectors.shape[1])
print("encoded:", vectors.shape, "| dim:", EMBEDDING_DIM)
assert vectors.shape[0] == len(chunk_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 - Build the embeddings DataFrame

# COMMAND ----------

from datetime import datetime, timezone

from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

now_iso = datetime.now(timezone.utc).isoformat()
emb_rows = []
for r, vec in zip(chunk_rows, vectors):
    emb_rows.append(
        {
            "chunk_id": r["chunk_id"],
            "document_id": r["document_id"],
            "section": r["section"],
            "page_number": r["page_number"],
            "source_path": r["source_path"],
            "content_hash": r["content_hash"],
            "chunk_text": r["chunk_text"],
            "embedding": [float(x) for x in vec.tolist()],
            "embedding_model": MODEL_NAME,
            "embedding_dim": int(EMBEDDING_DIM),
            "normalized": NORMALIZE,
            "created_at": now_iso,
        }
    )

embeddings_schema = StructType(
    [
        StructField("chunk_id", StringType(), False),
        StructField("document_id", StringType(), False),
        StructField("section", StringType(), True),
        StructField("page_number", IntegerType(), True),
        StructField("source_path", StringType(), False),
        StructField("content_hash", StringType(), False),
        StructField("chunk_text", StringType(), False),
        StructField("embedding", ArrayType(FloatType()), False),
        StructField("embedding_model", StringType(), False),
        StructField("embedding_dim", IntegerType(), False),
        StructField("normalized", BooleanType := __import__("pyspark.sql.types", fromlist=["BooleanType"]).BooleanType(), False),
        StructField("created_at", StringType(), False),
    ]
)

embeddings_df = spark.createDataFrame(emb_rows, schema=embeddings_schema)
display(embeddings_df.select("chunk_id", "document_id", "section", "embedding_model", "embedding_dim"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 - (Optional) Save the embeddings as a Delta table

# COMMAND ----------

if SAVE_EMBEDDINGS:
    (embeddings_df.write.mode(WRITE_MODE).saveAsTable(EMBEDDINGS_FQN))
    print(f"Saved embeddings to {EMBEDDINGS_FQN} (mode={WRITE_MODE}), rows={embeddings_df.count()}")
else:
    print("save_embeddings=false -> embeddings not persisted")
