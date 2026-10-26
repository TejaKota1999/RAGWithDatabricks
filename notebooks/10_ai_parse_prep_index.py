# Databricks notebook source
# MAGIC %md
# MAGIC # 10 - Databricks-recommended RAG pipeline (ai_parse_document + ai_prep_search)
# MAGIC
# MAGIC This is the **Databricks-recommended** ingestion path, replacing the custom
# MAGIC extract (02) + chunk (03) + embed (04) notebooks with built-in AI Functions:
# MAGIC
# MAGIC 1. `ai_parse_document(content)` - parse each file in the volume (PDF/Office/image).
# MAGIC 2. `ai_prep_search(parsed)` - semantic chunking + context enrichment. It returns
# MAGIC    `chunk_id`, `chunk_to_retrieve` (clean text for the LLM), and `chunk_to_embed`
# MAGIC    (enriched with title/headers for better retrieval).
# MAGIC 3. **Vector Search with managed embeddings** - the index embeds `chunk_to_embed`
# MAGIC    itself (via an embedding endpoint), so queries are by TEXT, not a vector.
# MAGIC
# MAGIC Cost note: each `ai_*` call is one LLM inference (metered against the serverless
# MAGIC quota). The corpus is tiny, so usage is small. Run cell by cell.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 - Parameters (widgets)

# COMMAND ----------

dbutils.widgets.text("catalog", "databricks_rag_dev", "1. Catalog")
dbutils.widgets.text("schema", "rag", "2. Schema")
dbutils.widgets.text("volume", "source_docs", "3. Volume with documents")
# Output table of ai_prep_search chunks.
dbutils.widgets.text("chunks_table", "document_chunks_ai", "4. Output chunks table")
# Vector Search endpoint + index name.
dbutils.widgets.text("endpoint_name", "rag_poc_vs", "5. Vector Search endpoint")
dbutils.widgets.text("index_table", "document_chunks_ai_index", "6. Index name")
# Embedding endpoint the MANAGED index uses to embed chunk_to_embed AND queries.
dbutils.widgets.text("embedding_endpoint", "databricks-gte-large-en", "7. Embedding endpoint (managed)")
# Rebuild the index if it already exists (needed if the embedding endpoint changes).
dbutils.widgets.dropdown("recreate_index", "false", ["false", "true"], "8. Recreate index if exists")
# Safety switch: create the Vector Search index (managed compute, included in Free Edition).
dbutils.widgets.dropdown("create_index", "true", ["false", "true"], "9. Create / sync the index")

print("Widgets declared.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 - Read parameters

# COMMAND ----------

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
VOLUME = dbutils.widgets.get("volume").strip()
CHUNKS_TABLE = dbutils.widgets.get("chunks_table").strip()
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name").strip()
INDEX_TABLE = dbutils.widgets.get("index_table").strip()
EMBEDDING_ENDPOINT = dbutils.widgets.get("embedding_endpoint").strip()
RECREATE_INDEX = dbutils.widgets.get("recreate_index") == "true"
CREATE_INDEX = dbutils.widgets.get("create_index") == "true"

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
CHUNKS_FQN = f"{CATALOG}.{SCHEMA}.{CHUNKS_TABLE}"
INDEX_FQN = f"{CATALOG}.{SCHEMA}.{INDEX_TABLE}"

print("VOLUME_PATH       :", VOLUME_PATH)
print("CHUNKS_FQN        :", CHUNKS_FQN)
print("INDEX_FQN         :", INDEX_FQN)
print("EMBEDDING_ENDPOINT:", EMBEDDING_ENDPOINT)
print("CREATE_INDEX      :", CREATE_INDEX, "| RECREATE_INDEX:", RECREATE_INDEX)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 - Parse + chunk with ai_parse_document + ai_prep_search
# MAGIC Reads every file in the volume as binary, parses it, and produces semantic
# MAGIC chunks. Writes the `document_chunks_ai` Delta table and enables Change Data
# MAGIC Feed (required for the Vector Search Delta Sync index).

# COMMAND ----------

# spark.sql runs the AI Functions on the serverless engine; no endpoints to wire up.
spark.sql(f"""
CREATE OR REPLACE TABLE {CHUNKS_FQN} AS
WITH prepped AS (
  SELECT path AS source_path,
         ai_prep_search(ai_parse_document(content)) AS prep
  FROM read_files('{VOLUME_PATH}/', format => 'binaryFile')
)
SELECT
  variant_get(chunk, '$.chunk_id', 'STRING')          AS chunk_id,
  variant_get(chunk, '$.chunk_to_retrieve', 'STRING') AS chunk_to_retrieve,
  variant_get(chunk, '$.chunk_to_embed', 'STRING')    AS chunk_to_embed,
  source_path
FROM prepped LATERAL VIEW explode(variant_get(prep, '$.document.contents', 'ARRAY<VARIANT>')) c AS chunk
""")

# Change Data Feed lets the Vector Search Delta Sync index track changes.
spark.sql(f"ALTER TABLE {CHUNKS_FQN} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print("Built", CHUNKS_FQN, "and enabled Change Data Feed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 - Preview the chunks

# COMMAND ----------

n = spark.table(CHUNKS_FQN).count()
print(f"{n} chunk(s) in {CHUNKS_FQN}")
display(spark.table(CHUNKS_FQN).select("chunk_id", "chunk_to_retrieve", "source_path"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 - Create / sync the MANAGED-embeddings Vector Search index
# MAGIC The index embeds `chunk_to_embed` with the embedding endpoint, so no manual
# MAGIC embedding step is needed. Guarded by `create_index`.

# COMMAND ----------

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,   # MANAGED embeddings: name the source column + endpoint
    PipelineType,
    VectorIndexType,
)

w = WorkspaceClient()


def _index_exists(name):
    try:
        w.vector_search_indexes.get_index(name)
        return True
    except Exception:
        return False


def _create():
    print(f"creating managed-embeddings index {INDEX_FQN} (embeds via {EMBEDDING_ENDPOINT}) ...")
    w.vector_search_indexes.create_index(
        name=INDEX_FQN,
        endpoint_name=ENDPOINT_NAME,
        primary_key="chunk_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=CHUNKS_FQN,
            pipeline_type=PipelineType.TRIGGERED,
            embedding_source_columns=[
                EmbeddingSourceColumn(name="chunk_to_embed", embedding_model_endpoint_name=EMBEDDING_ENDPOINT)
            ],
        ),
    )
    for _ in range(90):
        st = w.vector_search_indexes.get_index(INDEX_FQN).status
        if st and st.ready:
            print("index ready, indexed rows:", getattr(st, "indexed_row_count", None))
            return
        time.sleep(10)
    print("index not ready yet; check later with get-index.")


if not CREATE_INDEX:
    print("create_index=false -> not creating/syncing the index.")
elif _index_exists(INDEX_FQN) and RECREATE_INDEX:
    print("index exists; recreate_index=true -> deleting then recreating ...")
    w.vector_search_indexes.delete_index(INDEX_FQN)
    for _ in range(30):
        if not _index_exists(INDEX_FQN):
            break
        time.sleep(5)
    _create()
elif _index_exists(INDEX_FQN):
    print("index exists -> triggering a sync to pick up new/changed rows ...")
    try:
        w.vector_search_indexes.sync_index(INDEX_FQN)
    except Exception as exc:
        print("sync:", exc)
else:
    _create()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 - Smoke test: query the index by TEXT

# COMMAND ----------

if CREATE_INDEX:
    # Wait until the index is queryable, then search by text (it embeds the query).
    for _ in range(60):
        st = w.vector_search_indexes.get_index(INDEX_FQN).status
        if st and st.ready:
            break
        time.sleep(10)
    res = w.vector_search_indexes.query_index(
        index_name=INDEX_FQN,
        columns=["chunk_id", "chunk_to_retrieve", "source_path"],
        query_text="What are the criteria for getting a replacement iPhone?",
        num_results=3,
    )
    for row in (res.result.data_array or []):
        print(row[0], "| score:", round(row[-1], 3), "|", (row[1] or "")[:80].replace("\n", " "))
else:
    print("create_index=false -> skipping smoke test.")
