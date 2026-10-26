# Databricks notebook source
# MAGIC %md
# MAGIC # 10 - Databricks-recommended RAG pipeline (ai_parse_document + ai_prep_search)
# MAGIC
# MAGIC The **recommended** ingestion path, replacing the custom extract (02) +
# MAGIC chunk (03) + embed (04) notebooks with built-in AI Functions:
# MAGIC
# MAGIC 1. `ai_parse_document(content)` - parse each file (PDF/Office/image).
# MAGIC 2. `ai_prep_search(parsed)` - semantic chunking + context enrichment. Returns
# MAGIC    `chunk_id`, `chunk_to_retrieve` (clean text for the LLM) and `chunk_to_embed`
# MAGIC    (enriched with title/headers for better retrieval).
# MAGIC 3. **Vector Search with managed embeddings** - the index embeds `chunk_to_embed`
# MAGIC    itself, so the chatbot queries by TEXT, not a vector.
# MAGIC
# MAGIC Supports **incremental** runs: only new or changed files (by content hash)
# MAGIC are parsed/chunked, so the metered AI Functions run only on fresh text.
# MAGIC Run cell by cell, or as the `rag_ingest_ai` job.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 - Parameters (widgets)

# COMMAND ----------

# dbutils.widgets.text(name, default, label) makes an input box at the top of the notebook.
dbutils.widgets.text("catalog", "databricks_rag_dev", "1. Catalog")                 # Unity Catalog catalog
dbutils.widgets.text("schema", "rag", "2. Schema")                                   # schema inside the catalog
dbutils.widgets.text("volume", "source_docs", "3. Volume with documents")            # volume holding the files
dbutils.widgets.text("chunks_table", "document_chunks_ai", "4. Output chunks table") # output Delta table
dbutils.widgets.text("endpoint_name", "rag_poc_vs", "5. Vector Search endpoint")     # VS endpoint that hosts the index
dbutils.widgets.text("index_table", "document_chunks_ai_index", "6. Index name")     # the index to create
# The embedding endpoint the MANAGED index uses to embed chunk_to_embed AND queries.
dbutils.widgets.text("embedding_endpoint", "databricks-gte-large-en", "7. Embedding endpoint (managed)")
# Rebuild the index if it already exists (needed only if the embedding endpoint changes).
dbutils.widgets.dropdown("recreate_index", "false", ["false", "true"], "8. Recreate index if exists")
# Create / sync the Vector Search index as part of this run.
dbutils.widgets.dropdown("create_index", "true", ["false", "true"], "9. Create / sync the index")
# Incremental: only parse/chunk NEW or CHANGED files (keyed on the file content hash).
dbutils.widgets.dropdown("incremental", "true", ["false", "true"], "10. Incremental (only new/changed)")

print("Widgets declared.")                                                           # confirm the widgets exist

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 - Read parameters into variables

# COMMAND ----------

# dbutils.widgets.get(name) reads a widget's current value as text; .strip() trims spaces.
CATALOG = dbutils.widgets.get("catalog").strip()                                     # e.g. "databricks_rag_dev"
SCHEMA = dbutils.widgets.get("schema").strip()                                       # e.g. "rag"
VOLUME = dbutils.widgets.get("volume").strip()                                       # e.g. "source_docs"
CHUNKS_TABLE = dbutils.widgets.get("chunks_table").strip()                           # output table name
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name").strip()                         # VS endpoint
INDEX_TABLE = dbutils.widgets.get("index_table").strip()                             # index name
EMBEDDING_ENDPOINT = dbutils.widgets.get("embedding_endpoint").strip()               # embedding model endpoint
RECREATE_INDEX = dbutils.widgets.get("recreate_index") == "true"                     # text -> True/False
CREATE_INDEX = dbutils.widgets.get("create_index") == "true"                         # text -> True/False
INCREMENTAL = dbutils.widgets.get("incremental") == "true"                           # text -> True/False

# Build the full paths/names used below.
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"                                 # where the documents live
CHUNKS_FQN = f"{CATALOG}.{SCHEMA}.{CHUNKS_TABLE}"                                     # catalog.schema.table
INDEX_FQN = f"{CATALOG}.{SCHEMA}.{INDEX_TABLE}"                                       # catalog.schema.index

# Echo the resolved settings so you can confirm them before running on.
print("VOLUME_PATH       :", VOLUME_PATH)
print("CHUNKS_FQN        :", CHUNKS_FQN)
print("INDEX_FQN         :", INDEX_FQN)
print("EMBEDDING_ENDPOINT:", EMBEDDING_ENDPOINT)
print("INCREMENTAL       :", INCREMENTAL, "| CREATE_INDEX:", CREATE_INDEX, "| RECREATE_INDEX:", RECREATE_INDEX)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 - Parse + chunk with ai_parse_document + ai_prep_search
# MAGIC Reads every file as binary, parses it, and produces semantic chunks. In
# MAGIC incremental mode only new/changed files are processed; otherwise it rebuilds
# MAGIC the whole table. Change Data Feed is enabled (needed by the Delta Sync index).

# COMMAND ----------


# Returns the parse+chunk SQL for a given source view. Each row of the view must
# have: source_path (the file path), content (the file bytes), content_hash (a
# fingerprint of the bytes). We keep source_path + content_hash on every chunk so
# incremental runs can detect and replace a changed file's rows.
def _chunks_sql(source_view):
    return f"""
    WITH prepped AS (                                     -- one row per source file
      SELECT source_path, content_hash,
             ai_prep_search(ai_parse_document(content)) AS prep   -- parse then semantic-chunk
      FROM {source_view}
    )
    SELECT
      variant_get(chunk, '$.chunk_id', 'STRING')          AS chunk_id,           -- stable chunk id
      variant_get(chunk, '$.chunk_to_retrieve', 'STRING') AS chunk_to_retrieve,  -- clean text for the LLM
      variant_get(chunk, '$.chunk_to_embed', 'STRING')    AS chunk_to_embed,     -- enriched text to embed
      source_path,                                                               -- which file it came from
      content_hash                                                               -- the file's fingerprint
    FROM prepped
    LATERAL VIEW explode(variant_get(prep, '$.document.contents', 'ARRAY<VARIANT>')) c AS chunk  -- one row per chunk
    """


# List every file in the volume, with a SHA-256 fingerprint of its bytes.
# read_files(..., binaryFile) returns: path, content (bytes), modificationTime, length.
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW _all_files AS
SELECT path AS source_path, content, sha2(content, 256) AS content_hash
FROM read_files('{VOLUME_PATH}/', format => 'binaryFile')
""")

# Decide full rebuild vs incremental. Incremental needs the table to already exist
# AND have the content_hash column (older tables are rebuilt once to gain it).
table_exists = spark.catalog.tableExists(CHUNKS_FQN)                                  # does the output table exist?
has_hash = table_exists and "content_hash" in [f.name for f in spark.table(CHUNKS_FQN).schema]  # has the hash column?

if INCREMENTAL and table_exists and has_hash:
    # New/changed files = those whose content fingerprint is not already in the table.
    spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW _new_files AS
    SELECT * FROM _all_files
    WHERE content_hash NOT IN (SELECT DISTINCT content_hash FROM {CHUNKS_FQN})
    """)
    # Collect the file PATHS to process (deterministic: read_files + sha2, no AI calls).
    new_paths = [r[0] for r in spark.table("_new_files").select("source_path").distinct().collect()]

    if not new_paths:
        # Nothing changed -> do not call the (metered) AI Functions at all.
        print("Incremental: no new or changed files -> nothing to parse (no cost).")
    else:
        # Materialize the new chunks to a STAGING table so the AI Functions run ONCE
        # here (they are non-deterministic, so we cannot reference them in a DELETE
        # subquery - that is why we stage first, then delete by a literal path list).
        STAGING = f"{CHUNKS_FQN}_staging"
        spark.sql(f"DROP TABLE IF EXISTS {STAGING}")
        spark.sql(f"CREATE TABLE {STAGING} AS {_chunks_sql('_new_files')}")           # parse/chunk new files once

        # Build a literal IN-list of the processed paths (deterministic predicate).
        in_list = ", ".join("'" + p.replace("'", "''") + "'" for p in new_paths)
        # Remove any OLD rows for those files (handles a changed/overwritten file),
        # then append the fresh chunks. Untouched files are left exactly as they are.
        spark.sql(f"DELETE FROM {CHUNKS_FQN} WHERE source_path IN ({in_list})")
        spark.sql(f"INSERT INTO {CHUNKS_FQN} SELECT * FROM {STAGING}")
        spark.sql(f"DROP TABLE IF EXISTS {STAGING}")                                  # tidy up the staging table
        print(f"Incremental: parsed {len(new_paths)} new/changed file(s) and upserted their chunks.")
else:
    # Full rebuild: parse/chunk every file and replace the table (also used on the
    # first run and to add the content_hash column to an older table).
    spark.sql(f"CREATE OR REPLACE TABLE {CHUNKS_FQN} AS {_chunks_sql('_all_files')}")
    print("Full rebuild of", CHUNKS_FQN)

# Change Data Feed lets the Vector Search Delta Sync index track row changes.
spark.sql(f"ALTER TABLE {CHUNKS_FQN} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print("Change Data Feed enabled on", CHUNKS_FQN)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 - Preview the chunks

# COMMAND ----------

n = spark.table(CHUNKS_FQN).count()                                                   # how many chunks in total
print(f"{n} chunk(s) in {CHUNKS_FQN}")
# Show the clean text + which file each chunk came from (not the huge embed text).
display(spark.table(CHUNKS_FQN).select("chunk_id", "chunk_to_retrieve", "source_path"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 - Create / sync the MANAGED-embeddings Vector Search index
# MAGIC The index embeds `chunk_to_embed` with the embedding endpoint, so there is
# MAGIC no manual embedding step. Guarded by `create_index`.

# COMMAND ----------

# time.sleep is used to pause between status checks while the index builds.
import time

# WorkspaceClient is the entry point to the Databricks APIs from a notebook.
from databricks.sdk import WorkspaceClient
# These types describe a MANAGED-embeddings Delta Sync index.
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,   # the overall Delta Sync index spec
    EmbeddingSourceColumn,             # MANAGED embeddings: the source column + endpoint
    PipelineType,                      # TRIGGERED / CONTINUOUS
    VectorIndexType,                   # DELTA_SYNC / DIRECT_ACCESS
)

w = WorkspaceClient()                                                                 # uses the notebook's own creds


# Helper: True if an index with this name already exists.
def _index_exists(name):
    try:
        w.vector_search_indexes.get_index(name)                                       # succeeds -> exists
        return True
    except Exception:
        return False                                                                  # error -> does not exist


# Helper: trigger a sync and WAIT for it to finish. A Delta Sync index rejects a
# sync while a previous one is still RUNNING, so we RETRY until it is accepted
# (swallowing that error is what previously left new rows un-indexed), then poll
# until the row count stops changing.
def _sync_and_wait():
    for _ in range(30):                                 # retry (~10 min) until the sync is accepted
        try:
            w.vector_search_indexes.sync_index(INDEX_FQN)
            print("sync triggered")
            break
        except Exception as exc:                        # not COMPLETED/FAILED/CANCELED yet -> wait + retry
            print("index busy, waiting to sync ...", str(exc)[:90])
            time.sleep(20)
    last = -1                                           # wait until the indexed row count is stable
    for _ in range(90):                                 # up to ~15 min
        st = w.vector_search_indexes.get_index(INDEX_FQN).status
        n = (st.indexed_row_count if st else 0) or 0
        if st and st.ready and n == last:               # ready AND unchanged since last check -> done
            print("index synced, indexed rows:", n)
            return
        last = n
        time.sleep(10)
    print("index sync still finishing; it will complete shortly.")


# Helper: create the managed-embeddings index and wait until it is ready.
def _create():
    print(f"creating managed-embeddings index {INDEX_FQN} (embeds via {EMBEDDING_ENDPOINT}) ...")
    w.vector_search_indexes.create_index(
        name=INDEX_FQN,                                 # the index's full name
        endpoint_name=ENDPOINT_NAME,                    # which endpoint hosts it
        primary_key="chunk_id",                         # the unique id column
        index_type=VectorIndexType.DELTA_SYNC,          # auto-syncs from the Delta table
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=CHUNKS_FQN,                    # read chunks from this table
            pipeline_type=PipelineType.TRIGGERED,       # sync on demand
            embedding_source_columns=[                  # MANAGED: VS embeds this column itself
                EmbeddingSourceColumn(name="chunk_to_embed", embedding_model_endpoint_name=EMBEDDING_ENDPOINT)
            ],
        ),
    )
    # The index does an initial build/sync; poll until it reports ready.
    for _ in range(90):                                 # up to ~15 min
        st = w.vector_search_indexes.get_index(INDEX_FQN).status
        if st and st.ready:
            print("index ready, indexed rows:", getattr(st, "indexed_row_count", None))
            return
        time.sleep(10)
    print("index not ready yet; check later with get-index.")


if not CREATE_INDEX:
    # Safety: do nothing when index creation/sync is turned off.
    print("create_index=false -> not creating/syncing the index.")
elif _index_exists(INDEX_FQN) and RECREATE_INDEX:
    # Rebuild from scratch (needed only if the embedding endpoint/dimension changed).
    print("index exists; recreate_index=true -> deleting then recreating ...")
    w.vector_search_indexes.delete_index(INDEX_FQN)
    for _ in range(30):                                 # wait until it is actually gone
        if not _index_exists(INDEX_FQN):
            break
        time.sleep(5)
    _create()
elif _index_exists(INDEX_FQN):
    # Already exists -> sync it (robustly) so it picks up new/changed rows.
    print("index exists -> syncing to pick up new/changed rows ...")
    _sync_and_wait()
else:
    # First time -> create it.
    _create()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 - Smoke test: query the index by TEXT

# COMMAND ----------

if CREATE_INDEX:
    # Wait until the index is queryable (a fresh sync may still be running).
    for _ in range(60):                                 # up to ~10 min
        st = w.vector_search_indexes.get_index(INDEX_FQN).status
        if st and st.ready:
            break
        time.sleep(10)
    # Query by TEXT - the managed index embeds the question with the same endpoint.
    res = w.vector_search_indexes.query_index(
        index_name=INDEX_FQN,
        columns=["chunk_id", "chunk_to_retrieve", "source_path"],   # columns to return
        query_text="What are the criteria for getting a replacement iPhone?",
        num_results=3,
    )
    # Each row ends with the similarity score; print the id, score, and a snippet.
    for row in (res.result.data_array or []):
        print(row[0], "| score:", round(row[-1], 3), "|", (row[1] or "")[:80].replace("\n", " "))
else:
    print("create_index=false -> skipping smoke test.")
