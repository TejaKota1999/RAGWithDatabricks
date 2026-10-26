# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Create a Vector Search index (Databricks retrieval)
# MAGIC
# MAGIC Phase 6 of the RAG POC. This is now the **default retrieval path**: a
# MAGIC Databricks **Vector Search** Delta Sync index over the `document_embeddings`
# MAGIC table (self-managed 1024-dim vectors from notebook 04).
# MAGIC
# MAGIC A "Vector Search index" is a managed search engine over your embeddings:
# MAGIC you give it a question vector and it returns the most similar chunks fast.
# MAGIC
# MAGIC Cost note: Vector Search is included in Free Edition but provisions a
# MAGIC managed **endpoint (compute)** and is metered against the serverless quota.
# MAGIC `create_resources=false` only validates and prints the plan; nothing is
# MAGIC created until you set it to `true`.
# MAGIC
# MAGIC Important: the index dimension (1024) must match the embedding endpoint. If
# MAGIC an older 384-dim index exists under the same name, set `recreate_index=true`
# MAGIC to delete and rebuild it (dimension cannot be changed in place).
# MAGIC
# MAGIC Every configurable value is a widget / job parameter. Nothing is hardcoded.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 - Declare parameters (widgets)

# COMMAND ----------

# Catalog / schema holding the embeddings table and the new index.
dbutils.widgets.text("catalog", "databricks_rag_dev", "1. Catalog")
dbutils.widgets.text("schema", "rag", "2. Schema")
# The table of embeddings produced by notebook 04 (the search source).
dbutils.widgets.text("source_table", "document_embeddings", "3. Source embeddings table")
# The name of the Vector Search index to create (a Unity Catalog object).
dbutils.widgets.text("index_table", "document_chunks_index", "4. Index name (table)")
# The Vector Search endpoint (the managed compute that hosts the index).
dbutils.widgets.text("endpoint_name", "rag_poc_vs", "5. Vector Search endpoint")
# STANDARD = low latency; STORAGE_OPTIMIZED = cheaper for very large data.
dbutils.widgets.dropdown("endpoint_type", "STANDARD", ["STANDARD", "STORAGE_OPTIMIZED"], "6. Endpoint type")
# The unique id column in the source table (used as the index's key).
dbutils.widgets.text("primary_key", "chunk_id", "7. Primary key column")
# The column that holds the pre-computed embedding vector.
dbutils.widgets.text("embedding_column", "embedding", "8. Embedding vector column")
# The length of each vector; must match what notebook 04 produced (1024).
dbutils.widgets.text("embedding_dim", "1024", "9. Embedding dimension")
# TRIGGERED = sync on demand; CONTINUOUS = keep syncing automatically.
dbutils.widgets.dropdown("pipeline_type", "TRIGGERED", ["TRIGGERED", "CONTINUOUS"], "10. Sync pipeline type")
# The endpoint used to embed the QUESTION at query time - must match Phase 5.
dbutils.widgets.text("query_endpoint", "databricks-qwen3-embedding-0-6b", "11. Query embed endpoint (match Phase 5)")
# Delete and rebuild the index if one already exists (needed when the dim changed).
dbutils.widgets.dropdown("recreate_index", "false", ["false", "true"], "12. Recreate index if it exists")
# The safety switch: "false" creates nothing; "true" creates billable resources.
dbutils.widgets.dropdown("create_resources", "false", ["false", "true"], "13. CREATE resources")

print("Widgets declared.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 - Read parameters into variables

# COMMAND ----------

# Copy widget values into variables.
CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
SOURCE_TABLE = dbutils.widgets.get("source_table").strip()
INDEX_TABLE = dbutils.widgets.get("index_table").strip()
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name").strip()
ENDPOINT_TYPE = dbutils.widgets.get("endpoint_type")
PRIMARY_KEY = dbutils.widgets.get("primary_key").strip()
EMBEDDING_COLUMN = dbutils.widgets.get("embedding_column").strip()
EMBEDDING_DIM = int(dbutils.widgets.get("embedding_dim"))         # text -> integer
PIPELINE_TYPE = dbutils.widgets.get("pipeline_type")
QUERY_ENDPOINT = dbutils.widgets.get("query_endpoint").strip()
RECREATE_INDEX = dbutils.widgets.get("recreate_index") == "true"
# The guard: only True when the dropdown is exactly "true".
CREATE_RESOURCES = dbutils.widgets.get("create_resources") == "true"

# Full names for the source table and the index.
SOURCE_FQN = f"{CATALOG}.{SCHEMA}.{SOURCE_TABLE}"
INDEX_FQN = f"{CATALOG}.{SCHEMA}.{INDEX_TABLE}"

# Echo the resolved settings, including whether creation is enabled.
print("SOURCE_FQN      :", SOURCE_FQN)
print("INDEX_FQN       :", INDEX_FQN)
print("ENDPOINT        :", ENDPOINT_NAME, f"({ENDPOINT_TYPE})")
print("PRIMARY_KEY     :", PRIMARY_KEY)
print("EMBEDDING_COLUMN:", EMBEDDING_COLUMN, f"(dim={EMBEDDING_DIM})")
print("QUERY_ENDPOINT  :", QUERY_ENDPOINT)
print("PIPELINE_TYPE   :", PIPELINE_TYPE)
print("RECREATE_INDEX  :", RECREATE_INDEX)
print("CREATE_RESOURCES:", CREATE_RESOURCES)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 - COST / AVAILABILITY NOTICE - read before enabling creation
# MAGIC Vector Search provisions a managed **endpoint (compute)**, included in Free
# MAGIC Edition but metered against the serverless quota. To remove everything:
# MAGIC
# MAGIC ```
# MAGIC databricks vector-search-indexes delete-index <INDEX_FQN> --profile <p>
# MAGIC databricks vector-search-endpoints delete-endpoint <ENDPOINT> --profile <p>
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 - Validate the source table (read-only, always safe)
# MAGIC Confirms the primary key and embedding column exist and the dimension
# MAGIC matches before any resource is created.

# COMMAND ----------

# F gives Spark SQL functions; we use F.size to measure the vector length.
from pyspark.sql import functions as F

# Build a {column_name: type} map of the source table's columns.
cols = {f.name: f.dataType.simpleString() for f in spark.table(SOURCE_FQN).schema.fields}
# The key and embedding columns must exist, or stop with a clear message.
assert PRIMARY_KEY in cols, f"primary_key '{PRIMARY_KEY}' not in {SOURCE_FQN} ({list(cols)})"
assert EMBEDDING_COLUMN in cols, f"embedding_column '{EMBEDDING_COLUMN}' not in {SOURCE_FQN}"

# Count rows, count distinct keys, and read one vector's length.
n_rows = spark.table(SOURCE_FQN).count()
n_keys = spark.table(SOURCE_FQN).select(PRIMARY_KEY).distinct().count()
actual_dim = spark.table(SOURCE_FQN).select(F.size(EMBEDDING_COLUMN).alias("d")).first()["d"]
print(f"rows={n_rows}  distinct {PRIMARY_KEY}={n_keys}  vector length={actual_dim}")
# Delta Sync needs a unique key: distinct keys must equal total rows.
assert n_keys == n_rows, "primary key is not unique; Delta Sync requires a unique key"
# The declared dimension must match the real vector length.
assert actual_dim == EMBEDDING_DIM, f"embedding_dim param ({EMBEDDING_DIM}) != actual ({actual_dim})"
print("Source table is valid for a self-managed Delta Sync index.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 - Enable Change Data Feed on the source table
# MAGIC Delta Sync indexes require CDF on the source Delta table (cheap table
# MAGIC property change; no endpoint yet). Runs only when creating.

# COMMAND ----------

# Change Data Feed lets the index detect row changes to sync. Only do this when creating.
if CREATE_RESOURCES:
    spark.sql(f"ALTER TABLE {SOURCE_FQN} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    print(f"Change Data Feed enabled on {SOURCE_FQN}")
else:
    print("create_resources=false -> skipping CDF change (no resources will be created)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 - Vector Search client (SDK, pre-installed)
# MAGIC Uses the `databricks-sdk` already on the serverless runtime, so there is
# MAGIC **no pip install** and no dependency/protobuf conflict.

# COMMAND ----------

# WorkspaceClient is the main entry point to the Databricks APIs from a notebook.
from databricks.sdk import WorkspaceClient
# These types describe the index we want to create.
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,   # the overall "Delta Sync index" spec
    EmbeddingVectorColumn,             # says which column holds the vector and its length
    EndpointType,                      # STANDARD / STORAGE_OPTIMIZED
    PipelineType,                      # TRIGGERED / CONTINUOUS
    VectorIndexType,                   # DELTA_SYNC / DIRECT_ACCESS
)

# Create the client (uses the notebook's own credentials automatically).
w = WorkspaceClient()
print("WorkspaceClient ready (SDK vector search API).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 - Create the endpoint - guarded
# MAGIC Creates the Vector Search endpoint and waits until it is online. This is
# MAGIC the step that provisions managed compute.

# COMMAND ----------


# Helper: returns True if an endpoint with this name already exists.
def _endpoint_exists(name: str) -> bool:
    try:
        w.vector_search_endpoints.get_endpoint(name)   # succeeds -> it exists
        return True
    except Exception:
        return False                                   # error -> it does not exist


# If creation is off, do nothing (safe).
if not CREATE_RESOURCES:
    print("create_resources=false -> NOT creating an endpoint. Set it to 'true' to proceed.")
# If it already exists, reuse it.
elif _endpoint_exists(ENDPOINT_NAME):
    print(f"endpoint '{ENDPOINT_NAME}' already exists")
# Otherwise create it and wait until it is online (this can take several minutes).
else:
    print(f"creating endpoint '{ENDPOINT_NAME}' ({ENDPOINT_TYPE}) - waits until online ...")
    w.vector_search_endpoints.create_endpoint_and_wait(
        name=ENDPOINT_NAME,
        endpoint_type=EndpointType(ENDPOINT_TYPE),     # convert the text to the enum value
    )
    print("endpoint online")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8 - Create the Delta Sync index (self-managed 1024-dim) - guarded
# MAGIC If the index already exists and `recreate_index=true`, it is deleted and
# MAGIC rebuilt (needed when the embedding dimension changed, e.g. 384 -> 1024).

# COMMAND ----------

# time.sleep is used to pause between status checks while the index builds.
import time


# Helper: returns True if an index with this name already exists.
def _index_exists(name: str) -> bool:
    try:
        w.vector_search_indexes.get_index(name)
        return True
    except Exception:
        return False


# Create the Delta Sync index over our embeddings table.
def _create_index():
    print(f"creating Delta Sync index '{INDEX_FQN}' (dim={EMBEDDING_DIM}) ...")
    w.vector_search_indexes.create_index(
        name=INDEX_FQN,                                 # the index's full name
        endpoint_name=ENDPOINT_NAME,                    # which endpoint hosts it
        primary_key=PRIMARY_KEY,                        # the unique id column
        index_type=VectorIndexType.DELTA_SYNC,          # auto-syncs from a Delta table
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=SOURCE_FQN,                    # read vectors from this table
            pipeline_type=PipelineType(PIPELINE_TYPE),  # how it syncs
            embedding_vector_columns=[                  # we supply our own vectors (self-managed)
                EmbeddingVectorColumn(name=EMBEDDING_COLUMN, embedding_dimension=EMBEDDING_DIM)
            ],
        ),
    )
    # The index does an initial build/sync; poll its status until ready.
    print("waiting for the index to become ready (polling) ...")
    for _ in range(90):  # up to ~15 min (90 checks x 10 seconds)
        status = w.vector_search_indexes.get_index(INDEX_FQN).status
        if status and status.ready:                     # ready means it can be queried
            print("index ready")
            break
        time.sleep(10)                                  # wait 10s and check again
    else:
        print("index not ready yet; check status later with get-index.")


# If creation is off, do nothing.
if not CREATE_RESOURCES:
    print("create_resources=false -> NOT creating an index.")
# If it exists and we were asked to recreate it, delete then rebuild.
elif _index_exists(INDEX_FQN) and RECREATE_INDEX:
    print(f"index '{INDEX_FQN}' exists; recreate_index=true -> deleting it first ...")
    w.vector_search_indexes.delete_index(INDEX_FQN)
    # Wait until it is actually gone before recreating.
    for _ in range(30):
        if not _index_exists(INDEX_FQN):
            break
        time.sleep(5)
    _create_index()
# If it exists and we were NOT asked to recreate, trigger a SYNC so the index
# picks up any new/changed rows added by an incremental notebook-04 run.
elif _index_exists(INDEX_FQN):
    print(f"index '{INDEX_FQN}' already exists -> triggering a sync to pick up new/changed rows ...")
    try:
        w.vector_search_indexes.sync_index(INDEX_FQN)   # pulls the latest from the source table
        print("sync triggered (set recreate_index=true instead if the dimension changed)")
    except Exception as exc:
        print("could not trigger sync:", exc)
# Otherwise create it fresh.
else:
    _create_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9 - Smoke-test the index with a self-embedded query - guarded
# MAGIC Embeds a sample question with the SAME Databricks endpoint used in Phase 5
# MAGIC and runs a similarity search. Confirms the index returns grounded chunks.

# COMMAND ----------

# Skip the test if we did not create anything.
if not CREATE_RESOURCES:
    print("create_resources=false -> skipping index query smoke test.")
else:
    # Delta Sync runs an initial sync after creation; wait until ready before querying.
    print("waiting for the index to be queryable ...")
    for _ in range(90):  # up to ~15 min
        status = w.vector_search_indexes.get_index(INDEX_FQN).status
        if status and status.ready:
            break
        time.sleep(10)

    # A sample question to test retrieval.
    question = "What are the criteria for getting a replacement iPhone?"
    # Embed the question with the SAME Databricks endpoint used for the chunks.
    qresp = w.serving_endpoints.query(name=QUERY_ENDPOINT, input=[question])
    qvec = [float(x) for x in qresp.data[0].embedding]

    # Ask the index for the 3 most similar chunks to the question vector.
    res = w.vector_search_indexes.query_index(
        index_name=INDEX_FQN,
        columns=["chunk_id", "section", "chunk_text"],   # columns to return
        query_vector=qvec,
        num_results=3,
    )
    # The results come back as rows; the last value in each row is the similarity score.
    data = res.result.data_array if res.result else []
    for row in data:
        print(row[:2], "| score:", row[-1])
