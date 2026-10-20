# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Create a Vector Search index (OPTIONAL, managed variant)
# MAGIC
# MAGIC Phase 6 of the RAG POC, **managed path**. This is NOT part of the
# MAGIC zero-cost baseline. It uses **Databricks Vector Search**, a managed
# MAGIC service that provisions an **endpoint (compute)** and may incur cost
# MAGIC and/or be limited on Free Edition.
# MAGIC
# MAGIC A "Vector Search index" is a managed search engine over your embeddings:
# MAGIC you give it a question vector and it returns the most similar chunks fast.
# MAGIC The local alternative (no managed service) is FAISS in `src/retriever.py`.
# MAGIC
# MAGIC Safe by default: `create_resources=false` only prints the plan and
# MAGIC validates inputs. Nothing billable is created until you set it to
# MAGIC `true` **and** have confirmed availability/pricing for your workspace.
# MAGIC
# MAGIC Index design: **Delta Sync, self-managed embeddings** over the
# MAGIC `document_embeddings` table (we already computed the vectors in Phase 5),
# MAGIC so no managed embedding-model endpoint is required.
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
# The length of each vector; must match what notebook 04 produced.
dbutils.widgets.text("embedding_dim", "384", "9. Embedding dimension")
# TRIGGERED = sync on demand; CONTINUOUS = keep syncing automatically.
dbutils.widgets.dropdown("pipeline_type", "TRIGGERED", ["TRIGGERED", "CONTINUOUS"], "10. Sync pipeline type")
# The model used to embed the QUESTION at query time - must match Phase 5's model.
dbutils.widgets.text("query_model", "sentence-transformers/all-MiniLM-L6-v2", "11. Query embed model (match Phase 5)")
# The safety switch: "false" creates nothing; "true" creates billable resources.
dbutils.widgets.dropdown("create_resources", "false", ["false", "true"], "12. CREATE billable resources")

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
QUERY_MODEL = dbutils.widgets.get("query_model").strip()
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
print("PIPELINE_TYPE   :", PIPELINE_TYPE)
print("CREATE_RESOURCES:", CREATE_RESOURCES)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 - COST / AVAILABILITY NOTICE - read before enabling creation
# MAGIC Vector Search provisions a managed **endpoint (compute)** that can incur
# MAGIC charges and may be unavailable or limited on Free Edition. Confirm current
# MAGIC pricing and availability for this workspace before setting
# MAGIC `create_resources=true`. To remove everything afterward:
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
# MAGIC Delta Sync indexes require CDF on the source Delta table. This is a cheap
# MAGIC table property change (no endpoint yet). Runs only when creating.

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
# MAGIC **no pip install** and no dependency/protobuf conflict. Only the sample
# MAGIC query (Cell 9) needs an extra library.

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
# MAGIC ## Cell 7 - Create the endpoint (BILLABLE) - guarded
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
# MAGIC ## Cell 8 - Create the Delta Sync index (self-managed embeddings) - guarded
# MAGIC No `create_index_and_wait` exists in the SDK, so we poll `get_index`.

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


# If creation is off, do nothing.
if not CREATE_RESOURCES:
    print("create_resources=false -> NOT creating an index.")
# If it already exists, reuse it.
elif _index_exists(INDEX_FQN):
    print(f"index '{INDEX_FQN}' already exists")
# Otherwise create the Delta Sync index over our embeddings table.
else:
    print(f"creating Delta Sync index '{INDEX_FQN}' ...")
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
        # The for-loop's else runs only if we never "break" (still not ready).
        print("index not ready yet; check status later with get-index.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9 - Smoke-test the index with a self-embedded query - guarded
# MAGIC Embeds a sample question with the SAME model used in Phase 5 and runs a
# MAGIC similarity search via the SDK. Confirms the index returns grounded chunks.

# COMMAND ----------

# Skip the test if we did not create anything.
if not CREATE_RESOURCES:
    print("create_resources=false -> skipping index query smoke test.")
else:
    # Make sure sentence-transformers is available to embed the question.
    import importlib
    import subprocess
    import sys

    try:
        importlib.import_module("sentence_transformers")
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "sentence-transformers==3.3.1"])
        importlib.invalidate_caches()

    from sentence_transformers import SentenceTransformer

    # Delta Sync runs an initial sync after creation; wait until the index is
    # ready before querying (avoids "Vector index ... is not ready").
    print("waiting for the index to be queryable ...")
    for _ in range(90):  # up to ~15 min
        status = w.vector_search_indexes.get_index(INDEX_FQN).status
        if status and status.ready:
            break
        time.sleep(10)

    # Load the SAME embedding model used in Phase 5 (query and documents must match).
    model = SentenceTransformer(QUERY_MODEL)
    # A sample question to test retrieval.
    question = "What is the laptop or iPhone warranty and repair policy?"
    # Turn the question into a vector (normalized, like the stored vectors).
    qvec = model.encode([question], normalize_embeddings=True)[0].tolist()

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
