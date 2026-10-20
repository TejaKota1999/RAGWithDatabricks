# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Create a Vector Search index (OPTIONAL, managed variant)
# MAGIC
# MAGIC Phase 6 of the RAG POC, **managed path**. This is NOT part of the
# MAGIC zero-cost baseline. It uses **Databricks Vector Search**, a managed
# MAGIC service that provisions an **endpoint (compute)** and may incur cost
# MAGIC and/or be limited on Free Edition.
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

dbutils.widgets.text("catalog", "databricks_rag_dev", "1. Catalog")
dbutils.widgets.text("schema", "rag", "2. Schema")
dbutils.widgets.text("source_table", "document_embeddings", "3. Source embeddings table")
dbutils.widgets.text("index_table", "document_chunks_index", "4. Index name (table)")
dbutils.widgets.text("endpoint_name", "rag_poc_vs", "5. Vector Search endpoint")
dbutils.widgets.dropdown("endpoint_type", "STANDARD", ["STANDARD", "STORAGE_OPTIMIZED"], "6. Endpoint type")
dbutils.widgets.text("primary_key", "chunk_id", "7. Primary key column")
dbutils.widgets.text("embedding_column", "embedding", "8. Embedding vector column")
dbutils.widgets.text("embedding_dim", "384", "9. Embedding dimension")
dbutils.widgets.dropdown("pipeline_type", "TRIGGERED", ["TRIGGERED", "CONTINUOUS"], "10. Sync pipeline type")
dbutils.widgets.text("query_model", "sentence-transformers/all-MiniLM-L6-v2", "11. Query embed model (match Phase 5)")
dbutils.widgets.dropdown("create_resources", "false", ["false", "true"], "12. CREATE billable resources")

print("Widgets declared.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 - Read parameters into variables

# COMMAND ----------

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
SOURCE_TABLE = dbutils.widgets.get("source_table").strip()
INDEX_TABLE = dbutils.widgets.get("index_table").strip()
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name").strip()
ENDPOINT_TYPE = dbutils.widgets.get("endpoint_type")
PRIMARY_KEY = dbutils.widgets.get("primary_key").strip()
EMBEDDING_COLUMN = dbutils.widgets.get("embedding_column").strip()
EMBEDDING_DIM = int(dbutils.widgets.get("embedding_dim"))
PIPELINE_TYPE = dbutils.widgets.get("pipeline_type")
QUERY_MODEL = dbutils.widgets.get("query_model").strip()
CREATE_RESOURCES = dbutils.widgets.get("create_resources") == "true"

SOURCE_FQN = f"{CATALOG}.{SCHEMA}.{SOURCE_TABLE}"
INDEX_FQN = f"{CATALOG}.{SCHEMA}.{INDEX_TABLE}"

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

from pyspark.sql import functions as F

cols = {f.name: f.dataType.simpleString() for f in spark.table(SOURCE_FQN).schema.fields}
assert PRIMARY_KEY in cols, f"primary_key '{PRIMARY_KEY}' not in {SOURCE_FQN} ({list(cols)})"
assert EMBEDDING_COLUMN in cols, f"embedding_column '{EMBEDDING_COLUMN}' not in {SOURCE_FQN}"

n_rows = spark.table(SOURCE_FQN).count()
n_keys = spark.table(SOURCE_FQN).select(PRIMARY_KEY).distinct().count()
actual_dim = spark.table(SOURCE_FQN).select(F.size(EMBEDDING_COLUMN).alias("d")).first()["d"]
print(f"rows={n_rows}  distinct {PRIMARY_KEY}={n_keys}  vector length={actual_dim}")
assert n_keys == n_rows, "primary key is not unique; Delta Sync requires a unique key"
assert actual_dim == EMBEDDING_DIM, f"embedding_dim param ({EMBEDDING_DIM}) != actual ({actual_dim})"
print("Source table is valid for a self-managed Delta Sync index.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 - Enable Change Data Feed on the source table
# MAGIC Delta Sync indexes require CDF on the source Delta table. This is a cheap
# MAGIC table property change (no endpoint yet). Runs only when creating.

# COMMAND ----------

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

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
    EndpointType,
    PipelineType,
    VectorIndexType,
)

w = WorkspaceClient()
print("WorkspaceClient ready (SDK vector search API).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 - Create the endpoint (BILLABLE) - guarded
# MAGIC Creates the Vector Search endpoint and waits until it is online. This is
# MAGIC the step that provisions managed compute.

# COMMAND ----------


def _endpoint_exists(name: str) -> bool:
    try:
        w.vector_search_endpoints.get_endpoint(name)
        return True
    except Exception:
        return False


if not CREATE_RESOURCES:
    print("create_resources=false -> NOT creating an endpoint. Set it to 'true' to proceed.")
elif _endpoint_exists(ENDPOINT_NAME):
    print(f"endpoint '{ENDPOINT_NAME}' already exists")
else:
    print(f"creating endpoint '{ENDPOINT_NAME}' ({ENDPOINT_TYPE}) - waits until online ...")
    w.vector_search_endpoints.create_endpoint_and_wait(
        name=ENDPOINT_NAME,
        endpoint_type=EndpointType(ENDPOINT_TYPE),
    )
    print("endpoint online")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8 - Create the Delta Sync index (self-managed embeddings) - guarded
# MAGIC No `create_index_and_wait` exists in the SDK, so we poll `get_index`.

# COMMAND ----------

import time


def _index_exists(name: str) -> bool:
    try:
        w.vector_search_indexes.get_index(name)
        return True
    except Exception:
        return False


if not CREATE_RESOURCES:
    print("create_resources=false -> NOT creating an index.")
elif _index_exists(INDEX_FQN):
    print(f"index '{INDEX_FQN}' already exists")
else:
    print(f"creating Delta Sync index '{INDEX_FQN}' ...")
    w.vector_search_indexes.create_index(
        name=INDEX_FQN,
        endpoint_name=ENDPOINT_NAME,
        primary_key=PRIMARY_KEY,
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=SOURCE_FQN,
            pipeline_type=PipelineType(PIPELINE_TYPE),
            embedding_vector_columns=[
                EmbeddingVectorColumn(name=EMBEDDING_COLUMN, embedding_dimension=EMBEDDING_DIM)
            ],
        ),
    )
    print("waiting for the index to become ready (polling) ...")
    for _ in range(90):  # up to ~15 min
        status = w.vector_search_indexes.get_index(INDEX_FQN).status
        if status and status.ready:
            print("index ready")
            break
        time.sleep(10)
    else:
        print("index not ready yet; check status later with get-index.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9 - Smoke-test the index with a self-embedded query - guarded
# MAGIC Embeds a sample question with the SAME model used in Phase 5 and runs a
# MAGIC similarity search via the SDK. Confirms the index returns grounded chunks.

# COMMAND ----------

if not CREATE_RESOURCES:
    print("create_resources=false -> skipping index query smoke test.")
else:
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

    # Must match the model used to build the embeddings in Phase 5.
    model = SentenceTransformer(QUERY_MODEL)
    question = "What is the laptop or iPhone warranty and repair policy?"
    qvec = model.encode([question], normalize_embeddings=True)[0].tolist()

    res = w.vector_search_indexes.query_index(
        index_name=INDEX_FQN,
        columns=["chunk_id", "section", "chunk_text"],
        query_vector=qvec,
        num_results=3,
    )
    data = res.result.data_array if res.result else []
    for row in data:
        print(row[:2], "| score:", row[-1])
