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
# MAGIC ## Cell 6 - Install the Vector Search client
# MAGIC Only needed when actually creating/querying. In-process pip, no restart.

# COMMAND ----------

if CREATE_RESOURCES:
    import importlib
    import subprocess
    import sys

    try:
        importlib.import_module("databricks.vector_search")
        print("ready: databricks-vectorsearch")
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "databricks-vectorsearch==0.56"])
        importlib.invalidate_caches()
        print("installed databricks-vectorsearch")
else:
    print("create_resources=false -> skipping client install")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 - Create the endpoint (BILLABLE) - guarded
# MAGIC Creates the Vector Search endpoint and waits until it is online. This is
# MAGIC the step that provisions managed compute.

# COMMAND ----------

if not CREATE_RESOURCES:
    print("create_resources=false -> NOT creating an endpoint. Set it to 'true' to proceed.")
else:
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient(disable_notice=True)
    existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
    if ENDPOINT_NAME in existing:
        print(f"endpoint '{ENDPOINT_NAME}' already exists")
    else:
        print(f"creating endpoint '{ENDPOINT_NAME}' ({ENDPOINT_TYPE}) - waits until online ...")
        vsc.create_endpoint_and_wait(name=ENDPOINT_NAME, endpoint_type=ENDPOINT_TYPE)
        print("endpoint online")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8 - Create the Delta Sync index (self-managed embeddings) - guarded

# COMMAND ----------

if not CREATE_RESOURCES:
    print("create_resources=false -> NOT creating an index.")
else:
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient(disable_notice=True)
    existing_idx = [i["name"] for i in vsc.list_indexes(ENDPOINT_NAME).get("vector_indexes", [])]
    if INDEX_FQN in existing_idx:
        print(f"index '{INDEX_FQN}' already exists")
    else:
        print(f"creating Delta Sync index '{INDEX_FQN}' - waits until ready ...")
        vsc.create_delta_sync_index_and_wait(
            endpoint_name=ENDPOINT_NAME,
            index_name=INDEX_FQN,
            source_table_name=SOURCE_FQN,
            primary_key=PRIMARY_KEY,
            pipeline_type=PIPELINE_TYPE,
            embedding_vector_column=EMBEDDING_COLUMN,
            embedding_dimension=EMBEDDING_DIM,
        )
        print("index ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9 - Smoke-test the index with a self-embedded query - guarded
# MAGIC Embeds a sample question with the SAME model used in Phase 5 and runs a
# MAGIC similarity search. Confirms the index returns grounded chunks.

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

    from databricks.vector_search.client import VectorSearchClient
    from sentence_transformers import SentenceTransformer

    # Must match the model used to build the embeddings in Phase 5.
    model = SentenceTransformer(QUERY_MODEL)
    question = "What is the laptop or iPhone warranty and repair policy?"
    qvec = model.encode([question], normalize_embeddings=True)[0].tolist()

    vsc = VectorSearchClient(disable_notice=True)
    index = vsc.get_index(endpoint_name=ENDPOINT_NAME, index_name=INDEX_FQN)
    res = index.similarity_search(
        query_vector=qvec,
        columns=["chunk_id", "section", "chunk_text"],
        num_results=3,
    )
    for row in res.get("result", {}).get("data_array", []):
        print(row[:2], "| score:", row[-1])
