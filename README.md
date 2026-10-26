# RAG POC on Databricks Free Edition

A small Retrieval-Augmented Generation proof of concept that runs **entirely on
Databricks Free Edition — including the models**. It answers questions about a
collection of company documents, returns answers grounded in retrieved text with
source references, and abstains when the evidence is insufficient.

Every stage runs on Databricks:

| Stage | Databricks Free Edition component |
|-------|-----------------------------------|
| Ingest / clean / chunk | Serverless notebooks + Delta tables (Unity Catalog) |
| Embeddings | `databricks-qwen3-embedding-0-6b` serving endpoint (1024-dim) |
| Retrieval | Databricks **Vector Search** Delta Sync index |
| Generation | `databricks-meta-llama-3-1-8b-instruct` serving endpoint |
| Chat UI | Databricks **App** (Streamlit) |

There is **no dependency on local models** (no Ollama, no local
sentence-transformers, no local FAISS) in the default path. A fully local path is
retained only as an **optional offline fallback** (see the end of this file).

### Two ingestion pipelines

| Pipeline | How it works | Index (app default) |
|---|---|---|
| **AI Functions (recommended, default)** | `ai_parse_document` + `ai_prep_search` (notebook 10 / job `rag_ingest_ai`) produce context-enriched chunks; **Vector Search managed embeddings** (via `databricks-gte-large-en`) embeds them and the query | `document_chunks_ai_index` — queried by **text** |
| **Custom (alternative)** | notebooks 02→05: Python parse/clean/chunk, embed via `databricks-qwen3-embedding-0-6b`, self-managed 1024-dim index | `document_chunks_index` — queried by vector |

The chat app (`app/chatbot.py`) uses the **AI-Functions index** by default
(`VectorSearchTextRetriever`, query by text). Generation uses the chat endpoint
via the SDK (`ai_query` is the SQL-side equivalent; all three AI Functions are
verified working on this Free Edition workspace).

> Cost note: Foundation Model endpoints and Vector Search are **included** in Free
> Edition but **metered against the serverless quota** — not unlimited. This POC
> keeps usage tiny. Exceeding the quota can throttle compute for the day/month.

## Architecture

```
  Ingestion (serverless notebooks, run cell by cell)
  UC Volume ─▶ 02 clean ─▶ 03 chunk ─▶ 04 embed (Databricks endpoint, 1024-dim)
              documents_cleaned   document_chunks   document_embeddings
                                                          │
                                                 05 Vector Search index (Delta Sync)

  Question answering (Databricks App or local client)
  question ─▶ embed (Databricks endpoint) ─▶ Vector Search ─▶ prompt
           ─▶ chat endpoint (Llama 3.1 8B) ─▶ answer + citations (or abstain)
```

## Selected components (verified live, profile `sairam`)

- Catalog / schema: `databricks_rag_dev.rag`
- Tables: `documents_cleaned`, `document_chunks`, `document_embeddings`
- Vector Search endpoint: `rag_poc_vs`; index: `databricks_rag_dev.rag.document_chunks_index` (1024-dim)
- Embedding endpoint: `databricks-qwen3-embedding-0-6b`
- Chat endpoint: `databricks-meta-llama-3-1-8b-instruct`
- App: `rag-chatbot` (Databricks Apps)

## Layout

| Path | Purpose |
|------|---------|
| `notebooks/ai_functions/10_ai_parse_prep_index.py` | **Recommended** pipeline: `ai_parse_document` + `ai_prep_search` + managed Vector Search index (incremental) |
| `notebooks/custom/02_extract_and_clean.py` | (custom) Extract + clean text from the UC volume (txt/md/pdf/docx) |
| `notebooks/custom/03_chunk_documents.py` | (custom) Paragraph-aware chunking with stable ids + metadata |
| `notebooks/custom/04_generate_embeddings.py` | (custom) Embeddings via the **Databricks endpoint** (1024-dim) into Delta + CDF |
| `notebooks/custom/05_create_ai_search_index.py` | (custom) Create/rebuild the **Vector Search** index (gated) |
| `notebooks/run_tests.py` | Run the pytest suite on serverless; logs per-test rows to `test_log` |
| `src/databricks_client.py` | Shared Databricks `WorkspaceClient` (auth helper) |
| `src/embedder.py` | **DatabricksEmbedder** (default) + LocalEmbedder (fallback) |
| `src/retriever.py` | **VectorSearchRetriever** (default) + FaissRetriever (fallback) |
| `src/llm.py` | **DatabricksChatClient** (default) + OllamaClient (fallback) |
| `src/prompt_templates.py` | Grounded prompt + abstain message |
| `src/rag_chain.py` | retrieve → threshold → abstain/answer → sources (backend-agnostic) |
| `src/pipeline.py` | `build_databricks_retriever` (default) + local `build_retriever` |
| `src/chunker.py`, `src/document_loader.py`, `src/text_cleaning.py` | Shared / fallback ingestion logic |
| `src/evaluation.py` | Recall@K, groundedness, citation checks |
| `app/chatbot.py` | Streamlit UI (Databricks App + local) |
| `app.yaml` | Databricks Apps runtime config (command + env) |
| `tests/` | pytest unit tests (Databricks clients mocked; no network) |
| `databricks.yml` | Asset Bundle (dev/prod targets); includes `resources/*.yml` |
| `resources/rag_ingest.job.yml` | Ingestion pipeline job definition (edit here) |
| `resources/rag_tests.job.yml` | Test pipeline job definition (edit here) |

## Prerequisites

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

databricks auth login --host <workspace-url> --profile sairam
```

## 1) Build the data + index (notebooks)

Deploy the notebooks and run them on serverless (cell by cell, or as jobs):

```powershell
databricks bundle deploy -t dev --profile sairam
```

Run order (each reads the previous table):

1. `02_extract_and_clean` → `documents_cleaned`
2. `03_chunk_documents` → `document_chunks`
3. `04_generate_embeddings` → `document_embeddings` (1024-dim, CDF on)
4. `05_create_ai_search_index` → Vector Search index
   - Set `create_resources=true` to build it (or to sync an existing one).
   - Set `recreate_index=true` to rebuild after an embedding-dimension change.
   - If the index already exists, re-running 05 **triggers a sync** to pick up new rows.

### Incremental processing (cost control)

Notebooks 02–04 are **incremental by default** (`incremental=true`): they process
**only new or changed documents** and skip anything already done, keyed on each
file's `content_hash`. So uploading one new file to the volume and re-running the
pipeline embeds **only that file** — existing documents are never re-embedded
(the embedding endpoint is the metered cost). A changed file (new content, new
hash) is re-processed for that document only. Set `incremental=false` to force a
full rebuild. After an incremental run, re-run 05 to sync the index.

Note: incremental handles **added and changed** files. Deleting a file from the
volume does **not** remove its rows automatically — delete those rows explicitly
if needed.

Remove the managed index/endpoint when done:

```
databricks vector-search-indexes delete-index databricks_rag_dev.rag.document_chunks_index --profile sairam
databricks vector-search-endpoints delete-endpoint rag_poc_vs --profile sairam
```

## 2) Run the chatbot

### As a Databricks App (default)

```powershell
databricks bundle deploy -t dev --profile sairam
databricks apps create rag-chatbot --profile sairam
databricks apps deploy rag-chatbot --source-code-path "/Workspace/Users/<you>/.bundle/rag-poc/dev/files" --profile sairam
```

Grant the app's service principal access (one time): `USE CATALOG` + `USE SCHEMA`
on `databricks_rag_dev.rag`, `SELECT` on the index and embeddings table, and query
access to the two serving endpoints. Then open the app URL from
`databricks apps get rag-chatbot` (workspace SSO).

### Locally (thin client, still uses Databricks for everything)

```powershell
$env:DATABRICKS_CONFIG_PROFILE = "sairam"
streamlit run app/chatbot.py
```

The local app embeds, retrieves, and generates entirely via Databricks endpoints
and Vector Search — only the browser UI runs on your machine.

## Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Unit tests mock the Databricks SDK clients, so they run offline and consume no
quota.

## Limitations

- Not production-ready: Databricks Apps provides workspace SSO, but there is no
  per-document authorization, monitoring, or secret management beyond CLI/SP auth.
- Free Edition is serverless-only, non-commercial, no SLA. Foundation Model
  endpoints and Vector Search are metered against the serverless quota.
- Generation quality depends on the model and retrieval; prompt wording alone does
  not eliminate hallucinations.
- Use synthetic or approved documents only. Do not commit real documents, secrets,
  model files, or generated indexes.

## Optional offline fallback (not the default)

A fully local path is kept for offline demos: `LocalEmbedder` (sentence-transformers),
`FaissRetriever` (FAISS), and `OllamaClient` (Ollama). Enable it by uncommenting the
fallback dependencies in `requirements.txt` and using `src.pipeline.build_retriever`
with a local docs folder. This path does not touch Databricks and is clearly
separated from the default Databricks path.
