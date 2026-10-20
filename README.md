# RAG POC on Databricks Free Edition

A small, free-first Retrieval-Augmented Generation proof of concept. It answers
questions about a collection of company documents and returns answers grounded
in retrieved text, with source references — and abstains when the evidence is
insufficient.

The project runs two ways:

- **Databricks Free Edition** (data engineering): notebooks that ingest, clean,
  chunk, embed, and optionally index documents in Unity Catalog / Vector Search.
- **Local, zero-cost** (reusable Python): `src/` modules + a Streamlit app that
  build a local FAISS index and generate answers with a local Ollama model — no
  paid API, endpoint, or cloud vector database.

> Cost note: Databricks Free Edition is a no-cost tier with fair-use throttling.
> Local use still consumes your machine's CPU/RAM/disk and electricity.

## Architecture

```
          ┌─────────────────────── Databricks Free Edition ───────────────────────┐
docs ──▶  │ UC Volume ─▶ 02 clean ─▶ 03 chunk ─▶ 04 embed ─▶ (05 Vector Search)   │
          │ documents_cleaned   document_chunks   document_embeddings   index      │
          └───────────────────────────────────────────────────────────────────────┘

          ┌───────────────────────────── Local (zero-cost) ───────────────────────┐
docs ──▶  │ document_loader ─▶ chunker ─▶ embedder ─▶ FAISS retriever              │
question ─▶ embedder ─▶ retriever ─▶ prompt_templates ─▶ Ollama LLM ─▶ answer+cites│
          └───────────────────────────────────────────────────────────────────────┘
```

## Layout

| Path | Purpose |
|------|---------|
| `notebooks/02_extract_and_clean.py` | Extract + clean text from the UC volume (txt/md/pdf/docx) |
| `notebooks/03_chunk_documents.py` | Paragraph-aware chunking with stable ids + metadata |
| `notebooks/04_generate_embeddings.py` | Local sentence-transformers embeddings into Delta |
| `notebooks/05_create_ai_search_index.py` | **Optional** Databricks Vector Search (gated, billable-aware) |
| `src/document_loader.py` | Local document loading/extraction |
| `src/text_cleaning.py` | Whitespace-only cleaning (preserves meaning) |
| `src/chunker.py` | Chunking logic (shared by notebook + tests) |
| `src/embedder.py` | Local sentence-transformers adapter |
| `src/retriever.py` | Local FAISS cosine retriever |
| `src/prompt_templates.py` | Grounded prompt + abstain message |
| `src/llm.py` | Local Ollama client |
| `src/rag_chain.py` | retrieve → threshold → abstain/answer → sources |
| `src/pipeline.py` | Build a local retriever from a docs folder |
| `src/evaluation.py` | Recall@K, groundedness, citation checks |
| `app/chatbot.py` | Local Streamlit chat UI |
| `tests/` | pytest unit tests (no network/model needed) |
| `data/eval_questions.json` | Small evaluation set |
| `databricks.yml` | Asset Bundle (dev/prod targets) |

## Setup (local)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install Ollama (for generation) and pull a small model:

```powershell
# https://ollama.com/download
ollama pull llama3.2:3b
```

## Run the local app

1. Put documents in `data/sample_docs/` (`.txt`, `.md`, `.pdf`, `.docx`).
2. Start Ollama (the model above).
3. Launch the UI:

```powershell
streamlit run app/chatbot.py
```

Ask a question; the app retrieves grounded chunks, generates an answer, shows
sources, and abstains when nothing relevant is found.

## Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

All unit tests mock the LLM and avoid model downloads, so they run offline.

## Databricks path

Auth with the CLI, then deploy the notebooks as an Asset Bundle:

```powershell
databricks auth login --host <workspace-url> --profile <profile>
databricks bundle deploy -t dev
```

Open the deployed notebooks under
`/Workspace/Users/<you>/.bundle/rag-poc/dev/files/notebooks/` and run them
cell by cell on Serverless. Each value is a widget / job parameter.

The Vector Search notebook (`05`) is **optional** and **gated**: it only creates
the billable endpoint/index when `create_resources=true`. Remove it with:

```
databricks vector-search-indexes delete-index <catalog>.<schema>.<index> --profile <p>
databricks vector-search-endpoints delete-endpoint <endpoint> --profile <p>
```

## Limitations

- Not production-ready: no identity-aware authorization, monitoring, or secret
  management beyond the CLI profile. Review security before any multi-user use.
- Free Edition limits apply (serverless-only, one small SQL warehouse, one AI
  Search endpoint/unit, fair-use throttling).
- Generation quality depends on the local model and retrieval quality; prompt
  wording alone does not eliminate hallucinations.
- Use synthetic or approved documents only. Do not commit real documents,
  secrets, model files, or generated indexes.
