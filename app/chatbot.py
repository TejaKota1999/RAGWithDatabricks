"""Streamlit chat UI for the Databricks RAG POC.

Runs as a Databricks App (default) or locally. Either way, ALL models and
retrieval run on Databricks Free Edition:
  - embeddings  : databricks-qwen3-embedding-0-6b  (serving endpoint)
  - retrieval   : Databricks Vector Search index
  - generation  : databricks-meta-llama-3-1-8b-instruct  (serving endpoint)

No local model, FAISS, or Ollama is used.

Run as a Databricks App:
    databricks apps deploy ... (see README); auth is the app's service principal.

Run locally (reads your CLI profile):
    export DATABRICKS_CONFIG_PROFILE=sairam
    streamlit run app/chatbot.py

Configuration comes from environment variables (with sensible defaults):
    RAG_INDEX_NAME        full Vector Search index name
    RAG_EMBED_ENDPOINT    embedding serving endpoint
    RAG_CHAT_ENDPOINT     chat serving endpoint
    DATABRICKS_CONFIG_PROFILE  CLI profile for LOCAL runs (unset inside an App)
"""

from __future__ import annotations

# os/sys: read config from the environment and make `src` importable.
import os
import sys

# Add the project root to Python's import path so `from src...` works.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Streamlit builds the web UI from simple Python calls.
import streamlit as st

# Our building blocks (all Databricks-backed by default).
from src.embedder import DEFAULT_ENDPOINT as DEFAULT_EMBED_ENDPOINT
from src.embedder import DatabricksEmbedder
from src.llm import DEFAULT_ENDPOINT as DEFAULT_CHAT_ENDPOINT
from src.llm import DatabricksChatClient, LLMUnavailableError
from src.rag_chain import RagChain
from src.retriever import VectorSearchRetriever

# COMMAND ----------
# Read configuration from environment variables, with defaults for this project.
DEFAULT_INDEX = os.environ.get("RAG_INDEX_NAME", "databricks_rag_dev.rag.document_chunks_index")
EMBED_ENDPOINT = os.environ.get("RAG_EMBED_ENDPOINT", DEFAULT_EMBED_ENDPOINT)
CHAT_ENDPOINT = os.environ.get("RAG_CHAT_ENDPOINT", DEFAULT_CHAT_ENDPOINT)
# Inside a Databricks App this is unset (the App's identity is used automatically);
# for LOCAL runs, set DATABRICKS_CONFIG_PROFILE (e.g. "sairam").
PROFILE = os.environ.get("DATABRICKS_CONFIG_PROFILE")

# COMMAND ----------
# Page setup: browser tab title/icon, the on-page title, and a caption.
st.set_page_config(page_title="RAG POC", page_icon="chat")
st.title("Company Knowledge Assistant")
st.caption("RAG on Databricks Free Edition. Answers are grounded in retrieved documents.")

# COMMAND ----------
# The left sidebar holds the settings the user can change.
with st.sidebar:
    st.header("Settings")
    # Which Vector Search index to query.
    index_name = st.text_input("Vector Search index", value=DEFAULT_INDEX)
    # Which serving endpoints to use.
    embed_endpoint = st.text_input("Embedding endpoint", value=EMBED_ENDPOINT)
    chat_endpoint = st.text_input("Chat endpoint", value=CHAT_ENDPOINT)
    # How many chunks to retrieve per question (slider from 1 to 10, default 5).
    top_k = st.slider("Top-k chunks", 1, 10, 5)
    # Minimum similarity to treat a chunk as relevant (0.0 to 1.0, default 0.3).
    min_score = st.slider("Min relevance score", 0.0, 1.0, 0.3, 0.05)
    if PROFILE:
        st.caption(f"Local run using profile: `{PROFILE}`")


# COMMAND ----------
# Build the Databricks-backed chain once and cache it (re-created if settings change).
@st.cache_resource(show_spinner="Connecting to Databricks ...")
def _build_chain(index: str, embed_ep: str, chat_ep: str, k: int, score: float):
    # The query embedder (Databricks endpoint) - same model that built the index.
    embedder = DatabricksEmbedder(endpoint=embed_ep, profile=PROFILE)
    # The retriever queries the managed Vector Search index.
    retriever = VectorSearchRetriever(index_name=index, profile=PROFILE)
    # The generator (Databricks chat endpoint).
    llm = DatabricksChatClient(endpoint=chat_ep, profile=PROFILE)
    # Wire them together into the RAG chain (orchestration is backend-agnostic).
    return RagChain(retriever=retriever, embed_fn=embedder.encode_one, llm=llm, top_k=k, min_score=score)


# COMMAND ----------
# Build/load the chain; show a clear error and stop if it fails.
try:
    chain = _build_chain(index_name, embed_endpoint, chat_endpoint, top_k, min_score)
except Exception as exc:
    st.error(f"Could not connect to Databricks: {exc}")
    st.stop()

# COMMAND ----------
# The chat input box at the bottom of the page.
question = st.chat_input("Ask a question about your documents")
if question:                                   # runs when the user submits a question
    st.chat_message("user").write(question)    # show the user's message
    with st.chat_message("assistant"):         # start the assistant's reply bubble
        with st.spinner("Retrieving and generating ..."):   # loading indicator
            try:
                result = chain.answer(question)             # the actual RAG call
            except LLMUnavailableError as exc:              # endpoint/quota problem?
                st.error(str(exc))
                st.stop()

        st.write(result["answer"])             # show the answer text
        if result.get("abstained"):            # if it abstained, say so clearly
            st.info("No sufficiently relevant evidence was found, so the assistant abstained.")

        # Show the sources the answer was based on, inside a collapsible section.
        sources = result.get("sources", [])
        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for s in sources:
                    score = s.get("score")
                    # Format the score to 3 decimals if it is a number, else "n/a".
                    score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
                    st.markdown(
                        f"- `{s.get('chunk_id', '?')}` "
                        f"(section: {s.get('section') or 'n/a'}, score: {score_str})  \n"
                        f"  {s.get('source_path', '')}"
                    )
