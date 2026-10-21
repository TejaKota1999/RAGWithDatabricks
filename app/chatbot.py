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
    # Button to wipe the conversation transcript and start fresh.
    clear = st.button("Clear chat")
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
# Keep the whole conversation in st.session_state so it survives each rerun.
# (Streamlit re-runs the script top-to-bottom on every interaction; without this
# the previous questions and answers would disappear.)
if "messages" not in st.session_state:
    st.session_state.messages = []            # each item: {role, content, sources, abstained}
# The "Clear chat" button empties the transcript.
if clear:
    st.session_state.messages = []


# Helper: render the sources of one assistant turn inside a collapsible section.
def _render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            score = s.get("score")
            # Format the score to 3 decimals if it is a number, else "n/a".
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
            # Show just the file name, not the full volume path (which is long).
            path = s.get("source_path") or ""
            file_name = path.rsplit("/", 1)[-1] if path else ""
            section = s.get("section") or "n/a"
            chunk_id = s.get("chunk_id", "?")
            # Wrap in a div with overflow-wrap:anywhere so long ids/paths break
            # instead of spilling horizontally outside the box.
            st.markdown(
                "<div style='overflow-wrap:anywhere;word-break:break-word;"
                "margin-bottom:10px;line-height:1.4'>"
                f"<code style='white-space:normal'>{chunk_id}</code><br>"
                f"<span style='color:#888;font-size:0.85em'>"
                f"section: {section} &middot; score: {score_str} &middot; {file_name}"
                "</span></div>",
                unsafe_allow_html=True,
            )


# COMMAND ----------
# Re-draw the full transcript so far (every past question and answer stays visible).
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m["role"] == "assistant":
            if m.get("abstained"):
                st.info("No sufficiently relevant evidence was found, so the assistant abstained.")
            _render_sources(m.get("sources", []))

# COMMAND ----------
# The chat input box at the bottom of the page.
question = st.chat_input("Ask a question about your documents")
if question:                                   # runs when the user submits a question
    # Save + show the user's message.
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

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
        _render_sources(result.get("sources", []))

    # Save the assistant's answer (with its sources) so it persists on the next rerun.
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "abstained": result.get("abstained", False),
        }
    )
