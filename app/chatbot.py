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
import re
import sys

# Add the project root to Python's import path so `from src...` works.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Streamlit builds the web UI from simple Python calls.
import streamlit as st

# Our building blocks (all Databricks-backed by default).
from src.llm import DEFAULT_ENDPOINT as DEFAULT_CHAT_ENDPOINT
from src.llm import DatabricksChatClient, LLMUnavailableError
from src.rag_chain import RagChain
from src.retriever import VectorSearchTextRetriever

# COMMAND ----------
# Read configuration from environment variables, with defaults for this project.
# Default is the managed-embeddings index built by ai_parse_document + ai_prep_search;
# it embeds the query itself, so the app queries by TEXT (no embedding endpoint here).
DEFAULT_INDEX = os.environ.get("RAG_INDEX_NAME", "databricks_rag_dev.rag.document_chunks_ai_index")
CHAT_ENDPOINT = os.environ.get("RAG_CHAT_ENDPOINT", DEFAULT_CHAT_ENDPOINT)
# Inside a Databricks App this is unset (the App's identity is used automatically);
# for LOCAL runs, set DATABRICKS_CONFIG_PROFILE (e.g. "sairam").
PROFILE = os.environ.get("DATABRICKS_CONFIG_PROFILE")

# COMMAND ----------
# Page setup: browser tab title/icon, the on-page title, and a caption.
st.set_page_config(page_title="RAG POC", page_icon="chat")


# Work out whose name to show in the corner badge:
#   1) RAG_USER_NAME env var if set (lets you pin a specific name), else
#   2) the logged-in user forwarded by the Databricks App, else
#   3) "Guest" (e.g. a plain local run).
def _display_name():
    name = os.environ.get("RAG_USER_NAME")
    if name:
        return name
    try:
        h = st.context.headers or {}
        for key in ("X-Forwarded-Preferred-Username", "X-Forwarded-Email", "X-Forwarded-User"):
            val = h.get(key)
            if val:
                return val.split("@")[0]       # show the name part, not the full email
    except Exception:
        pass
    return "Guest"


# Show a styled name badge at the top-right of the content (reliable: in-flow,
# right-aligned, so it is not hidden behind Streamlit's header bar).
_spacer, _badge = st.columns([3, 1])
with _badge:
    st.markdown(
        f"""
        <div style="text-align:right; margin-top:-0.5rem;">
          <span style="display:inline-block;
                       background:linear-gradient(135deg,#6366f1,#8b5cf6);
                       color:#ffffff; padding:6px 16px; border-radius:999px;
                       font-weight:600; font-size:0.9rem; white-space:nowrap;
                       box-shadow:0 2px 8px rgba(0,0,0,0.25);">
            &#128100; {_display_name()}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.title("Company Knowledge Assistant")
st.caption("RAG on Databricks Free Edition. Answers are grounded in retrieved documents.")

# COMMAND ----------
# The left sidebar holds the settings the user can change.
with st.sidebar:
    st.header("Settings")
    # Which Vector Search index to query (managed-embeddings index).
    index_name = st.text_input("Vector Search index", value=DEFAULT_INDEX)
    # Which chat endpoint to use (the index embeds queries itself, so no embed endpoint).
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
def _build_chain(index: str, chat_ep: str, k: int, score: float):
    # Managed-embeddings index: Vector Search embeds the query itself, so there is
    # no separate embedder - we query by text.
    retriever = VectorSearchTextRetriever(index_name=index, profile=PROFILE)
    # The generator (Databricks chat endpoint).
    llm = DatabricksChatClient(endpoint=chat_ep, profile=PROFILE)
    # embed_fn=None -> RagChain passes the question TEXT to the retriever.
    return RagChain(retriever=retriever, embed_fn=None, llm=llm, top_k=k, min_score=score)


# COMMAND ----------
# Build/load the chain; show a clear error and stop if it fails.
try:
    chain = _build_chain(index_name, chat_endpoint, top_k, min_score)
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


# Matches inline citations like [Doc_Name:0:0003] that the model adds after claims.
_CITE_RE = re.compile(r"\s*\[[^\[\]]*:\d+:\d+[^\[\]]*\]")


# Remove the raw [chunk_id] citations from the answer text for a clean display.
# The Sources panel below already shows exactly which chunks were used.
def _clean_answer(text: str) -> str:
    text = _CITE_RE.sub("", text or "")      # drop the bracketed source ids
    text = re.sub(r"\s+([.,;:)])", r"\1", text)  # tidy space left before punctuation
    text = re.sub(r"[ \t]{2,}", " ", text)   # collapse double spaces
    return text.strip()


# COMMAND ----------
# Re-draw the full transcript so far (every past question and answer stays visible).
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m["role"] == "assistant":
            if m.get("abstained"):
                st.info("No sufficiently relevant evidence was found, so the assistant abstained.")

# COMMAND ----------
# A multi-line, paste-friendly input box with an "Ask" button. Using a form means
# pressing Enter inside the box adds a new line (so long/multi-line pasted text is
# fine), and the button submits. The box clears after each question.
with st.form("ask_form", clear_on_submit=True):
    question = st.text_area(
        "Ask a question about your documents",
        placeholder="Type or paste your question here ... (Shift+Enter for a new line)",
        height=120,
    )
    submitted = st.form_submit_button("Ask")

if submitted and question and question.strip():   # runs when the user clicks Ask
    question = question.strip()
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

        # Strip inline [chunk_id] citations; the Sources panel holds traceability.
        answer_display = _clean_answer(result["answer"])
        st.write(answer_display)               # show the clean answer text
        if result.get("abstained"):            # if it abstained, say so clearly
            st.info("No sufficiently relevant evidence was found, so the assistant abstained.")

    # Save the assistant's answer (with its sources) so it persists on the next rerun.
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_display,
            "sources": result.get("sources", []),
            "abstained": result.get("abstained", False),
        }
    )
