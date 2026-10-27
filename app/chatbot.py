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
from src.serving.llm import DEFAULT_ENDPOINT as DEFAULT_CHAT_ENDPOINT
from src.serving.llm import DatabricksChatClient, LLMUnavailableError
from src.serving.rag_chain import RagChain
from src.serving.retriever import VectorSearchTextRetriever

# The "How it works" live flow diagram (app/flow_view.py, next to this file).
import flow_view

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
# Fixed configuration (the Settings sidebar was removed for a clean UI). These come
# from environment variables / sensible defaults instead of on-screen controls.
index_name = DEFAULT_INDEX          # managed-embeddings Vector Search index
chat_endpoint = CHAT_ENDPOINT       # Databricks chat serving endpoint
top_k = 5                           # how many chunks to retrieve per question
min_score = 0.3                     # minimum similarity to treat a chunk as relevant

# A small "Clear chat" button in the main area (no settings panel). It still drives
# the session-clear behavior: it wipes the in-memory transcript for this session.
_sp, _clearcol = st.columns([5, 1])
with _clearcol:
    clear = st.button("Clear chat")


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
    st.session_state.pop("last_trace", None)


# Matches inline chunk-id citations the model adds after claims, in either style:
#   old custom chunks:  [Doc_Name:0:0003]  (name + :page:index)
#   ai_prep_search:     [72c7...a43_1]     (hex hash + _index)
_CITE_RE = re.compile(r"\s*\[(?:[^\[\]\s]*:\d+:\d+|[0-9a-fA-F]{6,}_\d+)\]")


# Remove the raw [chunk_id] citations from the answer text for a clean display.
# The chunk ids are an internal detail, confusing to the reader.
def _clean_answer(text: str, chunk_ids=None) -> str:
    text = text or ""
    # 1) Remove the EXACT ids that were retrieved (most reliable, any id format).
    for cid in (chunk_ids or []):
        if cid:
            text = text.replace(f"[{cid}]", "")
    # 2) Remove any remaining citation-shaped brackets the model invented.
    text = _CITE_RE.sub("", text)
    text = re.sub(r"\s+([.,;:)])", r"\1", text)  # tidy space left before punctuation
    text = re.sub(r"[ \t]{2,}", " ", text)       # collapse double spaces
    return text.strip()


# COMMAND ----------
# Two tabs: the chat itself, and a live "How it works" flow diagram that traces
# the ingestion pipeline and the last question asked (see app/flow_view.py).
chat_tab, flow_tab = st.tabs(["Chat", "How it works"])

with chat_tab:
    # Re-draw the full transcript so far (every past question and answer stays visible).
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m["role"] == "assistant":
                if m.get("abstained"):
                    st.info("No sufficiently relevant evidence was found, so the assistant abstained.")

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

        # Conversational memory: pass the PRIOR turns (everything except the question we
        # just appended) so a follow-up is understood in context. The chain reformulates
        # the follow-up for retrieval and gives the conversation to the model. History
        # lives only in st.session_state, so it is cleared on "Clear chat" / session end.
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        result = None
        with st.chat_message("assistant"):         # start the assistant's reply bubble
            with st.spinner("Retrieving and generating ..."):   # loading indicator
                try:
                    result = chain.answer(question, history=history)   # the actual RAG call
                except LLMUnavailableError as exc:              # endpoint/quota problem?
                    st.error(str(exc))

            if result is not None:
                # Strip inline [chunk_id] citations (confusing to the reader). Pass the
                # retrieved ids so the exact ones are removed regardless of id format.
                _ids = [s.get("chunk_id") for s in result.get("sources", [])]
                answer_display = _clean_answer(result["answer"], _ids)
                st.write(answer_display)               # show the clean answer text
                if result.get("abstained"):            # if it abstained, say so clearly
                    st.info("No sufficiently relevant evidence was found, so the assistant abstained.")

        if result is not None:
            # Save the assistant's answer (with its sources) so it persists on the next rerun.
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_display,
                    "sources": result.get("sources", []),
                    "abstained": result.get("abstained", False),
                }
            )
            # Keep a trace of this question for the "How it works" tab.
            st.session_state.last_trace = {
                "question": question,
                "sources": result.get("sources", []),
                "abstained": result.get("abstained", False),
                "error": result.get("error"),
                "diagnostics": result.get("diagnostics", {}),
            }

with flow_tab:
    flow_view.render(PROFILE, index_name, chat_endpoint, top_k, min_score)
