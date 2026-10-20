# Databricks notebook source
"""Local Streamlit chat UI for the RAG POC (zero-cost baseline).

Builds a local FAISS retriever from a documents folder and answers questions
with a local Ollama model, showing source references and abstaining when the
evidence is weak.

Run:
    streamlit run app/chatbot.py

Prerequisites (local, zero-cost):
    - Documents in data/sample_docs/ (.txt, .md, .pdf, .docx)
    - Ollama running locally with a model pulled, e.g. `ollama pull llama3.2:3b`

Note: this is a Streamlit web app, so run it with `streamlit run` (not cell by
cell). The `# COMMAND ----------` markers just make the sections easy to read.
"""

from __future__ import annotations

# os/sys are used to make the `src` package importable when launched by Streamlit.
import os
import sys

# Add the project root to Python's import path so `from src...` works.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Streamlit builds the web UI (text boxes, chat, buttons) from simple Python calls.
import streamlit as st

# Our own building blocks.
from src.embedder import LocalEmbedder
from src.llm import DEFAULT_MODEL, LLMUnavailableError, OllamaClient
from src.pipeline import build_retriever
from src.rag_chain import RagChain

# COMMAND ----------
# Page setup: browser tab title/icon, the on-page title, and a caption.
st.set_page_config(page_title="RAG POC", page_icon="chat")
st.title("Company Knowledge Assistant")
st.caption("Local, zero-cost RAG over your documents. Answers are grounded in retrieved text.")

# COMMAND ----------
# The left sidebar holds the settings the user can change.
with st.sidebar:
    st.header("Settings")
    # Which folder of documents to index.
    docs_dir = st.text_input("Documents folder", value="data/sample_docs")
    # Which local Ollama model to use for generating answers.
    model_name = st.text_input("Ollama model", value=DEFAULT_MODEL)
    # How many chunks to retrieve per question (slider from 1 to 10, default 5).
    top_k = st.slider("Top-k chunks", 1, 10, 5)
    # Minimum similarity to treat a chunk as relevant (0.0 to 1.0, default 0.3).
    min_score = st.slider("Min relevance score", 0.0, 1.0, 0.3, 0.05)
    # A button to force rebuilding the index (e.g. after changing documents).
    rebuild = st.button("Rebuild index")


# COMMAND ----------
# Build the retriever once and cache it, so it is not rebuilt on every interaction.
@st.cache_resource(show_spinner="Building local index ...")
def _load_retriever(folder: str):
    embedder = LocalEmbedder()                                 # the embedding model
    retriever, chunks = build_retriever(folder, embedder=embedder)  # load+chunk+embed+index
    return embedder, retriever, len(chunks)                    # return all three


# COMMAND ----------
# If the user clicked "Rebuild index", clear the cache so it rebuilds.
if rebuild:
    _load_retriever.clear()

# Try to build/load the index; show a clear error and stop if it fails.
try:
    embedder, retriever, n_chunks = _load_retriever(docs_dir)
    st.sidebar.success(f"Indexed {n_chunks} chunks from {docs_dir}")
except Exception as exc:
    st.error(f"Could not build the index from '{docs_dir}': {exc}")
    st.stop()

# COMMAND ----------
# Create the RAG chain, wiring together the retriever, the embedder, and the LLM.
chain = RagChain(
    retriever=retriever,
    embed_fn=embedder.encode_one,          # how to turn the question into a vector
    llm=OllamaClient(model=model_name),    # the local model that writes the answer
    top_k=top_k,
    min_score=min_score,
)

# COMMAND ----------
# The chat input box at the bottom of the page.
question = st.chat_input("Ask a question about your documents")
if question:                                   # runs when the user submits a question
    st.chat_message("user").write(question)    # show the user's message
    with st.chat_message("assistant"):         # start the assistant's reply bubble
        with st.spinner("Retrieving and generating ..."):   # loading indicator
            try:
                result = chain.answer(question)             # the actual RAG call
            except LLMUnavailableError as exc:              # Ollama not running?
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
