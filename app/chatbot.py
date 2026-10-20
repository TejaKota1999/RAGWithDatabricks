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
"""

from __future__ import annotations

import os
import sys

# Make `src` importable when run via `streamlit run app/chatbot.py`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from src.embedder import LocalEmbedder
from src.llm import DEFAULT_MODEL, LLMUnavailableError, OllamaClient
from src.pipeline import build_retriever
from src.rag_chain import RagChain

# COMMAND ----------
st.set_page_config(page_title="RAG POC", page_icon="chat")
st.title("Company Knowledge Assistant")
st.caption("Local, zero-cost RAG over your documents. Answers are grounded in retrieved text.")

# COMMAND ----------
with st.sidebar:
    st.header("Settings")
    docs_dir = st.text_input("Documents folder", value="data/sample_docs")
    model_name = st.text_input("Ollama model", value=DEFAULT_MODEL)
    top_k = st.slider("Top-k chunks", 1, 10, 5)
    min_score = st.slider("Min relevance score", 0.0, 1.0, 0.3, 0.05)
    rebuild = st.button("Rebuild index")


# COMMAND ----------
@st.cache_resource(show_spinner="Building local index ...")
def _load_retriever(folder: str):
    embedder = LocalEmbedder()
    retriever, chunks = build_retriever(folder, embedder=embedder)
    return embedder, retriever, len(chunks)


# COMMAND ----------
if rebuild:
    _load_retriever.clear()

try:
    embedder, retriever, n_chunks = _load_retriever(docs_dir)
    st.sidebar.success(f"Indexed {n_chunks} chunks from {docs_dir}")
except Exception as exc:
    st.error(f"Could not build the index from '{docs_dir}': {exc}")
    st.stop()

# COMMAND ----------
chain = RagChain(
    retriever=retriever,
    embed_fn=embedder.encode_one,
    llm=OllamaClient(model=model_name),
    top_k=top_k,
    min_score=min_score,
)

# COMMAND ----------
question = st.chat_input("Ask a question about your documents")
if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating ..."):
            try:
                result = chain.answer(question)
            except LLMUnavailableError as exc:
                st.error(str(exc))
                st.stop()

        st.write(result["answer"])
        if result.get("abstained"):
            st.info("No sufficiently relevant evidence was found, so the assistant abstained.")

        sources = result.get("sources", [])
        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for s in sources:
                    score = s.get("score")
                    score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
                    st.markdown(
                        f"- `{s.get('chunk_id', '?')}` "
                        f"(section: {s.get('section') or 'n/a'}, score: {score_str})  \n"
                        f"  {s.get('source_path', '')}"
                    )
