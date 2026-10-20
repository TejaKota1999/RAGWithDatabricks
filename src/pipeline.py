# Databricks notebook source
"""Build a local FAISS retriever from a folder of documents (zero-cost path).

Ties together: load_documents -> build_chunks -> embed -> FaissRetriever. Runs
fully on the local machine; no Databricks or paid service required.
"""

from __future__ import annotations

from src.chunker import build_chunks
from src.document_loader import load_documents
from src.embedder import LocalEmbedder
from src.retriever import FaissRetriever


# COMMAND ----------
def build_retriever(
    docs_dir: str,
    embedder: LocalEmbedder | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    min_chunk_chars: int = 60,
) -> tuple[FaissRetriever, list[dict]]:
    """Return a FaissRetriever over the chunks found under ``docs_dir``.

    Also returns the chunk records so callers (e.g. the UI) can show text.
    Raises ValueError if the folder yields no usable chunks.
    """
    embedder = embedder or LocalEmbedder()
    documents = load_documents(docs_dir)
    if not documents:
        raise ValueError(f"No supported documents found in {docs_dir}")

    chunks = build_chunks(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_chars=min_chunk_chars,
    )
    if not chunks:
        raise ValueError(f"No chunks produced from documents in {docs_dir}")

    vectors = embedder.encode([c["chunk_text"] for c in chunks])
    retriever = FaissRetriever(vectors, chunks, normalize=embedder.normalize)
    return retriever, chunks
