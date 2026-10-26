"""Wire up the retriever + embedder for the RAG app.

Default (Databricks) path: point at an existing Vector Search index and a
Databricks embedding endpoint -- no local load/chunk/embed needed, because the
documents were already processed by the notebooks.

Fallback (local) path: build a FAISS retriever from a folder of documents.

Plain English: these helpers return a ready-to-use (embedder, retriever) pair
for the chat app.
"""

from __future__ import annotations

# The building blocks from the other modules.
from src.offline.chunker import build_chunks
from src.offline.document_loader import load_documents
from src.offline.embedder import DatabricksEmbedder, LocalEmbedder
from src.serving.retriever import FaissRetriever, VectorSearchRetriever


# COMMAND ----------
# DEFAULT: use the Databricks embedding endpoint + an existing Vector Search index.
def build_databricks_retriever(
    index_name: str,
    embedding_endpoint: str | None = None,
    chat_profile: str | None = None,
    profile: str | None = None,
):
    """Return (embedder, retriever) backed by Databricks services.

    ``index_name`` is the full Vector Search index name (catalog.schema.index).
    ``profile`` is the CLI profile for local runs (None inside a Databricks App).
    No documents are loaded locally -- retrieval hits the managed index directly.
    """
    # The query embedder must be the SAME model that built the index.
    embedder = (
        DatabricksEmbedder(endpoint=embedding_endpoint, profile=profile)
        if embedding_endpoint
        else DatabricksEmbedder(profile=profile)
    )
    # The retriever queries the managed Vector Search index.
    retriever = VectorSearchRetriever(index_name=index_name, profile=profile)
    return embedder, retriever


# COMMAND ----------
# FALLBACK: build a local FAISS retriever from a folder of documents (offline).
def build_retriever(
    docs_dir: str,
    embedder: LocalEmbedder | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    min_chunk_chars: int = 60,
) -> tuple[FaissRetriever, list[dict]]:
    """Offline fallback: return a FaissRetriever over the chunks under ``docs_dir``.

    Also returns the chunk records so callers can show text. Raises ValueError if
    the folder yields no usable chunks. NOT the default path.
    """
    # Use the given local embedder, or create a default one.
    embedder = embedder or LocalEmbedder()
    # 1) Load + clean every document in the folder.
    documents = load_documents(docs_dir)
    if not documents:
        raise ValueError(f"No supported documents found in {docs_dir}")

    # 2) Split the documents into chunks.
    chunks = build_chunks(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_chars=min_chunk_chars,
    )
    if not chunks:
        raise ValueError(f"No chunks produced from documents in {docs_dir}")

    # 3) Embed each chunk's text into a vector.
    vectors = embedder.encode([c["chunk_text"] for c in chunks])
    # 4) Build the FAISS search index over those vectors + chunk metadata.
    retriever = FaissRetriever(vectors, chunks, normalize=embedder.normalize)
    return retriever, chunks
