"""Retrieval adapters: Databricks Vector Search (default) + local FAISS fallback.

The default path queries a Databricks Vector Search index in the workspace. A
local FAISS retriever is kept only as an optional offline fallback and for unit
tests (it needs no managed service).

Plain English: given a question vector, this finds the chunk vectors closest to
it (most similar in meaning) and returns those chunks with their metadata.
"""

from __future__ import annotations

# "Sequence" is just a type hint meaning "a list/tuple-like of items".
from typing import Sequence

# numpy is used only by the local FAISS fallback below.
import numpy as np

# The default columns returned with each retrieved chunk (source traceability).
DEFAULT_COLUMNS = ("chunk_id", "document_id", "chunk_text", "section", "page_number", "source_path")


# COMMAND ----------
# DEFAULT adapter: query a Databricks Vector Search index.
class VectorSearchRetriever:
    """Cosine-similarity retriever backed by a Databricks Vector Search index.

    Keeps the same ``search(query_vector, top_k, min_score)`` interface as the
    local FAISS retriever so the RAG chain does not change.
    """

    # Remember which index to query, which columns to return, and how to auth.
    def __init__(
        self,
        index_name: str,
        columns: Sequence[str] = DEFAULT_COLUMNS,
        profile: str | None = None,
    ):
        self.index_name = index_name     # full name: catalog.schema.index
        self.columns = list(columns)     # metadata columns to return per hit
        self.profile = profile           # optional CLI profile (None in app/notebook)
        self._client = None              # the WorkspaceClient, created lazily

    # Create (once) the Databricks client used to query the index.
    def _ws(self):
        if self._client is None:
            from src.databricks_client import get_workspace_client

            self._client = get_workspace_client(self.profile)
        return self._client

    # The main method: find the chunks most similar to a question vector.
    def search(self, query_vector, top_k: int = 5, min_score: float | None = None) -> list[dict]:
        """Return the top-k records most similar to the query vector.

        Each result is a dict of the requested columns plus a ``score``. Results
        below ``min_score`` are dropped, so an irrelevant query can return ``[]``
        (the caller then abstains).
        """
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        # Ask Vector Search for the nearest rows to this query vector.
        resp = self._ws().vector_search_indexes.query_index(
            index_name=self.index_name,
            columns=self.columns,
            query_vector=[float(x) for x in query_vector],
            num_results=top_k,
        )

        # The response lists the column order, and rows of values (score appended last).
        col_names = [c.name for c in resp.manifest.columns] if resp.manifest else list(self.columns)
        rows = resp.result.data_array if (resp.result and resp.result.data_array) else []

        results: list[dict] = []
        for row in rows:
            # Vector Search appends the similarity score as the final value in each row.
            score = float(row[-1]) if len(row) > len(self.columns) else None
            if min_score is not None and score is not None and score < min_score:
                continue  # drop weak matches
            # Pair each returned column name with its value for this row.
            record = {name: row[i] for i, name in enumerate(col_names) if i < len(self.columns)}
            record["score"] = score
            results.append(record)
        return results  # may be empty -> caller abstains


# COMMAND ----------
# RECOMMENDED adapter: query a MANAGED-embeddings Vector Search index by TEXT.
# Used with the ai_parse_document + ai_prep_search pipeline, where Vector Search
# computes embeddings itself (via an embedding endpoint), so we pass the question
# TEXT (not a vector) and VS embeds it with the same model used for the chunks.
class VectorSearchTextRetriever:
    """Retriever for a managed-embeddings Vector Search index (query by text).

    The index embeds both documents and queries, so no local/question embedding
    is needed. Exposes ``search(query_text, top_k, min_score)`` and maps the
    ai_prep_search ``chunk_to_retrieve`` column to ``chunk_text`` so the rest of
    the RAG pipeline (prompt building, sources) is unchanged.
    """

    # The columns the ai_prep_search index exposes (plus what we return as sources).
    DEFAULT_COLUMNS = ("chunk_id", "chunk_to_retrieve", "source_path")

    def __init__(
        self,
        index_name: str,
        columns: Sequence[str] = DEFAULT_COLUMNS,
        profile: str | None = None,
    ):
        self.index_name = index_name     # full name: catalog.schema.index
        self.columns = list(columns)     # columns to return per hit
        self.profile = profile           # optional CLI profile (None in app/notebook)
        self._client = None              # the WorkspaceClient, created lazily

    def _ws(self):
        if self._client is None:
            from src.databricks_client import get_workspace_client

            self._client = get_workspace_client(self.profile)
        return self._client

    # Find the chunks most relevant to the question TEXT (VS embeds the text).
    def search(self, query_text: str, top_k: int = 5, min_score: float | None = None) -> list[dict]:
        """Return the top-k records most relevant to ``query_text``.

        Each result is a dict with ``chunk_id``, ``chunk_text`` (the clean
        retrieve text), ``source_path`` and a ``score``. Weak matches below
        ``min_score`` are dropped, so an irrelevant query returns ``[]``.
        """
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not query_text or not str(query_text).strip():
            raise ValueError("query_text must not be empty")

        # Query by TEXT; the managed index embeds it with the same endpoint as the chunks.
        resp = self._ws().vector_search_indexes.query_index(
            index_name=self.index_name,
            columns=self.columns,
            query_text=str(query_text),
            num_results=top_k,
        )

        col_names = [c.name for c in resp.manifest.columns] if resp.manifest else list(self.columns)
        rows = resp.result.data_array if (resp.result and resp.result.data_array) else []

        results: list[dict] = []
        for row in rows:
            # Vector Search appends the similarity score as the final value in each row.
            score = float(row[-1]) if len(row) > len(self.columns) else None
            if min_score is not None and score is not None and score < min_score:
                continue  # drop weak matches
            record = {name: row[i] for i, name in enumerate(col_names) if i < len(self.columns)}
            # Expose the retrieve text under the standard "chunk_text" key.
            if "chunk_to_retrieve" in record:
                record["chunk_text"] = record.pop("chunk_to_retrieve")
            record["score"] = score
            results.append(record)
        return results  # may be empty -> caller abstains


# COMMAND ----------
# Convert a list of vectors into a 2D float32 numpy array (FAISS needs float32).
def _to_float32(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype="float32")
    if arr.ndim != 2:
        raise ValueError("vectors must be a 2D array (n, dim)")
    return arr


# Scale each vector to length 1 so that "inner product" equals "cosine similarity".
def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# COMMAND ----------
# OPTIONAL FALLBACK: local FAISS retriever (offline only + unit tests).
class FaissRetriever:
    """Cosine-similarity retriever over chunk embeddings held in a local FAISS index.

    NOT the default. Used for offline demos and unit tests (no managed service).

    Parameters
    ----------
    vectors : (n, dim) array of chunk embeddings.
    records : list of n metadata dicts (chunk_id, chunk_text, source_path, ...),
              aligned by row with ``vectors``.
    normalize : L2-normalize vectors so inner product == cosine similarity.
    """

    # Runs once when you create the retriever; it builds the search index.
    def __init__(self, vectors, records: list[dict], normalize: bool = True):
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover
            raise ImportError("faiss is not installed. pip install faiss-cpu") from exc
        arr = _to_float32(vectors)
        if len(records) != arr.shape[0]:
            raise ValueError("records length must match number of vectors")
        self.records = list(records)
        self.normalize = normalize
        self.dim = arr.shape[1]
        if normalize:
            arr = _l2_normalize(arr)
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(arr)

    # Lets you call len(retriever) to get how many chunks it holds.
    def __len__(self) -> int:
        return len(self.records)

    # The main method: find the chunks most similar to a question vector.
    def search(self, query_vector, top_k: int = 5, min_score: float | None = None) -> list[dict]:
        """Return the top-k records most similar to the query (may be empty)."""
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        q = _to_float32([list(query_vector)])
        if q.shape[1] != self.dim:
            raise ValueError(f"query dim {q.shape[1]} != index dim {self.dim}")
        if self.normalize:
            q = _l2_normalize(q)
        k = min(top_k, len(self.records))
        scores, idxs = self._index.search(q, k)
        results: list[dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            if min_score is not None and float(score) < min_score:
                continue
            hit = dict(self.records[idx])
            hit["score"] = float(score)
            results.append(hit)
        return results
