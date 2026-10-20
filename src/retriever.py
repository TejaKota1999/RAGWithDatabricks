# Databricks notebook source
"""Local vector retrieval over a FAISS index (zero-cost baseline).

Pure/local: no Databricks or managed endpoint required. The retriever works on
numpy vectors and chunk metadata, so embedding is decoupled and the retrieval
logic is unit-testable without downloading a model.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


# COMMAND ----------
def _to_float32(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype="float32")
    if arr.ndim != 2:
        raise ValueError("vectors must be a 2D array (n, dim)")
    return arr


# COMMAND ----------
def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# COMMAND ----------
class FaissRetriever:
    """Cosine-similarity retriever over chunk embeddings.

    Parameters
    ----------
    vectors : (n, dim) array of chunk embeddings.
    records : list of n metadata dicts (chunk_id, chunk_text, source_path, ...),
              aligned by row with ``vectors``.
    normalize : L2-normalize vectors so inner product == cosine similarity.
    """

    def __init__(self, vectors, records: list[dict], normalize: bool = True):
        if faiss is None:  # pragma: no cover
            raise ImportError("faiss is not installed. pip install faiss-cpu")
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

    def __len__(self) -> int:
        return len(self.records)

    def search(self, query_vector, top_k: int = 5, min_score: float | None = None) -> list[dict]:
        """Return the top-k records most similar to the query.

        Each result is a copy of the record plus a ``score``. Results below
        ``min_score`` are dropped, so an irrelevant query can return ``[]``
        (the caller then abstains).
        """
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
