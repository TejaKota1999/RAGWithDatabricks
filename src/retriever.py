# Databricks notebook source
"""Local vector retrieval over a FAISS index (zero-cost baseline).

Pure/local: no Databricks or managed endpoint required. The retriever works on
numpy vectors and chunk metadata, so embedding is decoupled and the retrieval
logic is unit-testable without downloading a model.

Plain English: given a question vector, this finds the chunk vectors closest to
it (most similar in meaning) and returns those chunks with their metadata.
"""

from __future__ import annotations

# "Sequence" is just a type hint meaning "a list/tuple-like of items".
from typing import Sequence

# numpy handles the number arrays (the vectors) efficiently.
import numpy as np

# FAISS is the local similarity-search library. If it is missing we set it to
# None here and raise a clear error later only if someone tries to use it.
try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


# COMMAND ----------
# Convert a list of vectors into a 2D float32 numpy array (FAISS needs float32).
def _to_float32(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype="float32")   # make a float32 array
    if arr.ndim != 2:                             # must be a grid: rows x columns
        raise ValueError("vectors must be a 2D array (n, dim)")
    return arr


# COMMAND ----------
# Scale each vector to length 1 so that "inner product" equals "cosine similarity".
def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)  # length of each row vector
    norms[norms == 0] = 1.0                             # avoid dividing by zero
    return arr / norms                                  # divide each vector by its length


# COMMAND ----------
# The retriever: builds a FAISS index from chunk vectors and searches it.
class FaissRetriever:
    """Cosine-similarity retriever over chunk embeddings.

    Parameters
    ----------
    vectors : (n, dim) array of chunk embeddings.
    records : list of n metadata dicts (chunk_id, chunk_text, source_path, ...),
              aligned by row with ``vectors``.
    normalize : L2-normalize vectors so inner product == cosine similarity.
    """

    # Runs once when you create the retriever; it builds the search index.
    def __init__(self, vectors, records: list[dict], normalize: bool = True):
        if faiss is None:  # pragma: no cover
            raise ImportError("faiss is not installed. pip install faiss-cpu")
        arr = _to_float32(vectors)                        # tidy the vectors
        if len(records) != arr.shape[0]:                  # one record per vector
            raise ValueError("records length must match number of vectors")
        self.records = list(records)                      # keep the chunk metadata
        self.normalize = normalize                        # remember the setting
        self.dim = arr.shape[1]                           # the vector length (e.g. 384)
        if normalize:
            arr = _l2_normalize(arr)                       # unit-normalize for cosine
        self._index = faiss.IndexFlatIP(self.dim)          # an inner-product index
        self._index.add(arr)                               # load all chunk vectors into it

    # Lets you call len(retriever) to get how many chunks it holds.
    def __len__(self) -> int:
        return len(self.records)

    # The main method: find the chunks most similar to a question vector.
    def search(self, query_vector, top_k: int = 5, min_score: float | None = None) -> list[dict]:
        """Return the top-k records most similar to the query.

        Each result is a copy of the record plus a ``score``. Results below
        ``min_score`` are dropped, so an irrelevant query can return ``[]``
        (the caller then abstains).
        """
        if top_k <= 0:                                    # must ask for at least 1 result
            raise ValueError("top_k must be > 0")
        q = _to_float32([list(query_vector)])             # shape the query as a 1-row array
        if q.shape[1] != self.dim:                        # query length must match the index
            raise ValueError(f"query dim {q.shape[1]} != index dim {self.dim}")
        if self.normalize:
            q = _l2_normalize(q)                           # normalize the query too
        k = min(top_k, len(self.records))                 # don't ask for more than we have
        scores, idxs = self._index.search(q, k)           # FAISS returns scores and row indexes
        results: list[dict] = []
        # Pair each returned score with the row index it refers to.
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:                                    # FAISS uses -1 for "no result"
                continue
            if min_score is not None and float(score) < min_score:  # drop weak matches
                continue
            hit = dict(self.records[idx])                 # copy that chunk's metadata
            hit["score"] = float(score)                   # attach its similarity score
            results.append(hit)
        return results                                     # may be empty -> caller abstains
