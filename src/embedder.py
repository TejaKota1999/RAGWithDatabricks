"""Embedding adapters: Databricks Foundation Model endpoint (default) + local fallback.

The SAME model must embed both document chunks and questions. The default path
calls a Databricks embedding serving endpoint (no local model download). A local
sentence-transformers class is kept only as an optional offline fallback.

Plain English: this turns text into a vector (a list of numbers) that captures
its meaning, so similar texts get similar vectors.
"""

from __future__ import annotations

# Type hint meaning "a list/tuple-like of strings".
from typing import Sequence

# The Databricks embedding endpoint used by this project (1024-dimensional vectors).
DEFAULT_ENDPOINT = "databricks-qwen3-embedding-0-6b"
# The dimension that endpoint returns; the Vector Search index must match this.
EMBEDDING_DIM = 1024

# The local fallback model name (only used by LocalEmbedder, not the default path).
DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# COMMAND ----------
# DEFAULT adapter: embeds text by calling a Databricks serving endpoint.
class DatabricksEmbedder:
    """Embeds text using a Databricks Foundation Model embedding endpoint.

    Keeps the same interface as the local embedder (``encode`` / ``encode_one``)
    so the rest of the RAG pipeline does not change.
    """

    # Remember which endpoint to call and how to authenticate; no model is loaded locally.
    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        profile: str | None = None,
        batch_size: int = 16,
        normalize: bool = True,
    ):
        self.endpoint = endpoint          # the serving endpoint name
        self.profile = profile            # optional CLI profile (None in app/notebook)
        self.batch_size = batch_size      # how many texts to send per request
        self.normalize = normalize        # Vector Search uses cosine; keep True
        self._client = None               # the WorkspaceClient, created lazily

    # Create (once) the Databricks client used to call the endpoint.
    def _ws(self):
        if self._client is None:
            from src.databricks_client import get_workspace_client

            self._client = get_workspace_client(self.profile)
        return self._client

    # Turn many texts into vectors by calling the endpoint in batches.
    def encode(self, texts: Sequence[str]):
        """Return a (n, 1024) list of embeddings (one row per input text)."""
        import numpy as np

        client = self._ws()
        texts = list(texts)
        vectors: list[list[float]] = []
        # Send the texts in small batches to stay within request limits.
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            # Query the embeddings endpoint; "input" accepts a list of strings.
            resp = client.serving_endpoints.query(name=self.endpoint, input=batch)
            # The response carries one embedding per input under ``data``.
            for item in resp.data:
                # Each item has an ``embedding`` (a list of floats).
                vectors.append([float(x) for x in item.embedding])
        arr = np.asarray(vectors, dtype="float32")
        if self.normalize:
            # Scale each vector to length 1 so inner product == cosine similarity.
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr

    # Convenience: embed ONE text and return a plain list of floats (for queries).
    def encode_one(self, text: str) -> list[float]:
        """Return a single embedding as a plain list of floats."""
        return [float(x) for x in self.encode([text])[0].tolist()]


# COMMAND ----------
# OPTIONAL FALLBACK: local sentence-transformers embedder (offline only).
class LocalEmbedder:
    """Offline fallback: a lazy wrapper around a local sentence-transformers model.

    NOT the default. Produces different (384-dim) vectors than the Databricks
    endpoint, so it cannot be mixed with the Databricks Vector Search index.
    """

    # Remember settings; the actual model is loaded later (lazily) to save time.
    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL, normalize: bool = True):
        self.model_name = model_name     # which local model to use
        self.normalize = normalize       # unit-normalize for cosine similarity
        self._model = None               # not loaded yet

    # Load the model the first time it is needed, then reuse it.
    def _load(self):
        if self._model is None:                               # only load once
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)  # download/open the model
        return self._model

    # Turn many texts into vectors at once; returns a numpy array.
    def encode(self, texts: Sequence[str]):
        """Return a (n, dim) numpy array of embeddings."""
        model = self._load()
        return model.encode(
            list(texts),
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    # Convenience: embed ONE text and return a plain list of floats.
    def encode_one(self, text: str) -> list[float]:
        """Return a single embedding as a plain list of floats."""
        return [float(x) for x in self.encode([text])[0].tolist()]
