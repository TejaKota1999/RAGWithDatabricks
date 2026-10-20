"""Local embedding adapter (sentence-transformers).

The SAME model must embed both document chunks and questions. Kept separate
from Databricks so the zero-cost local path needs no managed endpoint.
"""

from __future__ import annotations

from typing import Sequence

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbedder:
    """Lazy wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str = DEFAULT_MODEL, normalize: bool = True):
        self.model_name = model_name
        self.normalize = normalize
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]):
        """Return a (n, dim) numpy array of embeddings."""
        model = self._load()
        return model.encode(
            list(texts),
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def encode_one(self, text: str) -> list[float]:
        """Return a single embedding as a plain list of floats."""
        return [float(x) for x in self.encode([text])[0].tolist()]
