# Databricks notebook source
"""Local embedding adapter (sentence-transformers).

The SAME model must embed both document chunks and questions. Kept separate
from Databricks so the zero-cost local path needs no managed endpoint.

Plain English: this turns text into a vector (a list of numbers) that captures
its meaning, so similar texts get similar vectors.
"""

from __future__ import annotations

# Type hint meaning "a list/tuple-like of strings".
from typing import Sequence

# The default small, fast, free embedding model.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# COMMAND ----------
# A thin wrapper so the rest of the code does not deal with the model directly.
class LocalEmbedder:
    """Lazy wrapper around a sentence-transformers model."""

    # Remember settings; the actual model is loaded later (lazily) to save time.
    def __init__(self, model_name: str = DEFAULT_MODEL, normalize: bool = True):
        self.model_name = model_name     # which model to use
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
            list(texts),                       # the texts to embed
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    # Convenience: embed ONE text and return a plain list of floats (for queries).
    def encode_one(self, text: str) -> list[float]:
        """Return a single embedding as a plain list of floats."""
        # encode([text]) -> array of 1 row; [0] takes that row; tolist() makes it a list.
        return [float(x) for x in self.encode([text])[0].tolist()]
