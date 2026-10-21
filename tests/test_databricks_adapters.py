"""Unit tests for the Databricks-backed adapters (embedder, llm, retriever).

The Databricks SDK client is replaced with a fake, so these tests do NOT hit a
live workspace or consume quota. They verify how each adapter parses the SDK
response shapes and handles errors.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.embedder import DatabricksEmbedder
from src.llm import DatabricksChatClient, LLMUnavailableError
from src.retriever import VectorSearchRetriever


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------
class _FakeEmbedEndpoints:
    def query(self, name, input):  # mimics serving_endpoints.query for embeddings
        # Return one 4-dim embedding per input text.
        data = [SimpleNamespace(embedding=[float(len(t)), 1.0, 0.0, 0.0]) for t in input]
        return SimpleNamespace(data=data)


def _fake_embed_client():
    return SimpleNamespace(serving_endpoints=_FakeEmbedEndpoints())


def test_embedder_encode_shapes_and_normalizes():
    emb = DatabricksEmbedder(normalize=True)
    emb._client = _fake_embed_client()          # inject the fake SDK client
    arr = emb.encode(["aa", "bbbb"])
    assert arr.shape == (2, 4)                   # one row per text, 4-dim vectors
    # Normalized vectors have length ~1.
    import numpy as np
    assert np.allclose(np.linalg.norm(arr, axis=1), 1.0, atol=1e-5)


def test_embedder_encode_one_returns_list():
    emb = DatabricksEmbedder(normalize=False)
    emb._client = _fake_embed_client()
    vec = emb.encode_one("hello")
    assert isinstance(vec, list) and len(vec) == 4
    assert all(isinstance(x, float) for x in vec)


# ---------------------------------------------------------------------------
# Chat LLM
# ---------------------------------------------------------------------------
class _FakeChatEndpoints:
    def __init__(self, content="grounded answer", raise_exc=False):
        self.content = content
        self.raise_exc = raise_exc

    def query(self, name, messages, max_tokens, temperature):
        if self.raise_exc:
            raise RuntimeError("endpoint down")
        msg = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_chat_generate_returns_text():
    llm = DatabricksChatClient()
    llm._client = SimpleNamespace(serving_endpoints=_FakeChatEndpoints("hello world"))
    out = llm.generate("prompt", system="rules")
    assert out == "hello world"


def test_chat_generate_wraps_errors():
    llm = DatabricksChatClient()
    llm._client = SimpleNamespace(serving_endpoints=_FakeChatEndpoints(raise_exc=True))
    with pytest.raises(LLMUnavailableError):
        llm.generate("prompt")


# ---------------------------------------------------------------------------
# Vector Search retriever
# ---------------------------------------------------------------------------
class _FakeVSIndexes:
    def query_index(self, index_name, columns, query_vector, num_results):
        # manifest lists the requested columns; each row has the values + a score.
        manifest = SimpleNamespace(columns=[SimpleNamespace(name=c) for c in columns])
        rows = [
            ["c1", "doc", "text one", "S1", 0, "path1", 0.91],   # score 0.91
            ["c2", "doc", "text two", "S2", 0, "path2", 0.20],   # score 0.20 (weak)
        ]
        result = SimpleNamespace(data_array=rows)
        return SimpleNamespace(manifest=manifest, result=result)


def _fake_vs_client():
    return SimpleNamespace(vector_search_indexes=_FakeVSIndexes())


def test_vs_search_returns_records_with_scores():
    r = VectorSearchRetriever(index_name="cat.sch.idx")
    r._client = _fake_vs_client()
    hits = r.search([0.1, 0.2, 0.3], top_k=5)
    assert len(hits) == 2
    assert hits[0]["chunk_id"] == "c1"
    assert hits[0]["section"] == "S1"
    assert abs(hits[0]["score"] - 0.91) < 1e-6


def test_vs_search_applies_min_score():
    r = VectorSearchRetriever(index_name="cat.sch.idx")
    r._client = _fake_vs_client()
    hits = r.search([0.1, 0.2, 0.3], top_k=5, min_score=0.3)
    # The 0.20 row is dropped; only the strong hit remains.
    assert [h["chunk_id"] for h in hits] == ["c1"]


def test_vs_search_rejects_bad_top_k():
    r = VectorSearchRetriever(index_name="cat.sch.idx")
    r._client = _fake_vs_client()
    with pytest.raises(ValueError):
        r.search([0.1], top_k=0)
