"""Unit tests for the Databricks-backed adapters (embedder, llm, retriever).

The real Databricks SDK client is replaced with a FAKE object, so these tests do
NOT hit a live workspace or consume quota. They verify how each adapter parses
the SDK's response shapes and handles errors.
"""

from __future__ import annotations

# SimpleNamespace lets us build quick fake objects with attributes (like a stub response).
from types import SimpleNamespace

import pytest

# The three Databricks adapters under test.
from src.embedder import DatabricksEmbedder
from src.llm import DatabricksChatClient, LLMUnavailableError
from src.retriever import VectorSearchRetriever


# ---------------------------------------------------------------------------
# Embedder - turns text into vectors by calling a serving endpoint.
# ---------------------------------------------------------------------------
# A fake "serving_endpoints" that mimics the embeddings response shape.
class _FakeEmbedEndpoints:
    def query(self, name, input):  # same signature as the real SDK method
        # Return one fake 4-number embedding per input text.
        data = [SimpleNamespace(embedding=[float(len(t)), 1.0, 0.0, 0.0]) for t in input]
        return SimpleNamespace(data=data)


# A fake workspace client whose .serving_endpoints is our fake above.
def _fake_embed_client():
    return SimpleNamespace(serving_endpoints=_FakeEmbedEndpoints())


# encode() should return one vector per text, shaped (n, dim), and L2-normalized.
def test_embedder_encode_shapes_and_normalizes():
    emb = DatabricksEmbedder(normalize=True)
    emb._client = _fake_embed_client()           # inject the fake SDK client
    arr = emb.encode(["aa", "bbbb"])             # embed two texts
    assert arr.shape == (2, 4)                    # 2 rows, 4-number vectors
    import numpy as np
    # Normalized vectors have length ~1.
    assert np.allclose(np.linalg.norm(arr, axis=1), 1.0, atol=1e-5)


# encode_one() should return a single vector as a plain list of floats.
def test_embedder_encode_one_returns_list():
    emb = DatabricksEmbedder(normalize=False)
    emb._client = _fake_embed_client()
    vec = emb.encode_one("hello")
    assert isinstance(vec, list) and len(vec) == 4
    assert all(isinstance(x, float) for x in vec)


# ---------------------------------------------------------------------------
# Chat LLM - sends the prompt to a chat endpoint and returns the text.
# ---------------------------------------------------------------------------
# A fake chat endpoint that returns canned content, or raises if asked to.
class _FakeChatEndpoints:
    def __init__(self, content="grounded answer", raise_exc=False):
        self.content = content
        self.raise_exc = raise_exc

    def query(self, name, messages, max_tokens, temperature):
        if self.raise_exc:                        # simulate the endpoint being down
            raise RuntimeError("endpoint down")
        msg = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


# generate() should return the model's text from the response.
def test_chat_generate_returns_text():
    llm = DatabricksChatClient()
    llm._client = SimpleNamespace(serving_endpoints=_FakeChatEndpoints("hello world"))
    out = llm.generate("prompt", system="rules")
    assert out == "hello world"


# If the endpoint call fails, it should raise a clear LLMUnavailableError (not crash raw).
def test_chat_generate_wraps_errors():
    llm = DatabricksChatClient()
    llm._client = SimpleNamespace(serving_endpoints=_FakeChatEndpoints(raise_exc=True))
    with pytest.raises(LLMUnavailableError):
        llm.generate("prompt")


# ---------------------------------------------------------------------------
# Vector Search retriever - queries the managed index.
# ---------------------------------------------------------------------------
# A fake index that returns two rows; the last value in each row is the score.
class _FakeVSIndexes:
    def query_index(self, index_name, columns, query_vector, num_results):
        # manifest lists the requested columns in order.
        manifest = SimpleNamespace(columns=[SimpleNamespace(name=c) for c in columns])
        rows = [
            ["c1", "doc", "text one", "S1", 0, "path1", 0.91],   # strong match (0.91)
            ["c2", "doc", "text two", "S2", 0, "path2", 0.20],   # weak match (0.20)
        ]
        result = SimpleNamespace(data_array=rows)
        return SimpleNamespace(manifest=manifest, result=result)


def _fake_vs_client():
    return SimpleNamespace(vector_search_indexes=_FakeVSIndexes())


# search() should map columns to values and attach each row's score.
def test_vs_search_returns_records_with_scores():
    r = VectorSearchRetriever(index_name="cat.sch.idx")
    r._client = _fake_vs_client()
    hits = r.search([0.1, 0.2, 0.3], top_k=5)
    assert len(hits) == 2
    assert hits[0]["chunk_id"] == "c1"             # first column mapped correctly
    assert hits[0]["section"] == "S1"              # another column mapped correctly
    assert abs(hits[0]["score"] - 0.91) < 1e-6     # score taken from the last value


# min_score should drop the weak (0.20) row, leaving only the strong one.
def test_vs_search_applies_min_score():
    r = VectorSearchRetriever(index_name="cat.sch.idx")
    r._client = _fake_vs_client()
    hits = r.search([0.1, 0.2, 0.3], top_k=5, min_score=0.3)
    assert [h["chunk_id"] for h in hits] == ["c1"]  # only the 0.91 row survives


# top_k must be at least 1, or it raises.
def test_vs_search_rejects_bad_top_k():
    r = VectorSearchRetriever(index_name="cat.sch.idx")
    r._client = _fake_vs_client()
    with pytest.raises(ValueError):
        r.search([0.1], top_k=0)
