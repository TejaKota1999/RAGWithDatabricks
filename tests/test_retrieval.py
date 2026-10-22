"""Unit tests for src/retriever.py (local FAISS retrieval, no model download).

These tests build a tiny FAISS index from hand-made vectors and check that
search returns the closest chunk, respects top_k and min_score, and handles
bad inputs. No Databricks Vector Search is used here.
"""

# numpy builds the vectors; pytest checks that errors are raised.
import numpy as np
import pytest

# The retriever class under test (the local fallback / testable one).
from src.retriever import FaissRetriever


# Helper: build a small 3-vector index with known chunk metadata.
def _toy_index():
    # Three "orthogonal" vectors so one is clearly closest to a given query.
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],  # returns
            [0.0, 1.0, 0.0],  # expenses
            [0.0, 0.0, 1.0],  # leave
        ],
        dtype="float32",
    )
    # One metadata record per vector (aligned by row).
    records = [
        {"chunk_id": "returns:0:0000", "chunk_text": "Returns within 30 days.", "source_path": "returns.md"},
        {"chunk_id": "expenses:0:0000", "chunk_text": "Submit expenses in 60 days.", "source_path": "expenses.md"},
        {"chunk_id": "leave:0:0000", "chunk_text": "20 days paid leave.", "source_path": "leave.md"},
    ]
    return FaissRetriever(vectors, records)


# A query close to the first vector should return the "returns" chunk.
def test_retrieves_most_relevant_chunk():
    r = _toy_index()
    hits = r.search([0.9, 0.1, 0.0], top_k=1)   # ask for the single best match
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "returns:0:0000"   # correct chunk found
    assert hits[0]["score"] > 0.9                     # high similarity score
    assert hits[0]["source_path"] == "returns.md"     # metadata carried through


# top_k limits how many results come back, and they are ordered best-first.
def test_top_k_limits_and_orders_results():
    r = _toy_index()
    hits = r.search([0.8, 0.6, 0.0], top_k=2)   # want the top 2
    assert len(hits) == 2
    assert hits[0]["score"] >= hits[1]["score"]  # highest score first
    assert hits[0]["chunk_id"] == "returns:0:0000"


# min_score drops weak matches; only strong-enough chunks remain.
def test_min_score_filters_irrelevant_results():
    r = _toy_index()
    # Query points exactly at "leave"; with a 0.99 threshold only "leave" passes.
    hits = r.search([0.0, 0.0, 1.0], top_k=3, min_score=0.99)
    assert all(h["chunk_id"] == "leave:0:0000" for h in hits)
    # With a stricter 0.999 threshold, only a near-perfect match survives.
    none = r.search([1.0, 0.0, 0.0], top_k=3, min_score=0.999)
    assert len(none) == 1


# If the threshold is impossibly high, NO chunks pass -> empty list (abstain).
def test_no_results_when_threshold_excludes_all():
    r = _toy_index()
    hits = r.search([0.0, 1.0, 0.0], top_k=3, min_score=1.01)  # >1 is unreachable
    assert hits == []                                           # nothing returned


# Asking for more results than exist just returns all of them (capped).
def test_top_k_capped_to_index_size():
    r = _toy_index()
    hits = r.search([1.0, 0.0, 0.0], top_k=10)  # only 3 vectors exist
    assert len(hits) == 3


# A query vector of the wrong length must raise a clear error.
def test_dim_mismatch_raises():
    r = _toy_index()
    with pytest.raises(ValueError):
        r.search([1.0, 0.0], top_k=1)           # 2 numbers, index expects 3


# top_k must be at least 1.
def test_invalid_top_k_raises():
    r = _toy_index()
    with pytest.raises(ValueError):
        r.search([1.0, 0.0, 0.0], top_k=0)


# The number of metadata records must match the number of vectors.
def test_records_length_must_match_vectors():
    with pytest.raises(ValueError):
        # 2 vectors but only 1 record -> error.
        FaissRetriever(np.zeros((2, 3), dtype="float32"), [{"chunk_id": "x"}])
