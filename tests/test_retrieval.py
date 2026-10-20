"""Unit tests for src/retriever.py (local FAISS retrieval, no model download)."""

import numpy as np
import pytest

from src.retriever import FaissRetriever


def _toy_index():
    # Three orthogonal-ish vectors so similarity is unambiguous.
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],  # returns
            [0.0, 1.0, 0.0],  # expenses
            [0.0, 0.0, 1.0],  # leave
        ],
        dtype="float32",
    )
    records = [
        {"chunk_id": "returns:0:0000", "chunk_text": "Returns within 30 days.", "source_path": "returns.md"},
        {"chunk_id": "expenses:0:0000", "chunk_text": "Submit expenses in 60 days.", "source_path": "expenses.md"},
        {"chunk_id": "leave:0:0000", "chunk_text": "20 days paid leave.", "source_path": "leave.md"},
    ]
    return FaissRetriever(vectors, records)


def test_retrieves_most_relevant_chunk():
    r = _toy_index()
    hits = r.search([0.9, 0.1, 0.0], top_k=1)
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "returns:0:0000"
    assert hits[0]["score"] > 0.9
    # Metadata is carried through to the result.
    assert hits[0]["source_path"] == "returns.md"


def test_top_k_limits_and_orders_results():
    r = _toy_index()
    hits = r.search([0.8, 0.6, 0.0], top_k=2)
    assert len(hits) == 2
    # Highest score first.
    assert hits[0]["score"] >= hits[1]["score"]
    assert hits[0]["chunk_id"] == "returns:0:0000"


def test_min_score_filters_irrelevant_results():
    r = _toy_index()
    # A query orthogonal-ish to everything with a high threshold -> abstain.
    hits = r.search([0.0, 0.0, 1.0], top_k=3, min_score=0.99)
    assert all(h["chunk_id"] == "leave:0:0000" for h in hits)
    none = r.search([1.0, 0.0, 0.0], top_k=3, min_score=0.999)
    assert len(none) == 1  # only the (near-)perfect match survives


def test_no_results_when_threshold_excludes_all():
    r = _toy_index()
    hits = r.search([0.0, 1.0, 0.0], top_k=3, min_score=1.01)
    assert hits == []


def test_top_k_capped_to_index_size():
    r = _toy_index()
    hits = r.search([1.0, 0.0, 0.0], top_k=10)
    assert len(hits) == 3


def test_dim_mismatch_raises():
    r = _toy_index()
    with pytest.raises(ValueError):
        r.search([1.0, 0.0], top_k=1)


def test_invalid_top_k_raises():
    r = _toy_index()
    with pytest.raises(ValueError):
        r.search([1.0, 0.0, 0.0], top_k=0)


def test_records_length_must_match_vectors():
    with pytest.raises(ValueError):
        FaissRetriever(np.zeros((2, 3), dtype="float32"), [{"chunk_id": "x"}])
