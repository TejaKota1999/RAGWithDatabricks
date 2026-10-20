"""Unit tests for src/evaluation.py."""

from src.evaluation import (
    citation_precision,
    extract_citations,
    groundedness,
    is_grounded,
    recall_at_k,
)


def test_recall_at_k():
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, ["a"], k=1) == 1.0
    assert recall_at_k(retrieved, ["c"], k=2) == 0.0
    assert recall_at_k(retrieved, ["a", "c"], k=3) == 1.0
    assert recall_at_k(retrieved, [], k=3) == 0.0


def test_groundedness_and_is_grounded():
    chunks = [{"chunk_text": "Returns are allowed within 30 days of purchase."}]
    assert groundedness("Returns within 30 days", chunks) > 0.8
    assert is_grounded("Returns within 30 days", chunks)
    # An invented answer shares few words with the context.
    assert not is_grounded("Bananas are purple spaceships orbiting Jupiter", chunks)


def test_groundedness_empty_cases():
    assert groundedness("", [{"chunk_text": "anything"}]) == 0.0
    assert groundedness("something", []) == 0.0


def test_extract_and_score_citations():
    answer = "Returns in 30 days [returns:0:0000], see also [leave:0:0001]."
    cites = extract_citations(answer)
    assert cites == ["returns:0:0000", "leave:0:0001"]
    retrieved = ["returns:0:0000", "expenses:0:0000"]
    assert citation_precision(cites, retrieved) == 0.5
    assert citation_precision([], retrieved) == 0.0
