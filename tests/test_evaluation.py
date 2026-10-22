"""Unit tests for src/evaluation.py.

Checks the scoring helpers used to measure RAG quality: Recall@K (did we find
the right chunks), groundedness (is the answer supported by the context), and
citation precision (did the answer cite real retrieved chunks).
"""

# The scoring functions under test.
from src.evaluation import (
    citation_precision,
    extract_citations,
    groundedness,
    is_grounded,
    recall_at_k,
)


# Recall@K = fraction of the expected ("relevant") ids found in the top-k results.
def test_recall_at_k():
    # Example: search returned ["a","b","c","d"]. If the answer should have come
    #   from "a": looking at the top-1 ("a") -> found it -> recall 1.0.
    #   If it should have come from "c" but we only check top-2 ("a","b") -> 0.0.
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, ["a"], k=1) == 1.0
    assert recall_at_k(retrieved, ["c"], k=2) == 0.0
    assert recall_at_k(retrieved, ["a", "c"], k=3) == 1.0
    assert recall_at_k(retrieved, [], k=3) == 0.0


# Groundedness = how much of the answer's wording appears in the context.
def test_groundedness_and_is_grounded():
    # Example: context says "Returns are allowed within 30 days of purchase."
    #   Answer "Returns within 30 days" reuses those words -> grounded (high score).
    #   Answer "Bananas are purple spaceships..." shares no words -> NOT grounded.
    chunks = [{"chunk_text": "Returns are allowed within 30 days of purchase."}]
    assert groundedness("Returns within 30 days", chunks) > 0.8
    assert is_grounded("Returns within 30 days", chunks)
    assert not is_grounded("Bananas are purple spaceships orbiting Jupiter", chunks)


# Edge cases: empty answer or empty context should score 0 (not crash).
def test_groundedness_empty_cases():
    # Example: no answer text -> 0.0; or an answer but no context chunks -> 0.0.
    assert groundedness("", [{"chunk_text": "anything"}]) == 0.0
    assert groundedness("something", []) == 0.0


# Citations like [id] are pulled out, then scored against what was retrieved.
def test_extract_and_score_citations():
    # Example: answer cites [returns:0:0000] and [leave:0:0001]. If retrieval
    #   actually returned "returns:0:0000" and "expenses:0:0000", then 1 of the 2
    #   citations is real -> citation precision = 0.5.
    answer = "Returns in 30 days [returns:0:0000], see also [leave:0:0001]."
    cites = extract_citations(answer)
    assert cites == ["returns:0:0000", "leave:0:0001"]
    retrieved = ["returns:0:0000", "expenses:0:0000"]
    assert citation_precision(cites, retrieved) == 0.5
    assert citation_precision([], retrieved) == 0.0     # nothing cited -> 0.0
