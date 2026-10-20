"""Unit tests for src/rag_chain.py (retriever + LLM are faked/mocked)."""

import numpy as np
import pytest

from src.prompt_templates import ABSTAIN_MESSAGE
from src.rag_chain import RagChain
from src.retriever import FaissRetriever


class FakeLLM:
    def __init__(self, response="Returns are allowed within 30 days [returns:0:0000]."):
        self.response = response
        self.last_prompt = None
        self.last_system = None

    def generate(self, prompt, system=None):
        self.last_prompt = prompt
        self.last_system = system
        return self.response


class BrokenLLM:
    def generate(self, prompt, system=None):
        raise RuntimeError("model crashed")


def _retriever():
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype="float32")
    records = [
        {"chunk_id": "returns:0:0000", "chunk_text": "Returns within 30 days.",
         "source_path": "returns.md", "section": "Returns", "document_id": "returns"},
        {"chunk_id": "expenses:0:0000", "chunk_text": "Submit expenses in 60 days.",
         "source_path": "expenses.md", "section": "Expenses", "document_id": "expenses"},
        {"chunk_id": "leave:0:0000", "chunk_text": "20 days paid leave.",
         "source_path": "leave.md", "section": "Leave", "document_id": "leave"},
    ]
    return FaissRetriever(vectors, records)


def test_empty_question_raises():
    chain = RagChain(_retriever(), lambda q: [1.0, 0.0, 0.0], FakeLLM())
    with pytest.raises(ValueError):
        chain.answer("   ")


def test_answers_with_sources_when_relevant():
    llm = FakeLLM()
    chain = RagChain(_retriever(), lambda q: [1.0, 0.0, 0.0], llm, top_k=2, min_score=0.5)
    result = chain.answer("What is the returns policy?")
    assert result["abstained"] is False
    assert "30 days" in result["answer"]
    assert result["sources"][0]["chunk_id"] == "returns:0:0000"
    # The context and system rules were passed to the LLM.
    assert "Returns within 30 days." in llm.last_prompt
    assert "only" in (llm.last_system or "").lower()


def test_abstains_when_nothing_relevant():
    # Query vector is equidistant from all docs (cosine ~0.577), below 0.99.
    chain = RagChain(_retriever(), lambda q: [0.3, 0.3, 0.3], FakeLLM(), min_score=0.99)
    result = chain.answer("Totally unrelated question?")
    assert result["abstained"] is True
    assert result["answer"] == ABSTAIN_MESSAGE
    assert result["sources"] == []


def test_llm_failure_is_handled_gracefully():
    chain = RagChain(_retriever(), lambda q: [1.0, 0.0, 0.0], BrokenLLM(), min_score=0.5)
    result = chain.answer("What is the returns policy?")
    # No exception leaks; still returns sources and flags the error.
    assert result["abstained"] is False
    assert result["error"] == "RuntimeError"
    assert result["sources"][0]["chunk_id"] == "returns:0:0000"
    assert "unavailable" in result["answer"].lower()


def test_diagnostics_present():
    chain = RagChain(_retriever(), lambda q: [1.0, 0.0, 0.0], FakeLLM(), top_k=3, min_score=0.1)
    result = chain.answer("returns?")
    assert result["diagnostics"]["retrieved"] >= 1
    assert result["diagnostics"]["top_k"] == 3
