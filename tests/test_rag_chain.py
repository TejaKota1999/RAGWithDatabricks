"""Unit tests for src/rag_chain.py (retriever + LLM are faked/mocked).

The RAG chain is the "conductor": embed question -> retrieve -> abstain or
answer -> return sources. These tests pass in FAKE pieces (a fake retriever and
fake LLMs) so the orchestration logic is checked without any real model or
Databricks call.
"""

import numpy as np
import pytest

# The abstain message, the chain under test, and the local retriever used as a stand-in.
from src.prompt_templates import ABSTAIN_MESSAGE
from src.rag_chain import RagChain
from src.retriever import FaissRetriever


# A fake LLM that just returns a fixed answer and remembers what it was sent.
class FakeLLM:
    def __init__(self, response="Returns are allowed within 30 days [returns:0:0000]."):
        self.response = response
        self.last_prompt = None       # we can inspect the prompt the chain built
        self.last_system = None

    def generate(self, prompt, system=None):
        self.last_prompt = prompt      # record the inputs for assertions
        self.last_system = system
        return self.response           # always return the canned answer


# A fake LLM that always crashes - used to test error handling.
class BrokenLLM:
    def generate(self, prompt, system=None):
        raise RuntimeError("model crashed")


# Helper: a small FAISS retriever with 3 known chunks (our stand-in for search).
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


# An empty/blank question must be rejected up front with an error.
def test_empty_question_raises():
    # embed_fn is a fake lambda returning a fixed vector; the LLM is never reached.
    chain = RagChain(_retriever(), lambda q: [1.0, 0.0, 0.0], FakeLLM())
    with pytest.raises(ValueError):
        chain.answer("   ")                      # only spaces -> ValueError


# A relevant question should answer, attach the right source, and feed the LLM the context.
def test_answers_with_sources_when_relevant():
    llm = FakeLLM()
    # embed_fn returns a vector pointing at the "returns" chunk; min_score 0.5.
    chain = RagChain(_retriever(), lambda q: [1.0, 0.0, 0.0], llm, top_k=2, min_score=0.5)
    result = chain.answer("What is the returns policy?")
    assert result["abstained"] is False          # it answered (did not abstain)
    assert "30 days" in result["answer"]         # the canned answer came back
    assert result["sources"][0]["chunk_id"] == "returns:0:0000"   # correct source
    # The chain actually passed the retrieved context + the rules to the LLM.
    assert "Returns within 30 days." in llm.last_prompt
    assert "only" in (llm.last_system or "").lower()


# THE GUARDRAIL: when nothing is relevant enough, the chain must abstain (no LLM guess).
def test_abstains_when_nothing_relevant():
    # Query vector [0.3,0.3,0.3] is only ~0.577 similar to each doc; threshold 0.99
    # is impossible to reach, so every chunk is filtered out.
    chain = RagChain(_retriever(), lambda q: [0.3, 0.3, 0.3], FakeLLM(), min_score=0.99)
    result = chain.answer("Totally unrelated question?")
    assert result["abstained"] is True                 # abstained
    assert result["answer"] == ABSTAIN_MESSAGE         # the "I cannot answer" message
    assert result["sources"] == []                     # no sources returned


# If the LLM crashes, the chain must NOT leak the error - it returns a friendly result.
def test_llm_failure_is_handled_gracefully():
    chain = RagChain(_retriever(), lambda q: [1.0, 0.0, 0.0], BrokenLLM(), min_score=0.5)
    result = chain.answer("What is the returns policy?")
    assert result["abstained"] is False                # it did retrieve evidence
    assert result["error"] == "RuntimeError"           # the error type is recorded
    assert result["sources"][0]["chunk_id"] == "returns:0:0000"   # sources still shown
    assert "unavailable" in result["answer"].lower()   # a friendly message to the user


# The result should include diagnostics (how many retrieved, the top_k used, etc.).
def test_diagnostics_present():
    chain = RagChain(_retriever(), lambda q: [1.0, 0.0, 0.0], FakeLLM(), top_k=3, min_score=0.1)
    result = chain.answer("returns?")
    assert result["diagnostics"]["retrieved"] >= 1     # at least one chunk retrieved
    assert result["diagnostics"]["top_k"] == 3         # echoes the setting used
