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
    # Example: user submits "   " (just spaces) -> the chain raises ValueError
    #   immediately, before embedding or calling the model.
    chain = RagChain(_retriever(), lambda q: [1.0, 0.0, 0.0], FakeLLM())
    with pytest.raises(ValueError):
        chain.answer("   ")


# A relevant question should answer, attach the right source, and feed the LLM the context.
def test_answers_with_sources_when_relevant():
    # Example: "What is the returns policy?" -> embeds to [1,0,0] (near "returns"),
    #   retrieves the returns chunk, and the fake LLM answers "...30 days...".
    #   The result's first source is "returns:0:0000", and the chunk text + rules
    #   were passed to the LLM.
    llm = FakeLLM()
    chain = RagChain(_retriever(), lambda q: [1.0, 0.0, 0.0], llm, top_k=2, min_score=0.5)
    result = chain.answer("What is the returns policy?")
    assert result["abstained"] is False
    assert "30 days" in result["answer"]
    assert result["sources"][0]["chunk_id"] == "returns:0:0000"
    assert "Returns within 30 days." in llm.last_prompt
    assert "only" in (llm.last_system or "").lower()


# THE GUARDRAIL: when nothing is relevant enough, the chain must abstain (no LLM guess).
def test_abstains_when_nothing_relevant():
    # Example: an off-topic question embeds to [0.3,0.3,0.3], which is only ~0.577
    #   similar to every chunk. With min_score=0.99 nothing passes, so the chain
    #   returns "I cannot answer that..." with no sources - it does NOT guess.
    chain = RagChain(_retriever(), lambda q: [0.3, 0.3, 0.3], FakeLLM(), min_score=0.99)
    result = chain.answer("Totally unrelated question?")
    assert result["abstained"] is True
    assert result["answer"] == ABSTAIN_MESSAGE
    assert result["sources"] == []


# If the LLM crashes, the chain must NOT leak the error - it returns a friendly result.
def test_llm_failure_is_handled_gracefully():
    # Example: retrieval succeeds but the model throws (BrokenLLM) -> the chain
    #   catches it, records error="RuntimeError", still shows the sources, and
    #   returns a message containing "unavailable" instead of crashing.
    chain = RagChain(_retriever(), lambda q: [1.0, 0.0, 0.0], BrokenLLM(), min_score=0.5)
    result = chain.answer("What is the returns policy?")
    assert result["abstained"] is False
    assert result["error"] == "RuntimeError"
    assert result["sources"][0]["chunk_id"] == "returns:0:0000"
    assert "unavailable" in result["answer"].lower()


# A fake "text" retriever: it is searched by the QUESTION TEXT (managed-embeddings
# index), remembers what it received, and returns one known chunk.
class FakeTextRetriever:
    def __init__(self):
        self.last_query = None

    def search(self, query, top_k=5, min_score=None):
        self.last_query = query     # record what the chain passed in
        return [{"chunk_id": "returns:0:0000", "chunk_text": "Returns within 30 days.",
                 "source_path": "returns.md", "section": "Returns", "document_id": "returns"}]


# With embed_fn=None, the chain must pass the QUESTION TEXT (not a vector) to search.
def test_text_retriever_path_passes_question_text():
    # Example: a managed-embeddings index embeds the query itself, so RagChain is
    #   built with embed_fn=None and hands the raw question to the retriever.
    retr = FakeTextRetriever()
    chain = RagChain(retr, embed_fn=None, llm=FakeLLM(), top_k=3, min_score=0.1)
    result = chain.answer("What is the returns policy?")
    assert retr.last_query == "What is the returns policy?"   # text passed, not a vector
    assert result["abstained"] is False
    assert result["sources"][0]["chunk_id"] == "returns:0:0000"


# The result should include diagnostics (how many retrieved, the top_k used, etc.).
def test_diagnostics_present():
    # Example: asking "returns?" with top_k=3 -> result["diagnostics"] reports
    #   retrieved >= 1 and top_k == 3 (useful for debugging/tuning).
    chain = RagChain(_retriever(), lambda q: [1.0, 0.0, 0.0], FakeLLM(), top_k=3, min_score=0.1)
    result = chain.answer("returns?")
    assert result["diagnostics"]["retrieved"] >= 1
    assert result["diagnostics"]["top_k"] == 3
