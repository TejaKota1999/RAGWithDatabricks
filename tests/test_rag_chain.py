"""Unit tests for src/rag_chain.py (retriever + LLM are faked/mocked).

The RAG chain is the "conductor": embed question -> retrieve -> abstain or
answer -> return sources. These tests pass in FAKE pieces (a fake retriever and
fake LLMs) so the orchestration logic is checked without any real model or
Databricks call.
"""

import numpy as np
import pytest

# The abstain message, the chain under test, and the local retriever used as a stand-in.
from src.serving.prompt_templates import ABSTAIN_MESSAGE
from src.serving.rag_chain import RagChain
from src.serving.retriever import FaissRetriever


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


# Per-stage timings feed the app's "How it works" tab: condense is skipped (None)
# on a first question, retrieval and generation are measured in ms.
def test_diagnostics_timings():
    chain = RagChain(_retriever(), lambda q: [1.0, 0.0, 0.0], FakeLLM(), top_k=3, min_score=0.1)
    d = chain.answer("returns?")["diagnostics"]
    assert d["condensed"] is False
    assert d["timings"]["condense_ms"] is None
    assert d["timings"]["retrieve_ms"] >= 0
    assert d["timings"]["generate_ms"] >= 0


# A fake LLM that behaves differently for the two kinds of call the chain makes:
#   - the CONDENSE call (system == CONDENSE_SYSTEM_PROMPT) -> return the rewrite
#   - the ANSWER call -> return a canned answer, and record the prompt so we can
#     check the conversation history was included.
class ConversationLLM:
    def __init__(self, rewrite="What is the iPhone replacement policy for screen damage?"):
        self.rewrite = rewrite
        self.condense_calls = 0
        self.answer_prompt = None

    def generate(self, prompt, system=None):
        from src.serving.prompt_templates import CONDENSE_SYSTEM_PROMPT

        if system == CONDENSE_SYSTEM_PROMPT:   # this is the reformulation step
            self.condense_calls += 1
            return self.rewrite
        self.answer_prompt = prompt            # this is the answer step
        return "Answer [returns:0:0000]."


# CONVERSATIONAL MEMORY: a follow-up with history is rewritten into a standalone
# question BEFORE retrieval, and the conversation is included in the answer prompt.
def test_follow_up_is_reformulated_and_history_in_prompt():
    # Example: after asking about iPhone replacement, the user asks a bare
    #   follow-up "what about screen damage?". With history, the chain rewrites it
    #   into a full standalone question and searches with THAT (so retrieval works),
    #   and the prior turns appear in the answer prompt.
    retr = FakeTextRetriever()
    llm = ConversationLLM()
    chain = RagChain(retr, embed_fn=None, llm=llm, top_k=3, min_score=0.1)
    history = [
        {"role": "user", "content": "What are the criteria for a replacement iPhone?"},
        {"role": "assistant", "content": "A replacement is provided for manufacturing defects."},
    ]
    result = chain.answer("what about screen damage?", history=history)
    # The follow-up was reformulated and the rewrite was used for retrieval.
    assert llm.condense_calls == 1
    assert retr.last_query == "What is the iPhone replacement policy for screen damage?"
    assert result["diagnostics"]["search_query"] == retr.last_query
    # The prior conversation is present in the prompt sent to the model.
    assert "Conversation so far:" in llm.answer_prompt
    assert "replacement iPhone" in llm.answer_prompt


# No history (a fresh first question) must NOT trigger the extra condense call, and
# retrieval uses the raw question text unchanged.
def test_first_question_skips_reformulation():
    # Example: the very first question has no history, so the chain does not spend
    #   an extra model call reformulating it - it searches with the question as-is.
    retr = FakeTextRetriever()
    llm = ConversationLLM()
    chain = RagChain(retr, embed_fn=None, llm=llm, top_k=3, min_score=0.1)
    chain.answer("What is the returns policy?")
    assert llm.condense_calls == 0
    assert retr.last_query == "What is the returns policy?"
