"""RAG orchestration: retrieve -> threshold -> abstain/answer -> sources.

Dependencies are injected (retriever, embed function, LLM), so the chain is
unit-testable without a real model or server.

This is the "brain" that ties everything together: it turns a question into a
grounded answer by (1) embedding the question, (2) retrieving relevant chunks,
(3) deciding whether there is enough evidence, and (4) asking the LLM to answer.
"""

from __future__ import annotations

# Callable = "a function"; Protocol lets us describe the shape of injected objects.
from typing import Callable, Protocol

# Prompt text and the standard "I can't answer" message come from prompt_templates.
from src.prompt_templates import ABSTAIN_MESSAGE, SYSTEM_PROMPT, build_user_prompt


# Describes any retriever we accept: it must have a .search(...) method.
class _Retriever(Protocol):
    def search(self, query_vector, top_k: int, min_score: float | None) -> list[dict]: ...


# Describes any LLM we accept: it must have a .generate(...) method.
class _LLM(Protocol):
    def generate(self, prompt: str, system: str | None = None) -> str: ...


# Which metadata fields we expose as "sources" with each answer.
_SOURCE_FIELDS = ("chunk_id", "document_id", "section", "page_number", "source_path", "score")


# COMMAND ----------
# The chain object: created once with its parts, then used to answer questions.
class RagChain:
    # Store the injected pieces and settings. embed_fn turns text into a vector.
    def __init__(
        self,
        retriever: _Retriever,
        embed_fn: Callable[[str], list[float]],
        llm: _LLM,
        top_k: int = 5,                       # how many chunks to retrieve
        min_score: float | None = 0.3,        # minimum similarity to count as relevant
    ):
        self.retriever = retriever
        self.embed_fn = embed_fn
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score

    # Helper: pick only the "source" fields out of each retrieved chunk.
    @staticmethod
    def _sources(hits: list[dict]) -> list[dict]:
        return [{k: h.get(k) for k in _SOURCE_FIELDS if k in h} for h in hits]

    # The main method: take a question, return a structured answer dictionary.
    def answer(self, question: str) -> dict:
        """Return {answer, sources, abstained, diagnostics}."""
        # Reject empty/blank questions up front.
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        # 1) Turn the question into a vector using the same model as the documents.
        query_vector = self.embed_fn(question)
        # 2) Retrieve the most similar chunks above the score threshold.
        hits = self.retriever.search(query_vector, top_k=self.top_k, min_score=self.min_score)

        # Diagnostics are extra info for debugging (how many hits, best score, etc.).
        diagnostics = {
            "retrieved": len(hits),
            "top_k": self.top_k,
            "min_score": self.min_score,
            "top_score": hits[0].get("score") if hits else None,
        }

        # 3) No relevant evidence -> abstain instead of guessing.
        if not hits:
            return {
                "answer": ABSTAIN_MESSAGE,
                "sources": [],
                "abstained": True,
                "diagnostics": diagnostics,
            }

        # 4) Build the prompt (question + retrieved context) and call the LLM.
        prompt = build_user_prompt(question, hits)
        try:
            text = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        except Exception as exc:  # never leak stack traces to the user
            # If the LLM fails, return a friendly message but still show sources.
            return {
                "answer": "The answer service is unavailable right now. Please try again later.",
                "sources": self._sources(hits),
                "abstained": False,
                "error": type(exc).__name__,
                "diagnostics": diagnostics,
            }

        # Success: return the answer text plus the sources it was based on.
        return {
            "answer": text or ABSTAIN_MESSAGE,   # if the model returned nothing, abstain
            "sources": self._sources(hits),
            "abstained": False,
            "diagnostics": diagnostics,
        }
