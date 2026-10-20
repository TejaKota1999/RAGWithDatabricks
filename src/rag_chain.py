"""RAG orchestration: retrieve -> threshold -> abstain/answer -> sources.

Dependencies are injected (retriever, embed function, LLM), so the chain is
unit-testable without a real model or server.
"""

from __future__ import annotations

from typing import Callable, Protocol

from src.prompt_templates import ABSTAIN_MESSAGE, SYSTEM_PROMPT, build_user_prompt


class _Retriever(Protocol):
    def search(self, query_vector, top_k: int, min_score: float | None) -> list[dict]: ...


class _LLM(Protocol):
    def generate(self, prompt: str, system: str | None = None) -> str: ...


# Fields returned as sources for each supporting chunk.
_SOURCE_FIELDS = ("chunk_id", "document_id", "section", "page_number", "source_path", "score")


# %%
class RagChain:
    def __init__(
        self,
        retriever: _Retriever,
        embed_fn: Callable[[str], list[float]],
        llm: _LLM,
        top_k: int = 5,
        min_score: float | None = 0.3,
    ):
        self.retriever = retriever
        self.embed_fn = embed_fn
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score

    @staticmethod
    def _sources(hits: list[dict]) -> list[dict]:
        return [{k: h.get(k) for k in _SOURCE_FIELDS if k in h} for h in hits]

    def answer(self, question: str) -> dict:
        """Return {answer, sources, abstained, diagnostics}."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        query_vector = self.embed_fn(question)
        hits = self.retriever.search(query_vector, top_k=self.top_k, min_score=self.min_score)

        diagnostics = {
            "retrieved": len(hits),
            "top_k": self.top_k,
            "min_score": self.min_score,
            "top_score": hits[0].get("score") if hits else None,
        }

        if not hits:
            return {
                "answer": ABSTAIN_MESSAGE,
                "sources": [],
                "abstained": True,
                "diagnostics": diagnostics,
            }

        prompt = build_user_prompt(question, hits)
        try:
            text = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        except Exception as exc:  # never leak stack traces to the user
            return {
                "answer": "The answer service is unavailable right now. Please try again later.",
                "sources": self._sources(hits),
                "abstained": False,
                "error": type(exc).__name__,
                "diagnostics": diagnostics,
            }

        return {
            "answer": text or ABSTAIN_MESSAGE,
            "sources": self._sources(hits),
            "abstained": False,
            "diagnostics": diagnostics,
        }
