"""RAG orchestration: retrieve -> threshold -> abstain/answer -> sources.

Dependencies are injected (retriever, embed function, LLM), so the chain is
unit-testable without a real model or server.

This is the "brain" that ties everything together: it turns a question into a
grounded answer by (1) embedding the question, (2) retrieving relevant chunks,
(3) deciding whether there is enough evidence, and (4) asking the LLM to answer.
"""

from __future__ import annotations

# time: measure how long each stage takes (shown in the app's "How it works" tab).
import time

# Callable = "a function"; Protocol lets us describe the shape of injected objects.
from typing import Callable, Protocol

# Prompt text and the standard "I can't answer" message come from prompt_templates.
from src.serving.prompt_templates import (
    ABSTAIN_MESSAGE,
    CONDENSE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_condense_prompt,
    build_user_prompt,
)

# How many recent turns of history to use (keeps prompts small and quota tiny).
_MAX_HISTORY_TURNS = 6


# Describes any retriever we accept: it must have a .search(...) method.
class _Retriever(Protocol):
    def search(self, query_vector, top_k: int, min_score: float | None) -> list[dict]: ...


# Describes any LLM we accept: it must have a .generate(...) method.
class _LLM(Protocol):
    def generate(self, prompt: str, system: str | None = None) -> str: ...


# Milliseconds elapsed since a time.perf_counter() reading, rounded.
def _ms_since(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


# Which metadata fields we expose as "sources" with each answer.
_SOURCE_FIELDS = ("chunk_id", "document_id", "section", "page_number", "source_path", "score")


# COMMAND ----------
# The chain object: created once with its parts, then used to answer questions.
class RagChain:
    # Store the injected pieces and settings. embed_fn turns text into a vector.
    def __init__(
        self,
        retriever: _Retriever,
        embed_fn: Callable[[str], list[float]] | None,
        llm: _LLM,
        top_k: int = 5,                       # how many chunks to retrieve
        min_score: float | None = 0.3,        # minimum similarity to count as relevant
    ):
        # embed_fn is OPTIONAL: pass a function for self-managed indexes (we embed
        # the question ourselves), or None for a managed-embeddings index that
        # embeds the question itself (we then pass the question text to search).
        self.retriever = retriever
        self.embed_fn = embed_fn
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score

    # Helper: pick only the "source" fields out of each retrieved chunk.
    @staticmethod
    def _sources(hits: list[dict]) -> list[dict]:
        return [{k: h.get(k) for k in _SOURCE_FIELDS if k in h} for h in hits]

    # Rewrite a follow-up into a standalone question using the recent history, so
    # retrieval finds the right chunks. Only runs when there IS history (the first
    # question has none), so it adds no extra model call to a fresh conversation.
    # On any failure it safely falls back to the original question.
    def _condense(self, question: str, history: list[dict]) -> str:
        recent = history[-_MAX_HISTORY_TURNS:]
        try:
            rewritten = self.llm.generate(
                build_condense_prompt(recent, question),
                system=CONDENSE_SYSTEM_PROMPT,
            )
        except Exception:
            return question
        rewritten = (rewritten or "").strip()
        return rewritten or question

    # The main method: take a question (+ optional prior turns) and return a
    # structured answer dictionary.
    def answer(self, question: str, history: list[dict] | None = None) -> dict:
        """Return {answer, sources, abstained, diagnostics}.

        `history` is the prior conversation as [{role, content}, ...] (most recent
        last), excluding the current question. When present, the follow-up is
        reformulated into a standalone search query and the conversation is given
        to the model so it can answer "based on the context response".
        """
        # Reject empty/blank questions up front.
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        # Keep only the most recent turns (token/quota safety).
        history = (history or [])[-_MAX_HISTORY_TURNS:]

        # 0) If this is a follow-up, rewrite it into a standalone question so
        #    retrieval (which has no memory) searches for the right thing.
        # Per-stage timings in milliseconds (None = stage did not run).
        timings = {"condense_ms": None, "retrieve_ms": None, "generate_ms": None}
        t0 = time.perf_counter()
        search_question = self._condense(question, history) if history else question
        if history:
            timings["condense_ms"] = _ms_since(t0)

        # 1) Build the query the retriever expects: a VECTOR when we have an
        #    embed_fn (self-managed index), or the raw QUESTION TEXT when there is
        #    no embed_fn (managed-embeddings index embeds the text itself).
        t0 = time.perf_counter()
        query = self.embed_fn(search_question) if self.embed_fn else search_question
        # 2) Retrieve the most relevant chunks above the score threshold.
        hits = self.retriever.search(query, top_k=self.top_k, min_score=self.min_score)
        timings["retrieve_ms"] = _ms_since(t0)

        # Diagnostics are extra info for debugging (how many hits, best score, etc.).
        diagnostics = {
            "retrieved": len(hits),
            "top_k": self.top_k,
            "min_score": self.min_score,
            "top_score": hits[0].get("score") if hits else None,
            # The actual query used for retrieval (differs from `question` only when
            # a follow-up was reformulated); handy for debugging conversational memory.
            "search_query": search_question,
            # True when a follow-up was rewritten by the condense step.
            "condensed": bool(history),
            "timings": timings,
        }

        # 3) No relevant evidence -> abstain instead of guessing.
        if not hits:
            return {
                "answer": ABSTAIN_MESSAGE,
                "sources": [],
                "abstained": True,
                "diagnostics": diagnostics,
            }

        # 4) Build the prompt (recent conversation + question + retrieved context)
        #    and call the LLM. Passing the history lets the model answer a follow-up
        #    in light of what was said before, while still grounding on the context.
        prompt = build_user_prompt(question, hits, history=history)
        t0 = time.perf_counter()
        try:
            text = self.llm.generate(prompt, system=SYSTEM_PROMPT)
            timings["generate_ms"] = _ms_since(t0)
        except Exception as exc:  # never leak stack traces to the user
            timings["generate_ms"] = _ms_since(t0)
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
