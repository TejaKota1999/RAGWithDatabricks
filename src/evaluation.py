# Databricks notebook source
"""Evaluation helpers for retrieval and grounding (pure functions).

Measures retrieval quality (Recall@K), whether an answer is grounded in the
retrieved text, and citation correctness. No external services required.
"""

from __future__ import annotations

import re
from typing import Iterable

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall((text or "").lower()))


# COMMAND ----------
def recall_at_k(retrieved_ids: list[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of relevant ids found within the top-k retrieved ids."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    topk = set(retrieved_ids[:k])
    return len(relevant & topk) / len(relevant)


# COMMAND ----------
def groundedness(answer: str, chunks: list[dict], threshold: float = 0.5) -> float:
    """Share of answer content words that also appear in the retrieved context.

    A cheap, dependency-free proxy for whether the answer is supported by the
    context. Returns a value in [0, 1].
    """
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 0.0
    context_tokens: set[str] = set()
    for c in chunks:
        context_tokens |= _tokens(c.get("chunk_text", ""))
    if not context_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens)


# COMMAND ----------
def is_grounded(answer: str, chunks: list[dict], threshold: float = 0.5) -> bool:
    return groundedness(answer, chunks) >= threshold


# COMMAND ----------
def citation_precision(cited_ids: Iterable[str], retrieved_ids: Iterable[str]) -> float:
    """Fraction of cited ids that correspond to actually retrieved chunks."""
    cited = list(cited_ids)
    if not cited:
        return 0.0
    retrieved = set(retrieved_ids)
    valid = sum(1 for c in cited if c in retrieved)
    return valid / len(cited)


# COMMAND ----------
def extract_citations(answer: str) -> list[str]:
    """Pull [source-id] style citations out of an answer string."""
    return re.findall(r"\[([^\[\]]+)\]", answer or "")
