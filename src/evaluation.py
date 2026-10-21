"""Evaluation helpers for retrieval and grounding (pure functions).

Measures retrieval quality (Recall@K), whether an answer is grounded in the
retrieved text, and citation correctness. No external services required.

Plain English: these functions score how good the search and the answers are.
"""

from __future__ import annotations

# re to find words/citations; Iterable is a type hint for "a sequence of items".
import re
from typing import Iterable

# Pattern that matches a "word" (letters/digits) used to compare texts.
_WORD_RE = re.compile(r"[a-z0-9]+")


# Turn text into a set of lowercase words (for overlap comparisons).
def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall((text or "").lower()))


# COMMAND ----------
# Recall@K: of the chunks that SHOULD be found, how many are in the top-k results.
def recall_at_k(retrieved_ids: list[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of relevant ids found within the top-k retrieved ids."""
    relevant = set(relevant_ids)                 # the ids we expected to retrieve
    if not relevant:                             # nothing expected -> score 0
        return 0.0
    topk = set(retrieved_ids[:k])                # the first k retrieved ids
    return len(relevant & topk) / len(relevant)  # overlap / expected count


# COMMAND ----------
# Groundedness: how much of the answer's wording actually appears in the context.
def groundedness(answer: str, chunks: list[dict], threshold: float = 0.5) -> float:
    """Share of answer content words that also appear in the retrieved context.

    A cheap, dependency-free proxy for whether the answer is supported by the
    context. Returns a value in [0, 1].
    """
    answer_tokens = _tokens(answer)              # words in the answer
    if not answer_tokens:
        return 0.0
    context_tokens: set[str] = set()             # all words across the chunks
    for c in chunks:
        context_tokens |= _tokens(c.get("chunk_text", ""))
    if not context_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens     # words shared by both
    return len(overlap) / len(answer_tokens)     # fraction of answer words supported


# COMMAND ----------
# Simple yes/no version of groundedness against a threshold.
def is_grounded(answer: str, chunks: list[dict], threshold: float = 0.5) -> bool:
    return groundedness(answer, chunks) >= threshold


# COMMAND ----------
# Citation precision: of the ids the answer cited, how many were actually retrieved.
def citation_precision(cited_ids: Iterable[str], retrieved_ids: Iterable[str]) -> float:
    """Fraction of cited ids that correspond to actually retrieved chunks."""
    cited = list(cited_ids)
    if not cited:                                # nothing cited -> score 0
        return 0.0
    retrieved = set(retrieved_ids)
    valid = sum(1 for c in cited if c in retrieved)  # count the real ones
    return valid / len(cited)


# COMMAND ----------
# Pull [source-id] style citations out of an answer string.
def extract_citations(answer: str) -> list[str]:
    """Pull [source-id] style citations out of an answer string."""
    # Find everything inside square brackets, e.g. [returns:0:0003].
    return re.findall(r"\[([^\[\]]+)\]", answer or "")
