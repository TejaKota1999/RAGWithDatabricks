"""Unit tests for src/prompt_templates.py.

Checks how the prompt is built: retrieved chunks are tagged with their source
ids, the context respects a size budget, the final prompt contains the question
and context, and the system prompt enforces grounding + abstention.
"""

# The prompt pieces under test.
from src.prompt_templates import (
    ABSTAIN_MESSAGE,
    SYSTEM_PROMPT,
    build_user_prompt,
    format_context,
)


# Helper: two sample retrieved chunks to build prompts from.
def _chunks():
    return [
        {"chunk_id": "returns:0:0000", "chunk_text": "Returns within 30 days."},
        {"chunk_id": "leave:0:0001", "chunk_text": "20 days paid leave."},
    ]


# The context block should tag each chunk's text with its source id in [brackets].
def test_format_context_tags_sources():
    ctx = format_context(_chunks())
    assert "[returns:0:0000]" in ctx             # first source id tag present
    assert "Returns within 30 days." in ctx      # its text present
    assert "[leave:0:0001]" in ctx               # second source id tag present


# With a small character budget, only the first chunk should fit in the context.
def test_format_context_respects_max_chars():
    ctx = format_context(_chunks(), max_chars=30)
    assert "returns:0:0000" in ctx               # first block kept
    assert "leave:0:0001" not in ctx             # second block dropped (over budget)


# The final user prompt must include BOTH the question and the retrieved context.
def test_build_user_prompt_contains_question_and_context():
    prompt = build_user_prompt("What is the returns policy?", _chunks())
    assert "What is the returns policy?" in prompt   # the question
    assert "Returns within 30 days." in prompt       # the context


# The system prompt must tell the model to answer ONLY from context and how to abstain.
def test_system_prompt_enforces_grounding_and_abstention():
    assert "only" in SYSTEM_PROMPT.lower()           # "use only the context" rule
    assert ABSTAIN_MESSAGE in SYSTEM_PROMPT           # the exact abstain wording is included
