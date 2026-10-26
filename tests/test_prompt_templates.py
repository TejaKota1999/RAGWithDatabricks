"""Unit tests for src/prompt_templates.py.

Checks how the prompt is built: retrieved chunks are tagged with their source
ids, the context respects a size budget, the final prompt contains the question
and context, and the system prompt enforces grounding + abstention.
"""

# The prompt pieces under test.
from src.serving.prompt_templates import (
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
    # Example: two chunks -> a context block like
    #   "[returns:0:0000]\nReturns within 30 days.\n\n[leave:0:0001]\n20 days paid leave."
    #   (each chunk's text is labelled with its id so the answer can cite it).
    ctx = format_context(_chunks())
    assert "[returns:0:0000]" in ctx
    assert "Returns within 30 days." in ctx
    assert "[leave:0:0001]" in ctx


# With a small character budget, only the first chunk should fit in the context.
def test_format_context_respects_max_chars():
    # Example: max_chars=30 is only big enough for the first chunk, so the
    #   second ("leave:0:0001") is left out to keep the prompt within budget.
    ctx = format_context(_chunks(), max_chars=30)
    assert "returns:0:0000" in ctx
    assert "leave:0:0001" not in ctx


# The final user prompt must include BOTH the question and the retrieved context.
def test_build_user_prompt_contains_question_and_context():
    # Example: question "What is the returns policy?" + the two chunks ->
    #   one prompt string that contains the question AND "Returns within 30 days."
    prompt = build_user_prompt("What is the returns policy?", _chunks())
    assert "What is the returns policy?" in prompt
    assert "Returns within 30 days." in prompt


# The system prompt must tell the model to answer ONLY from context and how to abstain.
def test_system_prompt_enforces_grounding_and_abstention():
    # Example: the SYSTEM_PROMPT text includes the word "only" (use only the
    #   context) and the exact ABSTAIN_MESSAGE ("I cannot answer that...").
    assert "only" in SYSTEM_PROMPT.lower()
    assert ABSTAIN_MESSAGE in SYSTEM_PROMPT
