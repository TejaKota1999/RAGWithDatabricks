"""Unit tests for src/prompt_templates.py."""

from src.prompt_templates import (
    ABSTAIN_MESSAGE,
    SYSTEM_PROMPT,
    build_user_prompt,
    format_context,
)


def _chunks():
    return [
        {"chunk_id": "returns:0:0000", "chunk_text": "Returns within 30 days."},
        {"chunk_id": "leave:0:0001", "chunk_text": "20 days paid leave."},
    ]


def test_format_context_tags_sources():
    ctx = format_context(_chunks())
    assert "[returns:0:0000]" in ctx
    assert "Returns within 30 days." in ctx
    assert "[leave:0:0001]" in ctx


def test_format_context_respects_max_chars():
    ctx = format_context(_chunks(), max_chars=30)
    # Only the first block fits.
    assert "returns:0:0000" in ctx
    assert "leave:0:0001" not in ctx


def test_build_user_prompt_contains_question_and_context():
    prompt = build_user_prompt("What is the returns policy?", _chunks())
    assert "What is the returns policy?" in prompt
    assert "Returns within 30 days." in prompt


def test_system_prompt_enforces_grounding_and_abstention():
    assert "only" in SYSTEM_PROMPT.lower()
    assert ABSTAIN_MESSAGE in SYSTEM_PROMPT
