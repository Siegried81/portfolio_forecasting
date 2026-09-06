"""
Unit tests for src/llm_client.py — the single seam every LLM call in the app
goes through. Had NO test coverage before this pass, despite being exactly
the kind of "one seam to mock" module the file's own docstring describes as
the point of the design. Priority: the Groq -> Ollama fallback cascade (the
whole reason this module exists) and the token-budget truncation, not the
Groq/Ollama SDK internals themselves.
"""
import dataclasses

import pytest

import src.llm_client as llm_client
from src.llm_client import LLMUnavailableError, chat, truncate_to_token_budget


def _patch_settings(monkeypatch, **overrides):
    """LLM_SETTINGS is a frozen dataclass (config.py) — same pattern already
    used in test_market_data.py / test_macro_data.py for the same reason."""
    fake = dataclasses.replace(llm_client.LLM_SETTINGS, **overrides)
    monkeypatch.setattr(llm_client, "LLM_SETTINGS", fake)


# ---------------------------------------------------------------------------
# chat() — the Groq -> Ollama -> LLMUnavailableError cascade
# ---------------------------------------------------------------------------

def test_chat_returns_groq_result_and_backend_label_when_groq_succeeds(monkeypatch):
    monkeypatch.setattr(llm_client, "_call_groq", lambda messages, temperature, max_tokens: "groq answer")
    monkeypatch.setattr(llm_client, "_call_ollama", lambda *a, **k: pytest.fail("Ollama should not be called"))
    text, backend = chat([{"role": "user", "content": "hi"}])
    assert text == "groq answer"
    assert backend == "groq"


def test_chat_falls_back_to_ollama_when_groq_fails(monkeypatch):
    def _failing_groq(*a, **k):
        raise RuntimeError("Groq down")
    monkeypatch.setattr(llm_client, "_call_groq", _failing_groq)
    monkeypatch.setattr(llm_client, "_call_ollama", lambda messages, temperature, max_tokens: "ollama answer")
    text, backend = chat([{"role": "user", "content": "hi"}])
    assert text == "ollama answer"
    assert backend == "ollama (local fallback)"


def test_chat_raises_llm_unavailable_when_both_backends_fail(monkeypatch):
    monkeypatch.setattr(llm_client, "_call_groq", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Groq down")))
    monkeypatch.setattr(llm_client, "_call_ollama", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Ollama down")))
    with pytest.raises(LLMUnavailableError, match="Both LLM backends failed"):
        chat([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# _call_groq — key rotation on rate limit, fail-fast on other errors
# ---------------------------------------------------------------------------

def test_call_groq_raises_llm_unavailable_with_no_keys_configured(monkeypatch):
    _patch_settings(monkeypatch, groq_api_keys=[])
    with pytest.raises(LLMUnavailableError, match="No GROQ_API_KEY"):
        llm_client._call_groq([{"role": "user", "content": "hi"}], temperature=0.3, max_tokens=100)


# ---------------------------------------------------------------------------
# truncate_to_token_budget / _count_tokens
# ---------------------------------------------------------------------------

def test_truncate_to_token_budget_leaves_short_text_unchanged():
    short_text = "This is a short sentence."
    assert truncate_to_token_budget(short_text, max_tokens=1000) == short_text


def test_truncate_to_token_budget_shortens_long_text_and_marks_it():
    long_text = "word " * 5000
    truncated = truncate_to_token_budget(long_text, max_tokens=50)
    assert len(truncated) < len(long_text)
    assert truncated.endswith("[...truncated...]")


def test_count_tokens_falls_back_to_char_heuristic_when_tiktoken_unavailable(monkeypatch):
    def _broken_get_encoding(name):
        raise RuntimeError("no network access to fetch encoding")
    monkeypatch.setattr(llm_client.tiktoken, "get_encoding", _broken_get_encoding)
    # Fallback is len(text) // 4 — must not raise, must return a plausible count.
    assert llm_client._count_tokens("a" * 40) == 10


def test_truncate_to_token_budget_never_raises_when_tiktoken_unavailable(monkeypatch):
    def _broken_get_encoding(name):
        raise RuntimeError("no network access to fetch encoding")
    monkeypatch.setattr(llm_client.tiktoken, "get_encoding", _broken_get_encoding)
    # Short text: _count_tokens falls back cleanly, short-text branch taken.
    assert truncate_to_token_budget("short text", max_tokens=1000) == "short text"


def test_truncate_to_token_budget_degrades_to_char_truncation_when_encoding_unavailable(monkeypatch):
    # Regression test: the truncation branch itself used to call
    # tiktoken.get_encoding() unguarded and crashed with an uncaught
    # HTTPError when the encoding couldn't be fetched (confirmed live in
    # this project's own sandboxed network — openaipublic.blob.core.windows.net
    # is not on the allowed-domains list). Long text forces the truncation
    # branch to actually run.
    def _broken_get_encoding(name):
        raise RuntimeError("no network access to fetch encoding")
    monkeypatch.setattr(llm_client.tiktoken, "get_encoding", _broken_get_encoding)
    long_text = "word " * 5000
    result = truncate_to_token_budget(long_text, max_tokens=50)  # must not raise
    assert result.endswith("[...truncated...]")
    assert len(result) < len(long_text)