"""
Unified LLM client: Groq (hosted, fast, primary) with automatic fallback to a
local Ollama instance if Groq is unreachable, unauthenticated, or rate-limited.

This is the ONLY place in the codebase that talks to an LLM provider — every
other module calls `chat()` and doesn't know or care which backend answered.
That's what makes the fallback possible without duplicating prompt logic, and
it's what "LLM-friendly code" means in practice: one seam to swap providers,
mock in tests, or add a third backend later.
"""
from __future__ import annotations

import logging

import requests
import tiktoken

from src.config import LLM_SETTINGS, MAX_CONTEXT_TOKENS

logger = logging.getLogger(__name__)

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


class LLMUnavailableError(RuntimeError):
    """Raised only if BOTH Groq and Ollama fail — lets the UI show one clear message
    instead of a stack trace, without silently pretending everything is fine."""


def _count_tokens(text: str) -> int:
    """Approximate token count using OpenAI's cl100k_base encoding — close enough
    across providers for the purpose of staying under a context budget (we don't
    need exact provider-specific tokenisation here, just a safety margin)."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4  # crude fallback if tiktoken's encoding download fails offline


def truncate_to_token_budget(text: str, max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    """Hard-truncate free-text context (news articles, etc.) to a token budget
    before it's stuffed into a prompt, so a chatty NewsAPI response never blows
    past a small local model's context window."""
    if _count_tokens(text) <= max_tokens:
        return text
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)[:max_tokens]
    return encoding.decode(tokens) + "\n[...truncated...]"


def _call_groq(messages: list[Message], temperature: float, max_tokens: int) -> str:
    if not LLM_SETTINGS.groq_api_keys:
        raise LLMUnavailableError("No GROQ_API_KEY configured.")
    from groq import Groq, RateLimitError  # imported lazily — package is optional if only Ollama is used

    last_error: Exception | None = None
    for i, key in enumerate(LLM_SETTINGS.groq_api_keys):
        try:
            client = Groq(api_key=key)
            response = client.chat.completions.create(
                model=LLM_SETTINGS.groq_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except RateLimitError as exc:
            # This specific key is exhausted (429) — rotate to the next one, if any.
            logger.warning("Groq key #%d rate-limited, trying next key: %s", i + 1, exc)
            last_error = exc
            continue
        except Exception as exc:
            # Any OTHER error (bad key, model deprecated, network) affects every key
            # equally — rotating won't help, so fail fast to the Ollama fallback
            # instead of burning time cycling through all 5 keys for nothing.
            raise
    raise LLMUnavailableError(f"All {len(LLM_SETTINGS.groq_api_keys)} Groq keys are rate-limited: {last_error}")


def _call_ollama(messages: list[Message], temperature: float, max_tokens: int) -> str:
    response = requests.post(
        f"{LLM_SETTINGS.ollama_host}/api/chat",
        json={
            "model": LLM_SETTINGS.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")


def chat(messages: list[Message], temperature: float = 0.3, max_tokens: int = 600) -> tuple[str, str]:
    """
    Send a chat completion request. Tries Groq first; on ANY failure (missing key,
    network error, rate limit, model deprecation) falls back to a local Ollama
    instance. Returns (response_text, backend_used) so the UI can be transparent
    about which model actually answered — useful both for debugging and for
    honesty with the end user about provenance.

    Raises LLMUnavailableError only if both backends fail.
    """
    try:
        return _call_groq(messages, temperature, max_tokens), "groq"
    except Exception as groq_error:
        logger.warning("Groq call failed, falling back to Ollama: %s", groq_error)
        try:
            return _call_ollama(messages, temperature, max_tokens), "ollama (local fallback)"
        except Exception as ollama_error:
            raise LLMUnavailableError(
                f"Both LLM backends failed. Groq: {groq_error} | Ollama: {ollama_error}"
            ) from ollama_error
