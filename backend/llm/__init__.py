"""LLM providers: OpenAI (primary) and Gemini (fallback)."""

from __future__ import annotations

from collections.abc import Iterator

from backend.llm import gemini_provider, openai_provider
from backend.observability import event, observe
from backend.reliability import is_fallback_eligible, normalize_error


def generate_reply(message: str) -> str:
    """Generate a reply via OpenAI, falling back to Gemini on retryable failures."""
    with observe(
        "llm-request",
        input=message,
        metadata={"workflow": "chat", "streaming": False},
    ):
        try:
            return openai_provider.generate_reply(message)
        except Exception as exc:
            if not is_fallback_eligible(exc):
                raise normalize_error(exc, "OpenAI") from exc
            event(
                "llm-fallback",
                {
                    "primary_provider": "OpenAI",
                    "fallback_provider": "Gemini",
                    "reason": type(exc).__name__,
                },
            )
            try:
                return gemini_provider.generate_reply(message)
            except Exception as fallback_exc:
                raise normalize_error(fallback_exc, "Gemini") from fallback_exc


def stream_reply(message: str) -> Iterator[str]:
    """Stream a reply via OpenAI, falling back to Gemini on retryable failures.

    Fallback runs only if OpenAI fails before any token was yielded.
    """
    with observe(
        "llm-request",
        as_current=False,
        input=message,
        metadata={"workflow": "chat", "streaming": True},
    ):
        started = False
        try:
            for token in openai_provider.stream_reply(message):
                started = True
                yield token
        except Exception as exc:
            if started or not is_fallback_eligible(exc):
                raise normalize_error(exc, "OpenAI") from exc
            event(
                "llm-fallback",
                {
                    "primary_provider": "OpenAI",
                    "fallback_provider": "Gemini",
                    "reason": type(exc).__name__,
                    "streaming": True,
                },
            )
            try:
                yield from gemini_provider.stream_reply(message)
            except Exception as fallback_exc:
                raise normalize_error(fallback_exc, "Gemini") from fallback_exc


__all__ = ["generate_reply", "stream_reply"]
