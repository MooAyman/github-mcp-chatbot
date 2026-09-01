"""OpenAI LLM provider."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from backend.observability import observe, update
from backend.reliability import call_with_retry, iter_with_retry


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def _model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def generate_reply(message: str) -> str:
    """Send a user message to OpenAI and return the full assistant reply."""

    def _call() -> Any:
        return _client().chat.completions.create(
            model=_model(),
            messages=[{"role": "user", "content": message}],
        )

    with observe(
        "openai-chat-completion",
        as_type="generation",
        input=message,
        model=_model(),
    ) as generation:
        completion = call_with_retry(_call)
        content = completion.choices[0].message.content or ""
        update(generation, output=content, usage=completion.usage)
        return content


def stream_reply(message: str) -> Iterator[str]:
    """Yield assistant reply tokens from OpenAI as they arrive."""

    def _factory() -> Iterator[str]:
        with observe(
            "openai-chat-stream",
            as_type="generation",
            as_current=False,
            input=message,
            model=_model(),
        ) as generation:
            stream = _client().chat.completions.create(
                model=_model(),
                messages=[{"role": "user", "content": message}],
                stream=True,
                stream_options={"include_usage": True},
            )
            output: list[str] = []
            usage = None
            for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    output.append(delta)
                    yield delta
            update(generation, output="".join(output), usage=usage)

    yield from iter_with_retry(_factory)
