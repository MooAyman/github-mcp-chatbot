"""Gemini LLM provider (fallback)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from google import genai
from google.genai import types

from backend.observability import observe, update
from backend.reliability import call_with_retry, iter_with_retry


def _client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


def _model_name() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def generate_reply(message: str) -> str:
    """Send a user message to Gemini and return the full assistant reply."""

    def _call() -> Any:
        response = _client().models.generate_content(
            model=_model_name(),
            contents=message,
        )
        return response

    with observe(
        "gemini-content-generation",
        as_type="generation",
        input=message,
        model=_model_name(),
    ) as generation:
        response = call_with_retry(_call)
        content = getattr(response, "text", None) or ""
        update(
            generation,
            output=content,
            usage=getattr(response, "usage_metadata", None),
        )
        return content


def stream_reply(message: str) -> Iterator[str]:
    """Yield assistant reply tokens from Gemini as they arrive."""

    def _factory() -> Iterator[str]:
        with observe(
            "gemini-content-stream",
            as_type="generation",
            as_current=False,
            input=message,
            model=_model_name(),
        ) as generation:
            client = _client()
            stream = client.models.generate_content_stream(
                model=_model_name(),
                contents=message,
            )
            output: list[str] = []
            usage = None
            for chunk in stream:
                usage = getattr(chunk, "usage_metadata", None) or usage
                try:
                    text = chunk.text
                except (ValueError, AttributeError):
                    continue
                if text:
                    output.append(text)
                    yield text
            update(
                generation,
                output="".join(output),
                usage=usage,
            )

    yield from iter_with_retry(_factory)


def stream_tool_reply(
    contents: Any,
    tools: list[Any],
    *,
    system_instruction: str | None = None,
) -> Iterator[Any]:
    """Stream Gemini responses while exposing MCP function declarations."""

    def _factory() -> Iterator[Any]:
        with observe(
            "gemini-content-stream",
            as_type="generation",
            as_current=False,
            input=contents,
            model=_model_name(),
        ) as generation:
            client = _client()
            declarations = [
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description or "",
                    parameters_json_schema=tool.inputSchema,
                )
                for tool in tools
            ]
            config_values: dict[str, Any] = {}
            if declarations:
                config_values["tools"] = [
                    types.Tool(function_declarations=declarations)
                ]
            else:
                config_values["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.NONE
                    )
                )
            if system_instruction:
                config_values["system_instruction"] = system_instruction
            config = types.GenerateContentConfig(**config_values)
            stream = client.models.generate_content_stream(
                model=_model_name(),
                contents=contents,
                config=config,
            )
            output: list[str] = []
            usage = None
            for chunk in stream:
                usage = getattr(chunk, "usage_metadata", None) or usage
                try:
                    text = chunk.text
                except (ValueError, AttributeError):
                    text = None
                if text:
                    output.append(text)
                yield chunk
            update(generation, output="".join(output), usage=usage)

    yield from iter_with_retry(_factory)
