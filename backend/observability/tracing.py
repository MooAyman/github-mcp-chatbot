"""Best-effort Langfuse tracing for chatbot workflows."""

from __future__ import annotations

import backend.env  # noqa: F401
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:
    from langfuse import Langfuse
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    Langfuse = None

_client: Any | None = None


def _get_client() -> Any | None:
    """Return a configured client, or disable tracing when not configured."""
    global _client
    if _client is not None:
        return _client

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL")
    if not public_key or not secret_key or not base_url or Langfuse is None:
        return None

    try:
        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
        )
    except Exception:
        return None
    return _client


@contextmanager
def observe(
    name: str,
    *,
    as_type: str = "span",
    as_current: bool = True,
    input: Any | None = None,
    model: str | None = None,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> Iterator[Any | None]:
    """Create a Langfuse observation without affecting application behavior."""
    client = _get_client()
    if client is None:
        yield None
        return

    context_metadata = {
        key: value
        for key, value in {"session_id": session_id, "user_id": user_id}.items()
        if value
    }
    started_at = time.perf_counter()
    try:
        if as_current:
            observation_context = client.start_as_current_observation(
                as_type=as_type,
                name=name,
                model=model,
                input=input,
            )
        else:
            observation = client.start_observation(
                as_type=as_type,
                name=name,
                model=model,
                input=input,
            )
    except Exception:
        yield None
        return

    if not as_current:
        combined_metadata = {**(metadata or {}), **context_metadata}
        if combined_metadata:
            try:
                observation.update(metadata=combined_metadata)
            except Exception:
                pass
        try:
            yield observation
        except Exception as exc:
            try:
                observation.update(
                    level="ERROR",
                    status_message=type(exc).__name__,
                    metadata={"error_type": type(exc).__name__},
                )
            except Exception:
                pass
            raise
        finally:
            try:
                observation.update(
                    metadata={
                        "latency_ms": round(
                            (time.perf_counter() - started_at) * 1000,
                            2,
                        )
                    }
                )
            except Exception:
                pass
            try:
                observation.end()
            except Exception:
                pass
        return

    with observation_context as observation:
        combined_metadata = {**(metadata or {}), **context_metadata}
        if combined_metadata:
            try:
                observation.update(metadata=combined_metadata)
            except Exception:
                pass
        try:
            yield observation
        except Exception as exc:
            try:
                observation.update(
                    level="ERROR",
                    status_message=type(exc).__name__,
                    metadata={"error_type": type(exc).__name__},
                )
            except Exception:
                pass
            raise
        finally:
            try:
                observation.update(
                    metadata={
                        "latency_ms": round(
                            (time.perf_counter() - started_at) * 1000,
                            2,
                        )
                    }
                )
            except Exception:
                pass


def update(
    observation: Any | None,
    *,
    output: Any | None = None,
    usage: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Update an observation when Langfuse is enabled."""
    if observation is None:
        return

    values: dict[str, Any] = {}
    if output is not None:
        values["output"] = output
    usage_details = usage_details_from(usage)
    if usage_details:
        values["usage_details"] = usage_details
    if metadata:
        values["metadata"] = metadata
    if values:
        observation.update(**values)


def usage_details_from(usage: Any | None) -> dict[str, int]:
    """Normalize common OpenAI and Gemini usage objects."""
    if usage is None:
        return {}

    values = {
        "input": getattr(usage, "prompt_tokens", None)
        or getattr(usage, "prompt_token_count", None),
        "output": getattr(usage, "completion_tokens", None)
        or getattr(usage, "candidates_token_count", None),
        "total": getattr(usage, "total_tokens", None)
        or getattr(usage, "total_token_count", None),
    }
    return {key: value for key, value in values.items() if value is not None}


def event(name: str, metadata: dict[str, Any]) -> None:
    """Record a small child event such as a retry or provider fallback."""
    try:
        with observe(name, as_type="span", metadata=metadata):
            pass
    except Exception:
        pass


def flush() -> None:
    """Flush queued traces for short-lived scripts."""
    client = _get_client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass
