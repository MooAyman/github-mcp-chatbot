"""Retry helpers with exponential backoff for transient failures."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import ParamSpec, TypeVar

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from backend.observability import event

P = ParamSpec("P")
T = TypeVar("T")

MAX_ATTEMPTS = 3
BASE_DELAY_SECONDS = 1.0


def is_retryable(exc: BaseException) -> bool:
    """Return True for temporary LLM/network failures only."""
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError) and 500 <= exc.status_code < 600:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, httpx.RequestError)):
        return True

    try:
        from google.genai import errors as genai_errors
    except ImportError:
        genai_errors = None

    try:
        from google.api_core import exceptions as google_exceptions
    except ImportError:
        google_exceptions = None

    if genai_errors is not None and isinstance(exc, genai_errors.ClientError):
        return exc.code == 429 or 500 <= exc.code < 600
    if genai_errors is not None and isinstance(exc, genai_errors.ServerError):
        return True
    if google_exceptions is None:
        return False

    return isinstance(
        exc,
        (
            google_exceptions.TooManyRequests,
            google_exceptions.ServiceUnavailable,
            google_exceptions.DeadlineExceeded,
            google_exceptions.InternalServerError,
            google_exceptions.BadGateway,
            google_exceptions.GatewayTimeout,
        ),
    )


def is_fallback_eligible(exc: BaseException) -> bool:
    """Return True when OpenAI should be replaced by the fallback provider."""
    if is_retryable(exc) or isinstance(exc, AuthenticationError):
        return True
    return isinstance(exc, ValueError) and str(exc) == "OPENAI_API_KEY is not set"


def call_with_retry(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Call ``fn`` up to MAX_ATTEMPTS times with exponential backoff."""
    delay = BASE_DELAY_SECONDS
    last_exc: BaseException | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not is_retryable(exc) or attempt == MAX_ATTEMPTS:
                raise
            event(
                "llm-retry",
                {
                    "attempt": attempt,
                    "next_attempt": attempt + 1,
                    "error_type": type(exc).__name__,
                    "delay_seconds": delay,
                },
            )
            time.sleep(delay)
            delay *= 2

    assert last_exc is not None
    raise last_exc


def iter_with_retry(factory: Callable[[], Iterator[T]]) -> Iterator[T]:
    """Retry creating/consuming an iterator until the first item is produced.

    Once any item has been yielded, failures are not retried (avoids duplicate
    partial streams).
    """
    delay = BASE_DELAY_SECONDS

    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = False
        try:
            for item in factory():
                started = True
                yield item
            return
        except Exception as exc:
            if started or not is_retryable(exc) or attempt == MAX_ATTEMPTS:
                raise
            event(
                "llm-stream-retry",
                {
                    "attempt": attempt,
                    "next_attempt": attempt + 1,
                    "error_type": type(exc).__name__,
                    "delay_seconds": delay,
                },
            )
            time.sleep(delay)
            delay *= 2
