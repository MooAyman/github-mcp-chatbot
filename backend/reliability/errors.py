"""Provider-independent errors for the LLM layer."""

from __future__ import annotations

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from google.genai import errors as genai_errors


class LLMError(Exception):
    """Safe error that can be returned to the user."""

    status_code = 502
    user_message = "The language model is temporarily unavailable. Please try again."

    def __init__(self, provider: str) -> None:
        super().__init__(self.user_message)
        self.provider = provider


class LLMConfigurationError(LLMError):
    status_code = 500
    user_message = "The language model is not configured correctly."


class LLMTimeoutError(LLMError):
    status_code = 504
    user_message = "The language model took too long to respond. Please try again."


class LLMRateLimitError(LLMError):
    status_code = 429
    user_message = "The language model is busy. Please try again shortly."


class LLMProviderError(LLMError):
    """Safe wrapper for an unexpected provider failure."""


class LLMToolCallError(LLMError):
    """The model requested another tool after the supported tool turn."""

    user_message = (
        "The assistant requested another GitHub operation after the first one. "
        "Please try again."
    )


def normalize_error(exc: BaseException, provider: str) -> LLMError:
    """Convert provider/configuration exceptions to safe LLM errors."""
    if isinstance(exc, LLMError):
        return exc

    if isinstance(exc, ValueError):
        return LLMConfigurationError(provider)

    if isinstance(exc, (APITimeoutError, TimeoutError, httpx.TimeoutException)):
        return LLMTimeoutError(provider)

    if isinstance(exc, RateLimitError):
        return LLMRateLimitError(provider)

    if isinstance(exc, AuthenticationError):
        return LLMConfigurationError(provider)

    if isinstance(exc, (APIConnectionError, ConnectionError, httpx.RequestError)):
        return LLMProviderError(provider)

    if isinstance(exc, APIStatusError):
        if exc.status_code == 429:
            return LLMRateLimitError(provider)
        if 500 <= exc.status_code < 600:
            return LLMProviderError(provider)
        return LLMProviderError(provider)

    if isinstance(exc, APIError):
        return LLMProviderError(provider)

    if isinstance(exc, genai_errors.ClientError):
        if exc.code == 429:
            return LLMRateLimitError(provider)
        if exc.code in (401, 403):
            return LLMConfigurationError(provider)
        if exc.code == 408:
            return LLMTimeoutError(provider)
        return LLMProviderError(provider)

    if isinstance(exc, genai_errors.ServerError):
        return LLMProviderError(provider)

    return LLMProviderError(provider)
