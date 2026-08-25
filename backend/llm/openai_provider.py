"""OpenAI LLM provider."""

import os

from openai import OpenAI


def generate_reply(message: str) -> str:
    """Send a user message to OpenAI and return the assistant reply."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}],
    )
    content = completion.choices[0].message.content
    return content or ""
