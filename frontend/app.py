"""Chainlit frontend — sends messages to the FastAPI backend."""

import os

import chainlit as cl
import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


@cl.on_message
async def on_message(message: cl.Message) -> None:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(
                f"{BACKEND_URL}/chat",
                json={"message": message.content},
            )
            res.raise_for_status()
            data = res.json()
            reply = data.get("response", "No response from backend.")
    except httpx.HTTPError as exc:
        reply = f"Backend error: {exc}"

    await cl.Message(content=reply).send()
