"""Chainlit frontend connected to the FastAPI agent workflow."""

import json
import os

import chainlit as cl
import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


async def _request_chat(
    client: httpx.AsyncClient,
    payload: dict[str, object],
    *,
    show_response: bool = True,
) -> dict[str, object] | None:
    """Stream a normal response or return an approval request."""
    async with client.stream(
        "POST",
        f"{BACKEND_URL}/chat",
        json=payload,
    ) as response:
        response.raise_for_status()
        if response.headers.get("content-type", "").startswith("application/json"):
            body = await response.aread()
            return json.loads(body)

        if not show_response:
            await response.aread()
            return None

        reply = cl.Message(content="")
        await reply.send()
        async for chunk in response.aiter_text():
            if chunk:
                await reply.stream_token(chunk)
        await reply.update()
        return None


@cl.on_message
async def on_message(message: cl.Message) -> None:
    session = getattr(cl.context, "session", None)
    session_id = getattr(session, "id", "default")
    payload = {"message": message.content, "session_id": session_id}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            approval_request = await _request_chat(client, payload)
            if not approval_request or not approval_request.get("approval_required"):
                return

            tool = approval_request.get("tool") or {}
            approval = await cl.AskActionMessage(
                content=(
                    f"Approval required for `{tool.get('name', 'GitHub tool')}`.\n\n"
                    f"Arguments:\n```json\n"
                    f"{json.dumps(tool.get('arguments', {}), indent=2)}\n```"
                ),
                actions=[
                    cl.Action(name="approve", payload={}, label="Proceed"),
                    cl.Action(name="reject", payload={}, label="Reject"),
                ],
            ).send()
            approval_value = (
                approval.get("name") if approval else "reject"
            )
            await _request_chat(
                client,
                {**payload, "approval": approval_value},
                show_response=approval_value == "approve",
            )
            if approval_value == "reject":
                await cl.Message(content="Operation cancelled.").send()
    except httpx.HTTPError:
        await cl.Message(
            content="The request could not be completed. Please try again."
        ).send()
