"""FastAPI application entry point."""

from __future__ import annotations

import backend.env  # noqa: F401

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from backend.agent import AgentDecision, GitHubAgent
from backend.observability import flush
from backend.reliability import LLMError

app = FastAPI(title="GitHub MCP Production Chatbot")
_agents: dict[str, GitHubAgent] = {}
_pending_approvals: dict[str, tuple[str, AgentDecision]] = {}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(default="default", min_length=1)
    approval: Literal["approve", "reject"] | None = None


class ToolApproval(BaseModel):
    name: str
    arguments: dict[str, Any]


class ChatResponse(BaseModel):
    response: str = ""
    approval_required: bool = False
    tool: ToolApproval | None = None


def _get_agent(session_id: str) -> GitHubAgent:
    if session_id not in _agents:
        _agents[session_id] = GitHubAgent(session_id=session_id)
    return _agents[session_id]


def _stream_response(response: str) -> StreamingResponse:
    return StreamingResponse(
        iter([response]),
        media_type="text/plain",
        background=BackgroundTask(flush),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=None)
async def chat(request: ChatRequest) -> StreamingResponse | JSONResponse:
    try:
        if request.approval == "reject":
            _pending_approvals.pop(request.session_id, None)
            return _stream_response("Operation cancelled.")

        agent = _get_agent(request.session_id)
        if request.approval == "approve":
            pending = _pending_approvals.pop(request.session_id, None)
            if pending is None:
                return _stream_response("There is no pending operation to approve.")
            message, decision = pending
            response = await agent.complete_decision(
                message,
                decision,
                approved=True,
            )
            return _stream_response(response)

        events = agent.stream_response(request.message)
        try:
            first_event = await anext(events)
        except StopAsyncIteration:
            return _stream_response("")

        if first_event.approval is not None:
            await events.aclose()
            decision = first_event.approval
            _pending_approvals[request.session_id] = (
                request.message,
                decision,
            )
            approval_response = ChatResponse(
                approval_required=True,
                tool=ToolApproval(
                    name=decision.tool.name,
                    arguments=decision.tool.arguments,
                ),
            )
            return JSONResponse(
                content=approval_response.model_dump(),
                background=BackgroundTask(flush),
            )

        async def event_stream():
            try:
                if first_event.text:
                    yield first_event.text
                async for event in events:
                    if event.text:
                        yield event.text
            except LLMError as exc:
                yield f"\n\n{exc.user_message}"
            except Exception:
                yield (
                    "\n\nThe language model is temporarily unavailable. "
                    "Please try again."
                )

        return StreamingResponse(
            event_stream(),
            media_type="text/plain",
            background=BackgroundTask(flush),
        )
    except LLMError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.user_message) from exc
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="The language model is temporarily unavailable. Please try again.",
        )
