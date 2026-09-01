"""Minimal read-only GitHub tool-selection agent."""

from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from google.genai import types

from backend.mcp.github_client import GitHubMCPClient
from backend.observability import event, observe, update
from backend.reliability import (
    LLMToolCallError,
    is_fallback_eligible,
    iter_with_retry,
    normalize_error,
)

GITHUB_SYSTEM_INSTRUCTION = (
    "This is a GitHub assistant using Model Context Protocol (MCP) tools. "
    "For GitHub-related requests, use an available GitHub tool instead "
    "of claiming GitHub access is unavailable or giving manual GitHub UI steps. "
    "Use `search_issues` for issue search or list requests. "
    "Use `issue_write` with method `create` to create issues when the user "
    "asks to create, add, or file an issue. Include repo and title from the "
    "user message; omit optional fields rather than asking the user to create "
    "the issue manually. "
    "Use `get_me` only when the user asks about the authenticated GitHub user. "
    "Do not use `get_me` just because the user says 'my repository'."
)

TOOL_NAME_ALIASES = {
    "create_issue": "issue_write",
}

READ_ONLY_TOOL_NAMES = frozenset(
    {
        "get_commit",
        "get_file_contents",
        "get_label",
        "get_latest_release",
        "get_me",
        "get_release_by_tag",
        "get_tag",
        "get_team_members",
        "get_teams",
        "issue_read",
        "list_branches",
        "list_commits",
        "list_issue_fields",
        "list_issue_types",
        "list_issues",
        "list_pull_requests",
        "list_releases",
        "list_repository_collaborators",
        "list_tags",
        "pull_request_read",
        "search_code",
        "search_commits",
        "search_issues",
        "search_pull_requests",
        "search_repositories",
        "search_users",
    }
)
WRITE_TOOL_NAMES = frozenset(
    {
        "add_comment_to_pending_review",
        "add_issue_comment",
        "add_reply_to_pull_request_comment",
        "assign_copilot_to_issue",
        "create_branch",
        "create_or_update_file",
        "create_pull_request",
        "create_repository",
        "delete_file",
        "fork_repository",
        "issue_write",
        "merge_pull_request",
        "pull_request_review_write",
        "push_files",
        "request_copilot_review",
        "sub_issue_write",
        "update_pull_request",
        "update_pull_request_branch",
    }
)
SUPPORTED_TOOL_NAMES = READ_ONLY_TOOL_NAMES | WRITE_TOOL_NAMES


@dataclass(frozen=True)
class SelectedTool:
    """An MCP tool selected by the LLM."""

    name: str
    arguments: dict[str, Any]
    thought_signature: bytes | None = None
    model_content: types.Content | None = None


@dataclass(frozen=True)
class AgentDecision:
    """The LLM's response and optional tool selection."""

    response: str
    tool: SelectedTool | None = None
    approval_required: bool = False
    tool_call_id: str | None = None
    provider: str = "openai"


@dataclass(frozen=True)
class AgentStreamEvent:
    """A streamed response token or an approval request."""

    text: str | None = None
    approval: AgentDecision | None = None


def _openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def _normalize_tool_name(name: str) -> str:
    return TOOL_NAME_ALIASES.get(name, name)


def _as_openai_tool(tool: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


class GitHubAgent:
    """Connect to GitHub MCP and let OpenAI select a GitHub tool."""

    def __init__(
        self,
        mcp_client: GitHubMCPClient | None = None,
        llm_client: OpenAI | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        self._mcp_client = mcp_client or GitHubMCPClient()
        self._llm_client = llm_client
        self._session_id = session_id
        self._user_id = user_id
        self._tools: list[Any] | None = None
        self._conversation: list[dict[str, Any]] = []

    async def connect(self) -> None:
        """Connect to MCP and cache the available read-only tools."""
        await self._mcp_client.connect()
        tools = await self._mcp_client.list_tools()
        self._tools = [
            tool for tool in tools if tool.name in SUPPORTED_TOOL_NAMES
        ]

    def _messages(self, message: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": GITHUB_SYSTEM_INSTRUCTION,
            },
            *self._conversation,
            {"role": "user", "content": message},
        ]

    def _completion(self, messages: list[dict[str, Any]]) -> Any:
        if self._tools is None:
            raise RuntimeError("GitHub agent is not connected")
        if self._llm_client is None:
            self._llm_client = _openai_client()

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        with observe(
            "agent-tool-selection",
            as_type="generation",
            input=messages[-1]["content"],
            model=model,
            metadata={"workflow": "github-agent", "phase": "tool-selection"},
            session_id=self._session_id,
            user_id=self._user_id,
        ) as generation:
            completion = self._llm_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[
                    _as_openai_tool(tool)
                    for tool in self._tools_for_message(messages[-1]["content"])
                ],
                tool_choice="auto",
            )
            assistant_message = completion.choices[0].message
            update(
                generation,
                output={
                    "response": assistant_message.content or "",
                    "tool": (
                        assistant_message.tool_calls[0].function.name
                        if assistant_message.tool_calls
                        else None
                    ),
                },
                usage=getattr(completion, "usage", None),
            )
            return completion

    def _decision_from_completion(self, completion: Any) -> AgentDecision:
        assistant_message = completion.choices[0].message
        tool_calls = assistant_message.tool_calls or []
        if not tool_calls:
            return AgentDecision(response=assistant_message.content or "")

        selected = tool_calls[0].function
        selected_name = _normalize_tool_name(selected.name)
        allowed_names = {tool.name for tool in self._tools or []}
        if selected_name not in allowed_names:
            return AgentDecision(response=assistant_message.content or "")

        try:
            arguments = json.loads(selected.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}

        return AgentDecision(
            response=assistant_message.content or "",
            tool=SelectedTool(name=selected_name, arguments=arguments),
            approval_required=selected_name in WRITE_TOOL_NAMES,
            tool_call_id=getattr(tool_calls[0], "id", None),
        )

    def _stream_openai(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[Any]:
        if self._llm_client is None:
            self._llm_client = _openai_client()

        def factory() -> Iterator[Any]:
            request: dict[str, Any] = {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if tools is not None:
                request["tools"] = tools
                request["tool_choice"] = "auto"
            stream = self._llm_client.chat.completions.create(**request)
            yield from stream

        yield from iter_with_retry(factory)

    @staticmethod
    def _tool_messages(
        message: str,
        decision: AgentDecision,
        tool_result: Any,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if decision.tool is None:
            return history

        tool_call_id = decision.tool_call_id or "github-tool-call"
        history.extend(
            [
                {
                    "role": "assistant",
                    "content": decision.response or None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": decision.tool.name,
                                "arguments": json.dumps(decision.tool.arguments),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": GitHubAgent._serialize_tool_result(tool_result),
                },
            ]
        )
        return history

    @staticmethod
    def _gemini_tool_contents(
        message: str,
        tool: SelectedTool,
        tool_result: Any,
    ) -> list[types.Content]:
        serialized_result = GitHubAgent._serialize_tool_result(tool_result)
        try:
            result_value = json.loads(serialized_result)
        except json.JSONDecodeError:
            result_value = serialized_result
        if isinstance(result_value, dict) and isinstance(
            result_value.get("content"), list
        ):
            text_parts = [
                item["text"]
                for item in result_value["content"]
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            if len(text_parts) == 1:
                try:
                    result_value = json.loads(text_parts[0])
                except json.JSONDecodeError:
                    result_value = text_parts[0]
        final_prompt = (
            f"The MCP tool `{tool.name}` has already been executed. "
            "Do not call another tool. Answer the user's request using only "
            "the returned MCP result.\n\n"
            f"User request: {message}\n"
            f"MCP result: {json.dumps(result_value, default=str)}"
        )
        return [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=final_prompt)],
            ),
        ]

    @staticmethod
    def _gemini_thought_signature(
        chunk: Any,
        function_call: Any,
    ) -> bytes | None:
        for candidate in getattr(chunk, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                part_call = getattr(part, "function_call", None)
                if getattr(part_call, "name", None) == getattr(
                    function_call,
                    "name",
                    None,
                ):
                    signature = getattr(part, "thought_signature", None)
                    if isinstance(signature, bytes):
                        return signature
        return None

    def _tools_for_message(self, message: str) -> list[Any]:
        """Prefer a single relevant GitHub tool for common issue requests."""
        available_tools = self._tools or []
        normalized = message.lower()
        is_issue_request = bool(re.search(r"\bissues?\b", normalized))
        is_search_or_list_request = any(
            re.search(rf"\b{verb}\b", normalized)
            for verb in ("search", "find", "list", "show")
        ) or (
            re.search(r"\bopen\b", normalized)
            and re.search(r"\bissues\b", normalized)
            and not re.search(r"\bcreate\b", normalized)
        )
        if is_issue_request and is_search_or_list_request:
            issue_tools = [
                tool for tool in available_tools if tool.name == "search_issues"
            ]
            if issue_tools:
                return issue_tools

        is_create_issue_request = is_issue_request and (
            re.search(r"\b(create|add|file)\b", normalized)
            or re.search(r"\bnew issue\b", normalized)
        )
        if is_create_issue_request and not is_search_or_list_request:
            write_tools = [
                tool for tool in available_tools if tool.name == "issue_write"
            ]
            if write_tools:
                return write_tools

        return available_tools

    def _gemini_tools_for_message(self, message: str) -> list[Any]:
        return self._tools_for_message(message)

    async def stream_response(self, message: str) -> AsyncIterator[AgentStreamEvent]:
        """Stream normal replies while preserving MCP and approval decisions."""
        if self._tools is None:
            await self.connect()

        tools = [_as_openai_tool(tool) for tool in self._tools_for_message(message)]
        messages = self._messages(message)
        response_parts: list[str] = []
        tool_calls: dict[int, dict[str, str]] = {}
        emitted_text = False
        usage = None

        try:
            with observe(
                "agent-stream",
                as_type="generation",
                as_current=False,
                input=message,
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                metadata={"workflow": "github-agent", "streaming": True},
                session_id=self._session_id,
                user_id=self._user_id,
            ) as generation:
                for chunk in self._stream_openai(messages, tools):
                    usage = getattr(chunk, "usage", None) or usage
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        emitted_text = True
                        response_parts.append(delta.content)
                        yield AgentStreamEvent(text=delta.content)
                    for tool_delta in delta.tool_calls or []:
                        index = getattr(tool_delta, "index", 0)
                        call = tool_calls.setdefault(
                            index,
                            {"id": "", "name": "", "arguments": ""},
                        )
                        call["id"] += getattr(tool_delta, "id", "") or ""
                        function = getattr(tool_delta, "function", None)
                        if function is not None:
                            call["name"] += getattr(function, "name", "") or ""
                            call["arguments"] += (
                                getattr(function, "arguments", "") or ""
                            )

                if tool_calls:
                    selected = tool_calls[min(tool_calls)]
                    selected_name = _normalize_tool_name(selected["name"])
                    try:
                        arguments = json.loads(selected["arguments"] or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    decision = AgentDecision(
                        response="".join(response_parts),
                        tool=SelectedTool(
                            name=selected_name,
                            arguments=arguments,
                        ),
                        approval_required=selected_name in WRITE_TOOL_NAMES,
                        tool_call_id=selected["id"] or None,
                    )
                    if decision.tool.name not in {
                        tool.name for tool in self._tools or []
                    }:
                        return
                    if decision.approval_required:
                        decision = await self._normalize_decision(decision)
                        yield AgentStreamEvent(approval=decision)
                        return

                    tool_result = await self.execute_tool(decision.tool)
                    final_messages = self._tool_messages(
                        message,
                        decision,
                        tool_result,
                        messages,
                    )
                    final_parts: list[str] = []
                    for chunk in self._stream_openai(final_messages):
                        usage = getattr(chunk, "usage", None) or usage
                        if not chunk.choices:
                            continue
                        content = chunk.choices[0].delta.content
                        if content:
                            final_parts.append(content)
                            yield AgentStreamEvent(text=content)
                    final_response = "".join(final_parts)
                    self._record_tool_turn(
                        message,
                        decision,
                        tool_result,
                        final_response,
                    )
                    update(
                        generation,
                        output=final_response,
                        usage=usage,
                    )
                    return

                final_response = "".join(response_parts)
                self._conversation.extend(
                    [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": final_response},
                    ]
                )
                update(generation, output=final_response, usage=usage)
        except Exception as exc:
            if emitted_text or tool_calls or not is_fallback_eligible(exc):
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
            fallback_parts: list[str] = []
            fallback_tools = self._gemini_tools_for_message(message)
            fallback_function_call: Any | None = None
            fallback_thought_signature: bytes | None = None
            fallback_model_content: types.Content | None = None
            try:
                from backend.llm import gemini_provider

                for chunk in gemini_provider.stream_tool_reply(
                    message,
                    fallback_tools,
                    system_instruction=GITHUB_SYSTEM_INSTRUCTION,
                ):
                    try:
                        text = chunk.text
                    except (ValueError, AttributeError):
                        text = None
                    if text:
                        fallback_parts.append(text)
                    function_calls = getattr(chunk, "function_calls", None) or []
                    if function_calls and fallback_function_call is None:
                        fallback_function_call = function_calls[0]
                        fallback_thought_signature = self._gemini_thought_signature(
                            chunk,
                            fallback_function_call,
                        )
                        for candidate in getattr(chunk, "candidates", None) or []:
                            content = getattr(candidate, "content", None)
                            if content is not None:
                                fallback_model_content = content
                                break
            except Exception as fallback_exc:
                raise normalize_error(fallback_exc, "Gemini") from fallback_exc

            if fallback_function_call is not None:
                arguments = getattr(fallback_function_call, "args", None) or {}
                if not isinstance(arguments, dict):
                    arguments = dict(arguments)
                tool_name = _normalize_tool_name(fallback_function_call.name)
                decision = AgentDecision(
                    response="".join(fallback_parts),
                    tool=SelectedTool(
                        name=tool_name,
                        arguments=arguments,
                        thought_signature=fallback_thought_signature,
                        model_content=fallback_model_content,
                    ),
                    approval_required=(tool_name in WRITE_TOOL_NAMES),
                    provider="gemini",
                )
                if decision.tool.name not in {
                    tool.name for tool in fallback_tools
                }:
                    raise LLMToolCallError("Gemini")
                if decision.approval_required:
                    decision = await self._normalize_decision(decision)
                    yield AgentStreamEvent(approval=decision)
                    return

                tool_result = await self.execute_tool(decision.tool)
                final_contents = self._gemini_tool_contents(
                    message,
                    decision.tool,
                    tool_result,
                )
                final_parts: list[str] = []
                for chunk in gemini_provider.stream_tool_reply(
                    final_contents,
                    [],
                    system_instruction=GITHUB_SYSTEM_INSTRUCTION,
                ):
                    if getattr(chunk, "function_calls", None):
                        raise LLMToolCallError("Gemini")
                    try:
                        text = chunk.text
                    except (ValueError, AttributeError):
                        text = None
                    if text:
                        final_parts.append(text)
                        yield AgentStreamEvent(text=text)
                final_response = "".join(final_parts)
                self._record_tool_turn(
                    message,
                    decision,
                    tool_result,
                    final_response,
                )
                return

            for text in fallback_parts:
                yield AgentStreamEvent(text=text)
            self._conversation.extend(
                [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "".join(fallback_parts)},
                ]
            )

    async def decide(self, message: str) -> AgentDecision:
        """Ask the LLM whether a read-only GitHub tool is needed."""
        with observe(
            "github-agent-decision",
            input=message,
            metadata={"workflow": "github-agent"},
            session_id=self._session_id,
            user_id=self._user_id,
        ):
            if self._tools is None:
                await self.connect()
            completion = self._completion(self._messages(message))
            decision = self._decision_from_completion(completion)
            if decision.tool is None:
                self._conversation.extend(
                    [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": decision.response},
                    ]
                )
            return decision

    async def execute_tool(
        self, tool: SelectedTool, *, approved: bool = False
    ) -> Any:
        """Execute a validated tool through MCP after required approval."""
        if self._tools is None:
            await self.connect()

        allowed_names = {item.name for item in self._tools or []}
        if tool.name not in allowed_names:
            raise ValueError("Unsupported GitHub tool")
        if tool.name in WRITE_TOOL_NAMES and not approved:
            raise PermissionError("Approval is required for write tools")

        arguments = tool.arguments
        if tool.name == "issue_write":
            arguments = await self._resolve_issue_write_arguments(arguments)

        # GitHubMCPClient owns the active MCP session and intentionally exposes
        # only connection/tool-listing lifecycle methods at this stage.
        session = getattr(self._mcp_client, "_session", None)
        if session is None:
            raise RuntimeError("GitHub MCP client is not connected")
        with observe(
            "github-mcp-tool",
            input={"tool": tool.name, "arguments": tool.arguments},
            metadata={"workflow": "github-agent", "tool_name": tool.name},
            session_id=self._session_id,
            user_id=self._user_id,
        ) as observation:
            result = await session.call_tool(tool.name, arguments)
            update(observation, output=self._serialize_tool_result(result))
            return result

    @staticmethod
    def _serialize_tool_result(result: Any) -> str:
        if hasattr(result, "model_dump"):
            result = result.model_dump(mode="json")
        try:
            return json.dumps(result, default=str)
        except TypeError:
            return str(result)

    @classmethod
    def _extract_github_login(cls, result: Any) -> str | None:
        payload: Any = result
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        if isinstance(payload, dict) and "content" in payload:
            content = payload["content"]
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("text"):
                        try:
                            payload = json.loads(item["text"])
                        except json.JSONDecodeError:
                            payload = item["text"]
                        break
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return None
        if isinstance(payload, dict) and isinstance(payload.get("login"), str):
            return payload["login"]
        return None

    async def _resolve_issue_write_arguments(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        resolved = dict(arguments)
        if resolved.get("method") is None and resolved.get("title"):
            resolved["method"] = "create"
        if not resolved.get("repo"):
            return resolved

        owner = resolved.get("owner")
        if owner and owner != resolved.get("repo"):
            return resolved

        session = getattr(self._mcp_client, "_session", None)
        if session is None:
            return resolved

        me_result = await session.call_tool("get_me", {})
        login = self._extract_github_login(me_result)
        if login:
            resolved["owner"] = login
        return resolved

    async def _normalize_decision(self, decision: AgentDecision) -> AgentDecision:
        if decision.tool is None or decision.tool.name != "issue_write":
            return decision
        arguments = await self._resolve_issue_write_arguments(decision.tool.arguments)
        return AgentDecision(
            response=decision.response,
            tool=SelectedTool(
                name=decision.tool.name,
                arguments=arguments,
                thought_signature=decision.tool.thought_signature,
                model_content=decision.tool.model_content,
            ),
            approval_required=decision.approval_required,
            tool_call_id=decision.tool_call_id,
            provider=decision.provider,
        )

    def _record_tool_turn(
        self,
        message: str,
        decision: AgentDecision,
        tool_result: Any,
        final_response: str,
    ) -> None:
        """Add a completed tool-assisted turn to the session history."""
        if decision.tool is None:
            return

        tool_call_id = decision.tool_call_id or "github-tool-call"
        self._conversation.extend(
            [
                {"role": "user", "content": message},
                {
                    "role": "assistant",
                    "content": decision.response or None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": decision.tool.name,
                                "arguments": json.dumps(decision.tool.arguments),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": self._serialize_tool_result(tool_result),
                },
                {"role": "assistant", "content": final_response},
            ]
        )

    async def _complete_tool_call(
        self,
        message: str,
        decision: AgentDecision,
        *,
        approved: bool = False,
    ) -> str:
        if decision.tool is None:
            return decision.response
        if decision.provider == "gemini":
            return await self._complete_gemini_tool_call(
                message,
                decision,
                approved=approved,
            )

        tool_result = await self.execute_tool(decision.tool, approved=approved)
        tool_call_id = decision.tool_call_id or "github-tool-call"
        messages = self._messages(message)
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": decision.response or None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": decision.tool.name,
                                "arguments": json.dumps(decision.tool.arguments),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": self._serialize_tool_result(tool_result),
                },
            ]
        )

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        with observe(
            "agent-final-response",
            as_type="generation",
            input=message,
            model=model,
            metadata={"workflow": "github-agent", "phase": "final-response"},
            session_id=self._session_id,
            user_id=self._user_id,
        ) as generation:
            final_completion = self._llm_client.chat.completions.create(
                model=model,
                messages=messages,
            )
            update(
                generation,
                output=final_completion.choices[0].message.content or "",
                usage=getattr(final_completion, "usage", None),
            )
        final_response = final_completion.choices[0].message.content or ""
        self._record_tool_turn(message, decision, tool_result, final_response)
        return final_response

    async def _complete_gemini_tool_call(
        self,
        message: str,
        decision: AgentDecision,
        *,
        approved: bool = False,
    ) -> str:
        if decision.tool is None:
            return decision.response

        tool_result = await self.execute_tool(decision.tool, approved=approved)
        from backend.llm import gemini_provider

        contents = self._gemini_tool_contents(
            message,
            decision.tool,
            tool_result,
        )
        final_parts: list[str] = []
        for chunk in gemini_provider.stream_tool_reply(
            contents,
            [],
        ):
            if getattr(chunk, "function_calls", None):
                raise LLMToolCallError("Gemini")
            try:
                text = chunk.text
            except (ValueError, AttributeError):
                text = None
            if text:
                final_parts.append(text)
        final_response = "".join(final_parts)
        self._record_tool_turn(message, decision, tool_result, final_response)
        return final_response

    async def complete_decision(
        self,
        message: str,
        decision: AgentDecision,
        *,
        approved: bool = False,
    ) -> str:
        """Execute a selection and generate a final response."""
        with observe(
            "github-agent-completion",
            input=message,
            metadata={
                "workflow": "github-agent",
                "approval_required": decision.approval_required,
                "approved": approved,
            },
            session_id=self._session_id,
            user_id=self._user_id,
        ):
            if decision.tool is None:
                return decision.response
            if decision.approval_required and not approved:
                raise PermissionError("Approval is required for write tools")
            return await self._complete_tool_call(
                message,
                decision,
                approved=approved,
            )

    async def run(self, message: str) -> str:
        """Select a tool, execute it if allowed, and generate the final reply."""
        if self._tools is None:
            await self.connect()

        decision = await self.decide(message)
        if decision.tool is None:
            return decision.response
        if decision.approval_required:
            raise PermissionError("Approval is required for write tools")
        return await self._complete_tool_call(message, decision)

    async def close(self) -> None:
        """Close the underlying MCP connection."""
        await self._mcp_client.close()
