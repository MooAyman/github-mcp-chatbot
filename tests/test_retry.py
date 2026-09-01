"""Tests for retry behavior and the GitHub MCP connection."""

import json
import os
import shutil
import gc
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, AuthenticationError, RateLimitError

from backend.reliability.retry import (
    BASE_DELAY_SECONDS,
    MAX_ATTEMPTS,
    call_with_retry,
    is_fallback_eligible,
    is_retryable,
)
from backend.reliability.errors import (
    LLMConfigurationError,
    LLMRateLimitError,
    normalize_error,
)
from backend.observability import observe, update, usage_details_from


def _rate_limit_error() -> RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com"))
    return RateLimitError("rate limited", response=response, body=None)


def _auth_error() -> AuthenticationError:
    response = httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com"))
    return AuthenticationError("invalid api key", response=response, body=None)


def test_is_retryable_for_transient_and_permanent_errors() -> None:
    assert is_retryable(_rate_limit_error()) is True
    assert is_retryable(APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))) is True
    assert is_retryable(_auth_error()) is False


def test_call_with_retry_succeeds_after_transient_failures() -> None:
    fn = MagicMock(side_effect=[_rate_limit_error(), _rate_limit_error(), "ok"])

    with patch("backend.reliability.retry.time.sleep") as sleep_mock:
        assert call_with_retry(fn) == "ok"

    assert fn.call_count == 3
    assert sleep_mock.call_args_list == [
        ((BASE_DELAY_SECONDS,),),
        ((BASE_DELAY_SECONDS * 2,),),
    ]


def test_call_with_retry_raises_after_max_transient_failures() -> None:
    fn = MagicMock(side_effect=_rate_limit_error())

    with patch("backend.reliability.retry.time.sleep") as sleep_mock:
        with pytest.raises(RateLimitError):
            call_with_retry(fn)

    assert fn.call_count == MAX_ATTEMPTS
    assert sleep_mock.call_count == MAX_ATTEMPTS - 1


def test_call_with_retry_does_not_retry_permanent_errors() -> None:
    fn = MagicMock(side_effect=_auth_error())

    with patch("backend.reliability.retry.time.sleep") as sleep_mock:
        with pytest.raises(AuthenticationError):
            call_with_retry(fn)

    assert fn.call_count == 1
    sleep_mock.assert_not_called()


def test_provider_errors_are_normalized_without_raw_details() -> None:
    auth_error = normalize_error(_auth_error(), "OpenAI")
    rate_error = normalize_error(_rate_limit_error(), "OpenAI")

    assert isinstance(auth_error, LLMConfigurationError)
    assert isinstance(rate_error, LLMRateLimitError)
    assert "invalid api key" not in str(auth_error).lower()
    assert "429" not in str(rate_error)


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("OPENAI_API_KEY is not set"),
        _auth_error(),
    ],
    ids=["missing-key", "invalid-key"],
)
def test_openai_auth_failures_fall_back_without_being_retryable(
    failure: Exception,
) -> None:
    from backend.llm import generate_reply

    assert is_retryable(failure) is False
    assert is_fallback_eligible(failure) is True

    with (
        patch(
            "backend.llm.openai_provider.generate_reply",
            side_effect=failure,
        ) as openai_mock,
        patch(
            "backend.llm.gemini_provider.generate_reply",
            return_value="gemini reply",
        ) as gemini_mock,
    ):
        assert generate_reply("hi") == "gemini reply"

    openai_mock.assert_called_once_with("hi")
    gemini_mock.assert_called_once_with("hi")


def test_generate_reply_falls_back_to_gemini_after_openai_retries() -> None:
    from backend.llm import generate_reply

    with (
        patch(
            "backend.llm.openai_provider.generate_reply",
            side_effect=_rate_limit_error(),
        ) as openai_mock,
        patch(
            "backend.llm.gemini_provider.generate_reply",
            return_value="gemini reply",
        ) as gemini_mock,
    ):
        assert generate_reply("hi") == "gemini reply"

    openai_mock.assert_called_once_with("hi")
    gemini_mock.assert_called_once_with("hi")


def test_generate_reply_does_not_fallback_on_permanent_openai_error() -> None:
    from backend.llm import generate_reply

    with (
        patch(
            "backend.llm.openai_provider.generate_reply",
            side_effect=ValueError("unexpected provider configuration failure"),
        ),
        patch("backend.llm.gemini_provider.generate_reply") as gemini_mock,
    ):
        with pytest.raises(LLMConfigurationError):
            generate_reply("hi")

    gemini_mock.assert_not_called()


def test_stream_reply_falls_back_to_gemini_after_openai_retries() -> None:
    from backend.llm import stream_reply

    with (
        patch(
            "backend.llm.openai_provider.stream_reply",
            side_effect=_rate_limit_error(),
        ),
        patch(
            "backend.llm.gemini_provider.stream_reply",
            return_value=iter(["gem", "ini"]),
        ) as gemini_mock,
    ):
        assert "".join(stream_reply("hi")) == "gemini"

    gemini_mock.assert_called_once_with("hi")


def test_gemini_stream_keeps_client_alive_until_stream_is_consumed() -> None:
    from backend.llm import gemini_provider

    client_refs: list[weakref.ReferenceType[object]] = []

    class FakeStream:
        def __init__(self, client_ref: weakref.ReferenceType[object]) -> None:
            self._client_ref = client_ref

        def __iter__(self):
            gc.collect()
            if self._client_ref() is None:
                raise RuntimeError("Cannot send a request, as the client has been closed.")
            yield SimpleNamespace(text="hello", usage_metadata=None)

    class FakeModels:
        def __init__(self, client: object) -> None:
            self._client_ref = weakref.ref(client)

        def generate_content_stream(self, **_: object) -> FakeStream:
            return FakeStream(self._client_ref)

    class FakeClient:
        def __init__(self) -> None:
            self.models = FakeModels(self)

    def client_factory() -> FakeClient:
        client = FakeClient()
        client_refs.append(weakref.ref(client))
        return client

    with (
        patch.object(gemini_provider, "_client", side_effect=client_factory),
        patch("backend.observability.tracing._get_client", return_value=None),
    ):
        assert list(gemini_provider.stream_reply("hi")) == ["hello"]

    assert client_refs[0]() is None


def test_gemini_tool_stream_includes_mcp_function_declarations() -> None:
    from backend.llm import gemini_provider

    client = MagicMock()
    client.models.generate_content_stream.return_value = iter(
        [SimpleNamespace(text="hello", usage_metadata=None)]
    )
    tool = SimpleNamespace(
        name="search_issues",
        description="Search repository issues.",
        inputSchema={"type": "object", "properties": {}},
    )

    with (
        patch.object(gemini_provider, "_client", return_value=client),
        patch("backend.observability.tracing._get_client", return_value=None),
    ):
        assert [
            chunk.text
            for chunk in gemini_provider.stream_tool_reply(
                "hi",
                [tool],
                system_instruction="Use the appropriate GitHub tool.",
            )
        ] == ["hello"]

    config = client.models.generate_content_stream.call_args.kwargs["config"]
    declaration = config.tools[0].function_declarations[0]
    assert declaration.name == "search_issues"
    assert declaration.parameters_json_schema == tool.inputSchema
    assert config.system_instruction == "Use the appropriate GitHub tool."


@pytest.mark.anyio
async def test_github_mcp_client_prefers_http_transport_when_configured() -> None:
    from backend.mcp.github_client import GitHubMCPClient

    client = GitHubMCPClient()
    with (
        patch.dict(
            os.environ,
            {
                "GITHUB_MCP_URL": "http://github-mcp:8082",
                "GITHUB_PERSONAL_ACCESS_TOKEN": "test-token",
            },
            clear=False,
        ),
        patch.object(client, "_connect_http", AsyncMock()) as connect_http,
        patch.object(client, "_connect_stdio", AsyncMock()) as connect_stdio,
    ):
        await client.connect()
        connect_http.assert_awaited_once_with(
            "http://github-mcp:8082",
            "test-token",
        )
        connect_stdio.assert_not_called()
    await client.close()


@pytest.mark.anyio
async def test_github_mcp_client_lists_tools_and_closes() -> None:
    """Connect to the real Docker MCP server and verify its tool list."""
    from backend.mcp.github_client import GitHubMCPClient

    if os.getenv("GITHUB_MCP_URL"):
        pytest.skip("stdio integration test requires local docker-run MCP transport")
    if not os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"):
        pytest.skip("GITHUB_PERSONAL_ACCESS_TOKEN is required")
    if shutil.which("docker") is None:
        pytest.skip("Docker is required")

    client = GitHubMCPClient()
    try:
        await client.connect()
        tools = await client.list_tools()
        assert tools
    finally:
        await client.close()


@pytest.mark.anyio
async def test_agent_selects_read_only_tool_without_executing_it() -> None:
    from backend.agent import GitHubAgent

    mcp_client = MagicMock()
    mcp_client.connect = AsyncMock()
    mcp_client.list_tools = AsyncMock(
        return_value=[
            SimpleNamespace(
                name="get_me",
                description="Get the authenticated user.",
                inputSchema={"type": "object", "properties": {}},
            ),
            SimpleNamespace(
                name="create_issue",
                description="Create an issue.",
                inputSchema={"type": "object"},
            ),
        ]
    )
    mcp_client.close = AsyncMock()
    mcp_client.call_tool = AsyncMock()

    llm_client = MagicMock()
    llm_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="get_me",
                                arguments="{}",
                            )
                        )
                    ],
                )
            )
        ]
    )

    agent = GitHubAgent(mcp_client=mcp_client, llm_client=llm_client)
    await agent.connect()
    decision = await agent.decide("Who am I on GitHub?")

    assert decision.tool is not None
    assert decision.tool.name == "get_me"
    assert decision.tool.arguments == {}
    assert len(llm_client.chat.completions.create.call_args.kwargs["tools"]) == 1
    mcp_client.call_tool.assert_not_called()

    await agent.close()
    mcp_client.close.assert_awaited_once()


@pytest.mark.anyio
async def test_agent_returns_model_definition_for_mcp_question() -> None:
    from backend.agent import GitHubAgent

    mcp_client = MagicMock()
    mcp_client.connect = AsyncMock()
    mcp_client.list_tools = AsyncMock(return_value=[])
    llm_client = MagicMock()
    llm_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="MCP means Model Context Protocol in this project.",
                    tool_calls=None,
                )
            )
        ]
    )

    agent = GitHubAgent(mcp_client=mcp_client, llm_client=llm_client)
    decision = await agent.decide("What is MCP in this project?")

    assert decision.tool is None
    assert "Model Context Protocol" in decision.response


@pytest.mark.anyio
async def test_agent_executes_tool_and_returns_final_llm_response() -> None:
    from backend.agent import GitHubAgent

    mcp_client = MagicMock()
    mcp_client.connect = AsyncMock()
    mcp_client.list_tools = AsyncMock(
        return_value=[
            SimpleNamespace(
                name="get_file_contents",
                description="Read a file.",
                inputSchema={"type": "object"},
            )
        ]
    )
    mcp_client.close = AsyncMock()
    mcp_client._session = MagicMock()
    mcp_client._session.call_tool = AsyncMock(
        return_value={"content": [{"text": "file contents"}]}
    )

    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="get_file_contents",
            arguments='{"owner":"octo","repo":"demo","path":"README.md"}',
        ),
    )
    llm_client = MagicMock()
    llm_client.chat.completions.create.side_effect = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[tool_call],
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Here are the file contents.",
                        tool_calls=None,
                    )
                )
            ]
        ),
    ]

    agent = GitHubAgent(mcp_client=mcp_client, llm_client=llm_client)
    result = await agent.run("Show me README.md")

    assert result == "Here are the file contents."
    mcp_client._session.call_tool.assert_awaited_once_with(
        "get_file_contents",
        {"owner": "octo", "repo": "demo", "path": "README.md"},
    )
    final_messages = llm_client.chat.completions.create.call_args_list[1].kwargs[
        "messages"
    ]
    assert final_messages[-1]["role"] == "tool"
    assert "file contents" in final_messages[-1]["content"]


@pytest.mark.anyio
async def test_agent_rejects_write_tool_execution() -> None:
    from backend.agent import GitHubAgent, SelectedTool

    mcp_client = MagicMock()
    mcp_client._session = MagicMock()
    mcp_client._session.call_tool = AsyncMock()
    agent = GitHubAgent(mcp_client=mcp_client, llm_client=MagicMock())
    agent._tools = [
        SimpleNamespace(
            name="get_me",
            description="Read the current user.",
            inputSchema={"type": "object"},
        )
    ]

    with pytest.raises(ValueError, match="Unsupported GitHub tool"):
        await agent.execute_tool(SelectedTool(name="issue_write", arguments={}))

    mcp_client._session.call_tool.assert_not_called()


@pytest.mark.anyio
async def test_agent_requires_approval_before_write_and_approves_it() -> None:
    from backend.agent import GitHubAgent

    mcp_client = MagicMock()
    mcp_client.connect = AsyncMock()
    mcp_client.list_tools = AsyncMock(
        return_value=[
            SimpleNamespace(
                name="issue_write",
                description="Create or update an issue.",
                inputSchema={"type": "object"},
            )
        ]
    )
    mcp_client.close = AsyncMock()
    mcp_client._session = MagicMock()
    mcp_client._session.call_tool = AsyncMock(
        return_value={"content": [{"text": "issue created"}]}
    )

    tool_call = SimpleNamespace(
        id="write-call-1",
        function=SimpleNamespace(
            name="issue_write",
            arguments='{"owner":"octo","repo":"demo","title":"Bug"}',
        ),
    )
    llm_client = MagicMock()
    llm_client.chat.completions.create.side_effect = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[tool_call],
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="The issue was created.",
                        tool_calls=None,
                    )
                )
            ]
        ),
    ]

    agent = GitHubAgent(mcp_client=mcp_client, llm_client=llm_client)
    decision = await agent.decide("Create a bug issue")

    assert decision.approval_required is True
    assert decision.tool is not None
    mcp_client._session.call_tool.assert_not_called()

    with pytest.raises(PermissionError):
        await agent.complete_decision("Create a bug issue", decision)
    mcp_client._session.call_tool.assert_not_called()

    result = await agent.complete_decision(
        "Create a bug issue",
        decision,
        approved=True,
    )
    assert result == "The issue was created."
    mcp_client._session.call_tool.assert_awaited_once_with(
        "issue_write",
        {"owner": "octo", "repo": "demo", "title": "Bug", "method": "create"},
    )


@pytest.mark.anyio
async def test_agent_gemini_fallback_uses_mcp_tool_and_streams_final_response() -> None:
    from backend.agent import AgentStreamEvent, GitHubAgent
    from google.genai import types

    mcp_client = MagicMock()
    mcp_client.connect = AsyncMock()
    mcp_client.list_tools = AsyncMock(
        return_value=[
            SimpleNamespace(
                name="search_issues",
                description="Search repository issues.",
                inputSchema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            ),
            SimpleNamespace(
                name="search_repositories",
                description="Search repositories.",
                inputSchema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            ),
        ]
    )
    mcp_client._session = MagicMock()
    mcp_client._session.call_tool = AsyncMock(
        return_value={"content": [{"text": "matching issue"}]}
    )
    agent = GitHubAgent(mcp_client=mcp_client, llm_client=MagicMock())

    function_call = types.FunctionCall(
        name="search_issues",
        args={"query": "authentication"},
    )
    model_content = types.Content(
        role="model",
        parts=[
            types.Part(text="Search"),
            types.Part(
                function_call=function_call,
                thought_signature=b"gemini-signature",
            ),
        ],
    )
    initial_chunk = SimpleNamespace(
        text="Search",
        function_calls=[function_call],
        usage_metadata=None,
        candidates=[
            SimpleNamespace(
                content=model_content,
            )
        ],
    )
    final_chunk = SimpleNamespace(
        text="Found one matching issue.",
        function_calls=None,
        usage_metadata=None,
    )
    with (
        patch.object(
            agent,
            "_stream_openai",
            side_effect=ValueError("OPENAI_API_KEY is not set"),
        ),
        patch(
            "backend.llm.gemini_provider.stream_tool_reply",
            side_effect=[iter([initial_chunk]), iter([final_chunk])],
        ) as gemini_mock,
        patch("backend.observability.tracing._get_client", return_value=None),
    ):
        events = [
            event
            async for event in agent.stream_response(
                "Search authentication issues"
            )
        ]

    assert events == [AgentStreamEvent(text="Found one matching issue.")]
    mcp_client._session.call_tool.assert_awaited_once_with(
        "search_issues",
        {"query": "authentication"},
    )
    first_tools = gemini_mock.call_args_list[0].args[1]
    assert [tool.name for tool in first_tools] == ["search_issues"]
    assert (
        gemini_mock.call_args_list[0].kwargs["system_instruction"]
        == "This is a GitHub assistant using Model Context Protocol (MCP) tools. "
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
    final_contents = gemini_mock.call_args_list[1].args[0]
    assert [content.role for content in final_contents] == ["user"]
    final_prompt = final_contents[0].parts[0].text
    assert "Do not call another tool." in final_prompt
    assert '"matching issue"' in final_prompt


@pytest.mark.anyio
async def test_agent_fails_explicitly_on_unhandled_final_gemini_tool_call() -> None:
    from backend.agent import GitHubAgent
    from backend.reliability import LLMToolCallError
    from google.genai import types

    mcp_client = MagicMock()
    mcp_client.connect = AsyncMock()
    mcp_client.list_tools = AsyncMock(
        return_value=[
            SimpleNamespace(
                name="search_issues",
                description="Search repository issues.",
                inputSchema={"type": "object"},
            ),
            SimpleNamespace(
                name="search_repositories",
                description="Search repositories.",
                inputSchema={"type": "object"},
            ),
        ]
    )
    mcp_client._session = MagicMock()
    mcp_client._session.call_tool = AsyncMock(
        return_value={"content": [{"text": "matching issue"}]}
    )
    agent = GitHubAgent(mcp_client=mcp_client, llm_client=MagicMock())

    initial_call = types.FunctionCall(
        name="search_issues",
        args={"query": "authentication"},
    )
    initial_chunk = SimpleNamespace(
        text="Search",
        function_calls=[initial_call],
        usage_metadata=None,
        candidates=[
            SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=initial_call,
                            thought_signature=b"gemini-signature",
                        )
                    ],
                )
            )
        ],
    )
    final_chunk = SimpleNamespace(
        text=None,
        function_calls=[
            types.FunctionCall(
                name="search_repositories",
                args={"query": "github-mcp-chatbot"},
            )
        ],
        usage_metadata=None,
    )

    with (
        patch.object(
            agent,
            "_stream_openai",
            side_effect=ValueError("OPENAI_API_KEY is not set"),
        ),
        patch(
            "backend.llm.gemini_provider.stream_tool_reply",
            side_effect=[iter([initial_chunk]), iter([final_chunk])],
        ) as gemini_mock,
        patch("backend.observability.tracing._get_client", return_value=None),
    ):
        with pytest.raises(LLMToolCallError, match="another GitHub operation"):
            _ = [
                event
                async for event in agent.stream_response(
                    "Search open issues related to authentication"
                )
            ]

    mcp_client._session.call_tool.assert_awaited_once_with(
        "search_issues",
        {"query": "authentication"},
    )
    assert gemini_mock.call_count == 2


@pytest.mark.anyio
async def test_rejected_write_decision_does_not_call_github() -> None:
    from backend.agent import GitHubAgent

    mcp_client = MagicMock()
    mcp_client.connect = AsyncMock()
    mcp_client.list_tools = AsyncMock(
        return_value=[
            SimpleNamespace(
                name="add_issue_comment",
                description="Add an issue comment.",
                inputSchema={"type": "object"},
            )
        ]
    )
    mcp_client._session = MagicMock()
    mcp_client._session.call_tool = AsyncMock()

    llm_client = MagicMock()
    llm_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="comment-call-1",
                            function=SimpleNamespace(
                                name="add_issue_comment",
                                arguments="{}",
                            ),
                        )
                    ],
                )
            )
        ]
    )

    agent = GitHubAgent(mcp_client=mcp_client, llm_client=llm_client)
    decision = await agent.decide("Comment on the issue")

    assert decision.approval_required is True
    # A rejected Chainlit action does not call complete_decision.
    mcp_client._session.call_tool.assert_not_called()
    assert llm_client.chat.completions.create.call_count == 1


def test_tracing_records_generation_metadata_and_usage() -> None:
    from backend.observability import tracing

    observation = MagicMock()
    observation_context = MagicMock()
    observation_context.__enter__.return_value = observation
    observation_context.__exit__.return_value = False
    langfuse_client = MagicMock()
    langfuse_client.start_as_current_observation.return_value = observation_context
    usage = SimpleNamespace(
        prompt_tokens=4,
        completion_tokens=6,
        total_tokens=10,
    )

    with patch(
        "backend.observability.tracing._get_client",
        return_value=langfuse_client,
    ):
        with observe(
            "test-generation",
            as_type="generation",
            input="hello",
            model="test-model",
            session_id="session-1",
            user_id="user-1",
        ) as current:
            update(current, output="world", usage=usage)

    langfuse_client.start_as_current_observation.assert_called_once_with(
        as_type="generation",
        name="test-generation",
        model="test-model",
        input="hello",
    )
    assert any(
        call.kwargs.get("output") == "world"
        and call.kwargs.get("usage_details")
        == {"input": 4, "output": 6, "total": 10}
        for call in observation.update.call_args_list
    )


def test_tracing_uses_base_url_configuration() -> None:
    from backend.observability import tracing

    tracing._client = None
    with (
        patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": "public",
                "LANGFUSE_SECRET_KEY": "secret",
                "LANGFUSE_BASE_URL": "https://example.langfuse.com",
                "LANGFUSE_HOST": "https://wrong.example.com",
            },
            clear=False,
        ),
        patch("backend.observability.tracing.Langfuse") as langfuse,
    ):
        tracing._get_client()

    assert langfuse.call_args.kwargs["base_url"] == "https://example.langfuse.com"


def test_tracing_is_noop_without_configuration() -> None:
    with patch("backend.observability.tracing._get_client", return_value=None):
        with observe("disabled-trace") as current:
            assert current is None
    assert usage_details_from(None) == {}


def test_tracing_records_retry_and_fallback_events_with_supported_type() -> None:
    from backend.observability import tracing

    observation = MagicMock()
    observation_context = MagicMock()
    observation_context.__enter__.return_value = observation
    observation_context.__exit__.return_value = False
    langfuse_client = MagicMock()
    langfuse_client.start_as_current_observation.return_value = observation_context

    with patch(
        "backend.observability.tracing._get_client",
        return_value=langfuse_client,
    ):
        tracing.event("llm-fallback", {"reason": "AuthenticationError"})
        tracing.flush()

    langfuse_client.start_as_current_observation.assert_called_once_with(
        as_type="span",
        name="llm-fallback",
        model=None,
        input=None,
    )
    langfuse_client.flush.assert_called_once_with()


@pytest.mark.anyio
async def test_agent_includes_previous_turns_in_next_llm_decision() -> None:
    from backend.agent import GitHubAgent

    mcp_client = MagicMock()
    mcp_client.connect = AsyncMock()
    mcp_client.list_tools = AsyncMock(return_value=[])

    llm_client = MagicMock()
    llm_client.chat.completions.create.side_effect = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="First answer",
                        tool_calls=None,
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Second answer",
                        tool_calls=None,
                    )
                )
            ]
        ),
    ]

    agent = GitHubAgent(mcp_client=mcp_client, llm_client=llm_client)
    await agent.decide("First question")
    await agent.decide("Second question")

    second_messages = llm_client.chat.completions.create.call_args_list[1].kwargs[
        "messages"
    ]
    assert second_messages[1:4] == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
    ]


@pytest.mark.anyio
async def test_agent_streams_normal_response_tokens() -> None:
    from backend.agent import AgentStreamEvent, GitHubAgent

    mcp_client = MagicMock()
    mcp_client.connect = AsyncMock()
    mcp_client.list_tools = AsyncMock(return_value=[])
    agent = GitHubAgent(mcp_client=mcp_client, llm_client=MagicMock())

    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Hello ", tool_calls=None)
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="world", tool_calls=None)
                )
            ],
            usage=None,
        ),
    ]
    with patch.object(agent, "_stream_openai", return_value=iter(chunks)):
        events = [event async for event in agent.stream_response("Say hello")]

    assert events == [
        AgentStreamEvent(text="Hello "),
        AgentStreamEvent(text="world"),
    ]
    assert agent._conversation == [
        {"role": "user", "content": "Say hello"},
        {"role": "assistant", "content": "Hello world"},
    ]


@pytest.mark.anyio
async def test_agent_streaming_tracing_uses_manual_observation_lifecycle() -> None:
    from backend.agent import AgentStreamEvent, GitHubAgent

    mcp_client = MagicMock()
    mcp_client.connect = AsyncMock()
    mcp_client.list_tools = AsyncMock(return_value=[])
    agent = GitHubAgent(mcp_client=mcp_client, llm_client=MagicMock())
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Hello", tool_calls=None)
                )
            ],
            usage=None,
        )
    ]
    observation = MagicMock()
    langfuse_client = MagicMock()
    langfuse_client.start_observation.return_value = observation

    with (
        patch.object(agent, "_stream_openai", return_value=iter(chunks)),
        patch(
            "backend.observability.tracing._get_client",
            return_value=langfuse_client,
        ),
    ):
        events = [event async for event in agent.stream_response("Say hello")]

    assert events == [AgentStreamEvent(text="Hello")]
    langfuse_client.start_observation.assert_called_once()
    langfuse_client.start_as_current_observation.assert_not_called()
    observation.end.assert_called_once_with()


@pytest.mark.anyio
async def test_fastapi_chat_routes_normal_reply_through_session_agent() -> None:
    from backend import main
    from backend.agent import AgentStreamEvent

    agent = MagicMock()
    async def stream_events():
        yield AgentStreamEvent(text="agent ")
        yield AgentStreamEvent(text="reply")

    agent.stream_response = MagicMock(return_value=stream_events())

    async def collect(response: object) -> str:
        chunks = [
            chunk async for chunk in response.body_iterator  # type: ignore[attr-defined]
        ]
        return "".join(
            chunk.decode() if isinstance(chunk, bytes) else chunk for chunk in chunks
        )

    with patch("backend.main._get_agent", return_value=agent) as get_agent:
        response = await main.chat(
            main.ChatRequest(message="hello", session_id="session-1")
        )

    assert await collect(response) == "agent reply"
    get_agent.assert_called_once_with("session-1")
    agent.stream_response.assert_called_once_with("hello")


@pytest.mark.anyio
async def test_fastapi_chat_rejects_unhandled_final_gemini_tool_call() -> None:
    from fastapi import HTTPException

    from backend import main
    from backend.reliability import LLMToolCallError

    agent = MagicMock()

    async def failing_events():
        if False:
            yield None
        raise LLMToolCallError("Gemini")

    agent.stream_response = MagicMock(return_value=failing_events())

    with patch("backend.main._get_agent", return_value=agent):
        with pytest.raises(HTTPException) as error:
            await main.chat(
                main.ChatRequest(
                    message="Search open issues",
                    session_id="final-tool-error",
                )
            )

    assert error.value.status_code == 502
    assert (
        error.value.detail
        == "The assistant requested another GitHub operation after the first one. "
        "Please try again."
    )


@pytest.mark.anyio
async def test_agent_stream_read_only_tool_executes_without_approval() -> None:
    from backend.agent import AgentStreamEvent, GitHubAgent

    mcp_client = MagicMock()
    mcp_client.connect = AsyncMock()
    mcp_client.list_tools = AsyncMock(
        return_value=[
            SimpleNamespace(
                name="search_issues",
                description="Search repository issues.",
                inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ]
    )
    mcp_client.close = AsyncMock()
    mcp_client._session = MagicMock()
    mcp_client._session.call_tool = AsyncMock(
        return_value={"content": [{"text": '{"total_count": 0, "items": []}'}]}
    )

    agent = GitHubAgent(mcp_client=mcp_client, llm_client=MagicMock())
    tool_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id="search-call-1",
                            function=SimpleNamespace(
                                name="search_issues",
                                arguments='{"query":"authentication"}',
                            ),
                        )
                    ],
                )
            )
        ],
        usage=None,
    )
    final_chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Found 0 issues.", tool_calls=None)
                )
            ],
            usage=None,
        )
    ]

    with patch.object(
        agent,
        "_stream_openai",
        side_effect=[iter([tool_chunk]), iter(final_chunks)],
    ):
        events = [event async for event in agent.stream_response("Search issues")]

    assert events == [AgentStreamEvent(text="Found 0 issues.")]
    assert all(event.approval is None for event in events)
    mcp_client._session.call_tool.assert_awaited_once_with(
        "search_issues",
        {"query": "authentication"},
    )


@pytest.mark.anyio
async def test_agent_stream_write_tool_yields_approval_without_executing() -> None:
    from backend.agent import AgentStreamEvent, GitHubAgent

    mcp_client = MagicMock()
    mcp_client.connect = AsyncMock()
    mcp_client.list_tools = AsyncMock(
        return_value=[
            SimpleNamespace(
                name="issue_write",
                description="Create or update an issue.",
                inputSchema={"type": "object"},
            )
        ]
    )
    mcp_client.close = AsyncMock()
    mcp_client._session = MagicMock()
    mcp_client._session.call_tool = AsyncMock()

    agent = GitHubAgent(mcp_client=mcp_client, llm_client=MagicMock())
    tool_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id="write-call-1",
                            function=SimpleNamespace(
                                name="issue_write",
                                arguments='{"owner":"octo","repo":"demo","title":"Bug"}',
                            ),
                        )
                    ],
                )
            )
        ],
        usage=None,
    )

    with patch.object(agent, "_stream_openai", return_value=iter([tool_chunk])):
        events = [event async for event in agent.stream_response("Create a bug issue")]

    assert len(events) == 1
    assert events[0].approval is not None
    assert events[0].approval.approval_required is True
    assert events[0].approval.tool is not None
    assert events[0].approval.tool.name == "issue_write"
    mcp_client._session.call_tool.assert_not_called()


@pytest.mark.anyio
async def test_agent_stream_write_tool_proceed_executes_with_approved_true() -> None:
    from backend.agent import GitHubAgent

    mcp_client = MagicMock()
    mcp_client.connect = AsyncMock()
    mcp_client.list_tools = AsyncMock(
        return_value=[
            SimpleNamespace(
                name="issue_write",
                description="Create or update an issue.",
                inputSchema={"type": "object"},
            )
        ]
    )
    mcp_client.close = AsyncMock()
    mcp_client._session = MagicMock()
    mcp_client._session.call_tool = AsyncMock(
        return_value={"content": [{"text": "issue created"}]}
    )

    llm_client = MagicMock()
    llm_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="The issue was created.",
                    tool_calls=None,
                )
            )
        ]
    )

    agent = GitHubAgent(mcp_client=mcp_client, llm_client=llm_client)
    tool_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id="write-call-1",
                            function=SimpleNamespace(
                                name="issue_write",
                                arguments='{"owner":"octo","repo":"demo","title":"Bug"}',
                            ),
                        )
                    ],
                )
            )
        ],
        usage=None,
    )

    with patch.object(agent, "_stream_openai", return_value=iter([tool_chunk])):
        events = [event async for event in agent.stream_response("Create a bug issue")]

    decision = events[0].approval
    assert decision is not None
    mcp_client._session.call_tool.assert_not_called()

    result = await agent.complete_decision(
        "Create a bug issue",
        decision,
        approved=True,
    )
    assert result == "The issue was created."
    mcp_client._session.call_tool.assert_awaited_once_with(
        "issue_write",
        {"owner": "octo", "repo": "demo", "title": "Bug", "method": "create"},
    )


@pytest.mark.anyio
async def test_tools_for_message_routes_create_issue_to_issue_write() -> None:
    from backend.agent import GitHubAgent

    agent = GitHubAgent()
    agent._tools = [
        SimpleNamespace(name="search_issues", description="", inputSchema={}),
        SimpleNamespace(name="issue_write", description="", inputSchema={}),
        SimpleNamespace(name="get_me", description="", inputSchema={}),
    ]
    message = (
        'Create an issue in my repository github-mcp-chatbot titled "Test issue"'
    )

    routed = agent._tools_for_message(message)

    assert [tool.name for tool in routed] == ["issue_write"]


@pytest.mark.anyio
async def test_stream_create_issue_alias_maps_to_issue_write_approval() -> None:
    from backend.agent import AgentStreamEvent, GitHubAgent

    mcp_client = MagicMock()
    mcp_client.connect = AsyncMock()
    mcp_client.list_tools = AsyncMock(
        return_value=[
            SimpleNamespace(
                name="issue_write",
                description="Create or update an issue.",
                inputSchema={"type": "object"},
            )
        ]
    )
    mcp_client.close = AsyncMock()
    mcp_client._session = MagicMock()
    mcp_client._session.call_tool = AsyncMock(
        return_value={"content": [{"text": '{"login":"MooAyman"}'}]}
    )

    agent = GitHubAgent(mcp_client=mcp_client, llm_client=MagicMock())
    tool_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id="create-call-1",
                            function=SimpleNamespace(
                                name="create_issue",
                                arguments=(
                                    '{"method":"create","repo":"github-mcp-chatbot",'
                                    '"title":"Test issue"}'
                                ),
                            ),
                        )
                    ],
                )
            )
        ],
        usage=None,
    )

    with patch.object(agent, "_stream_openai", return_value=iter([tool_chunk])):
        events = [
            event
            async for event in agent.stream_response(
                'Create an issue in my repository github-mcp-chatbot titled "Test issue"'
            )
        ]

    assert len(events) == 1
    assert events[0].approval is not None
    assert events[0].approval.tool is not None
    assert events[0].approval.tool.name == "issue_write"
    assert events[0].approval.approval_required is True
    assert events[0].approval.tool.arguments["owner"] == "MooAyman"
    mcp_client._session.call_tool.assert_awaited_once_with("get_me", {})


@pytest.mark.anyio
async def test_fastapi_chat_create_issue_returns_approval_json() -> None:
    from backend import main
    from backend.agent import AgentDecision, AgentStreamEvent, SelectedTool

    decision = AgentDecision(
        response="",
        tool=SelectedTool(
            name="issue_write",
            arguments={
                "method": "create",
                "owner": "octo",
                "repo": "github-mcp-chatbot",
                "title": "Test issue",
            },
        ),
        approval_required=True,
        tool_call_id="call-create-1",
    )
    agent = MagicMock()

    async def approval_events():
        yield AgentStreamEvent(approval=decision)

    agent.stream_response = MagicMock(side_effect=lambda _: approval_events())
    agent.complete_decision = AsyncMock(return_value="Issue created.")
    main._pending_approvals.clear()

    message = (
        'Create an issue in my repository github-mcp-chatbot titled "Test issue"'
    )

    with patch("backend.main._get_agent", return_value=agent):
        approval_response = await main.chat(
            main.ChatRequest(message=message, session_id="session-create-issue")
        )
        assert approval_response.body is not None
        approval_data = json.loads(approval_response.body)

        assert approval_data["approval_required"] is True
        assert approval_data["tool"]["name"] == "issue_write"
        assert approval_data["tool"]["arguments"]["title"] == "Test issue"
        agent.complete_decision.assert_not_called()

        approve_response = await main.chat(
            main.ChatRequest(
                message=message,
                session_id="session-create-issue",
                approval="approve",
            )
        )
        body = "".join(
            [chunk async for chunk in approve_response.body_iterator]  # type: ignore[attr-defined]
        )
        assert body == "Issue created."

    agent.complete_decision.assert_awaited_once_with(
        message,
        decision,
        approved=True,
    )


@pytest.mark.anyio
async def test_stream_create_issue_approval_payload_resolves_owner() -> None:
    from backend.agent import GitHubAgent

    mcp_client = MagicMock()
    mcp_client.connect = AsyncMock()
    mcp_client.list_tools = AsyncMock(
        return_value=[
            SimpleNamespace(
                name="issue_write",
                description="Create or update an issue.",
                inputSchema={"type": "object"},
            ),
            SimpleNamespace(
                name="get_me",
                description="Get the authenticated user.",
                inputSchema={"type": "object"},
            ),
        ]
    )
    mcp_client.close = AsyncMock()
    mcp_client._session = MagicMock()
    mcp_client._session.call_tool = AsyncMock(
        return_value={"content": [{"text": '{"login":"MooAyman"}'}]}
    )

    agent = GitHubAgent(mcp_client=mcp_client, llm_client=MagicMock())
    tool_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id="create-call-1",
                            function=SimpleNamespace(
                                name="issue_write",
                                arguments=(
                                    '{"method":"create","owner":"github-mcp-chatbot",'
                                    '"repo":"github-mcp-chatbot","title":"Test issue"}'
                                ),
                            ),
                        )
                    ],
                )
            )
        ],
        usage=None,
    )
    message = (
        'Create an issue in my repository github-mcp-chatbot titled "Test issue"'
    )

    with patch.object(agent, "_stream_openai", return_value=iter([tool_chunk])):
        events = [event async for event in agent.stream_response(message)]

    assert len(events) == 1
    assert events[0].approval is not None
    assert events[0].approval.tool is not None
    assert events[0].approval.tool.arguments["owner"] == "MooAyman"
    assert events[0].approval.tool.arguments["repo"] == "github-mcp-chatbot"
    assert events[0].approval.tool.arguments["title"] == "Test issue"
    mcp_client._session.call_tool.assert_awaited_once_with("get_me", {})


@pytest.mark.anyio
async def test_issue_write_resolves_owner_when_missing_or_same_as_repo() -> None:
    from backend.agent import GitHubAgent, SelectedTool

    mcp_client = MagicMock()
    mcp_client.connect = AsyncMock()
    mcp_client.list_tools = AsyncMock(return_value=[])
    mcp_client._session = MagicMock()
    mcp_client._session.call_tool = AsyncMock(
        side_effect=[
            {"content": [{"text": '{"login":"octocat"}'}]},
            {"content": [{"text": "issue created"}]},
        ]
    )

    agent = GitHubAgent(mcp_client=mcp_client, llm_client=MagicMock())
    agent._tools = [
        SimpleNamespace(name="issue_write", description="", inputSchema={}),
        SimpleNamespace(name="get_me", description="", inputSchema={}),
    ]

    await agent.execute_tool(
        SelectedTool(
            name="issue_write",
            arguments={
                "method": "create",
                "owner": "github-mcp-chatbot",
                "repo": "github-mcp-chatbot",
                "title": "Test issue",
            },
        ),
        approved=True,
    )

    assert mcp_client._session.call_tool.await_args_list[0].args == ("get_me", {})
    assert mcp_client._session.call_tool.await_args_list[1].args == (
        "issue_write",
        {
            "method": "create",
            "owner": "octocat",
            "repo": "github-mcp-chatbot",
            "title": "Test issue",
        },
    )


@pytest.mark.anyio
async def test_fastapi_chat_approval_rejects_or_approves_pending_write() -> None:
    from backend import main
    from backend.agent import AgentDecision, AgentStreamEvent, SelectedTool

    decision = AgentDecision(
        response="",
        tool=SelectedTool(name="issue_write", arguments={"title": "Bug"}),
        approval_required=True,
        tool_call_id="call-1",
    )
    agent = MagicMock()
    async def approval_events():
        yield AgentStreamEvent(approval=decision)

    agent.stream_response = MagicMock(side_effect=lambda _: approval_events())
    agent.complete_decision = AsyncMock(return_value="Issue completed")
    main._pending_approvals.clear()

    with patch("backend.main._get_agent", return_value=agent):
        approval_response = await main.chat(
            main.ChatRequest(message="Create an issue", session_id="session-2")
        )
        assert approval_response.body is not None
        approval_data = json.loads(approval_response.body)

        assert approval_data["approval_required"] is True
        assert approval_data["tool"]["name"] == "issue_write"
        agent.complete_decision.assert_not_called()

        await main.chat(
            main.ChatRequest(
                message="Create an issue",
                session_id="session-2",
                approval="reject",
            )
        )
        agent.complete_decision.assert_not_called()

        await main.chat(
            main.ChatRequest(message="Create an issue", session_id="session-2")
        )
        await main.chat(
            main.ChatRequest(
                message="Create an issue",
                session_id="session-2",
                approval="approve",
            )
        )

    agent.complete_decision.assert_awaited_once_with(
        "Create an issue",
        decision,
        approved=True,
    )
