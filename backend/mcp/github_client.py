"""Client for the GitHub MCP Server over stdio (local) or HTTP (Docker Compose)."""

from __future__ import annotations

import os
from contextlib import AsyncExitStack

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

GITHUB_MCP_IMAGE = "ghcr.io/github/github-mcp-server"


class GitHubMCPClient:
    """Connect to GitHub MCP, inspect tools, and close the connection."""

    def __init__(self) -> None:
        self._exit_stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._http_client: httpx.AsyncClient | None = None

    async def connect(self) -> GitHubMCPClient:
        """Initialize MCP over HTTP (Compose) or Docker stdio (local dev)."""
        if self._session is not None:
            return self

        token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        if not token:
            raise RuntimeError("GITHUB_PERSONAL_ACCESS_TOKEN is not set")

        mcp_url = os.getenv("GITHUB_MCP_URL", "").strip()
        await self._exit_stack.__aenter__()
        try:
            if mcp_url:
                await self._connect_http(mcp_url, token)
            else:
                await self._connect_stdio(token)
        except Exception:
            await self.close()
            raise

        return self

    async def _connect_http(self, url: str, token: str) -> None:
        self._http_client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {token}"},
            timeout=60.0,
        )
        self._exit_stack.push_async_callback(self._http_client.aclose)
        read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
            streamable_http_client(url.rstrip("/"), http_client=self._http_client)
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

    async def _connect_stdio(self, token: str) -> None:
        server = StdioServerParameters(
            command="docker",
            args=[
                "run",
                "--rm",
                "-i",
                "-e",
                "GITHUB_PERSONAL_ACCESS_TOKEN",
                GITHUB_MCP_IMAGE,
                "stdio",
            ],
            env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": token},
        )
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(server)
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

    async def list_tools(self):
        """Return the tools advertised by the GitHub MCP Server."""
        if self._session is None:
            raise RuntimeError("GitHub MCP client is not connected")
        return (await self._session.list_tools()).tools

    async def close(self) -> None:
        """Close the MCP session and transport."""
        await self._exit_stack.aclose()
        self._session = None
        self._http_client = None
