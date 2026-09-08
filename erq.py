"""Print the tools advertised by the GitHub MCP Server."""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from backend.mcp.github_client import GitHubMCPClient


async def main() -> None:
    load_dotenv()
    client = GitHubMCPClient()
    try:
        await client.connect()
        tools = await client.list_tools()
        for tool in tools:
            description = tool.description or "(no description)"
            print(f"{tool.name}: {description}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
