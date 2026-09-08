"""Print the tools advertised by the GitHub MCP Server."""

from
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
