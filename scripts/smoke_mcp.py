import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "enterprise_ai_workflow_agent.mcp_server.server"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            tools = await client.list_tools()
            print("Available tools:", ", ".join(tool.name for tool in tools.tools))
            result = await client.call_tool(
                "search_knowledge", {"query": "SSO token expiry", "limit": 3}
            )
            print("Knowledge search:", result)
            if result.isError:
                raise RuntimeError("MCP tool returned an error")


if __name__ == "__main__":
    asyncio.run(main())
