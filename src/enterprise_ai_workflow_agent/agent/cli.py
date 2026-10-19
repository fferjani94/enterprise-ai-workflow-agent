"""Run a local Ollama model against OpsPilot's stdio MCP server."""

import argparse
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from enterprise_ai_workflow_agent.agent.graph import investigate


async def run(question: str, model_name: str, max_rounds: int) -> None:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise SystemExit('Install Ollama integration: pip install -e ".[dev,agent]"') from exc

    model = ChatOllama(
       model=model_name,
       base_url=os.getenv(
           "OLLAMA_BASE_URL",
           "http://127.0.0.1:11434",
           ),
       temperature=0.2,
       num_predict=512,
       num_ctx=4096,
       )
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "enterprise_ai_workflow_agent.mcp_server.server"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            tool_list = await client.list_tools()
            result = await investigate(question, model, client, tool_list.tools,
                                       max_rounds=max_rounds)
            print("\nANSWER\n" + str(result["answer"]))
            print("\nTOOL TRACE\n" + json.dumps(result["tool_trace"], indent=2))
            print(f"\nTool rounds: {result['tool_rounds']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigate SaaS issues with local Ollama")
    parser.add_argument("question", help="Customer support investigation question")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))
    parser.add_argument("--max-rounds", type=int, default=4)
    args = parser.parse_args()
    asyncio.run(run(args.question, args.model, args.max_rounds))


if __name__ == "__main__":
    main()
