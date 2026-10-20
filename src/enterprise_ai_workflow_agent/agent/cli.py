"""Run a local Ollama model against OpsPilot's stdio MCP server."""

import argparse
import asyncio
import json

from enterprise_ai_workflow_agent.agent.graph import investigate
from enterprise_ai_workflow_agent.agent.service import build_chat_model, mcp_session
from enterprise_ai_workflow_agent.core.config import settings


async def run(question: str, model_name: str, max_rounds: int) -> None:
    try:
        model = build_chat_model(settings, model_name)
    except ImportError as exc:
        raise SystemExit('Install Ollama integration: pip install -e ".[dev,agent]"') from exc

    async with mcp_session(settings) as client:
        tool_list = await client.list_tools()
        result = await investigate(question, model, client, tool_list.tools,
                                   max_rounds=max_rounds)
        print("\nANSWER\n" + str(result["answer"]))
        print("\nTOOL TRACE\n" + json.dumps(result["tool_trace"], indent=2))
        print(f"\nTool rounds: {result['tool_rounds']}")
        print(f"Status: {'limited' if result['limited'] else 'completed'}; "
              f"answer source: {result['answer_source']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigate SaaS issues with local Ollama")
    parser.add_argument("question", help="Customer support investigation question")
    parser.add_argument("--model", default=settings.ollama_model)
    parser.add_argument("--max-rounds", type=int, default=settings.investigation_max_rounds)
    args = parser.parse_args()
    asyncio.run(run(args.question, args.model, args.max_rounds))


if __name__ == "__main__":
    main()
