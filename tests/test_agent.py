"""Agent tests use fake model and MCP client; no Ollama or network required."""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from enterprise_ai_workflow_agent.agent.graph import investigate


class FakeBoundModel:
    def __init__(self, replies):
        self.replies = iter(replies)

    async def ainvoke(self, messages):
        return next(self.replies)


class FakeModel:
    synthesis = None  # reply of the tool-less synthesis call; None makes it fail

    def __init__(self, replies):
        self.replies = replies
        self.bound = None

    def bind_tools(self, tools):
        self.bound = tools
        return FakeBoundModel(self.replies)

    async def ainvoke(self, messages):
        if self.synthesis is None:
            raise RuntimeError("no synthesis reply scripted")
        return AIMessage(content=self.synthesis)


class FakeMCP:
    def __init__(self, error=False):
        self.calls = []
        self.error = error

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return SimpleNamespace(isError=self.error,
                               structuredContent={"result": [{"id": 1,
                                                                 "name": "Northstar Labs"}]},
                               content=[])


TOOLS = [SimpleNamespace(name="search_accounts", description="Find accounts",
                         inputSchema={"type": "object", "properties": {
                             "query": {"type": "string"}}, "required": ["query"]})]


def test_tool_call_and_grounded_answer():
    model = FakeModel([
        AIMessage(content="", tool_calls=[{"name": "search_accounts",
                                           "args": {"query": "Northstar"}, "id": "call1"}]),
        AIMessage(content="Northstar Labs is account 1 (search_accounts result)."),
    ])
    model.synthesis = "Northstar Labs is account 1."
    client = FakeMCP()
    result = asyncio.run(investigate("Find Northstar Labs", model, client, TOOLS))
    assert result["tool_rounds"] == 1
    assert result["tool_trace"][0]["success"] is True
    assert client.calls == [("search_accounts", {"query": "Northstar"})]
    assert result["answer_source"] == "synthesis"
    assert "account 1" in result["answer"]
    assert model.bound[0]["function"]["name"] == "search_accounts"


def test_unknown_tool_is_not_executed():
    model = FakeModel([
        AIMessage(content="", tool_calls=[{"name": "delete_account",
                                           "args": {"id": 1}, "id": "call2"}]),
        AIMessage(content="The requested tool is not available."),
    ])
    client = FakeMCP()
    result = asyncio.run(investigate("Delete account", model, client, TOOLS))
    assert client.calls == []
    assert result["tool_trace"][0]["success"] is False


def test_tool_error_reaches_agent():
    model = FakeModel([
        AIMessage(content="", tool_calls=[{"name": "search_accounts",
                                           "args": {"query": "Northstar"}, "id": "call3"}]),
        AIMessage(content="Lookup failed; no conclusion."),
    ])
    result = asyncio.run(investigate("Find Northstar", model, FakeMCP(error=True), TOOLS))
    assert result["tool_trace"][0]["success"] is False
    assert "failed" in result["answer"]


def test_tool_round_limit():
    def call(n):
        return AIMessage(content="", tool_calls=[{"name": "search_accounts",
                     "args": {"query": "Northstar"}, "id": f"call{n}"}])
    model = FakeModel([call(1), call(2)])
    client = FakeMCP()
    result = asyncio.run(investigate("Investigate", model, client, TOOLS, max_rounds=1))
    assert result["tool_rounds"] == 1
    assert len(client.calls) == 1
    assert "tool-round limit" in result["answer"]


def test_model_failure_returns_explanation():
    class BrokenModel(FakeModel):
        def bind_tools(self, tools):
            class Broken:
                async def ainvoke(self, messages):
                    raise ConnectionError("Ollama offline")
            return Broken()
    result = asyncio.run(investigate("Investigate", BrokenModel([]), FakeMCP(), TOOLS))
    assert "local model failed" in result["answer"].lower()


def test_invalid_question():
    try:
        asyncio.run(investigate("   ", FakeModel([]), FakeMCP(), TOOLS))
    except ValueError:
        pass
    else:
        raise AssertionError("Blank question must fail")
