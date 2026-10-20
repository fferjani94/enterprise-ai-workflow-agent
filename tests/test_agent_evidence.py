"""Regression tests for grounded MCP ID usage and final evidence checks."""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from enterprise_ai_workflow_agent.agent.graph import investigate, normalize_tool_arguments


class FakeModel:
    def __init__(self, replies):
        self.replies = iter(replies)

    def bind_tools(self, specs):
        return self

    async def ainvoke(self, messages):
        return next(self.replies)


class FakeMCP:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        data = {
            "search_accounts": [{"id": 1, "name": "Northstar Labs"}],
            "get_account_tickets": [{"id": 2, "account_id": 1, "product_id": 7,
                                     "subject": "SSO login failure", "status": "Open",
                                     "priority": "High"}],
            "search_knowledge": [{"id": 3, "product_id": 7,
                                  "title": "SSO troubleshooting", "snippet": "Check token expiry"}],
        }.get(name, [])
        return SimpleNamespace(isError=False, structuredContent={"result": data}, content=[])


TOOLS = [
    SimpleNamespace(name="search_accounts", description="", inputSchema={
        "type": "object", "properties": {"query": {"type": "string"}},
        "required": ["query"]}),
    SimpleNamespace(name="get_account_tickets", description="", inputSchema={
        "type": "object", "properties": {"account_id": {"type": "integer"}},
        "required": ["account_id"]}),
    SimpleNamespace(name="search_knowledge", description="", inputSchema={
        "type": "object", "properties": {"query": {"type": "string"},
                                          "product_id": {"type": "integer"}},
        "required": ["query"]}),
]


def call(name, args, id_):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_}])


def test_unverified_account_id_is_rejected_without_calling_mcp():
    client = FakeMCP()
    model = FakeModel([call("get_account_tickets", {"account_id": 12345}, "a"),
                       AIMessage(content="No tickets exist.")])
    result = asyncio.run(investigate("Find tickets", model, client, TOOLS))
    assert client.calls == []
    assert result["tool_trace"][0]["success"] is False
    assert result["tool_trace"][0]["record_count"] == 0
    assert "insufficient verified evidence" in result["answer"].lower()


def test_verified_account_and_tickets_are_used_and_reported():
    client = FakeMCP()
    model = FakeModel([
        call("search_accounts", {"query": "Northstar Labs"}, "a"),
        call("get_account_tickets", {"account_id": 1}, "b"),
        AIMessage(content="Northstar Labs has no support tickets."),
        # The contradicting answer goes to synthesis, which contradicts the evidence again.
        AIMessage(content="There are no support tickets for this account."),
    ])
    result = asyncio.run(investigate("Find tickets", model, client, TOOLS))
    assert client.calls[1] == ("get_account_tickets", {"account_id": 1})
    assert result["tool_trace"][1]["record_count"] == 1
    assert 'Ticket 2, "SSO login failure"' in result["answer"]
    assert "contained an unsupported" in result["answer"]
    assert result["answer_source"] == "evidence_summary"


def test_unknown_product_id_is_rejected_but_unfiltered_knowledge_is_allowed():
    client = FakeMCP()
    model = FakeModel([
        call("search_knowledge", {"query": "SSO", "product_id": 12345}, "a"),
        call("search_knowledge", {"query": "SSO"}, "b"),
        AIMessage(content="Knowledge document 3 covers SSO."),
    ])
    result = asyncio.run(investigate("Search SSO docs", model, client, TOOLS))
    assert client.calls == [("search_knowledge", {"query": "SSO"})]
    assert result["tool_trace"][0]["success"] is False
    assert result["tool_trace"][1]["record_count"] == 1
    assert "SSO troubleshooting" in result["answer"]


def test_wrapped_integer_is_normalized():
    result = normalize_tool_arguments(
        {"account_id": {"type": "integer", "value": 1}},
        {"properties": {"account_id": {"type": "integer"}}},
    )
    assert result == {"account_id": 1}


def test_invented_id_in_final_answer_is_replaced():
    client = FakeMCP()
    model = FakeModel([
        call("search_accounts", {"query": "Northstar Labs"}, "a"),
        AIMessage(content="The account ID is 12345."),
    ])
    result = asyncio.run(investigate("Find account", model, client, TOOLS))
    assert 'Account 1, "Northstar Labs"' in result["answer"]
    assert "12345" not in result["answer"]
