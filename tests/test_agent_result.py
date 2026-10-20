"""The agent result exposes structured evidence, trace errors and status signals."""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from enterprise_ai_workflow_agent.agent.graph import investigate


class FakeModel:
    def __init__(self, replies):
        self.replies = iter(replies)

    def bind_tools(self, specs):
        return self

    async def ainvoke(self, messages):
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


DATA = {
    "search_accounts": [{"id": 1, "name": "Northstar Labs", "industry": "Software",
                         "plan": "Enterprise", "status": "active"}],
    "get_ticket_details": {"found": True, "id": 2, "account_id": 1, "product_id": 7,
                           "subject": "SSO login failure", "description": "Token expired",
                           "status": "open", "priority": "high"},
    "search_knowledge": [{"id": 3, "product_id": 7, "title": "SSO troubleshooting",
                          "snippet": "Check token expiry", "score": 4}],
}


class FakeMCP:
    async def call_tool(self, name, args):
        return SimpleNamespace(isError=False, structuredContent={"result": DATA[name]},
                               content=[])


def schema(**props):
    return {"type": "object", "properties": {k: {"type": v} for k, v in props.items()}}


TOOLS = [
    SimpleNamespace(name="search_accounts", description="", inputSchema=schema(query="string")),
    SimpleNamespace(name="get_ticket_details", description="",
                    inputSchema=schema(ticket_id="integer")),
    SimpleNamespace(name="search_knowledge", description="", inputSchema=schema(query="string")),
]


def call(name, args, id_):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_}])


def test_structured_evidence_comes_from_tool_results():
    model = FakeModel([
        call("search_accounts", {"query": "Northstar"}, "a"),
        call("search_knowledge", {"query": "SSO"}, "b"),
        AIMessage(content="Northstar Labs is account 1."),
    ])
    result = asyncio.run(investigate("Find", model, FakeMCP(), TOOLS))
    by_type = {(item["type"], item["id"]): item for item in result["evidence"]}
    account = by_type[("account", 1)]
    assert account["title"] == "Northstar Labs"
    assert account["fields"] == {"industry": "Software", "plan": "Enterprise", "status": "active"}
    document = by_type[("document", 3)]
    assert document["title"] == "SSO troubleshooting"
    assert document["description"] == "Check token expiry"
    assert document["fields"]["product_id"] == 7
    # Grounded product ID referenced by the document, without invented metadata.
    assert by_type[("product", 7)]["title"] is None
    assert result["limited"] is False
    assert result["model_error"] is None


def test_evidence_is_empty_when_nothing_was_retrieved():
    result = asyncio.run(investigate("Hi", FakeModel([AIMessage(content="Hello")]),
                                     FakeMCP(), TOOLS))
    assert result["evidence"] == []


def test_failed_tool_call_records_error_reason():
    model = FakeModel([call("get_ticket_details", {"ticket_id": 99}, "a"),
                       AIMessage(content="Unknown.")])
    result = asyncio.run(investigate("Ticket 99", model, FakeMCP(), TOOLS))
    entry = result["tool_trace"][0]
    assert entry["success"] is False
    assert "Unverified record ID" in entry["error"]
    assert "ticket_id=99" in entry["error"]


def test_successful_tool_call_has_no_error_key():
    model = FakeModel([call("search_accounts", {"query": "N"}, "a"), AIMessage(content="Done")])
    result = asyncio.run(investigate("Find", model, FakeMCP(), TOOLS))
    assert "error" not in result["tool_trace"][0]


def test_round_limit_alone_does_not_mark_the_run_limited():
    model = FakeModel([call("search_accounts", {"query": "N"}, "a"),
                       call("search_accounts", {"query": "N"}, "b")])
    result = asyncio.run(investigate("Find", model, FakeMCP(), TOOLS, max_rounds=1))
    assert result["round_limit_reached"] is True
    assert result["limited"] is False
    assert result["evidence"][0]["type"] == "account"
    # FakeModel has no synthesis reply left, so the deterministic summary is returned.
    assert result["answer_source"] == "evidence_summary"
    assert 'Account 1, "Northstar Labs"' in result["answer"]


def test_model_failure_is_flagged():
    model = FakeModel([ConnectionError("offline")])
    result = asyncio.run(investigate("Find", model, FakeMCP(), TOOLS))
    assert result["model_error"] == "ConnectionError"
    assert "local model failed" in result["answer"].lower()


class FlakyModel(FakeModel):
    """Raises the listed errors first, then answers; records whether the system prompt was sent."""

    def __init__(self, errors):
        super().__init__([])
        self.errors = list(errors)
        self.system_prompt_flags = []

    async def ainvoke(self, messages):
        self.system_prompt_flags.append(type(messages[0]).__name__ == "SystemMessage")
        if self.errors:
            raise self.errors.pop(0)
        return AIMessage(content="Recovered.")


TOKEN_REPEAT = RuntimeError("token repeat limit reached (status code: 500)")


def test_token_repeat_errors_are_retried_with_alternating_prompts():
    model = FlakyModel([TOKEN_REPEAT, TOKEN_REPEAT, TOKEN_REPEAT])
    result = asyncio.run(investigate("Find", model, FakeMCP(), TOOLS))
    assert result["model_error"] is None
    assert result["answer"].startswith("Recovered.")
    assert model.system_prompt_flags == [True, False, True, False]


def test_persistent_token_repeat_errors_stop_after_bounded_attempts():
    model = FlakyModel([TOKEN_REPEAT] * 10)
    result = asyncio.run(investigate("Find", model, FakeMCP(), TOOLS))
    assert result["model_error"] == "RuntimeError"
    assert len(model.system_prompt_flags) == 4


def test_empty_ollama_stream_is_retried():
    model = FlakyModel([ValueError("No data received from Ollama stream.")])
    result = asyncio.run(investigate("Find", model, FakeMCP(), TOOLS))
    assert result["model_error"] is None
    assert model.system_prompt_flags == [True, False]


def test_empty_tool_loop_reply_is_retried_with_alternating_prompts():
    # Observed with qwen2.5:3b: an empty first reply looked like "the model stopped".
    model = FakeModel([AIMessage(content=""), call("search_accounts", {"query": "N"}, "a"),
                       AIMessage(content="Northstar Labs is account 1."),
                       AIMessage(content="Northstar Labs is account 1.")])
    result = asyncio.run(investigate("Find Northstar Labs", model, FakeMCP(), TOOLS))
    assert [entry["tool"] for entry in result["tool_trace"]] == ["search_accounts"]
    assert result["model_error"] is None


def test_persistent_empty_tool_loop_replies_are_not_an_infrastructure_failure():
    model = FakeModel([AIMessage(content="")] * 4)
    result = asyncio.run(investigate("Hello there", model, FakeMCP(), TOOLS))
    assert result["model_error"] is None
    assert result["tool_trace"] == []


def test_other_model_errors_are_not_retried():
    model = FlakyModel([ConnectionError("offline")])
    result = asyncio.run(investigate("Find", model, FakeMCP(), TOOLS))
    assert result["model_error"] == "ConnectionError"
    assert len(model.system_prompt_flags) == 1
