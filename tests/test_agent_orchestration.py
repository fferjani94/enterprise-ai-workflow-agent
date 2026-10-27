"""Reliability orchestration: argument bounds, clean tool errors, intent-aware follow-ups
and investigation completeness. No Ollama needed; MCP-level tests use the real server
in memory."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from mcp.shared.memory import create_connected_server_and_client_session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from ticket_pages import ticket_page

from enterprise_ai_workflow_agent.agent.followups import (
    account_references,
    classify_intent,
    knowledge_query,
    plan_evidence,
    unresolved_listed_accounts,
)
from enterprise_ai_workflow_agent.agent.graph import (
    _unmatched_account_searches,
    clamp_bounded_arguments,
    clean_tool_error,
    investigate,
)
from enterprise_ai_workflow_agent.agent.grounding import (
    build_analysis,
    requested_ticket_statuses,
)
from enterprise_ai_workflow_agent.db import seed as seed_module
from enterprise_ai_workflow_agent.db.base import Base
from enterprise_ai_workflow_agent.db.models import Account
from enterprise_ai_workflow_agent.mcp_server import server, services

LIMIT = {"type": "integer", "minimum": 1, "maximum": 20, "default": 10}
TICKET_LIMIT = {"type": "integer", "minimum": 1, "maximum": 50, "default": 10}

TOOLS = [
    SimpleNamespace(name="search_accounts", description="", inputSchema={
        "type": "object", "required": ["query"],
        "properties": {"query": {"type": "string"}, "limit": LIMIT}}),
    SimpleNamespace(name="get_account_tickets", description="", inputSchema={
        "type": "object", "required": ["account_id"],
        "properties": {"account_id": {"type": "integer"}, "limit": TICKET_LIMIT,
                       "offset": {"type": "integer", "minimum": 0, "default": 0}}}),
    SimpleNamespace(name="search_knowledge", description="", inputSchema={
        "type": "object", "required": ["query"],
        "properties": {"query": {"type": "string"},
                       "product_id": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                       "limit": LIMIT}}),
]

NORTHSTAR = {"id": 1, "name": "Northstar Labs", "plan": "Enterprise"}
NORTHWIND = {"id": 2, "name": "Northwind Retail", "plan": "Starter"}
SSO_TICKET = {"id": 11, "account_id": 1, "product_id": 5, "subject": "SSO login failures",
              "status": "open", "priority": "high"}
DASHBOARD_TICKET = {"id": 12, "account_id": 1, "product_id": 5,
                    "subject": "Dashboard loading slowly", "status": "open", "priority": "low"}
SSO_DOC = {"id": 21, "product_id": 5, "title": "SSO troubleshooting",
           "snippet": "Check identity provider certificates.", "score": 4}


class FakeModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.seen = []

    def bind_tools(self, specs):
        return self

    async def ainvoke(self, messages):
        self.seen.append(list(messages))
        return next(self.replies)


class FakeMCP:
    def __init__(self, accounts=(NORTHSTAR,), tickets=(SSO_TICKET,), docs=(SSO_DOC,),
                 failing=()):
        self.data = {"search_accounts": list(accounts), "get_account_tickets": list(tickets),
                     "search_knowledge": list(docs)}
        self.failing = set(failing)
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if name in self.failing:
            text = f"Error executing tool {name}: Internal error while reading data"
            return SimpleNamespace(isError=True, structuredContent=None,
                                   content=[SimpleNamespace(type="text", text=text)])
        result = (ticket_page(self.data[name], args) if name == "get_account_tickets"
                  else self.data[name])
        return SimpleNamespace(isError=False, structuredContent={"result": result},
                               content=[])


def call(name, args, id_):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_}])


def page(account_id, *status, offset=0):
    """The ticket page call the orchestrator plans."""
    args = {"account_id": account_id, "limit": 50}
    if status:
        args["status"] = list(status)
    if offset:
        args["offset"] = offset
    return ("get_account_tickets", args)


def run(question, replies, client=None, max_rounds=4):
    client = client or FakeMCP()
    model = FakeModel(replies)
    result = asyncio.run(investigate(question, model, client, TOOLS, max_rounds=max_rounds))
    return result, client, model


# --- Intent classification ---------------------------------------------------------

@pytest.mark.parametrize("question, tickets, knowledge", [
    ("What plan is Northstar Labs on?", False, False),
    ("Show the account status and industry for Northstar Labs.", False, False),
    ("Investigate SSO login failures for Northstar Labs.", True, False),
    ("Does Northstar Labs have any open tickets?", True, False),
    ("How do I configure SSO?", False, True),
    ("Investigate the SSO incident at Northstar Labs and suggest next steps.", True, True),
])
def test_intent_comes_from_explicit_question_terms(question, tickets, knowledge):
    intent = classify_intent(question)
    assert (intent.tickets, intent.knowledge) == (tickets, knowledge)


def test_knowledge_query_keeps_topic_terms_and_drops_entities_and_guidance_words():
    evidence = {"accounts": {1: NORTHSTAR}, "products": {}, "tickets": {}, "documents": {}}
    question = "Troubleshoot SSO login failures for Northstar Labs and suggest next steps."
    assert knowledge_query(question, evidence) == "sso login failures"


# --- Argument bounds and tool errors -----------------------------------------------

def test_out_of_range_limits_are_clamped_to_schema_bounds():
    schema = {"properties": {"limit": LIMIT}}
    assert clamp_bounded_arguments({"limit": 1000}, schema) == (
        {"limit": 20}, ["limit=1000 is outside the allowed range 1..20; used 20"])
    assert clamp_bounded_arguments({"limit": 0}, schema)[0] == {"limit": 1}
    assert clamp_bounded_arguments({"limit": 5}, schema) == ({"limit": 5}, [])


def test_record_ids_and_non_numeric_values_are_never_clamped():
    schema = {"properties": {"account_id": {"type": "integer", "minimum": 1, "maximum": 20},
                             "limit": LIMIT}}
    args = {"account_id": 1000, "limit": True}
    assert clamp_bounded_arguments(args, schema) == (args, [])
    assert clamp_bounded_arguments({"limit": "1000"}, schema) == ({"limit": "1000"}, [])


def test_model_limit_of_1000_is_clamped_before_the_mcp_call_and_traced():
    result, client, _ = run("What plan is Northstar Labs on?", [
        call("search_accounts", {"query": "Northstar", "limit": 1000}, "a"),
        AIMessage(content="Northstar Labs (account 1) is on Enterprise."),
    ])
    assert client.calls == [("search_accounts", {"query": "Northstar", "limit": 20})]
    entry = result["tool_trace"][0]
    assert entry["arguments"] == {"query": "Northstar", "limit": 1000}
    assert entry["normalized_arguments"] == {"query": "Northstar", "limit": 20}
    assert entry["adjustments"] == ["limit=1000 is outside the allowed range 1..20; used 20"]
    assert entry["success"] is True


def test_pydantic_validation_errors_are_condensed():
    raw = ("Error executing tool search_accounts: 1 validation error for "
           "search_accountsArguments\nlimit\n  Input should be less than or equal to 20 "
           "[type=less_than_equal, input_value=1000, input_type=int]\n    For further "
           "information visit https://errors.pydantic.dev/2.14/v/less_than_equal")
    assert clean_tool_error("search_accounts", raw) == (
        "Invalid arguments. limit: Input should be less than or equal to 20")


def test_mcp_schemas_publish_limit_bounds_but_not_id_bounds():
    tools = {tool.name: tool.inputSchema for tool in asyncio.run(server.mcp.list_tools())}
    for name in ("search_accounts", "search_knowledge"):
        limit = tools[name]["properties"]["limit"]
        assert (limit["minimum"], limit["maximum"]) == (1, 20)
    tickets = tools["get_account_tickets"]["properties"]
    assert (tickets["limit"]["minimum"], tickets["limit"]["maximum"]) == (1, 50)
    assert tickets["offset"]["minimum"] == 0
    assert "maximum" not in tools["get_account_tickets"]["properties"]["account_id"]
    assert "maximum" not in tools["get_ticket_details"]["properties"]["ticket_id"]


async def _call_real_server(name, arguments):
    async with create_connected_server_and_client_session(server.mcp._mcp_server) as client:
        result = await client.call_tool(name, arguments)
    return result.isError, "\n".join(item.text for item in result.content)


def test_real_server_rejects_out_of_range_limit_with_actionable_text():
    is_error, text = asyncio.run(_call_real_server(
        "search_accounts", {"query": "North", "limit": 1000}))
    assert is_error
    assert clean_tool_error("search_accounts", text) == (
        "Invalid arguments. limit: Input should be less than or equal to 20")


def test_real_server_masks_internal_exceptions(monkeypatch):
    class NullSession:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    def broken(db, query, limit):
        raise RuntimeError("sqlite3.OperationalError at C:\\secret\\opspilot.db")

    monkeypatch.setattr(server, "SessionLocal", NullSession)
    monkeypatch.setattr(server.services, "search_accounts", broken)
    is_error, text = asyncio.run(_call_real_server("search_accounts", {"query": "North"}))
    assert is_error
    assert "secret" not in text and "sqlite3" not in text
    assert "Internal error while reading data" in clean_tool_error("search_accounts", text)


# --- Intent-aware follow-ups -------------------------------------------------------

def test_premature_stop_on_ticket_question_triggers_ticket_lookup():
    question = "Investigate SSO login failures for Northstar Labs."
    result, client, model = run(question, [
        call("search_accounts", {"query": "Northstar Labs"}, "a"),
        AIMessage(content="PREMATURE: Northstar Labs is account 1."),
        AIMessage(content="Ticket 11 reports SSO login failures."),
    ])
    assert client.calls[1] == page(1)
    auto = result["tool_trace"][1]
    assert auto["origin"] == "orchestrator"
    assert "account 1" in auto["reason"]
    assert auto["success"] is True and auto["record_count"] == 1
    assert result["tool_trace"][0]["origin"] == "model"
    # The premature answer is replaced, not shown to the model as a prior answer.
    assert not any("PREMATURE" in str(m.content) for m in model.seen[-1])
    assert result["limited"] is False
    assert result["limited_reason"] is None
    assert result["narrative"] == "Ticket 11 reports SSO login failures."
    assert result["tool_rounds"] == 2


def test_account_only_question_does_not_fetch_tickets():
    result, client, _ = run("What plan is Northstar Labs on?", [
        call("search_accounts", {"query": "Northstar Labs"}, "a"),
        AIMessage(content="Northstar Labs (account 1) is on the Enterprise plan."),
    ])
    assert [name for name, _ in client.calls] == ["search_accounts"]
    assert result["limited"] is False


def test_model_that_fetched_tickets_itself_gets_no_extra_calls():
    result, client, _ = run("Investigate SSO login failures for Northstar Labs.", [
        call("search_accounts", {"query": "Northstar"}, "a"),
        call("get_account_tickets", {"account_id": 1, "limit": 5}, "b"),
        AIMessage(content="Ticket 11 reports SSO login failures."),
    ])
    assert len(client.calls) == 2
    assert all(entry["origin"] == "model" for entry in result["tool_trace"])
    assert result["limited"] is False


def test_troubleshooting_question_adds_one_knowledge_search():
    result, client, _ = run(
        "How do I troubleshoot SSO login for Northstar Labs?",
        [AIMessage(content="Check your IdP."), AIMessage(content="Document 21 covers SSO.")],
    )
    assert client.calls == [("search_knowledge", {"query": "sso login northstar labs"})]
    assert result["tool_trace"][0]["origin"] == "orchestrator"
    assert result["limited"] is False


def test_ticket_and_guidance_follow_ups_share_one_round():
    question = "Investigate the SSO incident at Northstar Labs and suggest next steps."
    result, client, _ = run(question, [
        call("search_accounts", {"query": "Northstar Labs"}, "a"),
        AIMessage(content="Found it."),
        AIMessage(content="Ticket 11 and document 21."),
    ])
    assert [name for name, _ in client.calls] == [
        "search_accounts", "get_account_tickets", "search_knowledge"]
    assert client.calls[2][1] == {"query": "sso incident"}
    assert result["tool_rounds"] == 2
    assert result["limited"] is False


def test_named_account_is_chosen_among_several_matches():
    client = FakeMCP(accounts=(NORTHSTAR, NORTHWIND))
    result, client, _ = run("Investigate SSO login failures for Northstar Labs.", [
        call("search_accounts", {"query": "North"}, "a"),
        AIMessage(content="Two accounts."),
        AIMessage(content="Ticket 11."),
    ], client=client)
    assert page(1) in client.calls
    assert not any(args.get("account_id") == 2 for _, args in client.calls)
    assert result["limited"] is False


def test_ambiguous_accounts_are_not_guessed_and_mark_the_run_limited():
    client = FakeMCP(accounts=(NORTHSTAR, NORTHWIND))
    result, client, _ = run("Investigate SSO login failures for North.", [
        call("search_accounts", {"query": "North"}, "a"),
        AIMessage(content="Two accounts matched."),
    ], client=client)
    assert [name for name, _ in client.calls] == ["search_accounts"]
    assert result["limited"] is True
    assert "Several accounts matched (1, 2)" in result["limited_reason"]


def test_follow_up_is_not_repeated_after_it_fails():
    client = FakeMCP(failing={"get_account_tickets"})
    result, client, _ = run("Investigate SSO login failures for Northstar Labs.", [
        call("search_accounts", {"query": "Northstar Labs"}, "a"),
        AIMessage(content="Found the account."),
        AIMessage(content="The ticket lookup failed."),
    ], client=client, max_rounds=8)
    assert [name for name, _ in client.calls] == ["search_accounts", "get_account_tickets"]
    assert result["tool_trace"][1]["error"].startswith("MCP tool returned an error: Internal")
    assert result["limited"] is True
    assert "tickets for verified account 1" in result["limited_reason"]


def test_exhausted_round_budget_reports_missing_tickets_as_limited():
    result, client, _ = run("Investigate SSO login failures for Northstar Labs.", [
        call("search_accounts", {"query": "Northstar Labs"}, "a"),
        AIMessage(content="Northstar Labs is account 1."),
    ], max_rounds=1)
    assert [name for name, _ in client.calls] == ["search_accounts"]
    assert result["limited"] is True
    assert "tickets for verified account 1" in result["limited_reason"]
    assert "tool-round limit (1) was reached" in result["limited_reason"]
    assert "Incomplete investigation:" in result["answer"]


def test_round_limit_with_complete_evidence_is_synthesized_not_limited():
    result, _, model = run("What plan is Northstar Labs on?", [
        call("search_accounts", {"query": "Northstar Labs"}, "a"),
        call("search_accounts", {"query": "Northstar"}, "b"),
        AIMessage(content="Northstar Labs (Account 1) is on the Enterprise plan."),
    ], max_rounds=1)
    assert result["round_limit_reached"] is True
    assert result["limited"] is False and result["limited_reason"] is None
    assert result["answer_source"] == "synthesis"
    assert result["narrative"] == "Northstar Labs (Account 1) is on the Enterprise plan."


def test_retrieved_tickets_without_a_match_are_a_complete_negative_result():
    client = FakeMCP(tickets=(DASHBOARD_TICKET,))
    result, _, _ = run("Investigate SSO login failures for Northstar Labs.", [
        call("search_accounts", {"query": "Northstar Labs"}, "a"),
        call("get_account_tickets", {"account_id": 1}, "b"),
        AIMessage(content="No retrieved ticket concerns SSO."),
    ], client=client)
    assert result["limited"] is False
    assert [item["text"] for item in result["analysis"]["findings"]] == [
        "Northstar Labs (Account 1): 1 ticket retrieved (complete).",
        "No retrieved ticket matches the topic of the question."]


def test_no_matching_account_is_a_complete_negative_result():
    client = FakeMCP(accounts=())
    result, client, _ = run("Investigate SSO login failures for Acme Unknown.", [
        call("search_accounts", {"query": "Acme Unknown"}, "a"),
        AIMessage(content="No such account."),
    ], client=client)
    assert len(client.calls) == 1
    assert result["limited"] is False
    assert "Insufficient verified evidence" in result["answer"]


def test_skipped_account_lookup_is_recovered_from_the_question():
    result, client, _ = run("Does Northstar Labs have open tickets?", [
        AIMessage(content="Yes, several."), AIMessage(content="Ticket 11 is open."),
    ])
    assert client.calls == [("search_accounts", {"query": "Northstar Labs"}),
                            page(1, "open")]
    assert [entry["origin"] for entry in result["tool_trace"]] == ["orchestrator"] * 2
    assert 'account "Northstar Labs"' in result["tool_trace"][0]["reason"]
    assert result["tool_rounds"] == 2
    assert result["limited"] is False
    assert result["answer_source"] == "synthesis"


def test_recovery_runs_with_required_guidance_and_chains_to_tickets():
    # Observed with qwen2.5:3b: the model replied without any tool call, so only the
    # guidance follow-up ran and the run ended limited.
    question = "Northstar Labs reports SSO login failures. What tickets exist and how do we fix it?"
    result, client, _ = run(question, [AIMessage(content="Try this."),
                                       AIMessage(content="Document 21 covers SSO.")])
    assert [name for name, _ in client.calls] == [
        "search_knowledge", "search_accounts", "get_account_tickets"]
    assert result["limited"] is False


def test_account_only_question_recovers_the_account_but_not_tickets():
    result, client, _ = run("What plan is Northstar Labs on?", [
        AIMessage(content="Enterprise, probably."),
        AIMessage(content="Northstar Labs (Account 1) is on the Enterprise plan."),
    ])
    assert client.calls == [("search_accounts", {"query": "Northstar Labs"})]
    assert result["limited"] is False
    assert result["analysis"]["findings"] == [] and result["analysis"]["guidance"] == []


def test_unknown_account_from_recovery_is_a_complete_negative_result():
    client = FakeMCP(accounts=())
    result, client, _ = run("Investigate SSO login failures for Acme Robotics.",
                            [AIMessage(content="No idea.")], client=client)
    assert client.calls == [("search_accounts", {"query": "Acme Robotics"})]
    assert result["limited"] is False
    assert result["answer_source"] == "evidence_summary"
    assert 'No account matched the search "Acme Robotics".' in result["answer"]
    notes = [item["text"] for item in result["analysis"]["unconfirmed"]]
    assert 'No account matched the search "Acme Robotics".' in notes


def test_listed_account_names_are_each_searched_independently():
    client = AccountDirectory(accounts=(NORTHSTAR, NORTHWIND))
    result, client, _ = run("Compare open tickets for Northstar Labs and Northwind Retail.",
                            [AIMessage(content="Both have tickets."),
                             AIMessage(content="Ticket 11 is open.")], client=client)
    assert client.calls == [
        ("search_accounts", {"query": "Northstar Labs"}),
        ("search_accounts", {"query": "Northwind Retail"}),
        page(1, "open"),
        page(2, "open"),
    ]
    assert result["limited"] is False


def test_unlisted_capitalised_phrases_are_not_searched_as_accounts():
    result, client, _ = run("Investigate CloudDesk tickets for Northstar Labs.",
                            [AIMessage(content="Both have tickets.")])
    assert client.calls == []
    assert result["limited"] is True
    assert '"CloudDesk", "Northstar Labs"' in result["limited_reason"]
    assert "Name one account" in result["limited_reason"]
    result, client, _ = run("Investigate CloudDesk tickets for Northstar Labs.", [
        call("search_accounts", {"query": "Northstar Labs"}, "a"),
        AIMessage(content="Found it."), AIMessage(content="Ticket 11.")])
    assert client.calls == [("search_accounts", {"query": "Northstar Labs"}), page(1)]
    assert result["limited"] is False


def test_question_without_an_account_name_is_limited_without_lookup():
    result, client, _ = run("Are there any open incidents?", [AIMessage(content="None.")])
    assert client.calls == []
    assert result["limited"] is True
    assert "names no identifiable account" in result["limited_reason"]


def test_model_invented_account_search_does_not_count_as_a_negative_result():
    # Observed with qwen2.5:3b: no account was named, and the model searched for
    # "SaaS customer with SSO login issues". Nothing matched, but nothing was verified.
    client = FakeMCP(accounts=())
    result, _, _ = run("Investigate the SSO login issues our customer reported. Check their "
                       "tickets.", [call("search_accounts",
                                         {"query": "SaaS customer with SSO login issues"}, "a"),
                                    AIMessage(content="Nothing found.")], client=client)
    assert result["limited"] is True
    assert "names no identifiable account" in result["limited_reason"]


class AccountDirectory(FakeMCP):
    """search_accounts matches account names by substring, like the real MCP tool."""

    async def call_tool(self, name, args):
        if name != "search_accounts":
            return await super().call_tool(name, args)
        self.calls.append((name, args))
        query = str(args.get("query", "")).casefold()
        found = [a for a in self.data["search_accounts"] if query in a["name"].casefold()]
        return SimpleNamespace(isError=False, structuredContent={"result": found}, content=[])


def test_combined_account_query_is_not_a_negative_result_for_each_name():
    # Observed with qwen2.5:3b on the dashboard's "Open tickets across both accounts" example:
    # one search for "Northstar Labs Northwind Retail" matched nothing, and the run was
    # reported as complete with no records.
    client = AccountDirectory(accounts=(NORTHSTAR, NORTHWIND))
    result, client, _ = run(
        "List all open support tickets for Northstar Labs and Northwind Retail.",
        [call("search_accounts", {"query": "Northstar Labs Northwind Retail"}, "a"),
         AIMessage(content="No accounts were found.")], client=client)
    assert client.calls == [
        ("search_accounts", {"query": "Northstar Labs Northwind Retail"}),
        ("search_accounts", {"query": "Northstar Labs"}),
        ("search_accounts", {"query": "Northwind Retail"}),
        page(1, "open"),
        page(2, "open"),
    ]
    assert [e["origin"] for e in result["tool_trace"]] == ["model"] + ["orchestrator"] * 4
    assert 'names the account "Northwind Retail"' in result["tool_trace"][2]["reason"]
    assert result["limited"] is False


def test_partial_name_search_that_matched_nothing_is_a_negative_result():
    client = AccountDirectory(accounts=())
    result, client, _ = run("Investigate SSO login failures for Globex Corporation.", [
        call("search_accounts", {"query": "Globex"}, "a"),
        AIMessage(content="No such account."),
    ], client=client)
    assert client.calls == [("search_accounts", {"query": "Globex"})]
    assert result["limited"] is False


def test_unsearched_names_without_rounds_left_make_the_run_limited():
    # With one round, the required account searches take it; no round is left for tickets.
    client = AccountDirectory(accounts=(NORTHSTAR, NORTHWIND))
    result, client, _ = run(
        "List all open support tickets for Northstar Labs and Northwind Retail.",
        [call("search_accounts", {"query": "Northstar Labs Northwind Retail"}, "a"),
         AIMessage(content="No accounts were found.")], client=client, max_rounds=1)
    assert [args["query"] for _, args in client.calls] == [
        "Northstar Labs", "Northwind Retail", "Northstar Labs Northwind Retail"]
    assert result["tool_rounds"] == 1
    assert result["limited"] is True
    assert "tickets for verified account 1" in result["limited_reason"]
    assert "tickets for verified account 2" in result["limited_reason"]


def test_model_repeating_a_combined_search_and_a_failed_lookup_is_treated_as_stalled():
    # Real qwen2.5:3b trace: the combined search, an invented account ID, then both again.
    # A repeated call cannot add evidence, so it neither runs nor covers required lookups.
    combined = {"query": "Northstar Labs Northwind Retail", "limit": 20}
    client = AccountDirectory(accounts=(NORTHSTAR, NORTHWIND))
    result, client, _ = run(
        "List all open support tickets for Northstar Labs and Northwind Retail.",
        [call("search_accounts", combined, "a"),
         call("get_account_tickets", {"account_id": 12345}, "b"),
         AIMessage(content="", tool_calls=[
             {"name": "search_accounts", "args": combined, "id": "c"},
             {"name": "get_account_tickets", "args": {"account_id": 12345}, "id": "d"}]),
         AIMessage(content="Ticket 11 is open for Northstar Labs.")],
        client=client)
    assert client.calls == [
        ("search_accounts", combined),
        ("search_accounts", {"query": "Northstar Labs"}),
        ("search_accounts", {"query": "Northwind Retail"}),
        page(1, "open"),
        page(2, "open"),
    ]
    assert result["tool_rounds"] == 4
    assert result["limited"] is False
    assert result["answer_source"] == "synthesis"


def test_unknown_account_stays_a_negative_result_when_the_rounds_run_out():
    # Real qwen2.5:3b run: "Globex Corporation" matched nothing, then the model spent the
    # remaining rounds on an invented account ID. The decisive search did run.
    client = AccountDirectory(accounts=())
    invented = [call("get_account_tickets", {"account_id": 12345}, f"x{i}") for i in range(4)]
    result, client, _ = run(
        "Investigate the SSO login failures at Globex Corporation. Check their support tickets.",
        [call("search_accounts", {"query": "Globex Corporation"}, "a"), *invented],
        client=client)
    assert result["round_limit_reached"] is True
    assert result["limited"] is False and result["limited_reason"] is None
    assert result["answer_source"] == "evidence_summary"
    assert result["answer"].startswith("Lookup failed or insufficient verified evidence")
    assert 'No account matched the search "Globex Corporation".' in result["answer"]


def test_round_limit_before_any_record_is_limited():
    invented = [call("get_account_tickets", {"account_id": 99}, f"x{i}") for i in range(3)]
    result, _, _ = run("Tell me about CloudDesk.", invented, max_rounds=2)
    assert result["round_limit_reached"] is True
    assert result["limited"] is True
    assert "before any records were retrieved" in result["limited_reason"]


def test_failed_account_search_gets_one_recovery_attempt_and_no_retry_loop():
    client = FakeMCP(failing={"search_accounts"})
    result, client, _ = run("Does Northstar Labs have open tickets?", [
        call("search_accounts", {"query": "Northstar"}, "a"),
        AIMessage(content="The lookup failed."),
    ], client=client, max_rounds=8)
    assert client.calls == [("search_accounts", {"query": "Northstar"}),
                            ("search_accounts", {"query": "Northstar Labs"})]
    assert result["tool_rounds"] == 2
    assert result["limited"] is True
    assert 'account search for "Northstar Labs" failed' in result["limited_reason"]


# --- Round budget ------------------------------------------------------------------

def test_last_round_runs_required_lookups_before_model_selected_ones():
    question = "Investigate SSO login failures for Northstar Labs and suggest next steps."
    result, client, _ = run(question, [
        call("search_accounts", {"query": "Northstar Labs"}, "a"),
        call("get_account_tickets", {"account_id": 1}, "b"),
        AIMessage(content="Ticket 11 and Document 21 concern SSO."),
    ], max_rounds=2)
    assert [(e["tool"], e["origin"]) for e in result["tool_trace"]] == [
        ("search_accounts", "model"), ("search_knowledge", "orchestrator"),
        ("get_account_tickets", "model")]
    assert result["tool_rounds"] == 2
    assert result["limited"] is False
    assert result["answer_source"] == "synthesis"


def test_last_round_keeps_the_model_request_when_it_covers_required_lookups():
    result, _, _ = run("Investigate SSO login failures for Northstar Labs.", [
        call("search_accounts", {"query": "Northstar Labs"}, "a"),
        call("get_account_tickets", {"account_id": 1, "limit": 5}, "b"),
        AIMessage(content="Ticket 11 concerns SSO."),
    ], max_rounds=2)
    assert all(entry["origin"] == "model" for entry in result["tool_trace"])
    assert result["limited"] is False


def test_symptom_question_without_account_is_complete_with_guidance():
    result, client, _ = run("How do I troubleshoot SSO login errors?", [
        AIMessage(content="Check your IdP."), AIMessage(content="Document 21 covers SSO."),
    ])
    assert [name for name, _ in client.calls] == ["search_knowledge"]
    assert result["limited"] is False


# --- Repeated lookups --------------------------------------------------------------

def test_repeated_successful_lookup_is_not_re_executed():
    result, client, _ = run("What plan is Northstar Labs on?", [
        call("search_accounts", {"query": "Northstar Labs"}, "a"),
        call("search_accounts", {"query": "Northstar Labs", "limit": 1000}, "b"),
        AIMessage(content="Northstar Labs (account 1) is on Enterprise."),
    ])
    # The second call differs only by a clamped limit, so it is a different lookup.
    assert len(client.calls) == 2
    result, client, _ = run("What plan is Northstar Labs on?", [
        call("search_accounts", {"query": "Northstar Labs"}, "a"),
        call("search_accounts", {"query": "Northstar Labs"}, "b"),
        AIMessage(content="Northstar Labs (account 1) is on Enterprise."),
    ])
    assert len(client.calls) == 1
    assert result["tool_trace"][1]["error"].startswith("Duplicate call skipped")
    assert result["limited"] is False


def test_identical_calls_in_one_round_run_once():
    model_turn = AIMessage(content="", tool_calls=[
        {"name": "search_accounts", "args": {"query": "Northstar Labs"}, "id": "a"},
        {"name": "search_accounts", "args": {"query": "Northstar Labs"}, "id": "b"},
    ])
    result, client, _ = run("What plan is Northstar Labs on?",
                            [model_turn, AIMessage(content="Account 1.")])
    assert len(client.calls) == 1
    assert [entry["success"] for entry in result["tool_trace"]] == [True, False]


def test_stalled_model_repeating_lookups_is_replaced_by_required_follow_up():
    result, client, model = run("Investigate SSO login failures for Northstar Labs.", [
        call("search_accounts", {"query": "Northstar Labs"}, "a"),
        call("search_accounts", {"query": "Northstar Labs"}, "b"),
        AIMessage(content="Ticket 11 reports SSO login failures."),
    ])
    assert client.calls == [("search_accounts", {"query": "Northstar Labs"}), page(1)]
    assert [entry["origin"] for entry in result["tool_trace"]] == ["model", "orchestrator"]
    assert result["tool_rounds"] == 2
    assert result["limited"] is False


# --- Multi-account resolution against the real seed data ----------------------------

class SeededMCP:
    """Runs the real MCP service functions on the production seed, in memory."""

    def __init__(self, monkeypatch, extra_accounts=()):
        engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool,
                               connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        self.sessions = sessionmaker(bind=engine, expire_on_commit=False)
        monkeypatch.setattr(seed_module, "SessionLocal", self.sessions)
        seed_module.seed()
        with self.sessions.begin() as db:
            db.add_all(Account(**account) for account in extra_accounts)
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        try:
            with self.sessions() as db:
                result = getattr(services, name)(db, **args)
        except (TypeError, ValueError) as exc:
            text = f"Error executing tool {name}: {exc}"
            return SimpleNamespace(isError=True, structuredContent=None,
                                   content=[SimpleNamespace(type="text", text=text)])
        return SimpleNamespace(isError=False, structuredContent={"result": result},
                               content=[])


TWO_ACCOUNTS = ("List all open support tickets for Northstar Labs and Bluebird Commerce, "
                "including their priority.")
# From db/seed.py: account 1 Northstar Labs (Software, Enterprise), account 2 Bluebird
# Commerce; tickets 1 (Northstar, open, high), 2 (Northstar, resolved, medium) and
# 3 (Bluebird, open, medium).
SEEDED_TICKETS = {1: (1, "open", "high"), 2: (1, "resolved", "medium"),
                  3: (2, "open", "medium")}
# What an "open tickets" question retrieves: the status filter is applied in SQL.
OPEN_TICKETS = {i: t for i, t in SEEDED_TICKETS.items() if t[1] == "open"}


def _records(result, kind):
    return {r["id"]: r for r in result["evidence"] if r["type"] == kind}


def _tickets(result):
    return {i: (r["fields"]["account_id"], r["fields"]["status"], r["fields"]["priority"])
            for i, r in _records(result, "ticket").items()}


def _searches(client):
    return [args.get("query") for name, args in client.calls if name == "search_accounts"]


def test_single_known_account_is_resolved_once(monkeypatch):
    client = SeededMCP(monkeypatch)
    result, client, _ = run(
        "What plan is Northstar Labs on, and what industry does it operate in?",
        [call("search_accounts", {"query": "Northstar Labs"}, "a"),
         AIMessage(content="Northstar Labs (Account 1) is on the Enterprise plan, Software.")],
        client=client)
    assert client.calls == [("search_accounts", {"query": "Northstar Labs"})]
    account = _records(result, "account")[1]
    assert (account["title"], account["fields"]["plan"], account["fields"]["industry"]) == (
        "Northstar Labs", "Enterprise", "Software")
    assert result["limited"] is False


def test_two_named_accounts_are_resolved_and_their_tickets_retrieved(monkeypatch):
    client = SeededMCP(monkeypatch)
    result, client, _ = run(TWO_ACCOUNTS, [
        AIMessage(content="Both have tickets."),
        AIMessage(content="Ticket 1 (high) and Ticket 3 (medium) are open."),
    ], client=client)
    assert client.calls == [
        ("search_accounts", {"query": "Northstar Labs"}),
        ("search_accounts", {"query": "Bluebird Commerce"}),
        page(1, "open"),
        page(2, "open"),
    ]
    assert {i: r["title"] for i, r in _records(result, "account").items()} == {
        1: "Northstar Labs", 2: "Bluebird Commerce"}
    assert _tickets(result) == OPEN_TICKETS
    assert result["tool_rounds"] == 2
    assert result["limited"] is False
    assert "several possible account names" not in result["answer"]
    # "including" is not a topic: the tickets are not labelled unrelated to the question.
    assert result["answer_source"] == "synthesis"
    assert result["narrative"] == "Ticket 1 (high) and Ticket 3 (medium) are open."
    findings = [item["text"] for item in result["analysis"]["findings"]]
    assert findings[:2] == [
        "Northstar Labs (Account 1): 1 open ticket retrieved (complete).",
        "Bluebird Commerce (Account 2): 1 open ticket retrieved (complete)."]
    assert "Ticket 1," in " ".join(findings) and "Ticket 3," in " ".join(findings)
    assert "Ticket 2" not in result["answer"]
    assert [c["state"] for c in result["ticket_coverage"]] == ["complete", "complete"]


def test_superseded_combined_search_is_not_reported_as_unmatched(monkeypatch):
    client = SeededMCP(monkeypatch)
    result, client, _ = run(TWO_ACCOUNTS, [
        call("search_accounts", {"query": "Northstar Labs Bluebird Commerce"}, "a"),
        AIMessage(content="No accounts were found."),
        AIMessage(content="Ticket 1 (high) and Ticket 3 (medium) are open."),
    ], client=client)
    assert _tickets(result) == OPEN_TICKETS
    assert "No account matched" not in result["answer"]
    assert result["limited"] is False


def test_combined_search_with_an_unknown_account_keeps_the_genuine_negative(monkeypatch):
    client = SeededMCP(monkeypatch)
    result, client, _ = run(
        "List open support tickets for Northstar Labs and Globex Corporation.",
        [call("search_accounts", {"query": "Northstar Labs Globex Corporation"}, "a"),
         AIMessage(content="No accounts were found."),
         AIMessage(content="Ticket 1 is open.")], client=client)
    notes = [item["text"] for item in result["analysis"]["unconfirmed"]]
    assert 'No account matched the search "Globex Corporation".' in notes
    assert not any("Northstar Labs Globex" in note for note in notes)
    assert result["limited"] is False


def test_combined_search_note_is_kept_until_a_named_account_is_verified():
    trace = [{"tool": "search_accounts", "success": True, "record_count": 0,
              "normalized_arguments": {"query": "Northstar Labs Bluebird Commerce"}}]
    unverified = {"accounts": {}}
    verified = {"accounts": {1: {"id": 1, "name": "Northstar Labs"}}}
    assert _unmatched_account_searches(trace, unverified) == (
        'No account matched the search "Northstar Labs Bluebird Commerce".',)
    assert _unmatched_account_searches(trace, verified) == ()


def test_unknown_listed_account_does_not_make_known_tickets_unrelated(monkeypatch):
    client = SeededMCP(monkeypatch)
    result, client, _ = run(
        "List all open support tickets for Northstar Labs and Globex Corporation, "
        "including their priority.",
        [AIMessage(content="Both have tickets."),
         AIMessage(content="Ticket 1 (high) is open for Northstar Labs.")], client=client)
    assert _tickets(result) == {1: SEEDED_TICKETS[1]}
    assert {r["relevance"] for r in _records(result, "ticket").values()} == {"unassessed"}
    assert result["answer_source"] == "synthesis"
    assert result["narrative"] == "Ticket 1 (high) is open for Northstar Labs."
    findings = " ".join(item["text"] for item in result["analysis"]["findings"])
    assert "Ticket 1," in findings and "Ticket 2," not in findings
    assert 'No account matched the search "Globex Corporation".' in result["answer"]
    assert result["limited"] is False


def test_topic_terms_still_apply_next_to_an_unknown_listed_account(monkeypatch):
    client = SeededMCP(monkeypatch)
    result, client, _ = run(
        "Investigate SSO login failures for Northstar Labs and Globex Corporation.",
        [AIMessage(content="Both have tickets."),
         AIMessage(content="Ticket 2 is related to the login failures.")], client=client)
    tickets = _records(result, "ticket")
    assert (tickets[1]["relevance"], tickets[2]["relevance"]) == ("relevant", "no_match")
    assert result["answer_source"] == "evidence_summary"
    assert "described records as related" in result["answer_notice"]


def test_ambiguous_listed_name_does_not_make_known_tickets_unrelated(monkeypatch):
    client = SeededMCP(monkeypatch, extra_accounts=[ROBOTICS])
    result, client, _ = run("List open tickets for Northstar and Bluebird Commerce.",
                            [AIMessage(content="Both have tickets.")], client=client)
    assert _records(result, "ticket")[3]["relevance"] == "unassessed"
    findings = " ".join(item["text"] for item in result["analysis"]["findings"])
    assert "Ticket 3," in findings
    assert set(_tickets(result)) == {3}
    assert result["limited"] is True


@pytest.mark.parametrize("question, accounts, unresolved", [
    ("Open tickets for Northstar Labs and Globex Corporation?", {1: NORTHSTAR},
     ("Globex Corporation",)),
    # Nothing in the list is a verified account, so the names may be topics.
    ("Open tickets for CloudDesk and FlowSync?", {}, ()),
    ("Investigate Webhook failures for Northstar Labs.", {1: NORTHSTAR}, ()),
])
def test_only_names_listed_with_a_verified_account_stop_being_topics(
        question, accounts, unresolved):
    evidence = {"accounts": accounts, "tickets": {}, "products": {}, "documents": {}}
    assert unresolved_listed_accounts(question, evidence, []) == unresolved

@pytest.mark.parametrize("question, statuses", [
    ("List all open support tickets for Northstar Labs.", {"open"}),
    ("Show closed tickets for Northstar Labs.", {"closed", "resolved"}),
    ("Why can't users open the dashboard at Northstar Labs?", set()),
    ("Investigate the tickets for Northstar Labs.", set()),
])
def test_ticket_status_filter_comes_from_explicit_wording(question, statuses):
    assert requested_ticket_statuses(question) == statuses


def test_status_filter_without_matching_tickets_says_so():
    records = [{"type": "ticket", "id": 2, "title": "Dashboard loading slowly",
                "description": None, "relevance": "unassessed",
                "fields": {"status": "resolved", "priority": "medium"}}]
    analysis = build_analysis(records, expects_tickets=True,
                              ticket_statuses=frozenset({"open"}))
    assert [item["text"] for item in analysis["findings"]] == [
        "No retrieved ticket has the requested status (open)."]
    assert analysis["context"][0]["evidence"] == [{"type": "ticket", "id": 2}]


def test_second_account_is_searched_after_the_model_resolved_the_first(monkeypatch):
    client = SeededMCP(monkeypatch)
    result, client, _ = run(TWO_ACCOUNTS, [
        call("search_accounts", {"query": "Northstar Labs"}, "a"),
        call("get_account_tickets", {"account_id": 1}, "b"),
        AIMessage(content="Ticket 1 is open with high priority."),
        AIMessage(content="Ticket 1 (high) and Ticket 3 (medium) are open."),
    ], client=client)
    assert client.calls[2:] == [("search_accounts", {"query": "Bluebird Commerce"}),
                                page(2, "open")]
    assert _searches(client).count("Northstar Labs") == 1
    # The model's unfiltered page for account 1 is complete, so it covers "open" too.
    assert _tickets(result) == SEEDED_TICKETS
    assert result["tool_rounds"] == 4
    assert result["limited"] is False


def test_mixed_known_and_unknown_accounts_keep_the_valid_evidence(monkeypatch):
    client = SeededMCP(monkeypatch)
    result, client, _ = run(
        "List open support tickets for Northstar Labs and Globex Corporation.",
        [AIMessage(content="Both have tickets."),
         AIMessage(content="Ticket 1 is open.")], client=client)
    assert client.calls == [
        ("search_accounts", {"query": "Northstar Labs"}),
        ("search_accounts", {"query": "Globex Corporation"}),
        page(1, "open"),
    ]
    assert list(_records(result, "account")) == [1]
    assert _tickets(result) == {1: SEEDED_TICKETS[1]}
    assert 'No account matched the search "Globex Corporation".' in result["answer"]
    assert result["limited"] is False


def test_unknown_account_is_a_clear_negative_result(monkeypatch):
    client = SeededMCP(monkeypatch)
    result, client, _ = run("Investigate the login failures reported by Globex Corporation.",
                            [AIMessage(content="They have SSO issues.")], client=client)
    assert client.calls == [("search_accounts", {"query": "Globex Corporation"})]
    assert result["evidence"] == []
    assert result["answer_source"] == "evidence_summary"
    assert 'No account matched the search "Globex Corporation".' in result["answer"]
    assert result["limited"] is False


ROBOTICS = {"name": "Northstar Robotics", "industry": "Hardware", "plan": "Business",
            "status": "active"}


def test_ambiguous_account_name_is_not_resolved_to_either_match(monkeypatch):
    client = SeededMCP(monkeypatch, extra_accounts=[ROBOTICS])
    result, client, _ = run("List the open tickets for Northstar.",
                            [AIMessage(content="Ticket 1 is open.")], client=client)
    assert client.calls == [("search_accounts", {"query": "Northstar"})]
    assert result["limited"] is True
    assert "Several accounts matched (1, 3)" in result["limited_reason"]


def test_ambiguous_name_listed_with_a_known_account_only_gets_the_known_tickets(monkeypatch):
    client = SeededMCP(monkeypatch, extra_accounts=[ROBOTICS])
    result, client, _ = run("List open tickets for Northstar and Bluebird Commerce.",
                            [AIMessage(content="Both have tickets.")], client=client)
    assert client.calls == [
        ("search_accounts", {"query": "Northstar"}),
        ("search_accounts", {"query": "Bluebird Commerce"}),
        page(2, "open"),
    ]
    assert _tickets(result) == {3: SEEDED_TICKETS[3]}
    assert result["limited"] is True
    assert 'The name "Northstar" matches several verified accounts (1, 3)' in (
        result["limited_reason"])


def test_repeated_identical_account_search_is_not_re_run(monkeypatch):
    client = SeededMCP(monkeypatch)
    northstar = call("search_accounts", {"query": "Northstar Labs"}, "a")
    result, client, _ = run(TWO_ACCOUNTS, [
        northstar, northstar,
        AIMessage(content="Ticket 1 (high) and Ticket 3 (medium) are open."),
    ], client=client)
    assert _searches(client) == ["Northstar Labs", "Bluebird Commerce"]
    assert [e["origin"] for e in result["tool_trace"]] == ["model"] + ["orchestrator"] * 3
    assert _tickets(result) == OPEN_TICKETS
    assert result["tool_rounds"] == 3
    assert result["limited"] is False


# Model turns that fail: invented account IDs, then invalid search arguments.
FAILING_TURNS = [
    AIMessage(content="", tool_calls=[
        {"name": "get_account_tickets", "args": {"account_id": 101}, "id": "f1"},
        {"name": "get_account_tickets", "args": {"account_id": 102}, "id": "f2"}]),
    AIMessage(content="", tool_calls=[
        {"name": "search_accounts", "args": {"name": "Northstar Labs"}, "id": "f3"},
        {"name": "search_accounts", "args": {"name": "Bluebird Commerce"}, "id": "f4"}]),
    call("get_account_tickets", {"account_id": 103}, "f5"),
]


@pytest.mark.parametrize("max_rounds, replies, rounds", [
    # The last two rounds go to the required search -> tickets chain.
    (4, FAILING_TURNS, 4),
    # More budget: the model repeats its failed call, which counts as a stall.
    (8, [*FAILING_TURNS, call("get_account_tickets", {"account_id": 103}, "f6")], 5),
])
def test_failed_model_calls_leave_rounds_for_both_accounts(
        monkeypatch, max_rounds, replies, rounds):
    client = SeededMCP(monkeypatch)
    result, client, _ = run(TWO_ACCOUNTS, [
        *replies, AIMessage(content="Ticket 1 (high) and Ticket 3 (medium) are open."),
    ], client=client, max_rounds=max_rounds)
    failed = [e for e in result["tool_trace"] if not e["success"]]
    assert len(failed) == 5
    assert [e["tool"] for e in result["tool_trace"] if e["origin"] == "orchestrator"] == [
        "search_accounts", "search_accounts", "get_account_tickets", "get_account_tickets"]
    assert len(client.calls) == len({(n, repr(sorted(a.items()))) for n, a in client.calls})
    assert _tickets(result) == OPEN_TICKETS
    assert result["tool_rounds"] == rounds
    assert result["round_limit_reached"] is False
    assert result["limited"] is False and result["limited_reason"] is None


def test_too_many_listed_accounts_are_reported_not_truncated():
    evidence = {"accounts": {}, "tickets": {}, "products": {}, "documents": {}}
    plan = plan_evidence(
        "Open tickets for Alpha One, Beta Two, Gamma Three and Delta Four?", evidence, [])
    assert plan.calls == []
    assert plan.gaps == ["The question names 4 accounts; at most 3 are investigated "
                         "automatically. Name fewer accounts."]


def test_shortened_name_listed_with_another_account_gets_its_tickets(monkeypatch):
    client = SeededMCP(monkeypatch)
    result, client, _ = run("List open tickets for Northstar and Bluebird Commerce.",
                            [AIMessage(content="Both have tickets."),
                             AIMessage(content="Ticket 1 and Ticket 3 are open.")],
                            client=client)
    assert client.calls[:2] == [("search_accounts", {"query": "Northstar"}),
                                ("search_accounts", {"query": "Bluebird Commerce"})]
    assert sorted(args["account_id"] for _, args in client.calls[2:]) == [1, 2]
    assert _tickets(result) == OPEN_TICKETS
    assert result["limited"] is False


@pytest.mark.parametrize("question, names", [
    ("Open tickets for Procter & Gamble and Northstar Labs?",
     ["Procter & Gamble", "Northstar Labs"]),
    ("Open tickets for Northstar Labs & Bluebird Commerce?",
     ["Northstar Labs", "Bluebird Commerce"]),
    ("List tickets for Northstar Labs, Bluebird Commerce.", ["Northstar Labs",
                                                             "Bluebird Commerce"]),
])
def test_account_names_with_ampersands_and_lists(question, names):
    assert account_references(question) == names


class FailsOnce(SeededMCP):
    """The first call of one tool fails, as a transient MCP error would."""

    def __init__(self, monkeypatch, tool):
        super().__init__(monkeypatch)
        self.pending_failure = tool

    async def call_tool(self, name, args):
        if name != self.pending_failure:
            return await super().call_tool(name, args)
        self.pending_failure = None
        self.calls.append((name, args))
        text = f"Error executing tool {name}: Internal error while reading data"
        return SimpleNamespace(isError=True, structuredContent=None,
                               content=[SimpleNamespace(type="text", text=text)])


def test_transient_search_failure_is_recovered_once_but_not_found_is_final(monkeypatch):
    client = FailsOnce(monkeypatch, "search_accounts")
    result, client, _ = run(
        "List open support tickets for Northstar Labs and Globex Corporation.",
        [call("search_accounts", {"query": "Northstar"}, "a"),
         AIMessage(content="The search failed."),
         AIMessage(content="Ticket 1 is open.")], client=client, max_rounds=8)
    assert _searches(client) == ["Northstar", "Northstar Labs", "Globex Corporation"]
    assert page(1, "open") in client.calls
    assert 'No account matched the search "Globex Corporation".' in result["answer"]
    assert result["tool_rounds"] == 3
    assert result["limited"] is False


def test_single_account_model_keeps_control_when_it_searches_the_account(monkeypatch):
    client = SeededMCP(monkeypatch)
    result, client, _ = run("Investigate the open tickets for Northstar Labs.", [
        call("search_knowledge", {"query": "login"}, "a"),
        call("search_accounts", {"query": "Northstar Labs"}, "b"),
        call("get_account_tickets", {"account_id": 1}, "c"),
        AIMessage(content="Ticket 1 is open."),
        AIMessage(content="Ticket 1 is open."),
    ], client=client)
    assert all(e["origin"] == "model" for e in result["tool_trace"])
    assert result["tool_rounds"] == 3
    assert result["limited"] is False