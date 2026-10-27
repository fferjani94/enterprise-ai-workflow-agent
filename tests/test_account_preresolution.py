"""Milestone 5.1C: named accounts are searched before the first model turn, model-proposed
IDs must be verified before the model's batch, and model proposals are recorded as
diagnostics. No Ollama needed; tool schemas come from the real MCP server."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from ticket_pages import ticket_page

from enterprise_ai_workflow_agent.agent.graph import ORCHESTRATOR_CALL_PREFIX, investigate
from enterprise_ai_workflow_agent.mcp_server import server

TOOLS = asyncio.run(server.mcp.list_tools())

# Deliberately not the seeded demo accounts.
ACME = {"id": 7, "name": "Acme Widgets", "plan": "Business"}
ZENITH = {"id": 8, "name": "Zenith Health Group", "plan": "Enterprise"}
ZENITH_LABS = {"id": 9, "name": "Zenith Labs", "plan": "Starter"}
TICKETS = [
    {"id": 70, "account_id": 7, "product_id": 1, "subject": "Export fails", "status": "open",
     "priority": "high"},
    {"id": 71, "account_id": 7, "product_id": 1, "subject": "Slow reports",
     "status": "resolved", "priority": "low"},
    {"id": 80, "account_id": 8, "product_id": 1, "subject": "Login loop", "status": "open",
     "priority": "medium"},
]
TWO = ("List all open support tickets for Acme Widgets and Zenith Health Group, "
       "including their priority.")


class Directory:
    """Substring account search, paged tickets and an empty knowledge base."""

    def __init__(self, accounts=(ACME, ZENITH, ZENITH_LABS)):
        self.accounts = list(accounts)
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if name == "search_accounts":
            query = args["query"].casefold()
            result = [a for a in self.accounts if query in a["name"].casefold()]
        elif name == "get_account_tickets":
            result = ticket_page(TICKETS, args)
        else:
            result = []
        return SimpleNamespace(isError=False, structuredContent={"result": result}, content=[])


class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def bind_tools(self, specs):
        return self

    async def ainvoke(self, messages):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def call(name, args, id_):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_}])


def batch(*calls):
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": id_} for name, args, id_ in calls])


def page(account_id, *status):
    args = {"account_id": account_id, "limit": 50}
    if status:
        args["status"] = list(status)
    return ("get_account_tickets", args)


def run(question, replies, client=None, max_rounds=4):
    client = client or Directory()
    model = FakeModel(replies)
    result = asyncio.run(investigate(question, model, client, TOOLS, max_rounds=max_rounds))
    return result, client, model


def searches(client):
    return [args["query"] for name, args in client.calls if name == "search_accounts"]


def ticket_ids(result):
    return {r["id"] for r in result["evidence"] if r["type"] == "ticket"}


def tool_messages(messages):
    return [m for m in messages if isinstance(m, ToolMessage)]


# --- Early resolution --------------------------------------------------------------

def test_two_named_accounts_are_resolved_before_the_first_model_turn():
    result, client, model = run(TWO, [
        AIMessage(content="Both have tickets."),
        AIMessage(content="Ticket 70 (high) and Ticket 80 (medium) are open."),
    ])
    assert client.calls == [("search_accounts", {"query": "Acme Widgets"}),
                            ("search_accounts", {"query": "Zenith Health Group"}),
                            page(7, "open"), page(8, "open")]
    # The model's first turn already sees both verified accounts.
    first = model.seen[0]
    assert [m.content for m in tool_messages(first)] == [
        '{"result": [{"id": 7, "name": "Acme Widgets", "plan": "Business"}]}',
        '{"result": [{"id": 8, "name": "Zenith Health Group", "plan": "Enterprise"}]}']
    assert [e["origin"] for e in result["tool_trace"][:2]] == ["orchestrator"] * 2
    assert "searched before the model planned" in result["tool_trace"][0]["reason"]
    assert ticket_ids(result) == {70, 80}
    assert result["tool_rounds"] == 2
    assert result["limited"] is False
    assert result["answer_source"] == "synthesis"
    assert result["model_proposals"] == []


def test_model_plans_dependent_calls_with_pre_resolved_ids():
    result, client, _ = run(TWO, [
        batch(("get_account_tickets", {"account_id": 7, "status": "open", "limit": 50}, "a"),
              ("get_account_tickets", {"account_id": 8, "status": "open", "limit": 50}, "b")),
        AIMessage(content="Ticket 70 (high) and Ticket 80 (medium) are open."),
    ])
    assert searches(client) == ["Acme Widgets", "Zenith Health Group"]
    assert len(client.calls) == 4
    assert [(p["tool"], p["outcome"], p["id_verification"], p["turn"], p["round"])
            for p in result["model_proposals"]] == [
        ("get_account_tickets", "executed", "before_batch", 1, 2),
        ("get_account_tickets", "executed", "before_batch", 1, 2)]
    assert [c["state"] for c in result["ticket_coverage"]] == ["complete", "complete"]
    assert result["tool_rounds"] == 2
    # The complete list goes straight to synthesis; the model is not asked again.
    assert result["model_turns"] == 1
    assert result["limited"] is False


def test_known_and_unknown_account():
    result, client, _ = run(
        "List open support tickets for Acme Widgets and Globex Corporation.",
        [AIMessage(content="Both have tickets."), AIMessage(content="Ticket 70 is open.")])
    assert client.calls == [("search_accounts", {"query": "Acme Widgets"}),
                            ("search_accounts", {"query": "Globex Corporation"}),
                            page(7, "open")]
    assert ticket_ids(result) == {70}
    assert 'No account matched the search "Globex Corporation".' in result["answer"]
    assert result["limited"] is False


def test_zero_matching_tickets_is_a_complete_result():
    result, client, _ = run("Show the resolved tickets for Zenith Health Group.",
                            [AIMessage(content="None."), AIMessage(content="No tickets.")])
    assert client.calls == [("search_accounts", {"query": "Zenith Health Group"}),
                            page(8, "closed", "resolved")]
    coverage = result["ticket_coverage"][0]
    assert (coverage["account_id"], coverage["state"], coverage["total"]) == (8, "complete", 0)
    assert ticket_ids(result) == set()
    assert result["limited"] is False


def test_ambiguous_account_name_is_searched_but_not_resolved():
    result, client, _ = run("List the open tickets for Zenith.",
                            [AIMessage(content="Ticket 80 is open.")])
    assert client.calls == [("search_accounts", {"query": "Zenith"})]
    assert result["limited"] is True
    assert "Several accounts matched (8, 9)" in result["limited_reason"]


def test_general_question_keeps_the_model_first_path():
    result, client, model = run("How do I troubleshoot export failures?",
                                [AIMessage(content="See the docs."), AIMessage(content="-")])
    first = model.seen[0]
    assert len(first) == 2
    assert isinstance(first[0], SystemMessage) and isinstance(first[1], HumanMessage)
    assert "search_accounts" not in [name for name, _ in client.calls]


def test_lowercase_account_names_fall_back_to_the_model():
    result, client, model = run("list open tickets for acme widgets", [
        call("search_accounts", {"query": "acme widgets"}, "a"),
        AIMessage(content="Account 7."),
        AIMessage(content="Ticket 70 is open."),
    ])
    assert tool_messages(model.seen[0]) == []
    assert result["tool_trace"][0]["origin"] == "model"
    assert client.calls[0] == ("search_accounts", {"query": "acme widgets"})


def test_more_accounts_than_the_cap_are_not_pre_resolved():
    result, client, model = run(
        "Open tickets for Alpha One, Beta Two, Gamma Three and Delta Four?",
        [AIMessage(content="Many tickets.")])
    assert client.calls == []
    assert tool_messages(model.seen[0]) == []
    assert "at most 3 are investigated automatically" in result["limited_reason"]


@pytest.mark.parametrize("max_rounds, pre_resolved, rounds, limited", [
    # One round stays with the model; the follow-up then searches but cannot page tickets.
    (1, False, 1, True),
    (2, True, 2, False),
    (4, True, 2, False),
])
def test_pre_resolution_is_budget_aware(max_rounds, pre_resolved, rounds, limited):
    result, client, model = run(TWO, [
        AIMessage(content="Both have tickets."),
        AIMessage(content="Ticket 70 (high) and Ticket 80 (medium) are open."),
    ], max_rounds=max_rounds)
    assert bool(tool_messages(model.seen[0])) is pre_resolved
    assert searches(client) == ["Acme Widgets", "Zenith Health Group"]
    assert result["tool_rounds"] == rounds
    assert result["limited"] is limited
    if not limited:
        assert ticket_ids(result) == {70, 80}


# --- Conversation history ----------------------------------------------------------

def test_model_receives_a_valid_assistant_tool_sequence():
    _, _, model = run(TWO, [AIMessage(content="Both."), AIMessage(content="Ticket 70.")])
    system, human, request, *results = model.seen[0]
    assert isinstance(system, SystemMessage) and isinstance(human, HumanMessage)
    assert isinstance(request, AIMessage) and request.content == ""
    ids = [c["id"] for c in request.tool_calls]
    assert len(set(ids)) == len(ids) == 2
    assert all(i.startswith(ORCHESTRATOR_CALL_PREFIX) for i in ids)
    assert all(isinstance(m, ToolMessage) for m in results)
    assert [m.tool_call_id for m in results] == ids


def test_empty_replies_after_pre_resolution_still_complete_the_evidence():
    result, _, model = run(TWO, [AIMessage(content="")] * 4 + [
        AIMessage(content="Ticket 70 (high) and Ticket 80 (medium) are open.")])
    # Retries alternate with and without the system prompt; the tool results stay.
    assert isinstance(model.seen[0][0], SystemMessage)
    assert not isinstance(model.seen[1][0], SystemMessage)
    assert len(tool_messages(model.seen[1])) == 2
    assert ticket_ids(result) == {70, 80}
    assert result["limited"] is False


# --- ID verification and corrective feedback ---------------------------------------

def test_unverified_id_after_pre_resolution_is_rejected_with_verified_ids():
    result, client, _ = run(TWO, [
        call("get_account_tickets", {"account_id": 999, "status": "open"}, "a"),
        AIMessage(content="Done."),
        AIMessage(content="Ticket 70 (high) and Ticket 80 (medium) are open."),
    ])
    assert ("get_account_tickets", {"account_id": 999, "status": "open"}) not in client.calls
    error = next(e["error"] for e in result["tool_trace"] if e["origin"] == "model")
    assert error.startswith("Unverified record ID: account_id=999")
    assert "Verified account IDs: 7 (Acme Widgets), 8 (Zenith Health Group)." in error
    assert "Search each account name separately" in error
    assert "Zenith Labs" not in error  # never retrieved in this investigation
    proposal = result["model_proposals"][0]
    assert (proposal["outcome"], proposal["id_verification"]) == ("rejected", "unverified")
    assert ticket_ids(result) == {70, 80}
    assert result["limited"] is False


def test_same_batch_guess_is_rejected_then_accepted_in_a_later_request():
    question = "list open tickets for acme widgets"  # not extracted: model-first path
    tickets = {"account_id": 7, "status": "open", "limit": 50}
    result, client, _ = run(question, [
        batch(("search_accounts", {"query": "acme widgets"}, "a"),
              ("get_account_tickets", tickets, "b")),
        call("get_account_tickets", tickets, "c"),
        AIMessage(content="Ticket 70 is open."),
        AIMessage(content="Ticket 70 is open."),
    ])
    assert client.calls == [("search_accounts", {"query": "acme widgets"}),
                            ("get_account_tickets", tickets)]
    rejected = result["tool_trace"][1]
    assert rejected["error"].startswith("Unverified record ID: account_id=7")
    assert "only returned by another call in the same request" in rejected["error"]
    assert [(p["tool"], p["outcome"], p.get("id_verification"))
            for p in result["model_proposals"]] == [
        ("search_accounts", "executed", None),
        ("get_account_tickets", "rejected", "same_batch"),
        ("get_account_tickets", "executed", "before_batch")]
    # The rejected guess is not a failed retrieval: coverage comes from the executed page.
    assert result["ticket_coverage"][0]["state"] == "complete"
    assert ticket_ids(result) == {70}


def test_orchestrator_lookups_are_not_subject_to_the_model_batch_rule():
    # The orchestrator pages tickets for IDs that earlier rounds verified.
    result, _, _ = run(TWO, [AIMessage(content="Both."), AIMessage(content="Ticket 70.")])
    pages = [e for e in result["tool_trace"] if e["tool"] == "get_account_tickets"]
    assert [(e["origin"], e["success"]) for e in pages] == [("orchestrator", True)] * 2


# --- Model proposal diagnostics -----------------------------------------------------

def test_superseded_model_proposals_are_recorded_but_not_traced():
    repeat = {"query": "Acme Widgets"}
    result, client, _ = run(TWO, [
        call("search_accounts", repeat, "a"),
        AIMessage(content="Ticket 70 (high) and Ticket 80 (medium) are open."),
    ])
    assert searches(client) == ["Acme Widgets", "Zenith Health Group"]
    assert [e["origin"] for e in result["tool_trace"]] == ["orchestrator"] * 4
    assert result["model_proposals"] == [
        {"turn": 1, "round": 2, "tool": "search_accounts", "arguments": repeat,
         "outcome": "superseded", "reason": "Repeats an earlier lookup."}]
    assert [c["state"] for c in result["ticket_coverage"]] == ["complete", "complete"]
    assert result["limited"] is False


def test_calls_proposed_at_the_round_limit_are_recorded_as_rejected():
    result, client, _ = run("What plan is Acme Widgets on?", [
        call("search_accounts", {"query": "Acme"}, "a"),
        call("search_accounts", {"query": "Widgets"}, "b"),
        AIMessage(content="Acme Widgets (Account 7) is on the Business plan."),
    ], max_rounds=2)
    assert searches(client) == ["Acme Widgets", "Acme"]
    assert result["round_limit_reached"] is True
    assert [(p["arguments"], p["outcome"], p.get("reason")) for p in result["model_proposals"]
            ] == [({"query": "Acme"}, "executed", None),
                  ({"query": "Widgets"}, "rejected", "Tool-round limit reached.")]
    assert len(result["tool_trace"]) == 2
    assert result["limited"] is False


def test_diagnostics_do_not_change_coverage_or_grounding():
    replies = [call("search_accounts", {"query": "Acme Widgets"}, "a"),
               AIMessage(content="Ticket 70 (high) and Ticket 80 (medium) are open.")]
    with_noise, _, _ = run(TWO, replies)
    quiet, _, _ = run(TWO, [replies[1], replies[1]])
    assert with_noise["model_proposals"] and not quiet["model_proposals"]
    for key in ("ticket_coverage", "evidence", "analysis", "narrative", "answer_source",
                "limited", "tool_rounds"):
        assert with_noise[key] == quiet[key], key


# --- MCP descriptions ---------------------------------------------------------------

def test_tool_descriptions_guide_account_search_and_ids():
    tools = {tool.name: tool for tool in TOOLS}
    search = tools["search_accounts"].description
    assert "case-insensitive substring" in search and "one account name per query" in search
    properties = tools["get_account_tickets"].inputSchema["properties"]
    assert "search_accounts" in properties["account_id"]["description"]
    assert "status" in properties["status"]["description"]
