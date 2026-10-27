"""Milestone 5.1C refinement: narrow ticket lists stop once complete, redundant ticket pages
are rejected, and model turns are logged. No Ollama needed."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from test_account_preresolution import (
    ACME,
    TICKETS,
    TOOLS,
    TWO,
    ZENITH,
    ZENITH_LABS,
    Directory,
    FakeModel,
    batch,
    call,
    ticket_ids,
)

from enterprise_ai_workflow_agent.agent.followups import (
    REDUNDANT_PAGE_ERROR,
    list_question_satisfied,
)
from enterprise_ai_workflow_agent.agent.graph import investigate

SYNTH = "Ticket 70 (high) and Ticket 80 (medium) are open."
EXPLORE = "Investigate the open support tickets for Acme Widgets."


def tickets(account_id, status=None, limit=50, offset=0):
    args = {"account_id": account_id, "limit": limit}
    if status:
        args["status"] = status
    if offset:
        args["offset"] = offset
    return args


def run(question, replies, client=None, max_rounds=4):
    client = client or Directory()
    result = asyncio.run(investigate(question, FakeModel(replies), client, TOOLS,
                                     max_rounds=max_rounds))
    return result, client


def ticket_calls(client):
    return [args for name, args in client.calls if name == "get_account_tickets"]


class FailsTicketsOnce(Directory):
    def __init__(self):
        super().__init__()
        self.failed = False

    async def call_tool(self, name, args):
        if name == "get_account_tickets" and not self.failed:
            self.failed = True
            self.calls.append((name, args))
            text = f"Error executing tool {name}: Internal error while reading data"
            return SimpleNamespace(isError=True, structuredContent=None,
                                   content=[SimpleNamespace(type="text", text=text)])
        return await super().call_tool(name, args)


# --- P1: narrow ticket lists stop once complete ---------------------------------------

def test_two_account_list_stops_after_complete_retrieval():
    result, client = run(TWO, [
        batch(("get_account_tickets", tickets(7, "open"), "a"),
              ("get_account_tickets", tickets(8, "open"), "b")),
        AIMessage(content=SYNTH),
    ])
    assert (result["model_turns"], result["tool_rounds"]) == (1, 2)
    assert result["round_limit_reached"] is False
    assert result["answer_source"] == "synthesis" and result["narrative"] == SYNTH
    assert ticket_ids(result) == {70, 80}
    assert [name for name, _ in client.calls].count("get_product_information") == 0
    assert result["limited"] is False


def test_zero_result_status_filtered_list_stops_after_complete_retrieval():
    result, client = run("Show the resolved tickets for Zenith Health Group.", [
        call("get_account_tickets", tickets(8, "resolved"), "a"),
        AIMessage(content="Zenith Health Group has no resolved tickets."),
    ])
    assert (result["model_turns"], result["tool_rounds"]) == (1, 2)
    coverage = result["ticket_coverage"][0]
    assert (coverage["state"], coverage["total"]) == ("complete", 0)
    assert result["narrative"] == "Zenith Health Group has no resolved tickets."
    assert result["limited"] is False


@pytest.mark.parametrize("max_rounds, rounds, turns, limited", [
    (1, 1, 2, True),  # no pre-resolution: the model's searches use the only round
    (2, 2, 1, False),
    (4, 2, 1, False),
])
def test_list_stop_is_safe_for_every_round_budget(max_rounds, rounds, turns, limited):
    first = (batch(("search_accounts", {"query": "Acme Widgets"}, "s1"),
                   ("search_accounts", {"query": "Zenith Health Group"}, "s2"))
             if max_rounds == 1 else
             batch(("get_account_tickets", tickets(7, "open"), "a"),
                   ("get_account_tickets", tickets(8, "open"), "b")))
    result, client = run(TWO, [first, AIMessage(content=SYNTH), AIMessage(content=SYNTH)],
                         max_rounds=max_rounds)
    assert (result["tool_rounds"], result["model_turns"]) == (rounds, turns)
    assert result["limited"] is limited
    if not limited:
        assert ticket_ids(result) == {70, 80}


def test_troubleshooting_question_still_gets_details_and_knowledge():
    question = "Investigate export failures for Acme Widgets. Check their open support tickets."
    result, client = run(question, [
        call("get_account_tickets", tickets(7, "open"), "a"),
        batch(("get_ticket_details", {"ticket_id": 70}, "b"),
              ("search_knowledge", {"query": "export failures"}, "c")),
        AIMessage(content="Done."), AIMessage(content="Ticket 70 reports export failures."),
    ])
    names = [name for name, _ in client.calls]
    assert "get_ticket_details" in names and "search_knowledge" in names
    assert result["model_turns"] == 3


@pytest.mark.parametrize("question", [
    "List all open support tickets for Acme Widgets with their descriptions.",
    "Explain the open support tickets for Acme Widgets.",
    "List the open support tickets for Acme Widgets and recommend what to do.",
    "List the open support tickets for Acme Widgets and their product information.",
    "Show the open support tickets for Acme Widgets and any guidance.",
    "Why are there open support tickets for Acme Widgets?",
])
def test_exploratory_ticket_questions_are_not_stopped(question):
    result, client = run(question, [
        call("get_account_tickets", tickets(7, "open"), "a"),
        call("get_product_information", {"product_id": 1}, "b"),
        AIMessage(content="Done."), AIMessage(content="Done."), AIMessage(content="Done."),
    ])
    assert ("get_product_information", {"product_id": 1}) in client.calls
    assert result["model_turns"] >= 2


def test_unknown_listed_account_keeps_the_model_in_the_loop():
    result, _ = run("List open support tickets for Acme Widgets and Globex Corporation.", [
        call("get_account_tickets", tickets(7, "open"), "a"),
        AIMessage(content="Ticket 70 is open."), AIMessage(content="Ticket 70 is open."),
    ])
    assert result["model_turns"] == 2
    assert ticket_ids(result) == {70}


# --- P1: helper preconditions -------------------------------------------------------

def _search(query, success=True, count=1):
    return {"tool": "search_accounts", "origin": "orchestrator", "success": success,
            "record_count": count, "normalized_arguments": {"query": query}}


def _page(account_id, total, ids, status=("open",), returned=None):
    returned = len(ids) if returned is None else returned
    return {"tool": "get_account_tickets", "origin": "model", "success": True,
            "normalized_arguments": {"account_id": account_id, "status": list(status),
                                     "limit": 50},
            "page": {"account_id": account_id, "status_filter": sorted(status),
                     "total": total, "offset": 0, "returned": returned, "ticket_ids": ids}}


def _evidence(*accounts, tickets_=()):
    return {"accounts": {a["id"]: a for a in accounts},
            "tickets": {t["id"]: t for t in tickets_}, "products": {}, "documents": {}}


ONE = "List the open support tickets for Acme Widgets."


def test_helper_accepts_a_complete_list():
    trace = [_search("Acme Widgets"), _page(7, 1, [70])]
    assert list_question_satisfied(ONE, _evidence(ACME, tickets_=TICKETS[:1]), trace)


@pytest.mark.parametrize("question, evidence, trace", [
    # Partial list: two pages needed.
    (ONE, _evidence(ACME, tickets_=TICKETS[:1]),
     [_search("Acme Widgets"), _page(7, 60, [70])]),
    # No ticket retrieval yet.
    (ONE, _evidence(ACME), [_search("Acme Widgets")]),
    # Ambiguous name.
    ("List the open support tickets for Zenith.", _evidence(ZENITH, ZENITH_LABS),
     [_search("Zenith", count=2)]),
    # Unknown listed account next to a complete one.
    ("List open support tickets for Acme Widgets and Globex Corporation.",
     _evidence(ACME, tickets_=TICKETS[:1]),
     [_search("Acme Widgets"), _search("Globex Corporation", count=0), _page(7, 1, [70])]),
    # Failed account search.
    (ONE, _evidence(), [_search("Acme Widgets", success=False, count=0)]),
    # Too many accounts for automatic investigation.
    ("Open support tickets for Alpha One, Beta Two, Gamma Three and Acme Widgets?",
     _evidence(ACME, tickets_=TICKETS[:1]), [_search("Acme Widgets"), _page(7, 1, [70])]),
    # Recency cannot be answered from ticket IDs.
    ("List the latest open support tickets for Acme Widgets.",
     _evidence(ACME, tickets_=TICKETS[:1]), [_search("Acme Widgets"), _page(7, 1, [70])]),
])
def test_helper_rejects_incomplete_or_unresolved_lists(question, evidence, trace):
    assert not list_question_satisfied(question, evidence, trace)


# --- P2: redundant ticket pages -----------------------------------------------------

def test_redundant_page_in_the_same_batch_is_rejected_without_mcp():
    result, client = run("Show the resolved tickets for Zenith Health Group.", [
        batch(("get_account_tickets", tickets(8, "resolved"), "a"),
              ("get_account_tickets", tickets(8, "resolved", offset=10), "b")),
        AIMessage(content="Zenith Health Group has no resolved tickets."),
    ])
    assert ticket_calls(client) == [tickets(8, "resolved")]
    rejected = result["tool_trace"][-1]
    assert rejected["executed"] is False and rejected["success"] is False
    assert rejected["error"].startswith(REDUNDANT_PAGE_ERROR)
    assert "All 0 resolved or closed tickets for account 8" in rejected["error"]
    assert [p["outcome"] for p in result["model_proposals"]] == ["executed", "rejected"]
    assert result["ticket_coverage"][0]["state"] == "complete"
    assert "Lookup failed" not in result["answer"]


def test_redundant_page_in_a_later_batch_is_rejected():
    result, client = run(EXPLORE, [
        call("get_account_tickets", tickets(7, "open"), "a"),
        call("get_account_tickets", tickets(7, "open", offset=10), "b"),
        AIMessage(content="Done."), AIMessage(content="Ticket 70 is open."),
    ])
    assert ticket_calls(client) == [tickets(7, "open")]
    assert result["model_proposals"][1]["outcome"] == "rejected"
    assert result["model_proposals"][1]["round"] == 3
    assert result["ticket_coverage"][0]["state"] == "complete"


def test_complete_open_list_does_not_block_an_unfiltered_request():
    _, client = run(EXPLORE, [
        call("get_account_tickets", tickets(7, "open"), "a"),
        call("get_account_tickets", tickets(7), "b"),
        AIMessage(content="Done."), AIMessage(content="Ticket 70 is open."),
    ])
    assert ticket_calls(client) == [tickets(7, "open"), tickets(7)]


def test_complete_unfiltered_list_satisfies_an_open_request():
    result, client = run(EXPLORE, [
        call("get_account_tickets", tickets(7), "a"),
        call("get_account_tickets", tickets(7, "open"), "b"),
        AIMessage(content="Done."), AIMessage(content="Ticket 70 is open."),
    ])
    assert ticket_calls(client) == [tickets(7)]
    assert result["tool_trace"][-1]["error"].startswith(REDUNDANT_PAGE_ERROR)


def test_partial_list_permits_the_next_page():
    _, client = run("Investigate the support tickets for Acme Widgets.", [
        call("get_account_tickets", tickets(7, limit=1), "a"),
        call("get_account_tickets", tickets(7, limit=1, offset=1), "b"),
        AIMessage(content="Done."), AIMessage(content="Tickets 70 and 71."),
    ])
    assert ticket_calls(client) == [tickets(7, limit=1), tickets(7, limit=1, offset=1)]


def test_failed_retrieval_is_not_treated_as_complete():
    client = FailsTicketsOnce()
    result, client = run(EXPLORE, [
        call("get_account_tickets", tickets(7, "open", limit=40), "a"),
        call("get_account_tickets", tickets(7, "open"), "b"),
        AIMessage(content="Done."), AIMessage(content="Ticket 70 is open."),
    ], client=client)
    assert ticket_calls(client) == [tickets(7, "open", limit=40), tickets(7, "open")]
    assert result["ticket_coverage"][0]["state"] == "complete"


def test_invalid_status_is_left_to_the_mcp_validation():
    result, client = run(EXPLORE, [
        call("get_account_tickets", tickets(7, "open"), "a"),
        call("get_account_tickets", tickets(7, "pending"), "b"),
        AIMessage(content="Done."), AIMessage(content="Ticket 70 is open."),
    ])
    assert tickets(7, "pending") in ticket_calls(client)
    assert result["tool_trace"][-1]["executed"] is True


# --- P3: model-turn diagnostics -----------------------------------------------------

def test_text_only_turn_discarded_by_a_follow_up_is_logged():
    result, _ = run(TWO, [AIMessage(content="Both have tickets."), AIMessage(content=SYNTH)])
    assert result["model_proposals"] == []
    assert result["model_turns"] == 1
    assert result["model_turn_log"] == [{"turn": 1, "response": "text", "tool_calls": 0,
                                         "invalid_tool_calls": 0, "discarded": True}]


def test_malformed_tool_calls_are_logged_without_affecting_evidence():
    malformed = AIMessage(content="", invalid_tool_calls=[
        {"name": "get_account_tickets", "args": '{"account_id": 7,', "id": "x",
         "error": "invalid JSON", "type": "invalid_tool_call"}])
    noisy, _ = run(TWO, [malformed, AIMessage(content="Both."), AIMessage(content=SYNTH)])
    quiet, _ = run(TWO, [AIMessage(content="Both."), AIMessage(content=SYNTH)])
    assert noisy["model_proposals"] == [
        {"turn": 1, "round": None, "tool": "get_account_tickets", "arguments": None,
         "outcome": "rejected",
         "reason": "Malformed tool call: its arguments could not be parsed."}]
    assert noisy["model_turn_log"][0]["invalid_tool_calls"] == 1
    assert noisy["model_turn_log"][0]["response"] == "text"
    for key in ("tool_trace", "evidence", "ticket_coverage", "analysis", "narrative",
                "limited", "tool_rounds"):
        assert noisy[key] == quiet[key], key


def test_executed_rejected_and_superseded_calls_are_counted():
    result, client = run(TWO, [
        batch(("get_account_tickets", tickets(7, "open"), "a"),
              ("get_account_tickets", tickets(999, "open"), "b")),
        call("search_accounts", {"query": "Acme Widgets"}, "c"),
        AIMessage(content=SYNTH),
    ])
    outcomes = [p["outcome"] for p in result["model_proposals"]]
    assert outcomes == ["executed", "rejected", "superseded"]
    executed = [e for e in result["tool_trace"] if e["executed"]]
    assert len(executed) == len(client.calls) == 4
    assert [e["executed"] for e in result["tool_trace"] if not e["success"]] == [False]
    assert [(t["turn"], t["response"], t["tool_calls"], t["discarded"])
            for t in result["model_turn_log"]] == [
        (1, "tool_calls", 2, False), (2, "tool_calls", 1, True)]
    assert result["model_turns"] == 2
    assert ticket_ids(result) == {70, 80}
