"""Milestone 5.1B: complete, paged ticket retrieval and coverage-aware answers.

Service and orchestration tests run the real MCP service functions on in-memory SQLite.
No Ollama needed.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from mcp.shared.memory import create_connected_server_and_client_session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from enterprise_ai_workflow_agent.agent.followups import parse_ticket_page, ticket_coverage
from enterprise_ai_workflow_agent.agent.graph import clean_tool_error, investigate
from enterprise_ai_workflow_agent.agent.grounding import (
    COMPLETE,
    FAILED,
    PARTIAL,
    UNRETRIEVED,
    TicketCoverage,
    coverage_statement,
    miscounted_records,
    requested_ticket_statuses,
    unsupported_completeness_claims,
    unsupported_no_ticket_claims,
)
from enterprise_ai_workflow_agent.db.base import Base
from enterprise_ai_workflow_agent.db.models import Account, Product, Ticket
from enterprise_ai_workflow_agent.mcp_server import server, services

OPEN_QUESTION = "List all open support tickets for Northstar Labs."
TWO_ACCOUNTS = ("List all open support tickets for Northstar Labs and Bluebird Commerce, "
                "including their priority.")


def make_sessions(accounts, subject="issue"):
    """{account name: [ticket status, ...]}; accounts get IDs 1, 2, ... in order."""
    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine)
    with sessions.begin() as db:
        product = Product(name="CloudDesk", version="2.4", description="Support workspace")
        db.add(product)
        db.flush()
        for name, statuses in accounts.items():
            account = Account(name=name, industry="Software", plan="Enterprise",
                              status="active")
            db.add(account)
            db.flush()
            for number, status in enumerate(statuses, start=1):
                db.add(Ticket(account_id=account.id, product_id=product.id,
                              subject=f"{subject} {number}", description="Details.",
                              status=status, priority="medium"))
                db.flush()
    return sessions


class ServiceMCP:
    """Runs the real service functions; dict results are returned unwrapped, like FastMCP."""

    def __init__(self, sessions, fail=lambda name, args: False):
        self.sessions = sessions
        self.fail = fail
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        try:
            if self.fail(name, args):
                raise ValueError("Internal error while reading data")
            with self.sessions() as db:
                result = getattr(services, name)(db, **args)
        except (TypeError, ValueError) as exc:
            text = f"Error executing tool {name}: {exc}"
            return SimpleNamespace(isError=True, structuredContent=None,
                                   content=[SimpleNamespace(type="text", text=text)])
        content = result if isinstance(result, dict) else {"result": result}
        return SimpleNamespace(isError=False, structuredContent=content, content=[])


class FakeModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.seen = []

    def bind_tools(self, specs):
        return self

    async def ainvoke(self, messages):
        self.seen.append(list(messages))
        return next(self.replies)


TOOLS = asyncio.run(server.mcp.list_tools())


def call(name, args, id_):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_}])


def run(question, replies, client, max_rounds=4):
    model = FakeModel(replies)
    result = asyncio.run(investigate(question, model, client, TOOLS, max_rounds=max_rounds))
    return result, model


def page(account_id, *status, offset=0):
    args = {"account_id": account_id, "limit": 50}
    if status:
        args["status"] = list(status)
    if offset:
        args["offset"] = offset
    return ("get_account_tickets", args)


def findings(result):
    return [item["text"] for item in result["analysis"]["findings"]]


# --- Service: SQL filtering, counting and paging ---------------------------------------

@pytest.mark.parametrize("count, has_more, next_offset", [
    (0, False, None), (20, False, None), (21, False, None), (120, True, 50),
])
def test_page_metadata_for_matching_ticket_counts(count, has_more, next_offset):
    sessions = make_sessions({"Northstar Labs": ["open"] * count + ["resolved"] * 3})
    with sessions() as db:
        result = services.get_account_tickets(db, 1, status="open", limit=50)
    assert result["status_filter"] == ["open"]
    assert (result["total"], result["offset"], result["limit"]) == (count, 0, 50)
    assert result["returned"] == len(result["tickets"]) == min(count, 50)
    assert (result["has_more"], result["next_offset"]) == (has_more, next_offset)
    assert all(t["status"] == "open" for t in result["tickets"])
    # Metadata precedes the records in the serialized result.
    assert list(result)[-1] == "tickets"


def test_status_is_filtered_in_sql_before_paging():
    sessions = make_sessions({"Northstar Labs": ["resolved"] * 60 + ["open"] * 5})
    with sessions() as db:
        result = services.get_account_tickets(db, 1, status="open", limit=50)
        unfiltered = services.get_account_tickets(db, 1, limit=50)
    assert (result["total"], result["returned"], result["has_more"]) == (5, 5, False)
    assert [t["id"] for t in result["tickets"]] == list(range(61, 66))
    assert (unfiltered["total"], unfiltered["has_more"]) == (65, True)


def test_closed_resolved_unresolved_and_multiple_statuses():
    sessions = make_sessions({"Northstar Labs": ["open", "resolved", "closed", "pending",
                                                 "Open", "RESOLVED"]})
    with sessions() as db:
        def ids(status):
            return [t["id"] for t in services.get_account_tickets(db, 1, status=status)["tickets"]]
        assert ids("closed") == ids("resolved") == [2, 3, 6]
        assert ids("open") == [1, 5]
        assert ids("unresolved") == [1, 4, 5]
        assert ids(["open", "closed"]) == [1, 2, 3, 5, 6]
        assert services.get_account_tickets(db, 1, status="closed")["status_filter"] == [
            "closed", "resolved"]
        assert ids(None) == [1, 2, 3, 4, 5, 6]


def test_pages_are_stable_contiguous_and_free_of_duplicates():
    sessions = make_sessions({"Northstar Labs": ["open"] * 120, "Bluebird Commerce": ["open"]})
    seen = []
    with sessions() as db:
        offset = 0
        while offset is not None:
            result = services.get_account_tickets(db, 1, status="open", limit=50, offset=offset)
            assert result["total"] == 120
            seen += [t["id"] for t in result["tickets"]]
            offset = result["next_offset"]
        beyond = services.get_account_tickets(db, 1, limit=50, offset=500)
    assert seen == sorted(set(seen)) and len(seen) == 120
    assert (beyond["returned"], beyond["has_more"], beyond["total"]) == (0, False, 120)


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": 51}, {"offset": -1},
                                    {"status": "pending"}, {"status": ["open", "later"]}])
def test_invalid_ticket_arguments_are_rejected(kwargs):
    sessions = make_sessions({"Northstar Labs": ["open"]})
    with sessions() as db, pytest.raises(ValueError):
        services.get_account_tickets(db, 1, **kwargs)


async def _real_server(name, arguments):
    async with create_connected_server_and_client_session(server.mcp._mcp_server) as client:
        return await client.call_tool(name, arguments)


def test_real_mcp_tool_returns_parseable_page_and_rejects_bad_bounds(monkeypatch):
    monkeypatch.setattr(server, "SessionLocal",
                        make_sessions({"Northstar Labs": ["open"] * 21 + ["closed"]}))
    result = asyncio.run(_real_server("get_account_tickets",
                                      {"account_id": 1, "status": ["open"], "limit": 50}))
    assert not result.isError
    parsed = parse_ticket_page(result.structuredContent)
    assert (parsed["total"], parsed["returned"], parsed["status_filter"]) == (21, 21, ["open"])
    # Clients without structured output get the same metadata from the text content.
    assert parse_ticket_page([result.content[0].text]) == parsed
    legacy = asyncio.run(_real_server("get_account_tickets", {"account_id": 1}))
    assert parse_ticket_page(legacy.structuredContent)["total"] == 22
    for bad, field in (({"limit": 51}, "limit"), ({"offset": -1}, "offset")):
        result = asyncio.run(_real_server("get_account_tickets", {"account_id": 1, **bad}))
        text = "\n".join(item.text for item in result.content)
        assert result.isError and field in clean_tool_error("get_account_tickets", text)


# --- Coverage from the trace -----------------------------------------------------------

EVIDENCE = {"accounts": {1: {"id": 1, "name": "Northstar Labs"}}, "tickets": {},
            "products": {}, "documents": {}}


def entry(offset, returned, total, status=("open",), origin="orchestrator", success=True):
    args = {"account_id": 1, "status": list(status), "limit": 50, "offset": offset}
    item = {"tool": "get_account_tickets", "origin": origin, "success": success,
            "normalized_arguments": args}
    if success:
        item["page"] = {"account_id": 1, "status_filter": sorted(status), "total": total,
                        "offset": offset, "returned": returned,
                        "ticket_ids": list(range(offset + 1, offset + returned + 1))}
    else:
        item["error"] = "MCP tool returned an error: Internal error"
    return item


def coverage(*trace, statuses=("open",)):
    return ticket_coverage(1, frozenset(statuses), EVIDENCE, list(trace))


def test_limit_one_page_is_partial_and_continues_from_its_end():
    result = coverage(entry(0, 1, 5, origin="model"))
    assert (result.state, result.total, result.next_offset, result.pages) == (PARTIAL, 5, 1, 0)


def test_missing_page_is_partial_even_when_the_last_page_says_no_more():
    result = coverage(entry(0, 50, 120), entry(100, 20, 120))
    assert (result.state, result.next_offset, result.retrieved) == (PARTIAL, 50, 70)


def test_overlapping_contiguous_pages_are_complete_but_inconsistent_totals_are_not():
    assert coverage(entry(0, 50, 90), entry(40, 50, 90)).state == COMPLETE
    assert coverage(entry(0, 50, 90), entry(40, 50, 95)).state == PARTIAL


def test_failed_page_after_a_good_one_is_partial_and_failed_first_page_is_failed():
    assert coverage(entry(0, 50, 120), entry(50, 0, 0, success=False)).state == PARTIAL
    assert coverage(entry(0, 0, 0, success=False)).state == FAILED
    assert coverage().state == UNRETRIEVED


def test_unverified_id_rejection_is_not_a_failed_retrieval():
    rejected = entry(0, 0, 0, success=False)
    rejected["error"] = "Unverified record ID: account_id=1 has not been established"
    assert coverage(rejected).state == UNRETRIEVED


def test_zero_matches_are_complete_only_after_a_successful_query():
    assert (coverage(entry(0, 0, 0)).state, coverage(entry(0, 0, 0)).total) == (COMPLETE, 0)


def test_coverage_is_specific_to_the_status_filter():
    assert coverage(entry(0, 3, 3, status=("closed", "resolved"))).state == UNRETRIEVED
    evidence = {**EVIDENCE, "tickets": {
        1: {"id": 1, "account_id": 1, "status": "open"},
        2: {"id": 2, "account_id": 1, "status": "resolved"}}}
    unfiltered = entry(0, 2, 2, status=())
    result = ticket_coverage(1, frozenset({"open"}), evidence, [unfiltered])
    assert (result.state, result.total) == (COMPLETE, 1)


def test_legacy_list_result_is_never_complete():
    legacy = {"tool": "get_account_tickets", "origin": "model", "success": True,
              "normalized_arguments": {"account_id": 1}}
    evidence = {**EVIDENCE, "tickets": {1: {"id": 1, "account_id": 1, "status": "open"}}}
    result = ticket_coverage(1, frozenset(), evidence, [legacy])
    assert (result.state, result.total, result.retrieved) == (PARTIAL, None, 1)
    assert parse_ticket_page({"result": [{"id": 1, "account_id": 1}]}) is None


def test_inconsistent_page_metadata_is_rejected():
    good = {"account_id": 1, "status_filter": [], "total": 1, "offset": 0, "returned": 1,
            "has_more": False, "next_offset": None, "tickets": [{"id": 7}]}
    assert parse_ticket_page(good)["ticket_ids"] == [7]
    assert parse_ticket_page(json.dumps({"result": good}))["total"] == 1
    assert parse_ticket_page({**good, "returned": 2}) is None
    assert parse_ticket_page({**good, "total": 0}) is None


# --- Orchestration ------------------------------------------------------------------

@pytest.mark.parametrize("count, offsets, state, statement", [
    (0, [0], COMPLETE, "Northstar Labs (Account 1): 0 open tickets (complete)."),
    (20, [0], COMPLETE, "Northstar Labs (Account 1): 20 open tickets retrieved (complete)."),
    (21, [0], COMPLETE, "Northstar Labs (Account 1): 21 open tickets retrieved (complete)."),
    (120, [0, 50], PARTIAL,
     "Northstar Labs (Account 1): 100 of 120 open tickets retrieved; detailed list "
     "incomplete."),
])
def test_orchestrator_pages_each_account_up_to_the_page_cap(count, offsets, state, statement):
    client = ServiceMCP(make_sessions({"Northstar Labs": ["open"] * count + ["resolved"]}))
    result, _ = run(OPEN_QUESTION, [AIMessage(content="Done."),
                                    AIMessage(content="Northstar Labs was investigated.")],
                    client)
    assert client.calls == [("search_accounts", {"query": "Northstar Labs"}),
                            *(page(1, "open", offset=o) for o in offsets)]
    assert result["ticket_coverage"][0]["state"] == state
    assert findings(result)[0] == statement
    assert result["limited"] is (state == PARTIAL)
    if state == PARTIAL:
        assert "100 of 120 open tickets" in result["limited_reason"]
        assert "page limit (2 per account)" in result["limited_reason"]
        assert result["answer_source"] == "evidence_summary"


def test_one_complete_and_one_partial_account_within_four_rounds():
    client = ServiceMCP(make_sessions({"Northstar Labs": ["open"] * 120,
                                       "Bluebird Commerce": ["resolved"] * 2}))
    result, _ = run(TWO_ACCOUNTS, [AIMessage(content="Done.")], client)
    assert client.calls[2:] == [page(1, "open"), page(2, "open"),
                                page(1, "open", offset=50)]
    assert result["tool_rounds"] == 3
    assert [c["state"] for c in result["ticket_coverage"]] == [PARTIAL, COMPLETE]
    assert findings(result)[:2] == [
        "Northstar Labs (Account 1): 100 of 120 open tickets retrieved; detailed list "
        "incomplete.",
        "Bluebird Commerce (Account 2): 0 open tickets (complete)."]
    assert result["limited"] is True
    assert "account 1" in result["limited_reason"]
    assert "account 2" not in result["limited_reason"]


def test_exhausted_round_budget_leaves_the_account_partial_and_says_so():
    invented = [call("get_account_tickets", {"account_id": 100 + i}, f"x{i}") for i in range(3)]
    client = ServiceMCP(make_sessions({"Northstar Labs": ["open"] * 120,
                                       "Bluebird Commerce": ["open"] * 2}))
    result, _ = run(TWO_ACCOUNTS, [*invented, AIMessage(content="Done.")], client)
    assert result["tool_rounds"] == 4
    assert [c["state"] for c in result["ticket_coverage"]] == [PARTIAL, COMPLETE]
    assert "the next page starts at offset 50" in result["limited_reason"]
    assert "tool-round limit (4) was reached" in result["limited_reason"]


def test_failed_second_page_is_partial_and_not_retried():
    client = ServiceMCP(make_sessions({"Northstar Labs": ["open"] * 120}),
                        fail=lambda name, args: args.get("offset") == 50)
    result, _ = run(OPEN_QUESTION, [AIMessage(content="Done.")], client, max_rounds=8)
    assert client.calls[1:] == [page(1, "open"), page(1, "open", offset=50)]
    assert result["ticket_coverage"][0]["state"] == PARTIAL
    assert findings(result)[0].endswith("50 of 120 open tickets retrieved; detailed list "
                                        "incomplete.")
    assert "partially retrieved (50 of 120 open tickets)" in result["limited_reason"]


def test_model_limit_one_does_not_end_retrieval():
    client = ServiceMCP(make_sessions({"Northstar Labs": ["open"] * 30}))
    result, _ = run(OPEN_QUESTION, [
        call("get_account_tickets", {"account_id": 1, "status": "open", "limit": 1}, "b"),
        AIMessage(content="Ticket 1 is open."), AIMessage(content="Ticket 1 is open.")],
        client)
    assert client.calls[0] == ("search_accounts", {"query": "Northstar Labs"})
    assert client.calls[2] == page(1, "open", offset=1)
    assert result["ticket_coverage"][0]["state"] == COMPLETE
    assert result["ticket_coverage"][0]["total"] == 30
    assert result["limited"] is False


@pytest.mark.parametrize("answer, kept", [
    ("These are all the open tickets for Northstar Labs, such as Ticket 1.", False),
    ("Northstar Labs has 50 open tickets in total, starting with Ticket 1.", False),
    ("Northstar Labs has 120 open tickets; 50 were retrieved, starting with Ticket 1.", True),
])
def test_partial_retrieval_claims_are_checked_on_the_displayed_answer(answer, kept):
    client = ServiceMCP(make_sessions({"Northstar Labs": ["open"] * 120}))
    result, _ = run(OPEN_QUESTION, [
        call("get_account_tickets", {"account_id": 1, "status": "open", "limit": 50}, "b"),
        AIMessage(content=answer)], client, max_rounds=2)
    assert result["limited"] is True
    assert (answer in result["answer"]) is kept
    assert result["answer_source"] == ("model" if kept else "evidence_summary")


def test_no_tickets_claim_after_a_failed_lookup_is_withheld():
    client = ServiceMCP(make_sessions({"Northstar Labs": ["open"] * 3}),
                        fail=lambda name, args: name == "get_account_tickets")
    result, _ = run(OPEN_QUESTION, [
        call("search_accounts", {"query": "Northstar Labs"}, "a"),
        call("get_account_tickets", {"account_id": 1, "status": "open"}, "b"),
        AIMessage(content="Northstar Labs has no open tickets.")], client, max_rounds=2)
    assert "has no open tickets" not in result["answer"]
    assert findings(result)[0] == ("Northstar Labs (Account 1): ticket retrieval failed; "
                                   "ticket count unknown.")
    assert result["limited"] is True


def test_per_account_zero_claim_is_kept_when_both_accounts_are_complete():
    client = ServiceMCP(make_sessions({"Northstar Labs": ["open"] * 3,
                                       "Bluebird Commerce": ["resolved"]}))
    answer = "Bluebird Commerce has no open tickets, while Northstar Labs has 3 (Ticket 1)."
    result, _ = run(TWO_ACCOUNTS, [AIMessage(content="Done."), AIMessage(content=answer)],
                    client)
    assert result["answer_source"] == "synthesis"
    assert result["narrative"] == answer
    assert result["limited"] is False


def test_legacy_list_results_never_make_the_run_complete():
    class LegacyMCP(ServiceMCP):
        async def call_tool(self, name, args):
            result = await super().call_tool(name, args)
            if name == "get_account_tickets":
                result.structuredContent = {"result": result.structuredContent["tickets"]}
            return result

    client = LegacyMCP(make_sessions({"Northstar Labs": ["open"]}))
    result, _ = run(OPEN_QUESTION, [AIMessage(content="Done.")], client)
    assert findings(result)[0] == ("Northstar Labs (Account 1): 1 open ticket retrieved; "
                                   "total unknown, detailed list incomplete.")
    assert result["limited"] is True


def test_unresolved_wording_filters_in_sql():
    assert requested_ticket_statuses("How many unresolved support tickets are there?") == {
        "unresolved"}
    client = ServiceMCP(make_sessions({"Northstar Labs": ["open", "pending", "resolved",
                                                         "closed"]}))
    result, _ = run("How many unresolved tickets does Northstar Labs have?",
                    [AIMessage(content="Done."),
                     AIMessage(content="Northstar Labs has 2 unresolved tickets.")], client)
    assert client.calls[1] == page(1, "unresolved")
    assert findings(result)[0] == ("Northstar Labs (Account 1): 2 unresolved tickets "
                                   "retrieved (complete).")
    assert result["narrative"] == "Northstar Labs has 2 unresolved tickets."


def test_recency_request_is_reported_not_answered_by_ticket_id():
    client = ServiceMCP(make_sessions({"Northstar Labs": ["open"] * 3}))
    result, _ = run("Show the 50 most recent tickets for Northstar Labs.",
                    [AIMessage(content="Done.")], client)
    assert result["limited"] is True
    assert "no creation date" in result["limited_reason"]


def test_truncated_tool_message_does_not_affect_coverage():
    client = ServiceMCP(make_sessions({"Northstar Labs": ["open"] * 50}, subject="x" * 180))
    # An investigation, not a plain list, so the model is shown the page after it completes.
    result, model = run("Investigate the open support tickets for Northstar Labs.", [
        call("get_account_tickets", {"account_id": 1, "status": "open", "limit": 50}, "b"),
        AIMessage(content="Done."), AIMessage(content="Ticket 1 is open.")], client)
    shown = [m.content for m in model.seen[1] if isinstance(m, ToolMessage)][-1]
    assert shown.endswith("[TRUNCATED]") and '"total": 50' in shown
    assert result["ticket_coverage"][0]["state"] == COMPLETE
    assert result["tool_trace"][1]["record_count"] == 50
    assert result["limited"] is False


# --- Grounding of answers against coverage --------------------------------------------

def cov(account_id, name, state, total, retrieved=None):
    return TicketCoverage(account_id, name, ("open",), state, total,
                          total if retrieved is None else retrieved)


NORTHSTAR_PARTIAL = cov(1, "Northstar Labs", PARTIAL, 73, 20)
NORTHSTAR_COMPLETE = cov(1, "Northstar Labs", COMPLETE, 3)
BLUEBIRD_ZERO = cov(2, "Bluebird Commerce", COMPLETE, 0)
TICKETS = [{"type": "ticket", "id": i, "relevance": "unassessed",
            "fields": {"status": "open", "priority": "high"}} for i in range(1, 21)]


def test_coverage_statements():
    assert coverage_statement(cov(1, "Northstar Labs", FAILED, None, 0)) == (
        "Northstar Labs (Account 1): ticket retrieval failed; ticket count unknown.")
    assert coverage_statement(cov(2, None, UNRETRIEVED, None, 0)) == (
        "Account 2: tickets not retrieved; ticket count unknown.")


@pytest.mark.parametrize("text, flagged", [
    ("These are all the tickets.", True),
    ("Both accounts were fully investigated.", True),
    ("Here is the complete list of open tickets.", True),
    ("Only the first 20 tickets are listed; the list is not complete.", False),
    ("Ticket 1 is open.", False),
])
def test_completeness_claims_with_partial_coverage(text, flagged):
    assert bool(unsupported_completeness_claims(text, [NORTHSTAR_PARTIAL])) is flagged
    assert unsupported_completeness_claims(text, [NORTHSTAR_COMPLETE]) == []


@pytest.mark.parametrize("text, coverage, flagged", [
    ("Bluebird Commerce has no open tickets, while Northstar Labs has 3.",
     [NORTHSTAR_COMPLETE, BLUEBIRD_ZERO], False),
    ("For Bluebird Commerce, there are no open tickets.", [NORTHSTAR_COMPLETE, BLUEBIRD_ZERO],
     False),
    ("There are no open tickets.", [NORTHSTAR_COMPLETE, BLUEBIRD_ZERO], True),
    ("Northstar Labs has no open tickets.", [cov(1, "Northstar Labs", FAILED, None, 0)], True),
    ("Northstar Labs has no open tickets.", [NORTHSTAR_PARTIAL], True),
    ("Northstar Labs has no other open tickets.", [NORTHSTAR_PARTIAL], False),
])
def test_no_ticket_claims_are_checked_per_account(text, coverage, flagged):
    assert bool(unsupported_no_ticket_claims(text, coverage)) is flagged


@pytest.mark.parametrize("text, problems", [
    ("Northstar Labs has 20 tickets in total.", ["20 tickets"]),
    ("Northstar Labs has 73 open tickets.", []),
    ("20 open tickets were retrieved so far.", []),
    ("Northstar Labs has 21 open tickets.", ["21 tickets"]),
])
def test_ticket_counts_must_match_sql_totals_while_partial(text, problems):
    assert miscounted_records(text, TICKETS, [NORTHSTAR_PARTIAL]) == problems


# --- Final review fixes ------------------------------------------------------------------

@pytest.mark.parametrize("text, problems", [
    ("20 tickets retrieved.", []),
    ("The first 20 tickets.", []),
    ("20 of 73 tickets were retrieved.", []),
    # A retrieval word elsewhere in the sentence does not qualify the count.
    ("Northstar Labs has 20 open tickets; the first is Ticket 1.", ["20 tickets"]),
    ("Northstar Labs has 50 open tickets, listed below.", ["50 tickets"]),
])
def test_retrieved_count_qualifier_must_be_attached_to_the_count(text, problems):
    assert miscounted_records(text, TICKETS, [NORTHSTAR_PARTIAL]) == problems


@pytest.mark.parametrize("text, flagged", [
    ("These are all the open tickets for Northstar Labs, not just the urgent ones.", True),
    ("This is the complete list, showing only open tickets.", True),
    ("Both accounts were fully investigated.", True),
    ("This is not a complete list.", False),
    ("Not all open tickets were retrieved.", False),
    ("Only the first 20 tickets were retrieved.", False),
])
def test_only_a_negation_attached_to_the_claim_disarms_it(text, flagged):
    assert bool(unsupported_completeness_claims(text, [NORTHSTAR_PARTIAL])) is flagged


OPEN_ZERO = cov(1, "Northstar Labs", COMPLETE, 0)
CLOSED_ZERO = TicketCoverage(2, "Bluebird Commerce", ("closed", "resolved"), COMPLETE, 0, 0)
ANY_ZERO = TicketCoverage(1, "Northstar Labs", (), COMPLETE, 0, 0)


@pytest.mark.parametrize("text, coverage, flagged", [
    ("Northstar Labs has no tickets at all.", [OPEN_ZERO], True),
    ("Bluebird Commerce has no tickets.", [CLOSED_ZERO], True),
    ("Bluebird Commerce has no closed tickets.", [CLOSED_ZERO], False),
    ("Bluebird Commerce has no resolved tickets.", [CLOSED_ZERO], False),
    ("Bluebird Commerce has no open tickets.", [CLOSED_ZERO], True),
    ("Northstar Labs has no tickets.", [ANY_ZERO], False),
    ("Bluebird Commerce has no tickets, while Northstar Labs has 3.",
     [NORTHSTAR_COMPLETE, BLUEBIRD_ZERO], True),
    ("Bluebird Commerce has no open tickets, while Northstar Labs has 3.",
     [NORTHSTAR_COMPLETE, BLUEBIRD_ZERO], False),
])
def test_zero_ticket_claims_cover_only_the_retrieved_statuses(text, coverage, flagged):
    assert bool(unsupported_no_ticket_claims(text, coverage)) is flagged


BLUEBIRD_ONE = cov(2, "Bluebird Commerce", COMPLETE, 1)
TWO_ACCOUNT_TICKETS = [
    {"type": "ticket", "id": i, "relevance": "unassessed",
     "fields": {"account_id": 1 if i <= 3 else 2, "status": "open", "priority": "high"}}
    for i in range(1, 5)]


@pytest.mark.parametrize("text, problems", [
    ("Bluebird Commerce has 3 open tickets.", ["3 tickets"]),
    ("Bluebird Commerce has 1 open ticket.", []),
    ("Northstar Labs has 4 open tickets.", ["4 tickets"]),
    ("Northstar Labs has 3 open tickets and Bluebird Commerce has 1 open ticket.", []),
    ("Northstar Labs and Bluebird Commerce have 4 open tickets.", []),
    ("For Bluebird Commerce, there are 3 open tickets.", ["3 tickets"]),
    ("There are 4 open tickets across both accounts.", []),
])
def test_ticket_counts_are_scoped_to_the_named_account(text, problems):
    coverage = [NORTHSTAR_COMPLETE, BLUEBIRD_ONE]
    assert miscounted_records(text, TWO_ACCOUNT_TICKETS, coverage) == problems


@pytest.mark.parametrize("text", [
    "Both accounts were fully investigated.",
    "The investigation is complete.",
    "All requested accounts were checked.",
])
def test_investigation_gaps_make_completeness_claims_unsupported(text):
    assert unsupported_completeness_claims(text, [BLUEBIRD_ONE], outstanding=True)
    assert unsupported_completeness_claims(text, [BLUEBIRD_ONE]) == []


def test_pages_for_another_account_are_rejected():
    page = {"account_id": 1, "status_filter": ["open"], "total": 1, "offset": 0,
            "returned": 1, "tickets": [{"id": 7, "account_id": 1}]}
    assert parse_ticket_page(page, 1)["ticket_ids"] == [7]
    assert parse_ticket_page({**page, "account_id": 2}, 1) is None
    assert parse_ticket_page({**page, "tickets": [{"id": 7, "account_id": 2}]}, 1) is None
    assert parse_ticket_page({**page, "tickets": [{"id": 7}]}, 1) is None
    # Coverage also ignores a stored page whose owner differs from the request.
    foreign = entry(0, 1, 1)
    foreign["page"]["account_id"] = 2
    assert coverage(foreign).state == PARTIAL


def test_mismatched_ticket_owner_never_makes_the_account_complete():
    class WrongOwnerMCP(ServiceMCP):
        async def call_tool(self, name, args):
            result = await super().call_tool(name, args)
            if name == "get_account_tickets" and not result.isError:
                for ticket in result.structuredContent["tickets"]:
                    ticket["account_id"] = 99
            return result

    client = WrongOwnerMCP(make_sessions({"Northstar Labs": ["open"] * 3}))
    result, _ = run(OPEN_QUESTION, [AIMessage(content="Done.")], client)
    assert result["ticket_coverage"][0]["state"] == PARTIAL
    assert result["limited"] is True


def test_displayed_narrative_is_checked_again_after_cleanup(monkeypatch):
    from enterprise_ai_workflow_agent.agent import graph

    monkeypatch.setattr(graph, "qualify_fix_claims",
                        lambda text: (text + " These are all the tickets.", 0))
    evidence = {"accounts": {}, "tickets": {}, "products": {}, "documents": {}}
    narrative, problem = graph._ground_narrative(
        "Northstar Labs was investigated.", evidence, [], [NORTHSTAR_PARTIAL])
    assert narrative == "" and "complete" in problem
    narrative, problem = graph._ground_narrative(
        "Northstar Labs was investigated.", evidence, [], [NORTHSTAR_COMPLETE], outstanding=True)
    assert narrative == "" and "investigation is incomplete" in problem


def test_false_completeness_claim_is_withheld_while_an_account_is_ambiguous():
    client = ServiceMCP(make_sessions({"Northstar Labs": ["open"], "Northstar Robotics": ["open"],
                                       "Bluebird Commerce": ["open"]}))
    claim = "Both accounts were fully investigated: Ticket 3 is the only open ticket."
    result, _ = run("List open tickets for Northstar and Bluebird Commerce.", [
        call("search_accounts", {"query": "Northstar"}, "a"),
        call("search_accounts", {"query": "Bluebird Commerce"}, "b"),
        call("get_account_tickets", {"account_id": 3, "status": "open", "limit": 50}, "c"),
        AIMessage(content=claim)], client)
    # Bluebird Commerce's tickets are complete; the ambiguous name is the outstanding gap.
    assert [c["state"] for c in result["ticket_coverage"]] == [COMPLETE]
    assert result["limited"] is True
    assert claim not in result["answer"]
    assert result["answer_source"] == "evidence_summary"
    assert "no conclusion was generated" in result["answer_notice"]
    assert 'The name "Northstar" matches several verified accounts' in result["limited_reason"]
