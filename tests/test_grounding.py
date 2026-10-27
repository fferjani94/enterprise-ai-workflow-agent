"""Regression tests: unrelated account tickets must not be presented as evidence of the issue."""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage
from ticket_pages import ticket_page

from enterprise_ai_workflow_agent.agent.graph import investigate
from enterprise_ai_workflow_agent.agent.grounding import (
    _same_term,
    classify_relevance,
    topic_terms,
)

SSO_QUESTION = (
    "Investigate the SSO login issues affecting Northstar Labs. Check their support tickets "
    "and the documentation, and cite the evidence."
)

# Mirrors what the real MCP tools return for the seeded Northstar Labs data.
DATA = {
    "search_accounts": [{"id": 1, "name": "Northstar Labs", "industry": "Software",
                         "plan": "Enterprise", "status": "active"}],
    "get_account_tickets": [
        {"id": 1, "account_id": 1, "product_id": 1, "subject": "Intermittent login failures",
         "status": "open", "priority": "high"},
        {"id": 2, "account_id": 1, "product_id": 1, "subject": "Dashboard loading slowly",
         "status": "resolved", "priority": "medium"},
    ],
    "search_knowledge": [{"id": 1, "product_id": 1, "title": "SSO troubleshooting",
                          "snippet": "Check identity provider settings and token expiry.",
                          "source_url": "https://docs.example.invalid/clouddesk/sso", "score": 3}],
}


class FakeModel:
    def __init__(self, replies):
        self.replies = iter(replies)

    def bind_tools(self, specs):
        return self

    async def ainvoke(self, messages):
        return next(self.replies)


class FakeMCP:
    async def call_tool(self, name, args):
        result = (ticket_page(DATA[name], args) if name == "get_account_tickets"
                  else DATA[name])
        return SimpleNamespace(isError=False, structuredContent={"result": result},
                               content=[])


def _schema(**props):
    return {"type": "object", "properties": {k: {"type": v} for k, v in props.items()}}


TOOLS = [
    SimpleNamespace(name="search_accounts", description="", inputSchema=_schema(query="string")),
    SimpleNamespace(name="get_account_tickets", description="",
                    inputSchema=_schema(account_id="integer")),
    SimpleNamespace(name="search_knowledge", description="", inputSchema=_schema(query="string")),
]


def _call(name, args, id_):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_}])


def run(question, final_answer, synthesis=None):
    """The model stops with complete evidence, so its answer goes to synthesis. By default
    the synthesis repeats the answer, so the grounding checks below apply to that text.
    The named account is searched before the model's first turn."""
    replies = [
        _call("get_account_tickets", {"account_id": 1}, "b"),
        _call("search_knowledge", {"query": "SSO login issues"}, "c"),
        AIMessage(content=final_answer),
        AIMessage(content=final_answer if synthesis is None else synthesis),
    ]
    return asyncio.run(investigate(question, FakeModel(replies), FakeMCP(), TOOLS))


def relevance(result):
    return {(r["type"], r["id"]): r["relevance"] for r in result["evidence"]}


# The answer the real model produced in the browser run that exposed the problem.
MISLEADING = (
    "Based on the information gathered:\n\n"
    "- Northstar Labs has an active account with account ID 1.\n"
    "- There are two support tickets related to SSO login issues affecting Northstar Labs:\n"
    "  - Ticket ID 1 is an open ticket with the subject \"Intermittent login failures\".\n"
    "  - Ticket ID 2 is a resolved ticket with the subject \"Dashboard loading slowly\".\n"
)


def test_sso_ticket_is_relevant_and_dashboard_ticket_is_not():
    labels = relevance(run(SSO_QUESTION, "Ticket 1 concerns login failures."))
    assert labels[("account", 1)] == "context"
    assert labels[("product", 1)] == "context"
    assert labels[("ticket", 1)] == "relevant"
    assert labels[("ticket", 2)] == "no_match"
    assert labels[("document", 1)] == "relevant"


def test_dashboard_ticket_is_not_presented_as_an_sso_issue():
    # The tool loop's answer goes to synthesis; a synthesis that repeats the claim is
    # withheld too, and the answer falls back to the verified records.
    result = run(SSO_QUESTION, MISLEADING, synthesis=MISLEADING)
    assert result["answer_source"] == "evidence_summary"
    assert result["narrative"].startswith(
        "The generated answer was withheld because it described records as related")
    assert "two support tickets related to SSO" not in result["answer"]
    relevant = result["answer"].split("**Relevant support tickets**")[1].split("**")[0]
    other = result["answer"].split("**Other retrieved records**")[1].split("**")[0]
    assert "Ticket 1" in relevant and "Ticket 2" not in relevant
    assert 'Ticket 2, "Dashboard loading slowly"' in other
    assert "not evidence of this issue" in other


def test_misleading_answer_is_replaced_by_grounded_synthesis():
    synthesis = ("**Verified findings**\nTicket 1 (Intermittent login failures) is open.\n"
                 "**Unrelated or unconfirmed records**\n- Ticket 2: Dashboard loading slowly\n"
                 "**Root cause**\nUnknown.")
    result = run(SSO_QUESTION, MISLEADING, synthesis=synthesis)
    assert result["answer_source"] == "synthesis"
    assert result["narrative"] == synthesis
    assert "two support tickets related to SSO" not in result["answer"]


def test_unrelated_records_stay_in_evidence_and_trace():
    result = run(SSO_QUESTION, MISLEADING)
    assert {("ticket", 1), ("ticket", 2)} <= set(relevance(result))
    assert [entry["tool"] for entry in result["tool_trace"]] == [
        "search_accounts", "get_account_tickets", "search_knowledge"]
    assert result["tool_trace"][1]["record_count"] == 2


def test_answer_that_disclaims_the_dashboard_ticket_is_kept():
    answer = ("Ticket 1 (Intermittent login failures) is open with high priority. "
              "Ticket 2 (Dashboard loading slowly) is unrelated to SSO.")
    result = run(SSO_QUESTION, answer)
    assert result["narrative"] == answer
    assert result["answer"].startswith(answer + "\n\n**Relevant support tickets**\n")


def test_answer_mentioning_only_relevant_records_is_kept():
    answer = "Ticket 1 is an open, high priority login failure; see SSO troubleshooting."
    result = run(SSO_QUESTION, answer)
    assert result["narrative"] == answer


def test_api_answer_keeps_the_evidence_appendix_but_narrative_does_not():
    result = run(SSO_QUESTION, "Ticket 1 is an open login failure.")
    assert "**Relevant support tickets**" in result["answer"]
    assert "**Relevant support tickets**" not in result["narrative"]


def test_unconfirmed_root_cause_is_labelled():
    result = run(SSO_QUESTION, "The root cause is token expiry on the identity provider.")
    assert "no retrieved record confirms a root cause" in result["narrative"].lower()
    assert "unconfirmed hypothesis" in result["narrative"]


def test_analysis_separates_findings_guidance_and_unsupported_records():
    analysis = run(SSO_QUESTION, "Ticket 1 is a login failure.")["analysis"]
    findings = " ".join(item["text"] for item in analysis["findings"])
    assert "Ticket 1" in findings and "Ticket 2" not in findings
    assert analysis["findings"][0]["text"] == (
        "Northstar Labs (Account 1): 2 tickets retrieved (complete).")
    assert analysis["findings"][1]["evidence"] == [{"type": "ticket", "id": 1}]

    unsupported = " ".join(item["text"] for item in analysis["unsupported"])
    assert "Ticket 2" in unsupported and "not evidence of this issue" in unsupported

    guidance = analysis["guidance"][0]
    assert "SSO troubleshooting" in guidance["text"]
    assert "not a confirmed diagnosis" in guidance["text"]

    unconfirmed = " ".join(item["text"] for item in analysis["unconfirmed"])
    assert "No retrieved record establishes a root cause" in unconfirmed
    assert "not confirmed" in unconfirmed


def test_topic_follows_the_question():
    result = run("Why is the Northstar Labs dashboard loading slowly? Check their tickets.",
                 "Ticket 2 is a resolved dashboard performance ticket.")
    labels = relevance(result)
    assert labels[("ticket", 2)] == "relevant"
    assert labels[("ticket", 1)] == "no_match"
    assert labels[("document", 1)] == "no_match"


def test_question_without_a_topic_does_not_assess_relevance():
    answer = "Ticket 1 is open (high) and ticket 2 is resolved (medium)."
    result = run("List all open support tickets for Northstar Labs, with their priority.", answer)
    assert {r["relevance"] for r in result["evidence"] if r["type"] == "ticket"} == {"unassessed"}
    assert result["narrative"] == answer
    assert result["analysis"]["unsupported"] == []


def test_no_evidence_gives_an_empty_analysis():
    # No ticket or guidance terms, so the orchestrator adds no lookups.
    model = FakeModel([AIMessage(content="Hello.")])
    result = asyncio.run(investigate("Hello there", model, FakeMCP(), TOOLS))
    assert result["evidence"] == []
    assert result["analysis"] == {
        "findings": [], "guidance": [], "context": [], "unsupported": [], "unconfirmed": []}


def test_topic_terms_ignore_generic_words_and_entity_names():
    records = [{"type": "account", "id": 1, "title": "Northstar Labs"}]
    assert topic_terms(SSO_QUESTION, records) == {"sso", "login"}


def test_term_matching_handles_simple_inflections():
    assert _same_term("failure", "failures")
    assert _same_term("slowly", "slow")
    assert _same_term("loading", "load")
    assert not _same_term("sso", "ssl")
    assert not _same_term("log", "login")


def test_classify_relevance_uses_retrieved_description_when_available():
    records = [{"type": "ticket", "id": 9, "title": "Cannot sign in",
                "description": "SSO login fails for some users.", "fields": {}}]
    assert classify_relevance(SSO_QUESTION, records)[0]["relevance"] == "relevant"
