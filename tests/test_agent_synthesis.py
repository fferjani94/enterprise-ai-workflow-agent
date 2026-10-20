"""Final synthesis stage: when it runs, what it sees, its bounds, fallbacks and status."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, SystemMessage

from enterprise_ai_workflow_agent.agent.graph import investigate
from enterprise_ai_workflow_agent.agent.grounding import (
    DOCS_UNAVAILABLE,
    FIX_CLAIM_NOTE,
    UNCITED_GUIDANCE_NOTE,
    build_analysis,
    classify_relevance,
    has_causal_claim,
    miscounted_records,
    qualify_fix_claims,
    remove_uncited_guidance,
    unrelated_mentions,
)
from enterprise_ai_workflow_agent.agent.synthesis import SYNTHESIS_PROMPT

QUESTION = "Investigate SSO issues for Northstar Labs. Check their support tickets."

ACCOUNT = {"id": 1, "name": "Northstar Labs", "plan": "Enterprise", "status": "active"}
LOGIN_TICKET = {"id": 11, "account_id": 1, "product_id": 5,
                "subject": "Intermittent login failures", "status": "open", "priority": "high"}
DASHBOARD_TICKET = {"id": 12, "account_id": 1, "product_id": 5,
                    "subject": "Dashboard loading slowly", "status": "resolved",
                    "priority": "medium"}

TOOLS = [
    SimpleNamespace(name="search_accounts", description="", inputSchema={
        "type": "object", "properties": {"query": {"type": "string"}}}),
    SimpleNamespace(name="get_account_tickets", description="", inputSchema={
        "type": "object", "properties": {"account_id": {"type": "integer"},
                                         "limit": {"type": "integer", "minimum": 1,
                                                   "maximum": 20}}}),
    SimpleNamespace(name="search_knowledge", description="", inputSchema={
        "type": "object", "properties": {"query": {"type": "string"}}}),
]

GOOD_SYNTHESIS = (
    "Northstar Labs has an open, high-priority sign-in problem in Ticket 11 (Intermittent "
    "login failures). Ticket 12 is unrelated. No record establishes the root cause."
)


class Scripted:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class SplitModel(Scripted):
    """Tool-loop replies come from the bound model; synthesis uses the raw model."""

    def __init__(self, tool_replies, synthesis_replies=()):
        super().__init__(synthesis_replies)
        self.bound = Scripted(tool_replies)
        self.bind_count = 0

    def bind_tools(self, specs):
        self.bind_count += 1
        return self.bound


class FakeMCP:
    def __init__(self, tickets=(LOGIN_TICKET, DASHBOARD_TICKET), docs=()):
        self.data = {"search_accounts": [ACCOUNT], "get_account_tickets": list(tickets),
                     "search_knowledge": list(docs)}
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return SimpleNamespace(isError=False, structuredContent={"result": self.data[name]},
                               content=[])


def call(name, args, id_):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_}])


def run(tool_replies, synthesis_replies=(), question=QUESTION, max_rounds=4, client=None):
    model = SplitModel(tool_replies, synthesis_replies)
    client = client or FakeMCP()
    result = asyncio.run(investigate(question, model, client, TOOLS, max_rounds=max_rounds))
    return result, model, client


LOOPING = [
    call("search_accounts", {"query": "Northstar Labs"}, "a"),
    call("get_account_tickets", {"account_id": 1}, "b"),
    call("get_account_tickets", {"account_id": 1, "limit": 5}, "c"),
]


# --- When synthesis runs -----------------------------------------------------------

def test_synthesis_after_exhausting_tool_rounds():
    result, model, client = run(LOOPING, [AIMessage(content=GOOD_SYNTHESIS)], max_rounds=2)
    assert len(client.calls) == 2  # the third request is never executed
    assert result["round_limit_reached"] is True
    assert result["answer_source"] == "synthesis"
    assert result["narrative"] == GOOD_SYNTHESIS
    assert result["limited"] is False and result["limited_reason"] is None
    assert result["answer"].startswith(GOOD_SYNTHESIS + "\n\n**Relevant support tickets**\n")
    assert len(model.calls) == 1


def test_synthesis_after_orchestrator_completes_the_evidence():
    result, model, client = run(
        [call("search_accounts", {"query": "Northstar Labs"}, "a"),
         AIMessage(content="Northstar Labs is account 1.")],
        [AIMessage(content=GOOD_SYNTHESIS)])
    assert [name for name, _ in client.calls] == ["search_accounts", "get_account_tickets"]
    assert len(model.bound.calls) == 2  # the tool loop is not asked again
    assert result["answer_source"] == "synthesis"
    assert result["round_limit_reached"] is False
    assert result["limited"] is False


def test_model_stop_with_complete_evidence_goes_to_synthesis():
    answer = "Ticket 11 (Intermittent login failures) is open. Ticket 12 is unrelated."
    result, model, _ = run(LOOPING[:2] + [AIMessage(content=answer)],
                           [AIMessage(content=GOOD_SYNTHESIS)])
    assert len(model.calls) == 1
    assert result["answer_source"] == "synthesis"
    assert result["narrative"] == GOOD_SYNTHESIS


def test_grounded_model_answer_is_kept_when_evidence_is_incomplete():
    answer = "Northstar Labs is Account 1. Its tickets were not retrieved."
    result, model, _ = run([LOOPING[0], AIMessage(content=answer)],
                           [AIMessage(content=GOOD_SYNTHESIS)], max_rounds=1)
    assert model.calls == []
    assert result["answer_source"] == "model"
    assert result["narrative"] == answer
    assert result["limited"] is True


ADVICE = " Clear the browser cache. You should review the server logs."


def _no_uncited_advice(result):
    assert "browser cache" not in result["answer"] and "server logs" not in result["answer"]
    assert UNCITED_GUIDANCE_NOTE in result["narrative"]
    assert DOCS_UNAVAILABLE in result["answer"]


def test_uncited_guidance_is_removed_from_a_synthesis():
    result, _, _ = run(LOOPING[:2] + [AIMessage(content="Done.")],
                       [AIMessage(content=GOOD_SYNTHESIS + ADVICE)])
    assert result["answer_source"] == "synthesis"
    assert result["narrative"].startswith(GOOD_SYNTHESIS)
    _no_uncited_advice(result)


def test_uncited_guidance_is_removed_from_a_synthesis_after_the_round_limit():
    result, _, _ = run(LOOPING, [AIMessage(content=GOOD_SYNTHESIS + ADVICE)], max_rounds=2)
    assert result["round_limit_reached"] is True
    assert result["answer_source"] == "synthesis"
    _no_uncited_advice(result)


def test_uncited_guidance_is_removed_from_a_kept_model_answer():
    answer = "Northstar Labs is Account 1. Its tickets were not retrieved." + ADVICE
    result, _, _ = run([LOOPING[0], AIMessage(content=answer)], max_rounds=1)
    assert result["answer_source"] == "model"
    assert result["narrative"].startswith("Northstar Labs is Account 1.")
    _no_uncited_advice(result)


def test_answer_with_only_uncited_guidance_falls_back_to_the_records():
    result, _, _ = run(LOOPING, [AIMessage(content=ADVICE.strip())], max_rounds=2)
    assert result["answer_source"] == "evidence_summary"
    assert "no longer referred to any record" in result["answer_notice"]
    assert 'Ticket 11, "Intermittent login failures"' in result["answer"]


def test_dangling_text_left_after_removing_advice_falls_back_to_the_records():
    # Observed with qwen2.5:3b: removing the advice left "This should provide insights...".
    text = "Consider reviewing the dashboard settings. This should provide useful insights."
    result, _, _ = run(LOOPING, [AIMessage(content=text)], max_rounds=2)
    assert result["answer_source"] == "evidence_summary"
    assert "useful insights" not in result["answer"]


def test_record_sections_are_deterministic_whatever_the_narrative_says():
    # The narrative mentions no tickets; the sections still list them, so the answer can
    # never say "Relevant tickets: None" while a relevant ticket exists.
    result, _, _ = run(LOOPING, [AIMessage(content="Northstar Labs reported a problem.")],
                       max_rounds=2)
    relevant = result["answer"].split("**Relevant support tickets**")[1].split("**")[0]
    other = result["answer"].split("**Other retrieved records**")[1].split("**")[0]
    assert 'Ticket 11, "Intermittent login failures" (open, high)' in relevant
    assert "Ticket 12" not in relevant
    assert 'Ticket 12, "Dashboard loading slowly"' in other
    assert 'Account 1, "Northstar Labs" (Enterprise, active)' in other
    assert "**Unknown or unconfirmed**" in result["answer"]


def test_wrong_record_count_in_a_narrative_is_excluded():
    claim = "There are two support tickets related to SSO for Northstar Labs (Ticket 11)."
    result, _, _ = run(LOOPING, [AIMessage(content=claim)], max_rounds=2)
    assert result["answer_source"] == "evidence_summary"
    assert "record count" in result["answer_notice"]
    assert claim not in result["answer"]


def test_synthesis_sees_only_question_and_labelled_evidence():
    result, model, _ = run(
        [call("search_accounts", {"query": "Northstar Labs"}, "a"),
         AIMessage(content="UNVERIFIED CLAIM: ticket 99 caused it.")],
        [AIMessage(content=GOOD_SYNTHESIS)])
    system, human = model.calls[0]
    assert isinstance(system, SystemMessage) and system.content == SYNTHESIS_PROMPT
    assert "UNVERIFIED CLAIM" not in human.content
    assert f"Question: {QUESTION}" in human.content
    relevant, unrelated = human.content.split("UNRELATED")
    assert 'Ticket 11: "Intermittent login failures"' in relevant
    assert "Related topic" in relevant
    assert 'Ticket 12: "Dashboard loading slowly"' in unrelated
    assert "Account 1" in human.content


# --- Bounds ------------------------------------------------------------------------

def test_no_tool_calls_are_executed_during_synthesis():
    tool_request = AIMessage(content=GOOD_SYNTHESIS, tool_calls=[
        {"name": "search_knowledge", "args": {"query": "sso"}, "id": "x"}])
    result, model, client = run(LOOPING, [tool_request], max_rounds=2)
    assert model.bind_count == 1  # only the tool loop is bound to tools
    assert len(model.calls) == 1
    assert len(client.calls) == 2
    assert all(entry["tool"] != "search_knowledge" for entry in result["tool_trace"])
    assert result["answer_source"] == "synthesis"


def test_synthesis_runs_at_most_once_even_when_it_is_rejected():
    bad = "Ticket 99 shows the SSO outage."
    result, model, _ = run(
        [*LOOPING[:2], AIMessage(content="Ticket 98 is the cause.")], [AIMessage(content=bad)])
    assert len(model.calls) == 1
    assert result["answer_source"] == "evidence_summary"
    assert "Ticket 99" not in result["answer"] and "Ticket 98" not in result["answer"]


def test_transient_synthesis_errors_are_retried_within_the_attempt_budget():
    repeat = RuntimeError("token repeat limit reached (status code: 500)")
    result, model, _ = run(LOOPING, [repeat, AIMessage(content=""), AIMessage(
        content=GOOD_SYNTHESIS)], max_rounds=2)
    assert len(model.calls) == 3
    assert result["answer_source"] == "synthesis"


# --- Fallbacks ---------------------------------------------------------------------

def test_synthesis_failure_returns_deterministic_grounded_summary():
    result, model, _ = run(LOOPING, [ConnectionError("Ollama offline")], max_rounds=2)
    assert len(model.calls) == 1  # not transient: no retry
    assert result["model_error"] is None  # evidence was collected; not an infrastructure fail
    assert result["limited"] is False
    assert result["answer_source"] == "evidence_summary"
    assert "could not write a summary (ConnectionError)" in result["answer_notice"]
    assert "there is no model-generated conclusion" in result["narrative"]
    assert "tool-round limit was reached" in result["narrative"]
    relevant = result["answer"].split("**Relevant support tickets**")[1].split("**")[0]
    other = result["answer"].split("**Other retrieved records**")[1]
    assert 'Ticket 11, "Intermittent login failures"' in relevant
    assert 'Ticket 12, "Dashboard loading slowly"' in other
    assert "not evidence of this issue" in other


def test_persistent_empty_synthesis_replies_fall_back_after_bounded_attempts():
    result, model, _ = run(LOOPING, [AIMessage(content=" ")] * 10, max_rounds=2)
    assert len(model.calls) == 4
    assert result["answer_source"] == "evidence_summary"


def test_incomplete_evidence_stays_limited_and_skips_synthesis():
    # With one round, the tickets cannot be fetched after the account is found.
    looping_without_tickets = [
        call("search_accounts", {"query": "Northstar Labs"}, "a"),
        call("search_accounts", {"query": "Northstar"}, "b"),
    ]
    result, model, _ = run(looping_without_tickets, [AIMessage(content=GOOD_SYNTHESIS)],
                           max_rounds=1)
    assert model.calls == []
    assert result["limited"] is True
    assert "tickets for verified account 1" in result["limited_reason"]
    assert "tool-round limit (1) was reached" in result["limited_reason"]
    assert result["answer_source"] == "evidence_summary"
    assert "retrieved so far are listed below" in result["narrative"]


def test_last_round_fetches_required_tickets_instead_of_ending_incomplete():
    looping_without_tickets = [
        call("search_accounts", {"query": "Northstar Labs"}, "a"),
        call("search_accounts", {"query": "Northstar"}, "b"),
    ]
    result, model, client = run(looping_without_tickets, [AIMessage(content=GOOD_SYNTHESIS)],
                                max_rounds=2)
    assert ("get_account_tickets", {"account_id": 1}) in client.calls
    assert result["limited"] is False
    assert result["answer_source"] == "synthesis"


# --- Unsupported claims ------------------------------------------------------------

def test_synthesis_with_invented_id_is_excluded():
    result, _, _ = run(LOOPING, [AIMessage(content="Ticket 404 confirms an SSO outage.")],
                       max_rounds=2)
    assert result["answer_source"] == "evidence_summary"
    assert "404" not in result["answer"]
    assert "contained an unsupported ID" in result["answer_notice"]


def test_cited_id_must_match_the_record_type():
    # Observed with qwen2.5:3b: only Document 1 was retrieved, and synthesis invented
    # "Ticket 1". The shared number must not make the ticket look verified.
    doc = {"id": 1, "product_id": 1, "title": "SSO troubleshooting",
           "snippet": "Check identity provider settings.", "score": 3}
    invented = ("**Relevant support tickets**\n- Ticket 1: \"SSO login issues\".\n"
                "**Troubleshooting guidance**\n- Document 1: SSO troubleshooting.")
    result, _, client = run(
        [AIMessage(content="Check your IdP.")], [AIMessage(content=invented)],
        question="Why are users failing to log in with SSO? Suggest next steps.",
        client=FakeMCP(docs=[doc]))
    assert [name for name, _ in client.calls] == ["search_knowledge"]
    assert result["answer_source"] == "evidence_summary"
    assert "SSO login issues" not in result["answer"]
    assert 'Document 1, "SSO troubleshooting"' in result["answer"]


def test_synthesis_presenting_dashboard_ticket_as_sso_evidence_is_excluded():
    claim = ("**Relevant support tickets**\n- Ticket 11: Intermittent login failures\n"
             "- Ticket 12: Dashboard loading slowly")
    result, _, _ = run(LOOPING, [AIMessage(content=claim)], max_rounds=2)
    assert result["answer_source"] == "evidence_summary"
    assert "described records as related" in result["answer_notice"]


def test_unrelated_heading_does_not_excuse_an_asserted_link():
    records = classify_relevance(QUESTION, [
        {"type": "ticket", "id": 12, "title": "Dashboard loading slowly", "fields": {}}])
    listed = "**Unrelated or unconfirmed records**\n- Ticket 12: Dashboard loading slowly"
    linked = ("**Unrelated or unconfirmed records**\n"
              "- Ticket 12 is related to the SSO failures.")
    assert unrelated_mentions(listed, records) == []
    assert unrelated_mentions(linked, records) == records


def test_root_cause_heading_and_unknown_statement_are_not_causal_claims():
    assert not has_causal_claim("**Root cause**\nUnknown; no record establishes it.")
    assert not has_causal_claim("Root cause: not yet determined.")
    assert has_causal_claim("The root cause is token expiry.")


# --- SSO / login terminology -------------------------------------------------------

def _tickets(*subjects):
    return [{"type": "ticket", "id": i, "title": s, "description": None, "fields": {}}
            for i, s in enumerate(subjects, start=1)]


def test_login_ticket_is_related_to_an_sso_question_without_ticket_details():
    records = classify_relevance("Investigate SSO issues for Northstar Labs.",
                                 _tickets("Intermittent login failures",
                                          "Dashboard loading slowly"))
    login, dashboard = records
    assert login["relevance"] == "relevant"
    assert "'login'" in login["relevance_basis"] and "'sso'" in login["relevance_basis"]
    assert "does not establish a cause" in login["relevance_basis"]
    assert dashboard["relevance"] == "no_match"
    assert dashboard["relevance_basis"] is None
    finding = build_analysis(records)["findings"][0]["text"]
    assert "related to the question" in finding and "matches the topic" not in finding


def test_direct_matches_are_labelled_as_direct():
    login = classify_relevance("Why do SSO logins fail?",
                               _tickets("Intermittent login failures"))[0]
    assert login["relevance_basis"] == "Shares the question term(s): failures, login"


def test_related_terms_do_not_cross_topics():
    records = classify_relevance("Why is the dashboard slow for Northstar Labs?",
                                 _tickets("Intermittent login failures",
                                          "Dashboard loading slowly"))
    assert [r["relevance"] for r in records] == ["no_match", "relevant"]
    unrelated = classify_relevance("Investigate webhook retries.",
                                   _tickets("Intermittent login failures"))
    assert unrelated[0]["relevance"] == "no_match"


def test_shared_symptom_word_alone_is_not_a_topic_match():
    # Observed with qwen2.5:3b: "Webhook delivery ... backoff on failures" was labelled
    # relevant to "SSO login failures" and shown as SSO troubleshooting guidance.
    webhook = {"type": "document", "id": 2, "title": "Webhook delivery",
               "description": "Webhook deliveries use exponential backoff on failures.",
               "fields": {}}
    login = _tickets("Intermittent login failures")[0]
    labelled = classify_relevance("Investigate SSO login failures for Northstar Labs.",
                                  [webhook, login])
    assert [r["relevance"] for r in labelled] == ["no_match", "relevant"]
    assert labelled[1]["relevance_basis"] == "Shares the question term(s): failures, login"


def test_symptom_words_still_match_when_the_question_names_no_other_topic():
    webhook = {"type": "document", "id": 2, "title": "Webhook delivery",
               "description": "Deliveries use exponential backoff on failures.", "fields": {}}
    assert classify_relevance("Show recent failures.", [webhook])[0]["relevance"] == "relevant"


DOC = {"type": "document", "id": 7, "title": "SSO troubleshooting",
       "description": "Check identity provider settings and token expiry.", "fields": {},
       "relevance": "relevant"}


def test_cited_guidance_is_kept_and_uncited_guidance_is_removed():
    text = ("Ticket 1 reports a login failure. Document 7 recommends checking identity provider "
            "settings. Check the token expiry in SSO troubleshooting.\n"
            "- Restart the server.\n- You should rotate all keys.\n"
            "The ticket subject suggests a sign-in problem.")
    cleaned, removed = remove_uncited_guidance(text, [DOC])
    assert removed == 2
    assert "Document 7 recommends checking identity provider settings." in cleaned
    assert "Check the token expiry in SSO troubleshooting." in cleaned
    assert "The ticket subject suggests a sign-in problem." in cleaned
    assert "Restart" not in cleaned and "rotate" not in cleaned


def test_guidance_citing_an_unrelated_document_is_not_supported():
    unrelated = {**DOC, "relevance": "no_match"}
    cleaned, removed = remove_uncited_guidance("Check the settings in Document 7.", [unrelated])
    assert (cleaned, removed) == ("", 1)


def test_statement_that_no_steps_exist_is_not_removed_as_advice():
    # Real qwen2.5:3b synthesis for "What plan is Northstar Labs on, and what industry is it
    # in?": the accurate answer was withheld because the second sentence was taken as advice.
    text = ("The account shows Northstar Labs is on the Enterprise plan and operates in the "
            "Software industry. No further information or steps to resolve are available from "
            "the provided records.")
    assert remove_uncited_guidance(text, []) == (text, 0)
    for statement in ("There are no next steps in the records.",
                      "None of the records recommends a workaround."):
        assert remove_uncited_guidance(statement, []) == (statement, 0)
    assert remove_uncited_guidance("Next steps: restart the server.", [])[1] == 1
    # Real qwen2.5:3b synthesis for the slow-dashboard example: the absence statement
    # carries advice, so it is still removed.
    advice = ("No troubleshooting steps are provided in the available documents, so you "
              "might need to explore other areas or consult with your account administrator.")
    assert remove_uncited_guidance(advice, []) == ("", 1)


def test_accurate_account_only_answer_without_ids_is_kept():
    client = FakeMCP()
    client.data["search_accounts"] = [{**ACCOUNT, "industry": "Software"}]
    text = ("The account shows Northstar Labs is on the Enterprise plan and operates in the "
            "Software industry. No further information or steps to resolve are available from "
            "the provided records.")
    result, _, _ = run(
        [call("search_accounts", {"query": "Northstar Labs"}, "a"), AIMessage(content="Done.")],
        [AIMessage(content=text)],
        question="What plan is Northstar Labs on, and what industry is it in?", client=client)
    assert result["answer_source"] == "synthesis"
    assert result["narrative"] == text


def test_record_counts_must_match_a_grouping_of_the_records():
    records = classify_relevance(QUESTION, [
        {"type": "ticket", "id": 11, "title": "Intermittent login failures",
         "fields": {"status": "open"}},
        {"type": "ticket", "id": 12, "title": "Dashboard loading slowly",
         "fields": {"status": "resolved"}},
    ])
    assert miscounted_records("Two tickets were retrieved.", records) == []
    assert miscounted_records("One open ticket concerns sign-in.", records) == []
    assert miscounted_records("Ticket 11 and Document 21 concern SSO.", records) == []
    assert miscounted_records("Two tickets relate to SSO.", records) == []  # "relate" != claim
    assert miscounted_records("There are two tickets related to SSO.", records) == ["two tickets"]
    assert miscounted_records("Three tickets were retrieved.", records) == ["Three tickets"]


# --- Regressions from real runs ----------------------------------------------------

def test_guidance_words_do_not_make_a_document_relevant():
    # Observed with qwen2.5:3b: "troubleshoot" matched "SSO troubleshooting" for a dashboard
    # performance question, and the SSO document was offered as guidance.
    sso_doc = {"type": "document", "id": 1, "title": "SSO troubleshooting",
               "description": "Check identity provider settings and token expiry.", "fields": {}}
    dashboard = _tickets("Dashboard loading slowly")[0]
    labelled = classify_relevance(
        "How do we troubleshoot the dashboard performance problems at Northstar Labs?",
        [sso_doc, dashboard])
    assert [r["relevance"] for r in labelled] == ["no_match", "relevant"]
    analysis = build_analysis(labelled, expects_tickets=True, issue_question=True)
    assert [item["text"] for item in analysis["guidance"]] == [DOCS_UNAVAILABLE]


def test_account_only_answer_has_industry_and_no_issue_sections():
    account = {**ACCOUNT, "industry": "Software"}
    client = FakeMCP()
    client.data["search_accounts"] = [account]
    result, model, _ = run(
        [call("search_accounts", {"query": "Northstar Labs"}, "a"), AIMessage(content="Done.")],
        [AIMessage(content="Account 1, Northstar Labs, is a Software company on Enterprise.")],
        question="What plan is Northstar Labs on, and what industry is it in?", client=client)
    assert "industry: Software" in model.calls[0][1].content
    assert 'Account 1, "Northstar Labs" (Enterprise, Software, active)' in result["answer"]
    assert "root cause" not in result["answer"]
    assert "documentation" not in result["answer"]
    assert result["limited"] is False


def test_qualified_negative_about_other_tickets_is_not_a_contradiction():
    narrative = "Ticket 11 is open. No other relevant tickets were found in the records."
    result, _, _ = run(LOOPING, [AIMessage(content=narrative)], max_rounds=2)
    assert result["answer_source"] == "synthesis"
    assert result["narrative"] == narrative


def test_stated_unknown_cause_gets_no_unconfirmed_cause_note():
    assert not has_causal_claim("No other evidence points to a specific root cause.")


# --- Unsupported claims that an action fixes the problem ---------------------------

@pytest.mark.parametrize("claim, softened", [
    # Observed on the dashboard with qwen2.5:3b: a documented step presented as the fix.
    ("Per Document 7, checking the token expiry resolves the issue.",
     "Per Document 7, checking the token expiry may help resolve the issue."),
    ("Updating the IdP settings will fix the SSO login failures.",
     "Updating the IdP settings may help fix the SSO login failures."),
    ("This will definitely solve their problem.", "This may help solve their problem."),
    ("Re-uploading the certificate is guaranteed to fix it.",
     "Re-uploading the certificate may help fix it."),
    ("Following Document 7 should resolve Northstar's login errors.",
     "Following Document 7 may help resolve Northstar's login errors."),
    ("The issue will be resolved once the token is renewed.",
     "The issue may be resolved once the token is renewed."),
    ("Clearing the cache fixes this.", "Clearing the cache may help fix this."),
])
def test_definite_fix_claims_are_softened(claim, softened):
    assert qualify_fix_claims(claim) == (softened, 1)


@pytest.mark.parametrize("statement", [
    "Ticket 12 is resolved.",
    "Ticket 12 (Dashboard loading slowly) was resolved.",
    "Ticket 12 has been resolved and is marked as resolved.",
    "Ticket 12, a resolved ticket, resolved the dashboard issue for the customer.",
    "The status of Ticket 12 is resolved; it was fixed last week.",
    "Document 7 may help resolve the issue.",
    "Checking the settings might fix the login problem.",
    "This does not resolve it.",
    "No record shows that this step resolves the issue.",
    "It is unknown whether renewing the token fixes the login errors.",
    "Restarting never fixes the problem.",
    'Ticket 4 is titled "Patch resolves the outage".',
    "The hostname resolves to 10.0.0.1.",
])
def test_status_hedged_negated_and_quoted_wording_is_kept(statement):
    assert qualify_fix_claims(statement) == (statement, 0)


def test_fix_claim_in_a_synthesis_is_softened_with_a_visible_note():
    synthesis = ("Ticket 11 reports intermittent login failures. According to Document 7, "
                 "checking identity provider settings resolves the issue.")
    sso_doc = {"id": 7, "product_id": 5, "title": "SSO troubleshooting",
               "snippet": "Check identity provider settings and token expiry."}
    question = QUESTION + " Check the documentation."
    result, _, _ = run(
        [*LOOPING[:2], call("search_knowledge", {"query": "SSO"}, "k"),
         AIMessage(content="Done.")],
        [AIMessage(content=synthesis)], question=question,
        client=FakeMCP(docs=[sso_doc]))
    assert result["answer_source"] == "synthesis"
    assert "settings may help resolve the issue." in result["narrative"]
    assert "resolves the issue" not in result["answer"]
    assert FIX_CLAIM_NOTE in result["narrative"]
    assert "Ticket 11" in result["answer"]


def test_resolved_ticket_status_in_a_synthesis_gets_no_fix_note():
    resolved = {**LOGIN_TICKET, "status": "resolved"}
    synthesis = ("Ticket 11 (Intermittent login failures) is resolved. It was resolved by "
                 "the support team and has been marked as resolved.")
    result, _, _ = run(LOOPING, [AIMessage(content=synthesis)], max_rounds=2,
                       client=FakeMCP(tickets=[resolved]))
    assert result["answer_source"] == "synthesis"
    assert result["narrative"] == synthesis
    assert FIX_CLAIM_NOTE not in result["answer"]
    assert '"Intermittent login failures" (resolved, high)' in result["answer"]


def test_synthesis_prompt_forbids_fix_claims():
    assert "will fix or resolve" in SYNTHESIS_PROMPT
