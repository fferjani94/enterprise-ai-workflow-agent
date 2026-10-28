"""Milestone 6A: offline benchmark dataset, harness, metrics, reports and CLI.

Runs the real FastMCP server in memory on isolated SQLite fixtures with a scripted model.
No Ollama, external API or network needed.
"""

import copy
import json
import socket

import pytest
from pydantic import ValidationError

from enterprise_ai_workflow_agent.agent import graph
from enterprise_ai_workflow_agent.agent.followups import EvidencePlan
from enterprise_ai_workflow_agent.evaluation import __main__ as cli
from enterprise_ai_workflow_agent.evaluation import harness
from enterprise_ai_workflow_agent.evaluation.dataset import (
    CATEGORIES,
    Benchmark,
    CompletionExpectation,
    Pattern,
    ScriptedReply,
    benchmark_digest,
    load_benchmark,
)
from enterprise_ai_workflow_agent.evaluation.harness import (
    FIXTURES,
    build_database,
    fixture_ids,
    run_case,
    run_case_safely,
)
from enterprise_ai_workflow_agent.evaluation.metrics import (
    ANSWER,
    COMPLETION,
    EFFICIENCY,
    GROUNDING,
    Run,
    build_report,
    cited_records,
    render_markdown,
    score_case,
)
from enterprise_ai_workflow_agent.mcp_server import server, services

BENCHMARK = load_benchmark()
CASES = {case.id: case for case in BENCHMARK.cases}
INTEGRITY = "harness_integrity"


def failed_checks(case, result, **run):
    score = score_case(case, Run(result=result, **run))
    groups = [(INTEGRITY, score.integrity),
              *((name, d.checks) for name, d in score.dimensions.items())]
    return {(group, check.name) for group, checks in groups for check in checks
            if not check.passed}


def with_script(case, **script):
    return case.model_copy(update={"script": case.script.model_copy(update=script)})


# --- Dataset -------------------------------------------------------------------------------

def test_benchmark_covers_all_categories():
    counts = {c: sum(case.category == c for case in BENCHMARK.cases) for c in CATEGORIES}
    assert counts == {"success": 6, "missing_account": 4, "incomplete_evidence": 5,
                      "unsafe_conclusion": 7, "tool_discipline": 3}
    assert BENCHMARK.name == "benchmark_v1" and BENCHMARK.schema_version == 1
    assert any(not case.pre_resolve for case in BENCHMARK.cases)


@pytest.mark.parametrize("case", BENCHMARK.cases, ids=lambda case: case.id)
def test_expected_records_exist_in_the_case_fixture(case):
    assert case.fixture in FIXTURES
    ids = fixture_ids(case.fixture)
    grounding = case.expect.grounding
    for ref in [*grounding.required_evidence, *grounding.forbidden_evidence]:
        assert (ref.type, ref.id) in ids, f"{case.id}: {ref} is not in {case.fixture}"
    for item in case.expect.completion.coverage:
        assert ("account", item.account_id) in ids


def test_every_unsafe_case_checks_the_model_written_text_or_names_a_forbidden_claim():
    for case in BENCHMARK.cases:
        if case.category == "unsafe_conclusion":
            answer = case.expect.answer
            assert answer.source_in, case.id
            assert answer.forbidden_claims or any(
                fact.scope == "narrative" for fact in answer.required_facts), case.id


def _raw():
    return json.loads(json.dumps(BENCHMARK.model_dump(mode="json")))


@pytest.mark.parametrize("mutate, message", [
    (lambda raw: raw["cases"].append(raw["cases"][0]), "duplicate case IDs: S1"),
    (lambda raw: raw["cases"][0].update(unexpected=True), "Extra inputs are not permitted"),
    (lambda raw: raw["cases"][0].update(failing_tools=["delete_account"]), "failing_tools"),
    (lambda raw: raw["cases"][0]["expect"]["answer"]["required_facts"][0].update(
        any_of=["(unclosed"]), r"missing \)"),
    (lambda raw: raw["cases"][0]["expect"]["answer"]["required_facts"][0].update(
        scope="sections"), "scope"),
    (lambda raw: raw["cases"][0].update(category="other"), "category"),
])
def test_malformed_benchmarks_are_rejected(mutate, message):
    raw = _raw()
    mutate(raw)
    with pytest.raises(ValidationError, match=message):
        Benchmark.model_validate(raw)


def test_benchmark_digest_tracks_content_not_formatting():
    raw = _raw()
    reformatted = Benchmark.model_validate(json.loads(json.dumps(raw, indent=4)))
    assert benchmark_digest(reformatted) == benchmark_digest(BENCHMARK)
    raw["cases"][0]["question"] += " "
    assert benchmark_digest(Benchmark.model_validate(raw)) != benchmark_digest(BENCHMARK)


def test_fixtures_use_explicit_ids_independent_of_production_seeding():
    sessions = build_database("bulk")
    try:
        with sessions() as db:
            assert services.search_accounts(db, "Northstar") == [
                {"id": 1, "name": "Northstar Labs", "industry": "Software",
                 "plan": "Enterprise", "status": "active"}]
            page = services.get_account_tickets(db, 1, "open", limit=50)
    finally:
        sessions.kw["bind"].dispose()
    assert page["total"] == 61 and page["tickets"][0]["id"] == 101


# --- Benchmark gate: every deterministic offline case must pass ----------------------------

@pytest.mark.parametrize("case", BENCHMARK.cases, ids=lambda case: case.id)
def test_offline_benchmark_case(case):
    score = score_case(case, run_case(case))
    assert score.passed, score.explain()


def test_offline_runs_are_reproducible():
    cases = [CASES[i] for i in ("S3", "I1", "U3", "T3")]
    first, second = ([score_case(case, run_case(case)) for case in cases] for _ in range(2))
    assert json.dumps(build_report(BENCHMARK, first, "offline"), sort_keys=True) == \
        json.dumps(build_report(BENCHMARK, second, "offline"), sort_keys=True)


def test_offline_runs_need_no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    assert score_case(CASES["S3"], run_case(CASES["S3"])).passed


def test_injected_failures_go_through_the_real_server_error_masking():
    trace = run_case(CASES["I2"]).result["tool_trace"]
    failed = [entry for entry in trace if entry["tool"] == "get_account_tickets"]
    assert failed and all(not entry["success"] for entry in failed)
    assert "Internal error while reading data" in failed[0]["error"]
    assert "injected" not in failed[0]["error"]


# --- Adversarial mutations: disabling a guard must fail its case ---------------------------

def _drop_planned(predicate):
    original = graph.plan_evidence

    def plan(question, evidence, trace):
        result = original(question, evidence, trace)
        return EvidencePlan([c for c in result.calls if not predicate(c)], result.gaps)
    return plan


MUTATIONS = {
    "U1": ("_unsupported_final", lambda *a, **k: False),
    "U2": ("miscounted_records", lambda *a, **k: []),
    "U3": ("unsupported_completeness_claims", lambda *a, **k: []),
    "U4": ("unsupported_no_ticket_claims", lambda *a, **k: []),
    "U5": ("remove_uncited_guidance", lambda text, records: (text, 0)),
    "U6": ("has_causal_claim", lambda *a, **k: False),
    "U7": ("unrelated_mentions", lambda *a, **k: []),
    "T1": ("_id_verification", lambda *a, **k: graph.VERIFIED_BEFORE_BATCH),
    "T2": ("_succeeded_keys", lambda *a, **k: set()),
    "S6": ("plan_evidence", _drop_planned(lambda call: call.arguments.get("offset"))),
    "T3": ("plan_evidence", _drop_planned(lambda call: call.tool == "search_accounts")),
}


@pytest.mark.parametrize("case_id", MUTATIONS)
def test_disabling_the_guard_fails_its_case(monkeypatch, case_id):
    attribute, replacement = MUTATIONS[case_id]
    monkeypatch.setattr(graph, attribute, replacement)
    score = score_case(CASES[case_id], run_case(CASES[case_id]))
    behavioural = [f for f in score.failures() if not f.startswith(INTEGRITY)]
    assert behavioural, f"{case_id} still passes with graph.{attribute} disabled"


GENERIC = "Here is a summary of the account."


@pytest.mark.parametrize("case", [
    case for case in BENCHMARK.cases
    if case.script.synthesis and case.expect.answer.source_in == ["synthesis"]
], ids=lambda case: case.id)
def test_generic_synthesis_fails_narrative_scoped_facts(case):
    generic = with_script(case, synthesis=[ScriptedReply(text=GENERIC)])
    score = score_case(generic, run_case(generic))
    assert score.dimensions[ANSWER].passed is False
    assert all(f.startswith(f"{ANSWER}/required_fact") for f in score.failures()), \
        score.explain()


# --- Harness integrity and state restoration -----------------------------------------------

def test_missing_scripted_reply_is_a_harness_failure():
    case = with_script(CASES["S3"], synthesis=[])
    run = run_case(case)
    assert run.exhausted == ["synthesis"]
    assert (INTEGRITY, "script_not_exhausted") in failed_checks(
        case, run.result, exhausted=run.exhausted)


def test_unused_scripted_reply_is_a_harness_failure():
    # An unsafe synthesis that is never requested would make a case pass vacuously.
    extra = ScriptedReply(text="Ticket 999 is the cause.")
    case = with_script(CASES["S1"], synthesis=[*CASES["S1"].script.synthesis, extra])
    run = run_case(case)
    assert run.unused_replies == {"synthesis": 1}
    assert failed_checks(case, run.result, unused_replies=run.unused_replies) == {
        (INTEGRITY, "script_fully_used")}


@pytest.fixture
def tracked_engines(monkeypatch):
    disposed, original = [], harness.build_database

    def build(name):
        sessions = original(name)
        engine = sessions.kw["bind"]
        dispose = engine.dispose
        monkeypatch.setattr(engine, "dispose", lambda *a, **k: (disposed.append(engine),
                                                                dispose(*a, **k)))
        return sessions
    monkeypatch.setattr(harness, "build_database", build)
    return disposed


def test_server_state_and_database_are_released_after_a_run(tracked_engines):
    session_factory, lookup = server.SessionLocal, services.get_account_tickets
    run_case(CASES["I2"])
    assert server.SessionLocal is session_factory
    assert services.get_account_tickets is lookup
    assert len(tracked_engines) == 1


def test_server_state_and_database_are_released_after_an_exception(monkeypatch,
                                                                   tracked_engines):
    session_factory, lookup = server.SessionLocal, services.get_account_tickets
    seen = {}

    async def crash(*args, **kwargs):
        seen["patched"] = (server.SessionLocal is not session_factory
                           and services.get_account_tickets is not lookup)
        raise RuntimeError("boom")
    monkeypatch.setattr(harness, "investigate", crash)
    with pytest.raises(RuntimeError, match="boom"):
        run_case(CASES["I2"])
    assert seen == {"patched": True}
    assert server.SessionLocal is session_factory
    assert services.get_account_tickets is lookup
    assert len(tracked_engines) == 1


def test_crashing_run_becomes_an_actionable_failure(monkeypatch):
    async def crash(*args, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(harness, "investigate", crash)
    run = run_case_safely(CASES["S1"])
    assert run.error.startswith("RuntimeError: boom (at test_evaluation.py:")
    score = score_case(CASES["S1"], run)
    assert not score.passed
    assert score.observed["status"] == "error"
    assert f"{INTEGRITY}/run_completed" in score.explain()


# --- Citation parsing ----------------------------------------------------------------------

RETRIEVED = {("account", 1), ("ticket", 1), ("ticket", 101), ("document", 201)}


@pytest.mark.parametrize("text, cited, ambiguous", [
    ("Ticket 101 is open.", {("ticket", 101)}, []),
    ("Tickets 101 and 999 are open.", {("ticket", 101), ("ticket", 999)}, []),
    ("Ticket IDs: 101, 999", {("ticket", 101), ("ticket", 999)}, []),
    ("tickets #101, #102, and #103", {("ticket", 101), ("ticket", 102), ("ticket", 103)}, []),
    ("Document ID 201 / ticket 101", {("document", 201), ("ticket", 101)}, []),
    ("See #101.", {("ticket", 101)}, []),
    ("See #999.", {("record", 999)}, []),
    ("See #1.", set(), ["#1 (account or ticket)"]),
    ('Ticket 101, "Intermittent login failures" (open, high)', {("ticket", 101)}, []),
    ("Northstar Labs (Account 1): 1 open ticket retrieved (complete).", {("account", 1)}, []),
    ("Several accounts matched (1, 3)", set(), []),
])
def test_citation_parsing(text, cited, ambiguous):
    assert cited_records(text, RETRIEVED) == (cited, ambiguous)


# --- Scorers detect failures (negative tests on a real passing result) ---------------------

@pytest.fixture(scope="module")
def s1():
    run = run_case(CASES["S1"])
    assert score_case(CASES["S1"], run).passed
    return run.result


def mutated(result, **changes):
    copied = copy.deepcopy(result)
    copied.update(changes)
    return copied


def test_narrative_fact_is_not_satisfied_by_the_record_sections(s1):
    narrative = s1["narrative"].replace("Ticket 101", "the ticket")
    assert "Ticket 101" in s1["answer"].replace(s1["narrative"], "")
    assert failed_checks(CASES["S1"], mutated(s1, narrative=narrative)) == {
        (ANSWER, "required_fact: ticket 101 is named")}


def test_answer_scorer_detects_a_forbidden_claim(s1):
    answer = s1["answer"] + "\nDashboard loading slowly is also open."
    assert failed_checks(CASES["S1"], mutated(s1, answer=answer)) == {
        (ANSWER, "forbidden_claim: resolved ticket presented as open")}


def test_answer_scorer_detects_an_unexpected_answer_source(s1):
    assert failed_checks(CASES["S1"], mutated(s1, answer_source="model")) == {
        (ANSWER, "answer_source")}


def test_answer_scorer_accepts_equivalent_wording(s1):
    narrative = "Northstar Labs has a single open ticket: #101 (Intermittent login failures)."
    assert failed_checks(CASES["S1"], mutated(
        s1, narrative=narrative, answer=narrative + s1["answer"][len(s1["narrative"]):])) \
        == set()


def test_model_and_answer_citations_are_scored_separately(s1):
    extra = " See also Tickets 101 and 999."
    both = mutated(s1, narrative=s1["narrative"] + extra, answer=s1["answer"] + extra)
    assert failed_checks(CASES["S1"], both) == {
        (GROUNDING, "model_citation_validity"), (GROUNDING, "answer_citation_validity")}
    sections_only = mutated(s1, answer=s1["answer"] + "\nDocument 101 has more detail.")
    assert failed_checks(CASES["S1"], sections_only) == {
        (GROUNDING, "answer_citation_validity")}


def test_model_citations_are_not_checked_for_deterministic_narratives(s1):
    result = mutated(s1, answer_source="evidence_summary", narrative="Ticket 999.")
    assert (GROUNDING, "model_citation_validity") not in failed_checks(CASES["S1"], result)


def test_evidence_retrieval_is_scored_separately_from_citation_validity(s1):
    evidence = [r for r in s1["evidence"] if (r["type"], r["id"]) != ("ticket", 101)]
    # Still cited, but no longer retrieved: retrieval and both citation checks fail.
    assert failed_checks(CASES["S1"], mutated(s1, evidence=evidence)) == {
        (GROUNDING, "required_evidence: ticket 101"), (GROUNDING, "model_citation_validity"),
        (GROUNDING, "answer_citation_validity")}


def test_grounding_scorer_detects_forbidden_evidence():
    result = run_case(CASES["S5"]).result
    extra = {"type": "account", "id": 1, "title": "Northstar Labs", "description": None,
             "fields": {}, "relevance": "context", "relevance_basis": None}
    assert failed_checks(CASES["S5"], mutated(result, evidence=[*result["evidence"], extra])) \
        == {(GROUNDING, "forbidden_evidence: account 1")}


@pytest.mark.parametrize("narrative", [
    "Northstar Labs has three open tickets, including Ticket 101.",
    "Northstar Labs has no open tickets besides Ticket 101.",
])
def test_unsupported_claim_detectors_flag_model_written_narratives(s1, narrative):
    failures = failed_checks(CASES["S1"], mutated(s1, narrative=narrative))
    assert (GROUNDING, "unsupported_claims") in failures


def test_unsupported_claim_detectors_flag_completeness_with_partial_coverage(s1):
    coverage = [{**s1["ticket_coverage"][0], "state": "partial", "total": 61, "retrieved": 1}]
    failures = failed_checks(CASES["S1"], mutated(
        s1, narrative="These are all the open tickets.", ticket_coverage=coverage))
    assert (GROUNDING, "unsupported_claims") in failures


def test_completion_scorer_detects_a_wrong_status(s1):
    assert failed_checks(CASES["S1"], mutated(s1, limited=True)) == {(COMPLETION, "status")}
    failures = failed_checks(CASES["S1"], mutated(s1, model_error="ConnectionError"))
    assert (COMPLETION, "status") in failures


def test_completion_scorer_detects_a_coverage_mismatch(s1):
    coverage = [{**s1["ticket_coverage"][0], "state": "partial"}]
    assert (COMPLETION, "coverage: account 1") in failed_checks(
        CASES["S1"], mutated(s1, ticket_coverage=coverage))


def test_completion_scorer_detects_a_missing_limitation_reason(s1):
    expectation = CompletionExpectation(
        status="completed",
        limited_reason=Pattern(description="partial retrieval", any_of=["partially"]))
    case = CASES["S1"].model_copy(update={
        "expect": CASES["S1"].expect.model_copy(update={"completion": expectation})})
    assert failed_checks(case, s1) == {(COMPLETION, "limited_reason: partial retrieval")}


def _entry(tool, success=True, **args):
    return {"tool": tool, "origin": "model", "arguments": args, "normalized_arguments": args,
            "success": success, "executed": True, "record_count": 0}


@pytest.mark.parametrize("change, expected", [
    (lambda trace: trace + [_entry("search_knowledge", query="sso")],
     {"forbidden_tool: search_knowledge", "max_executed_calls"}),
    (lambda trace: trace + [copy.deepcopy(trace[-1])],
     {"max_redundant_calls", "max_executed_calls"}),
    (lambda trace: [*trace[:-1], {**trace[-1], "success": False, "error": "boom"}],
     {"max_failed_calls"}),
    (lambda trace: [e for e in trace if e["tool"] != "get_account_tickets"],
     {"required_tool: get_account_tickets"}),
])
def test_efficiency_scorer_detects_budget_and_tool_violations(s1, change, expected):
    failures = failed_checks(CASES["S1"], mutated(s1, tool_trace=change(s1["tool_trace"])))
    assert {name for group, name in failures if group == EFFICIENCY} == expected


def test_efficiency_scorer_detects_too_many_rounds(s1):
    assert failed_checks(CASES["S1"], mutated(s1, tool_rounds=3)) == {
        (EFFICIENCY, "max_rounds")}


def test_efficiency_scorer_checks_rejections_and_orchestrator_calls():
    result = run_case(CASES["T1"]).result
    proposals = [p for p in result["model_proposals"] if p["outcome"] != "rejected"]
    assert failed_checks(CASES["T1"], mutated(result, model_proposals=proposals)) == {
        (EFFICIENCY, "required_rejection: get_ticket_details (unverified_id)")}
    extra = {"tool": "get_ticket_details", "outcome": "rejected",
             "reason": "Duplicate call skipped"}
    assert failed_checks(CASES["T1"], mutated(
        result, model_proposals=[*result["model_proposals"], extra])) == {
        (EFFICIENCY, "max_rejected_proposals")}
    result = run_case(CASES["T3"]).result
    trace = [{**e, "origin": "model"} for e in result["tool_trace"]]
    assert failed_checks(CASES["T3"], mutated(result, tool_trace=trace)) == {
        (EFFICIENCY, "min_orchestrator_calls")}


def test_rejected_calls_do_not_count_as_executed(s1):
    rejected = {**_entry("get_ticket_details", ticket_id=999), "executed": False,
                "success": False, "error": "Unverified record ID"}
    assert failed_checks(CASES["S1"], mutated(s1, tool_trace=[*s1["tool_trace"], rejected])) \
        == set()


# --- Diagnostics and reports ---------------------------------------------------------------

def test_failure_explanation_is_actionable(s1):
    score = score_case(CASES["S1"], Run(result=mutated(s1, limited=True,
                                                       limited_reason="Required evidence")))
    text = score.explain()
    assert text.startswith("Case S1 (success) failed:")
    assert "task_completion/status: expected 'completed', observed 'limited'" in text
    assert "limited_reason: 'Required evidence'" in text
    assert '[orchestrator] search_accounts {"query": "Northstar Labs"} -> ok, 1 record(s)' \
        in text


def test_report_aggregates_and_identifies_the_benchmark(s1):
    good = score_case(CASES["S1"], Run(result=s1))
    bad = score_case(CASES["S1"], Run(result=mutated(s1, answer=s1["answer"] + " Ticket 999")))
    report = build_report(BENCHMARK, [good, bad], "offline")
    assert report["benchmark"]["sha256"] == benchmark_digest(BENCHMARK)
    assert report["benchmark"]["cases_total"] == len(BENCHMARK.cases)
    assert report["benchmark"]["selection"] == ["S1", "S1"]
    assert report["summary"] == {"cases": 2, "passed": 1, "pass_rate": 0.5}
    assert report["by_category"]["success"] == {"cases": 2, "passed": 1, "pass_rate": 0.5}
    assert report["dimensions"][GROUNDING]["passed"] == 1
    assert report["metrics"][GROUNDING]["answer_citation_validity"] < 1
    assert report["metrics"][GROUNDING]["model_citation_validity"] == 1.0
    json.dumps(report)
    markdown = render_markdown(report)
    assert "**subset** of" in markdown and benchmark_digest(BENCHMARK) in markdown
    assert "**1/2 cases passed** (50.0%)" in markdown
    assert "## Failures" in markdown and "evidence_grounding/answer_citation_validity" in markdown


# --- CLI -----------------------------------------------------------------------------------

def test_cli_subset_writes_its_own_reports_and_never_the_full_suite_report(tmp_path, capsys):
    full = tmp_path / "benchmark_v1-offline.json"
    full.write_text("full-suite report", encoding="utf-8")
    assert cli.main(["--case", "S1", "--case", "U1", "--out", str(tmp_path)]) == 0
    assert full.read_text(encoding="utf-8") == "full-suite report"
    subset = tmp_path / "benchmark_v1-offline-subset-S1-U1.json"
    report = json.loads(subset.read_text(encoding="utf-8"))
    assert report["summary"] == {"cases": 2, "passed": 2, "pass_rate": 1.0}
    assert report["benchmark"]["selection"] == ["S1", "U1"]
    assert subset.with_suffix(".md").read_text(encoding="utf-8").startswith(
        "# benchmark_v1 (offline)")
    assert "**2/2 cases passed**" in capsys.readouterr().out


def test_cli_exits_non_zero_with_diagnostics_when_a_case_fails(tmp_path, capsys):
    raw = _raw()
    raw["cases"] = [case for case in raw["cases"] if case["id"] == "S1"]
    raw["cases"][0]["expect"]["completion"]["status"] = "limited"
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert cli.main(["--benchmark", str(path), "--out", str(tmp_path)]) == 1
    report = json.loads((tmp_path / "benchmark_v1-offline.json").read_text(encoding="utf-8"))
    assert report["benchmark"]["selection"] == "full"
    err = capsys.readouterr().err
    assert "Case S1 (success) failed:" in err and "tool trace:" in err


def test_cli_keeps_running_after_a_crashing_case(tmp_path, capsys, monkeypatch):
    original = harness.investigate

    async def crash_on_acme(question, *args, **kwargs):
        if "Acme" in question:
            raise RuntimeError("boom")
        return await original(question, *args, **kwargs)
    monkeypatch.setattr(harness, "investigate", crash_on_acme)
    assert cli.main(["--case", "M1", "--case", "S1", "--out", str(tmp_path)]) == 1
    report = json.loads((tmp_path / "benchmark_v1-offline-subset-S1-M1.json").read_text(
        encoding="utf-8"))
    assert [(c["id"], c["passed"]) for c in report["cases"]] == [("S1", True), ("M1", False)]
    assert any("run_completed" in f and "RuntimeError: boom" in f
               for f in report["cases"][1]["failures"])
    assert "Case M1 (missing_account) failed:" in capsys.readouterr().err


def test_cli_rejects_unknown_case_ids(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["--case", "Z9", "--out", str(tmp_path)])
