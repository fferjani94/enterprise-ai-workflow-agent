"""Milestone 6B.1: per-investigation metrics and tool-call outcomes.

A fake clock is advanced by the fake model and tool client, so every timing is exact.
No Ollama, network or sleeps.
"""

import asyncio
import contextlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import get_args
from unittest import mock

import httpx
import pytest
from langchain_core.messages import AIMessage
from mcp.shared.memory import create_connected_server_and_client_session
from test_account_preresolution import TOOLS as SERVER_TOOLS
from test_account_preresolution import TWO, Directory

from enterprise_ai_workflow_agent.agent import telemetry
from enterprise_ai_workflow_agent.agent.graph import investigate, tool_error_outcome
from enterprise_ai_workflow_agent.agent.telemetry import (
    InvestigationRecorder,
    error_category,
    provider_usage,
)
from enterprise_ai_workflow_agent.api.investigation_schemas import (
    ErrorCategory,
    InvestigationMetrics,
    LlmAttemptMetrics,
    ToolOutcome,
    ToolTraceEntry,
)
from enterprise_ai_workflow_agent.evaluation.harness import build_database
from enterprise_ai_workflow_agent.mcp_server import server, services

HANG = object()  # a model reply or tool behaviour that never completes


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class TimedModel:
    """Replays (seconds, reply) pairs; the bound model and synthesis share the queue."""

    def __init__(self, clock, replies):
        self.clock = clock
        self.replies = list(replies)
        self.entered = None

    def bind_tools(self, specs):
        return self

    async def ainvoke(self, messages):
        seconds, reply = self.replies.pop(0)
        self.clock.advance(seconds)
        if reply is HANG:
            self.entered.set()
            await asyncio.Event().wait()
        if isinstance(reply, BaseException):
            raise reply
        return reply


DATA = {
    "search_accounts": [{"id": 1, "name": "Northstar Labs", "plan": "Enterprise"}],
    "search_knowledge": [{"id": 3, "product_id": 7, "title": "SSO troubleshooting",
                          "snippet": "Check token expiry", "score": 4}],
}


class TimedMCP:
    """Per tool: (seconds, behaviour). Behaviour: "ok", "empty", an MCP error text, an
    exception, or HANG."""

    def __init__(self, clock, behaviours):
        self.clock = clock
        self.behaviours = behaviours
        self.calls = []
        self.entered = None

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        seconds, behaviour = self.behaviours[name]
        self.clock.advance(seconds)
        if behaviour is HANG:
            self.entered.set()
            await asyncio.Event().wait()
        if isinstance(behaviour, BaseException):
            raise behaviour
        if behaviour in ("ok", "empty"):
            result = DATA[name] if behaviour == "ok" else []
            return SimpleNamespace(isError=False, structuredContent={"result": result},
                                   content=[])
        return SimpleNamespace(isError=True, structuredContent=None,
                               content=[SimpleNamespace(type="text", text=behaviour)])


def schema(**props):
    return {"type": "object", "properties": {k: {"type": v} for k, v in props.items()}}


TOOLS = [
    SimpleNamespace(name="search_accounts", description="", inputSchema=schema(query="string")),
    SimpleNamespace(name="get_ticket_details", description="",
                    inputSchema=schema(ticket_id="integer")),
    SimpleNamespace(name="search_knowledge", description="", inputSchema=schema(query="string")),
]


def reply(content="", calls=(), tokens=None, meta=None):
    extra = {}
    if tokens is not None:
        extra["usage_metadata"] = {"input_tokens": tokens[0], "output_tokens": tokens[1],
                                   "total_tokens": tokens[0] + tokens[1]}
    return AIMessage(content=content, response_metadata=meta or {}, tool_calls=[
        {"name": name, "args": args, "id": f"m-{index}-{name}"}
        for index, (name, args) in enumerate(calls)], **extra)


def run(question, model, client, clock, **kwargs):
    recorder = InvestigationRecorder(clock=clock)
    result = asyncio.run(investigate(question, model, client, TOOLS, recorder=recorder,
                                     **kwargs))
    return result, recorder


def validated(metrics):
    """The metrics match the API schema exactly: nothing is dropped or renamed."""
    assert InvestigationMetrics.model_validate(metrics).model_dump() == metrics
    return metrics


# --- Timings and token usage -----------------------------------------------------------

def test_per_attempt_latency_tokens_and_phase_timings_are_exact():
    clock = Clock()
    ollama = {"done_reason": "stop", "load_duration": 250_000_000,
              "prompt_eval_duration": 40_000_000, "eval_duration": 900_000_000,
              "total_duration": 1_200_000_000}
    model = TimedModel(clock, [
        (1.5, reply(calls=[("search_accounts", {"query": "Northstar"})], tokens=(100, 10),
                    meta=ollama)),
        (2.0, reply(calls=[("search_knowledge", {"query": "SSO"})], tokens=(150, 12))),
        (0.5, reply("Northstar Labs is account 1.", tokens=(200, 8))),
        (3.0, reply("Account 1 is Northstar Labs.", tokens=(300, 40))),
    ])
    client = TimedMCP(clock, {"search_accounts": (0.25, "ok"), "search_knowledge": (0.125, "ok")})
    result, _ = run("Find", model, client, clock)
    metrics = validated(result["metrics"])

    attempts = metrics["llm_attempts"]
    assert [(a["stage"], a["turn"], a["call"], a["latency_ms"]) for a in attempts] == [
        ("tool_loop", 1, 1, 1500.0), ("tool_loop", 2, 2, 2000.0), ("tool_loop", 3, 3, 500.0),
        ("synthesis", None, 4, 3000.0)]
    assert [a["outcome"] for a in attempts] == ["ok"] * 4
    assert [a["tool_calls"] for a in attempts] == [1, 1, 0, 0]
    assert [a["prompt_variant"] for a in attempts] == ["with_system"] * 3 + [None]
    first = attempts[0]
    assert (first["input_tokens"], first["output_tokens"], first["total_tokens"]) == (100, 10, 110)
    assert (first["provider_load_ms"], first["provider_prompt_eval_ms"],
            first["provider_eval_ms"], first["provider_total_ms"]) == (250.0, 40.0, 900.0, 1200.0)
    assert first["done_reason"] == "stop"
    assert attempts[1]["done_reason"] is None

    assert metrics["tokens"] == {
        "input": 750, "output": 70, "total": 820, "reported_attempts": 4,
        "unreported_attempts": 0, "by_stage": {
            "tool_loop": {"input": 450, "output": 30, "total": 480, "reported_attempts": 3,
                          "unreported_attempts": 0},
            "synthesis": {"input": 300, "output": 40, "total": 340, "reported_attempts": 1,
                          "unreported_attempts": 0}}}
    assert metrics["llm"] == {
        "calls": 4, "attempts": 4, "retries": 0, "ok_attempts": 4, "error_attempts": 0,
        "transient_error_attempts": 0, "empty_attempts": 0, "cancelled_attempts": 0}
    assert metrics["synthesis"] == {"status": "completed", "latency_ms": 3000.0, "attempts": 1}
    assert metrics["timings_ms"] == {
        "total": None, "queue_wait": None, "preflight": None, "mcp_startup": None,
        "graph": 7375.0, "llm_tool_loop": 4000.0, "llm_synthesis": 3000.0,
        "tool_execution": 375.0, "orchestration_overhead": 0.0}

    assert [e["latency_ms"] for e in result["tool_trace"]] == [250.0, 125.0]
    tools = metrics["tools"]
    assert tools["latency"] == {"count": 2, "total_ms": 375.0, "max_ms": 250.0,
                                "mean_ms": 187.5}
    assert tools["latency_by_tool"]["search_knowledge"]["total_ms"] == 125.0
    assert (tools["trace_entries"], tools["executed"], tools["cancelled"]) == (2, 2, 0)
    assert tools["by_outcome"] == {"executed_success": 2}
    assert tools["by_origin"] == {"model": 2}
    assert (tools["rejected_proposals"], tools["superseded_proposals"]) == (0, 0)
    assert metrics["partial"] is False


def test_retries_are_separate_attempts_and_missing_usage_is_not_zero():
    clock = Clock()
    model = TimedModel(clock, [
        (0.25, RuntimeError("token repeat limit reached (status code: 500)")),
        (0.5, reply("", tokens=(50, 0))),  # empty reply: retried, tokens still counted
        (0.75, reply("Hello.")),  # no usage reported
    ])
    result, _ = run("Hi", model, TimedMCP(clock, {}), clock)
    metrics = validated(result["metrics"])
    attempts = metrics["llm_attempts"]
    assert [(a["attempt"], a["outcome"], a["error_category"], a["prompt_variant"],
             a["latency_ms"]) for a in attempts] == [
        (1, "transient_error", "transient", "with_system", 250.0),
        (2, "empty", None, "without_system", 500.0),
        (3, "ok", None, "with_system", 750.0)]
    assert {a["call"] for a in attempts} == {1}
    assert metrics["llm"] == {
        "calls": 1, "attempts": 3, "retries": 2, "ok_attempts": 1, "error_attempts": 0,
        "transient_error_attempts": 1, "empty_attempts": 1, "cancelled_attempts": 0}
    assert metrics["tokens"]["by_stage"].keys() == {"tool_loop"}
    assert {k: v for k, v in metrics["tokens"].items() if k != "by_stage"} == {
        "input": 50, "output": 0, "total": 50, "reported_attempts": 1,
        "unreported_attempts": 1}
    assert result["model_error"] is None


def test_no_reported_usage_gives_null_token_totals():
    clock = Clock()
    result, _ = run("Hi", TimedModel(clock, [(0.1, reply("Hello."))]), TimedMCP(clock, {}),
                    clock)
    assert validated(result["metrics"])["tokens"] == {
        "input": None, "output": None, "total": None,
        "reported_attempts": 0, "unreported_attempts": 1,
        "by_stage": {"tool_loop": {"input": None, "output": None, "total": None,
                                   "reported_attempts": 0, "unreported_attempts": 1}}}


def test_total_only_usage_is_counted_and_stages_are_separate():
    def partial_usage(content, usage):
        message = AIMessage(content=content)
        message.usage_metadata = usage  # providers may omit fields; bypasses validation
        return message

    clock = Clock()
    model = TimedModel(clock, [
        (0.1, reply(calls=[("search_accounts", {"query": "Northstar"})], tokens=(10, 5))),
        (0.1, partial_usage("Northstar Labs is account 1.", {"total_tokens": 40})),
        (0.1, partial_usage("Account 1 is Northstar Labs.", {"input_tokens": 7})),
    ])
    result, _ = run("Find", model, TimedMCP(clock, {"search_accounts": (0.1, "ok")}), clock)
    tokens = validated(result["metrics"])["tokens"]
    assert (tokens["input"], tokens["output"], tokens["total"]) == (17, 5, 62)
    assert (tokens["reported_attempts"], tokens["unreported_attempts"]) == (3, 0)
    assert tokens["by_stage"]["tool_loop"]["total"] == 55
    assert tokens["by_stage"]["synthesis"] == {
        "input": 7, "output": None, "total": 7, "reported_attempts": 1,
        "unreported_attempts": 0}


def test_non_transient_model_failure_still_returns_metrics():
    clock = Clock()
    result, _ = run("Find", TimedModel(clock, [(0.5, ConnectionError("refused at 10.0.0.1"))]),
                    TimedMCP(clock, {}), clock)
    metrics = validated(result["metrics"])
    assert result["model_error"] == "ConnectionError"
    assert [(a["outcome"], a["error_category"]) for a in metrics["llm_attempts"]] == [
        ("error", "connection")]
    assert metrics["llm"]["error_attempts"] == 1
    assert metrics["synthesis"]["status"] == "not_reached"
    assert "10.0.0.1" not in json.dumps(metrics)
    assert "ConnectionError" not in json.dumps(metrics)


def test_failed_synthesis_is_reported_separately():
    clock = Clock()
    model = TimedModel(clock, [
        (1.0, reply(calls=[("search_accounts", {"query": "Northstar"})])),
        (1.0, reply("Northstar Labs is account 1.")),
        (2.0, ConnectionError("down")),
    ])
    client = TimedMCP(clock, {"search_accounts": (0.5, "ok")})
    result, _ = run("Find", model, client, clock)
    metrics = validated(result["metrics"])
    assert metrics["synthesis"] == {"status": "failed", "latency_ms": 2000.0, "attempts": 1}
    assert metrics["timings_ms"]["llm_tool_loop"] == 2000.0
    assert result["answer_source"] == "evidence_summary"


def test_skipped_synthesis_makes_no_model_call():
    clock = Clock()
    model = TimedModel(clock, [
        (1.0, reply(calls=[("search_accounts", {"query": "Nobody"})])),
        (1.0, reply(calls=[("search_accounts", {"query": "Still nobody"})])),
    ])
    client = TimedMCP(clock, {"search_accounts": (0.5, "empty")})
    result, _ = run("Find", model, client, clock, max_rounds=1)
    metrics = validated(result["metrics"])
    assert metrics["synthesis"] == {"status": "skipped", "latency_ms": None, "attempts": 0}
    assert metrics["tools"]["rejected_proposals"] == 1  # the call at the round limit


# --- Tool outcomes ---------------------------------------------------------------------

def test_duplicate_is_rejected_not_failed():
    """Mirrors the live run: the model repeats a successful search_knowledge call."""
    clock = Clock()
    model = TimedModel(clock, [
        (1.0, reply(calls=[("search_knowledge", {"query": "SSO"})])),
        (1.0, reply(calls=[("search_knowledge", {"query": "SSO"})])),
        (1.0, reply("Document 3 covers SSO.")),
        (1.0, reply("Document 3 covers SSO.")),
    ])
    client = TimedMCP(clock, {"search_knowledge": (0.2, "ok")})
    result, _ = run("Find", model, client, clock)
    first, duplicate = result["tool_trace"]
    assert first["outcome"] == "executed_success" and first["latency_ms"] == 200.0
    assert duplicate["outcome"] == "duplicate_rejected"
    assert (duplicate["executed"], duplicate["success"]) == (False, False)
    assert duplicate["error"].startswith("Duplicate call skipped")
    assert "latency_ms" not in duplicate
    assert len(client.calls) == 1
    tools = validated(result["metrics"])["tools"]
    assert tools["by_outcome"] == {"duplicate_rejected": 1, "executed_success": 1}
    assert (tools["trace_entries"], tools["executed"]) == (2, 1)
    assert tools["rejected_proposals"] == 1


def test_rejections_and_execution_errors_are_classified_without_changing_legacy_fields():
    clock = Clock()
    batch = reply(calls=[
        ("no_such_tool", {}),
        ("get_ticket_details", {"ticket_id": 99}),
        ("search_knowledge", {"query": "SSO"}),
        ("search_accounts", {"query": "Northstar"}),
        ("get_ticket_details", {"ticket_id": 98}),
        ("search_accounts", {"query": "over the limit"}),
    ])
    model = TimedModel(clock, [(1.0, batch), (1.0, reply("Nothing found."))])
    client = TimedMCP(clock, {
        "search_knowledge": (0.5, "Error executing tool search_knowledge: Internal error "
                                  "while reading data; the lookup was not completed."),
        "search_accounts": (2.0, TimeoutError("read timed out from 10.0.0.1")),
    })
    result, _ = run("Find", model, client, clock)
    trace = result["tool_trace"]
    observed = [(e["outcome"], e["executed"], e["success"], e["error"].split(":")[0])
                for e in trace]
    assert observed == [
        ("unknown_tool", False, False, "Unknown tool"),
        ("unverified_id_rejected", False, False, "Unverified record ID"),
        ("executed_error", True, False, "MCP tool returned an error"),
        ("executed_error", True, False, "Tool unavailable"),
        ("unverified_id_rejected", False, False, "Unverified record ID"),
        ("call_limit_rejected", False, False, "Tool-call limit exceeded"),
    ]
    assert trace[2]["latency_ms"] == 500.0 and "error_category" not in trace[2]
    assert (trace[3]["latency_ms"], trace[3]["error_category"]) == (2000.0, "timeout")
    assert all("latency_ms" not in e for e in (trace[0], trace[1], trace[4], trace[5]))
    for entry in trace:
        ToolTraceEntry.model_validate(entry)
    tools = validated(result["metrics"])["tools"]
    assert tools["by_outcome"] == {
        "call_limit_rejected": 1, "executed_error": 2, "unknown_tool": 1,
        "unverified_id_rejected": 2}
    assert tools["executed"] == 2
    assert tools["latency_by_tool"] == {
        "search_accounts": {"count": 1, "total_ms": 2000.0, "max_ms": 2000.0,
                            "mean_ms": 2000.0},
        "search_knowledge": {"count": 1, "total_ms": 500.0, "max_ms": 500.0,
                             "mean_ms": 500.0}}
    assert "10.0.0.1" not in json.dumps(result["metrics"])


def test_orchestrator_calls_have_an_origin_and_trigger_but_a_normal_outcome():
    model = TimedModel(Clock(), [
        (0, AIMessage(content="Both have tickets.")),
        (0, AIMessage(content="Ticket 70 (high) and Ticket 80 (medium) are open.")),
    ])
    result = asyncio.run(investigate(TWO, model, Directory(), SERVER_TOOLS))
    assert [(e["origin"], e.get("trigger"), e["outcome"]) for e in result["tool_trace"]] == [
        ("orchestrator", "pre_resolution", "executed_success"),
        ("orchestrator", "pre_resolution", "executed_success"),
        ("orchestrator", "followup", "executed_success"),
        ("orchestrator", "followup", "executed_success"),
    ]
    tools = validated(result["metrics"])["tools"]
    assert tools["by_origin"] == {"orchestrator": 4}
    assert tools["by_trigger"] == {"followup": 2, "pre_resolution": 2}
    assert tools["by_outcome"] == {"executed_success": 4}


@pytest.mark.parametrize("text, outcome", [
    ("Error executing tool search_knowledge: 1 validation error for search_knowledgeArguments"
     "\nlimit\n  Input should be a valid integer [type=int_parsing]", "invalid_arguments"),
    ("Error executing tool search_knowledge: query must not be empty", "invalid_arguments"),
    ("Error executing tool search_knowledge: limit must be between 1 and 20",
     "invalid_arguments"),
    ("Error executing tool search_knowledge: Internal error while reading data; the lookup "
     "was not completed.", "executed_error"),
    # Server-side output validation is a tool failure, not a caller error.
    ("Error executing tool search_knowledge: Output validation error: 'id' is a required "
     "property", "executed_error"),
    ("Error executing tool search_knowledge: 1 validation error for search_knowledgeOutput"
     "\nresult\n  Input should be a valid list", "executed_error"),
    ("Error executing tool search_knowledge: 2 validation errors for search_knowledgeArguments"
     "\nquery\n  Field required", "invalid_arguments"),
    ("", "executed_error"),
    ("Something unexpected happened", "executed_error"),
])
def test_mcp_error_text_classification(text, outcome):
    assert tool_error_outcome("search_knowledge", text) == outcome


def test_every_service_argument_error_is_classified_as_invalid_arguments():
    source = Path(services.__file__).read_text(encoding="utf-8")
    messages = re.findall(r'raise ValueError\(\s*f?"([^"]+)"', source)
    assert messages
    for message in messages:
        assert tool_error_outcome("x", f"Error executing tool x: {message}") == \
            "invalid_arguments", message


def test_real_mcp_validation_errors_are_invalid_arguments():
    def broken_search(*args, **kwargs):
        raise RuntimeError("db at /secret/path")

    async def scenario():
        sessions = build_database("default")
        try:
            with mock.patch.object(server, "SessionLocal", sessions), \
                    mock.patch.object(services, "search_accounts", broken_search):
                async with create_connected_server_and_client_session(
                        server.mcp._mcp_server) as client:
                    tools = (await client.list_tools()).tools
                    model = TimedModel(Clock(), [
                        (0, reply(calls=[
                            ("search_knowledge", {"query": "SSO", "limit": "many"}),
                            ("search_knowledge", {"query": "   "}),
                            ("search_accounts", {"query": "Northstar"}),
                        ])),
                        (0, reply("Nothing usable.")),
                    ])
                    return await investigate("Find", model, client, tools)
        finally:
            sessions.kw["bind"].dispose()

    result = asyncio.run(scenario())
    trace = result["tool_trace"]
    assert [(e["outcome"], e["executed"]) for e in trace] == [
        ("invalid_arguments", True), ("invalid_arguments", True), ("executed_error", True)]
    assert trace[0]["error"].startswith("MCP tool returned an error: Invalid arguments.")
    assert "secret" not in json.dumps(result["metrics"])


# --- Cancellation ----------------------------------------------------------------------

def _cancel_when_entered(clock, model, client, wait_for, seconds=2.0):
    recorder = InvestigationRecorder(clock=clock)

    async def scenario():
        wait_for.entered = asyncio.Event()
        task = asyncio.create_task(investigate("Find", model, client, TOOLS, recorder=recorder))
        await wait_for.entered.wait()
        clock.advance(seconds)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    return validated(recorder.snapshot(partial=True))


def test_cancelled_tool_call_is_kept_in_partial_metrics():
    clock = Clock()
    model = TimedModel(clock, [(0.5, reply(calls=[("search_accounts", {"query": "N"})]))])
    client = TimedMCP(clock, {"search_accounts": (0.0, HANG)})
    metrics = _cancel_when_entered(clock, model, client, client)
    assert metrics["partial"] is True
    assert metrics["tools"]["by_outcome"] == {"cancelled": 1}
    assert metrics["tools"]["cancelled"] == 1
    assert metrics["tools"]["latency"]["count"] == 0
    assert (metrics["tools"]["trace_entries"], metrics["tools"]["executed"]) == (0, 0)
    assert metrics["timings_ms"]["tool_execution"] == 2000.0
    assert metrics["timings_ms"]["graph"] == 2500.0
    assert metrics["tools"]["rejected_proposals"] is None


def test_cancelled_model_attempt_is_kept_in_partial_metrics():
    clock = Clock()
    model = TimedModel(clock, [(0.25, HANG)])
    metrics = _cancel_when_entered(clock, model, TimedMCP(clock, {}), model)
    assert [(a["outcome"], a["latency_ms"]) for a in metrics["llm_attempts"]] == [
        ("cancelled", 2250.0)]
    assert metrics["synthesis"]["status"] == "not_reached"


def test_cancelled_synthesis_is_reported_as_cancelled():
    clock = Clock()
    model = TimedModel(clock, [
        (0.5, reply(calls=[("search_accounts", {"query": "Northstar"})])),
        (0.5, reply("Northstar Labs is account 1.")),
        (0.0, HANG),
    ])
    client = TimedMCP(clock, {"search_accounts": (0.1, "ok")})
    metrics = _cancel_when_entered(clock, model, client, model, seconds=1.0)
    assert metrics["synthesis"] == {"status": "cancelled", "latency_ms": 1000.0, "attempts": 1}


# --- Privacy and schema ----------------------------------------------------------------

def test_metrics_contain_no_question_arguments_results_or_error_messages():
    secret = "SENTINEL-7f3a"
    clock = Clock()
    DATA["search_knowledge"][0]["snippet"] = secret
    try:
        model = TimedModel(clock, [
            (1.0, reply(calls=[("search_knowledge", {"query": secret}),
                               ("search_accounts", {"query": secret})])),
            (1.0, reply(f"Document 3 says {secret}.")),
            (1.0, reply(f"Document 3 says {secret}.")),
        ])
        client = TimedMCP(clock, {"search_knowledge": (0.1, "ok"),
                                  "search_accounts": (0.1, OSError(secret))})
        result, _ = run(f"Find {secret}", model, client, clock)
    finally:
        DATA["search_knowledge"][0]["snippet"] = "Check token expiry"
    assert secret in json.dumps(result["tool_trace"])  # the trace keeps arguments, as before
    assert secret not in json.dumps(validated(result["metrics"]))


class ResponseError(Exception):
    """Stands in for a provider-specific exception class."""


@pytest.mark.parametrize("error, transient, category", [
    (httpx.ConnectError("refused"), False, "connection"),
    (httpx.RemoteProtocolError("closed"), False, "connection"),
    (ConnectionRefusedError(), False, "connection"),
    (httpx.ReadTimeout("slow"), False, "timeout"),
    (TimeoutError(), True, "timeout"),
    (ResponseError("token repeat limit reached"), True, "transient"),
    (ResponseError("model crashed"), False, "other"),
    (RuntimeError("x"), False, "other"),
])
def test_error_categories_hide_exception_class_names(error, transient, category):
    assert error_category(error, transient) == category


# --- Best-effort instrumentation -----------------------------------------------------------

class BrokenClock:
    def __call__(self):
        raise RuntimeError("clock failure")


def _simple_run(recorder, model_error=None):
    model = TimedModel(Clock(), [
        (0, reply(calls=[("search_accounts", {"query": "Northstar"})])),
        (0, model_error or reply("Northstar Labs is account 1.")),
        (0, reply("Account 1 is Northstar Labs.")),
    ])
    client = TimedMCP(Clock(), {"search_accounts": (0, "ok")})
    return asyncio.run(investigate("Find", model, client, TOOLS, recorder=recorder))


def test_a_failing_clock_does_not_change_the_investigation():
    baseline = _simple_run(InvestigationRecorder(clock=Clock()))
    broken = _simple_run(InvestigationRecorder(clock=BrokenClock()))
    for key in ("answer", "tool_trace", "evidence", "model_proposals", "model_turn_log"):
        assert broken[key] == baseline[key], key
    validated(broken["metrics"])


def test_a_failing_snapshot_omits_metrics_without_changing_the_result(monkeypatch):
    def explode(self, partial=False):
        raise RuntimeError("snapshot failure")

    monkeypatch.setattr(InvestigationRecorder, "snapshot", explode)
    result = _simple_run(InvestigationRecorder(clock=Clock()))
    assert result["metrics"] is None
    assert result["model_error"] is None and result["tool_trace"][0]["success"] is True


class FailingList(list):
    def append(self, item):
        raise RuntimeError("recording failure")


def _failing_recorder(clock):
    recorder = InvestigationRecorder(clock=clock)
    recorder.llm_attempts = FailingList()
    recorder.tool_calls = FailingList()
    return recorder


def test_failing_recording_never_replaces_a_model_error():
    result = _simple_run(_failing_recorder(Clock()), model_error=ConnectionError("down"))
    assert result["model_error"] == "ConnectionError"
    assert result["tool_trace"][0]["outcome"] == "executed_success"


def test_failing_recording_never_blocks_cancellation():
    clock = Clock()
    recorder = _failing_recorder(clock)
    model = TimedModel(clock, [(0.5, reply(calls=[("search_accounts", {"query": "N"})]))])
    client = TimedMCP(clock, {"search_accounts": (0.0, HANG)})

    async def scenario():
        client.entered = asyncio.Event()
        task = asyncio.create_task(investigate("Find", model, client, TOOLS, recorder=recorder))
        await client.entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert recorder.phases["graph"] == 500.0


def test_provider_usage_ignores_unexpected_values():
    message = SimpleNamespace(
        usage_metadata={"input_tokens": -1, "output_tokens": "12", "total_tokens": True},
        response_metadata={"done_reason": "secret text", "load_duration": "1",
                           "eval_duration": -5, "message": {"content": "secret"}})
    usage = provider_usage(message)
    assert usage == {"input_tokens": None, "output_tokens": None, "total_tokens": None,
                     "provider_load_ms": None, "provider_prompt_eval_ms": None,
                     "provider_eval_ms": None, "provider_total_ms": None,
                     "done_reason": "unknown"}
    assert provider_usage(None)["input_tokens"] is None


def test_schema_literals_match_recorder_constants():
    assert set(get_args(ToolOutcome)) == set(telemetry.TOOL_OUTCOMES)
    attempt = LlmAttemptMetrics.model_fields
    assert set(get_args(attempt["outcome"].annotation)) == set(telemetry.ATTEMPT_OUTCOMES)
    trigger = ToolTraceEntry.model_fields["trigger"].annotation
    assert set(get_args(get_args(trigger)[0])) == set(telemetry.TOOL_TRIGGERS)
    assert set(get_args(ErrorCategory)) == set(telemetry.ERROR_CATEGORIES)


def test_recorder_phase_is_recorded_even_when_the_block_raises():
    clock = Clock()
    recorder = InvestigationRecorder(clock=clock)
    with contextlib.suppress(RuntimeError), recorder.phase("preflight"):
        clock.advance(0.004)
        raise RuntimeError
    assert recorder.snapshot()["timings_ms"]["preflight"] == 4.0
    assert recorder.has_data()
    assert not InvestigationRecorder().has_data()
