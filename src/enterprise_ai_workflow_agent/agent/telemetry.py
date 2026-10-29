"""Per-investigation diagnostics: timings, provider token usage and tool-call outcomes.

Nothing here is read by routing, evidence, coverage or grounding. Only counts, durations,
tool names and fixed error categories are kept: never prompts, replies, arguments, results,
exception messages or exception class names. Recording is best effort and never raises.
"""

import logging
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Tool-call outcomes. Who made the call is recorded separately as origin/trigger.
EXECUTED_SUCCESS = "executed_success"
EXECUTED_ERROR = "executed_error"  # the tool ran (or was sent) and failed
INVALID_ARGUMENTS = "invalid_arguments"  # rejected locally or by MCP-side validation
UNKNOWN_TOOL = "unknown_tool"
UNVERIFIED_ID_REJECTED = "unverified_id_rejected"
DUPLICATE_REJECTED = "duplicate_rejected"
REDUNDANT_PAGE_REJECTED = "redundant_page_rejected"
CALL_LIMIT_REJECTED = "call_limit_rejected"
TOOL_OUTCOMES = (EXECUTED_SUCCESS, EXECUTED_ERROR, INVALID_ARGUMENTS, UNKNOWN_TOOL,
                 UNVERIFIED_ID_REJECTED, DUPLICATE_REJECTED, REDUNDANT_PAGE_REJECTED,
                 CALL_LIMIT_REJECTED)
# Recorder only: a call still in flight when the investigation was cancelled.
CANCELLED = "cancelled"

TRIGGER_PRE_RESOLUTION = "pre_resolution"
TRIGGER_FOLLOWUP = "followup"
TOOL_TRIGGERS = (TRIGGER_PRE_RESOLUTION, TRIGGER_FOLLOWUP)

STAGE_TOOL_LOOP = "tool_loop"
STAGE_SYNTHESIS = "synthesis"
STAGES = (STAGE_TOOL_LOOP, STAGE_SYNTHESIS)

ATTEMPT_OK = "ok"
ATTEMPT_EMPTY = "empty"
ATTEMPT_TRANSIENT = "transient_error"
ATTEMPT_ERROR = "error"
ATTEMPT_OUTCOMES = (ATTEMPT_OK, ATTEMPT_EMPTY, ATTEMPT_TRANSIENT, ATTEMPT_ERROR, CANCELLED)

# Client-safe replacements for exception class names.
ERROR_TIMEOUT = "timeout"
ERROR_CONNECTION = "connection"
ERROR_TRANSIENT = "transient"
ERROR_OTHER = "other"
ERROR_CATEGORIES = (ERROR_TIMEOUT, ERROR_CONNECTION, ERROR_TRANSIENT, ERROR_OTHER)
# httpx/httpcore classes that are not subclasses of the built-in connection errors.
_CONNECTION_ERROR_NAMES = frozenset({"ConnectError", "RemoteProtocolError", "ReadError",
                                     "WriteError", "NetworkError", "ProxyError"})

SYNTHESIS_COMPLETED = "completed"
SYNTHESIS_FAILED = "failed"
SYNTHESIS_SKIPPED = "skipped"  # deterministic evidence summary; the model was not called
SYNTHESIS_NOT_REACHED = "not_reached"

PHASES = ("total", "queue_wait", "preflight", "mcp_startup", "graph")
DONE_REASONS = ("stop", "length", "load")
_MAX_NAME = 100


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _ns_to_ms(value: Any) -> float | None:
    return round(value / 1e6, 3) if type(value) in (int, float) and value >= 0 else None


def _name(value: Any) -> str | None:
    return None if value is None else str(value)[:_MAX_NAME]


def error_category(error: BaseException, transient: bool = False) -> str:
    """A fixed category for an exception, so no provider-specific class name is exposed."""
    name = type(error).__name__
    if isinstance(error, TimeoutError) or "Timeout" in name:
        return ERROR_TIMEOUT
    if isinstance(error, ConnectionError) or name in _CONNECTION_ERROR_NAMES:
        return ERROR_CONNECTION
    return ERROR_TRANSIENT if transient else ERROR_OTHER


def provider_usage(reply: Any) -> dict[str, Any]:
    """Allow-listed numeric fields of a chat reply; None where the provider reports nothing.

    Token counts come from LangChain's ``usage_metadata``; durations (nanoseconds) and
    ``done_reason`` from Ollama's ``response_metadata``.
    """
    usage = getattr(reply, "usage_metadata", None)
    usage = usage if isinstance(usage, dict) else {}
    meta = getattr(reply, "response_metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    done = meta.get("done_reason")
    return {
        "input_tokens": _count(usage.get("input_tokens")),
        "output_tokens": _count(usage.get("output_tokens")),
        "total_tokens": _count(usage.get("total_tokens")),
        "provider_load_ms": _ns_to_ms(meta.get("load_duration")),
        "provider_prompt_eval_ms": _ns_to_ms(meta.get("prompt_eval_duration")),
        "provider_eval_ms": _ns_to_ms(meta.get("eval_duration")),
        "provider_total_ms": _ns_to_ms(meta.get("total_duration")),
        "done_reason": None if done is None else (
            done if isinstance(done, str) and done in DONE_REASONS else "unknown"),
    }


def _len(value: Any) -> int:
    try:
        return len(value or [])
    except TypeError:
        return 0


def _sum(values: Iterable[float]) -> float:
    return round(sum(values), 3)


def _latency(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "total_ms": 0.0, "max_ms": None, "mean_ms": None}
    total = _sum(values)
    return {"count": len(values), "total_ms": total, "max_ms": max(values),
            "mean_ms": round(total / len(values), 3)}


def _counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _token_total(values: Iterable[int | None]) -> int | None:
    reported = [value for value in values if value is not None]
    return sum(reported) if reported else None


def _reported(attempt: dict[str, Any]) -> bool:
    return any(attempt[key] is not None
               for key in ("input_tokens", "output_tokens", "total_tokens"))


def _attempt_total(attempt: dict[str, Any]) -> int | None:
    if attempt["total_tokens"] is not None:
        return attempt["total_tokens"]
    if attempt["input_tokens"] is None and attempt["output_tokens"] is None:
        return None
    return (attempt["input_tokens"] or 0) + (attempt["output_tokens"] or 0)


def _tokens(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Token totals over attempts, failed and empty ones included (the provider did the work)."""
    reported = [a for a in attempts if _reported(a)]
    replied = [a for a in attempts if a["outcome"] in (ATTEMPT_OK, ATTEMPT_EMPTY)]
    return {
        "input": _token_total(a["input_tokens"] for a in reported),
        "output": _token_total(a["output_tokens"] for a in reported),
        "total": _token_total(_attempt_total(a) for a in reported),
        "reported_attempts": len(reported),
        "unreported_attempts": sum(1 for a in replied if not _reported(a)),
    }


class InvestigationRecorder:
    """Collects one investigation's diagnostics. Not thread-safe; one per investigation.

    It is held by the caller, not the graph state, so what was recorded before a timeout or
    cancellation can still be read. Every method is best effort: a recording failure is
    logged and swallowed, so instrumentation never replaces an exception or blocks
    cancellation. ``clock`` is injectable for deterministic tests.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self.clock = clock
        self.phases: dict[str, float] = {}
        self.llm_attempts: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.synthesis_status = SYNTHESIS_NOT_REACHED
        self.proposals: list[dict[str, Any]] | None = None
        self._llm_calls = 0

    def now(self) -> float:
        try:
            return float(self.clock())
        except Exception:
            logger.debug("Telemetry clock failed", exc_info=True)
            return 0.0

    def elapsed_ms(self, started: float) -> float:
        return max(_ms(self.now() - started), 0.0)

    def add_phase(self, name: str, ms: float) -> None:
        try:
            self.phases[name] = round(self.phases.get(name, 0.0) + ms, 3)
        except Exception:
            logger.debug("Telemetry phase %s not recorded", name, exc_info=True)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = self.now()
        try:
            yield
        finally:
            self.add_phase(name, self.elapsed_ms(started))

    def start_llm_call(self) -> int:
        """Number one logical model call; its retries share the number."""
        self._llm_calls += 1
        return self._llm_calls

    def record_llm_attempt(
        self, *, stage: str, call: int, attempt: int, outcome: str, latency_ms: float,
        turn: int | None = None, prompt_variant: str | None = None,
        error: BaseException | None = None, reply: Any = None,
    ) -> None:
        try:
            self.llm_attempts.append({
                "stage": stage, "call": call, "turn": turn, "attempt": attempt,
                "prompt_variant": prompt_variant, "outcome": outcome,
                "error_category": None if error is None else error_category(
                    error, transient=outcome == ATTEMPT_TRANSIENT),
                "latency_ms": latency_ms,
                "tool_calls": _len(getattr(reply, "tool_calls", None)),
                "invalid_tool_calls": _len(getattr(reply, "invalid_tool_calls", None)),
                **provider_usage(reply),
            })
        except Exception:
            logger.debug("Telemetry model attempt not recorded", exc_info=True)

    def record_tool(
        self, *, tool: Any, origin: str, trigger: str | None, outcome: str, executed: bool,
        latency_ms: float | None = None, error_category: str | None = None,
    ) -> None:
        try:
            self.tool_calls.append({
                "tool": _name(tool), "origin": origin, "trigger": trigger, "outcome": outcome,
                "executed": executed, "latency_ms": latency_ms,
                "error_category": error_category,
            })
        except Exception:
            logger.debug("Telemetry tool call not recorded", exc_info=True)

    def has_data(self) -> bool:
        return bool(self.phases or self.llm_attempts or self.tool_calls)

    def safe_snapshot(self, partial: bool = False) -> dict[str, Any] | None:
        """``snapshot()``, or None if it fails: metrics are optional."""
        try:
            return self.snapshot(partial)
        except Exception:
            logger.warning("Investigation metrics could not be collected", exc_info=True)
            return None

    def snapshot(self, partial: bool = False) -> dict[str, Any]:
        """Machine-readable metrics. ``partial``: the investigation did not finish normally.

        Calls and attempts in flight at cancellation are counted as ``cancelled``; their
        elapsed time is included in the stage timings but not in tool latency statistics.
        Anything recorded after the snapshot is not reflected in it.
        """
        attempts = list(self.llm_attempts)
        tool_calls = list(self.tool_calls)
        synthesis = [a for a in attempts if a["stage"] == STAGE_SYNTHESIS]
        loop_ms = _sum(a["latency_ms"] for a in attempts if a["stage"] == STAGE_TOOL_LOOP)
        synthesis_ms = _sum(a["latency_ms"] for a in synthesis)
        tool_ms = _sum(t["latency_ms"] for t in tool_calls if t["latency_ms"] is not None)
        graph = self.phases.get("graph")
        timings = {name: self.phases.get(name) for name in PHASES}
        timings.update(
            llm_tool_loop=loop_ms, llm_synthesis=synthesis_ms, tool_execution=tool_ms,
            orchestration_overhead=None if graph is None else round(
                max(graph - loop_ms - synthesis_ms - tool_ms, 0.0), 3),
        )

        outcomes = Counter(a["outcome"] for a in attempts)
        calls = len({a["call"] for a in attempts})
        tokens = _tokens(attempts)
        tokens["by_stage"] = {stage: _tokens([a for a in attempts if a["stage"] == stage])
                              for stage in STAGES if any(a["stage"] == stage for a in attempts)}

        traced = [t for t in tool_calls if t["outcome"] != CANCELLED]
        executed = [t for t in traced if t["executed"]]
        by_tool: dict[str, list[float]] = {}
        for item in executed:
            if item["latency_ms"] is not None:
                by_tool.setdefault(item["tool"], []).append(item["latency_ms"])
        proposals = self.proposals

        status = self.synthesis_status
        if status == SYNTHESIS_NOT_REACHED and synthesis:
            status = CANCELLED  # synthesis started but never returned
        return {
            "schema_version": SCHEMA_VERSION,
            "partial": partial,
            "timings_ms": timings,
            "llm": {
                "calls": calls,
                "attempts": len(attempts),
                "retries": len(attempts) - calls,
                "ok_attempts": outcomes[ATTEMPT_OK],
                "error_attempts": outcomes[ATTEMPT_ERROR],
                "transient_error_attempts": outcomes[ATTEMPT_TRANSIENT],
                "empty_attempts": outcomes[ATTEMPT_EMPTY],
                "cancelled_attempts": outcomes[CANCELLED],
            },
            "tokens": tokens,
            "llm_attempts": [dict(a) for a in attempts],
            "synthesis": {"status": status,
                          "latency_ms": synthesis_ms if synthesis else None,
                          "attempts": len(synthesis)},
            "tools": {
                "trace_entries": len(traced),
                "executed": len(executed),
                "cancelled": len(tool_calls) - len(traced),
                "by_outcome": _counts(t["outcome"] for t in tool_calls),
                "by_origin": _counts(t["origin"] for t in traced),
                "by_trigger": _counts(t["trigger"] for t in traced if t["trigger"]),
                "rejected_proposals": None if proposals is None else sum(
                    1 for p in proposals if p.get("outcome") == "rejected"),
                "superseded_proposals": None if proposals is None else sum(
                    1 for p in proposals if p.get("outcome") == "superseded"),
                "latency": _latency([t["latency_ms"] for t in executed
                                     if t["latency_ms"] is not None]),
                "latency_by_tool": {name: _latency(values)
                                    for name, values in sorted(by_tool.items())},
            },
        }
