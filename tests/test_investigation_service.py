"""Service tests: timeouts, cancellation, concurrency and error mapping without Ollama or MCP."""

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage

from enterprise_ai_workflow_agent.agent import service as service_module
from enterprise_ai_workflow_agent.agent.graph import investigate
from enterprise_ai_workflow_agent.agent.service import (
    InvestigationError,
    InvestigationService,
    check_ollama,
    run_agent,
)
from enterprise_ai_workflow_agent.agent.telemetry import InvestigationRecorder
from enterprise_ai_workflow_agent.api.investigation_schemas import InvestigationMetrics
from enterprise_ai_workflow_agent.core.config import Settings

RESULT = {"answer": "ok", "tool_rounds": 0, "tool_trace": [], "evidence": [],
          "limited": False, "model_error": None}


def make_settings(**overrides):
    return Settings(_env_file=None, **overrides)


def ollama_transport(models=None, status=200, connect_error=False):
    def handler(request):
        if connect_error:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(status, json={"models": models or []})
    return httpx.MockTransport(handler)


# --- Ollama preflight -------------------------------------------------------------------

def preflight(transport, model="qwen2.5:3b"):
    return asyncio.run(check_ollama("http://ollama.test", model, 1.0, transport))


def test_preflight_accepts_installed_model():
    preflight(ollama_transport([{"name": "qwen2.5:3b"}]))


def test_preflight_accepts_implicit_latest_tag():
    preflight(ollama_transport([{"name": "llama3:latest"}]), model="llama3")


def test_preflight_reports_ollama_down():
    with pytest.raises(InvestigationError) as caught:
        preflight(ollama_transport(connect_error=True))
    assert caught.value.code == "ollama_unavailable"
    assert "ollama.test" not in caught.value.message


def test_preflight_reports_bad_status():
    with pytest.raises(InvestigationError) as caught:
        preflight(ollama_transport(status=500))
    assert caught.value.code == "ollama_unavailable"


def test_preflight_reports_missing_model():
    with pytest.raises(InvestigationError) as caught:
        preflight(ollama_transport([{"name": "other:1b"}]))
    assert caught.value.code == "model_unavailable"
    assert "ollama pull qwen2.5:3b" in caught.value.message


# --- run_agent wiring -------------------------------------------------------------------

@pytest.fixture
def wiring(monkeypatch):
    state = SimpleNamespace(session_closed=False, models_built=0, result=dict(RESULT),
                            session_error=None)

    def build(settings, model_name):
        state.models_built += 1
        return object()

    async def no_preflight(*args, **kwargs):
        return None

    @asynccontextmanager
    async def session(settings):
        if state.session_error:
            raise state.session_error
        try:
            async def list_tools():
                return SimpleNamespace(tools=[])
            yield SimpleNamespace(list_tools=list_tools)
        finally:
            state.session_closed = True

    async def fake_investigate(question, model, client, tools, max_rounds, recorder=None):
        state.recorder = recorder
        return state.result

    monkeypatch.setattr(service_module, "build_chat_model", build)
    monkeypatch.setattr(service_module, "check_ollama", no_preflight)
    monkeypatch.setattr(service_module, "mcp_session", session)
    monkeypatch.setattr(service_module, "investigate", fake_investigate)
    return state


def test_run_agent_returns_result_and_closes_session(wiring):
    result = asyncio.run(run_agent(make_settings(), "q", "qwen2.5:3b", 4))
    assert result["answer"] == "ok"
    assert wiring.session_closed
    assert wiring.models_built == 1


@pytest.mark.parametrize("model_error, code", [
    ("ConnectError", "ollama_unavailable"),
    ("ResponseError", "model_inference_failed"),
])
def test_model_failure_inside_graph_becomes_error(wiring, model_error, code):
    wiring.result = {**RESULT, "model_error": model_error}
    with pytest.raises(InvestigationError) as caught:
        asyncio.run(run_agent(make_settings(), "q", "qwen2.5:3b", 4))
    assert caught.value.code == code
    assert model_error not in caught.value.message
    assert wiring.session_closed


def test_mcp_startup_failure_is_reported(wiring):
    wiring.session_error = OSError("cannot spawn C:\\secret\\python.exe")
    with pytest.raises(InvestigationError) as caught:
        asyncio.run(run_agent(make_settings(), "q", "qwen2.5:3b", 4))
    assert caught.value.code == "mcp_unavailable"
    assert "secret" not in caught.value.message


def test_missing_ollama_integration_is_reported(wiring, monkeypatch):
    def missing(settings, model_name):
        raise ImportError("langchain_ollama")
    monkeypatch.setattr(service_module, "build_chat_model", missing)
    with pytest.raises(InvestigationError) as caught:
        asyncio.run(run_agent(make_settings(), "q", "qwen2.5:3b", 4))
    assert caught.value.code == "ollama_integration_missing"


# --- InvestigationService ---------------------------------------------------------------

def test_service_runs_agent_off_the_event_loop_thread():
    seen = {}

    async def agent(question, model, max_rounds):
        seen["thread"] = threading.current_thread()
        seen["args"] = (question, model, max_rounds)
        return RESULT

    service = InvestigationService(make_settings(), agent)
    assert asyncio.run(service.run("q", "m", 2)) == RESULT
    assert seen["args"] == ("q", "m", 2)
    assert seen["thread"] is not threading.main_thread()


def test_agent_errors_propagate():
    async def agent(question, model, max_rounds):
        raise InvestigationError("ollama_unavailable", "down")

    service = InvestigationService(make_settings(), agent)
    with pytest.raises(InvestigationError) as caught:
        asyncio.run(service.run("q", "m", 1))
    assert caught.value.code == "ollama_unavailable"


def test_timeout_cancels_agent_and_releases_slot():
    cleaned_up = threading.Event()

    async def hanging(question, model, max_rounds):
        try:
            await asyncio.sleep(30)
        finally:
            cleaned_up.set()  # stands in for MCP session / subprocess teardown

    async def scenario():
        service = InvestigationService(make_settings(investigation_timeout_seconds=0.3), hanging)
        started = time.monotonic()
        with pytest.raises(InvestigationError) as caught:
            await service.run("q", "m", 1)
        assert caught.value.code == "timeout"
        assert time.monotonic() - started < 5
        assert cleaned_up.is_set()
        assert not service._slots.locked()

    asyncio.run(scenario())


def test_waiting_for_a_slot_counts_towards_the_timeout():
    async def hanging(question, model, max_rounds):
        await asyncio.sleep(30)

    async def scenario():
        service = InvestigationService(make_settings(investigation_timeout_seconds=0.4), hanging)
        started = time.monotonic()
        results = await asyncio.gather(
            service.run("a", "m", 1), service.run("b", "m", 1), return_exceptions=True
        )
        assert all(isinstance(r, InvestigationError) and r.code == "timeout" for r in results)
        assert time.monotonic() - started < 5

    asyncio.run(scenario())


def test_concurrency_is_limited():
    lock = threading.Lock()
    active = {"now": 0, "peak": 0}

    async def agent(question, model, max_rounds):
        with lock:
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0.05)
        with lock:
            active["now"] -= 1
        return RESULT

    async def scenario():
        service = InvestigationService(make_settings(investigation_max_concurrency=1), agent)
        await asyncio.gather(*(service.run("q", "m", 1) for _ in range(4)))

    asyncio.run(scenario())
    assert active["peak"] == 1


def test_cancelling_the_request_cancels_the_agent_and_frees_the_slot():
    started, cleaned_up = threading.Event(), threading.Event()

    async def hanging(question, model, max_rounds):
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            cleaned_up.set()

    async def scenario():
        service = InvestigationService(make_settings(), hanging)
        request = asyncio.create_task(service.run("q", "m", 1))
        while not started.is_set():
            await asyncio.sleep(0.01)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        for _ in range(300):
            if cleaned_up.is_set() and not service._slots.locked():
                break
            await asyncio.sleep(0.01)
        assert cleaned_up.is_set()
        assert not service._slots.locked()

    asyncio.run(scenario())


# --- Metrics -----------------------------------------------------------------------------

def test_run_agent_passes_its_recorder_to_the_graph(wiring):
    asyncio.run(run_agent(make_settings(), "q", "qwen2.5:3b", 4))
    assert isinstance(wiring.recorder, InvestigationRecorder)
    assert {"preflight", "mcp_startup"} <= set(wiring.recorder.phases)


def test_model_failure_error_carries_partial_metrics(wiring, monkeypatch):
    async def failing(question, model, client, tools, max_rounds, recorder):
        recorder.record_llm_attempt(stage="tool_loop", call=1, turn=1, attempt=1,
                                    outcome="error", latency_ms=12.0,
                                    error=httpx.ConnectError("refused at 10.0.0.1"))
        return {**RESULT, "model_error": "ConnectError"}

    monkeypatch.setattr(service_module, "investigate", failing)
    with pytest.raises(InvestigationError) as caught:
        asyncio.run(run_agent(make_settings(), "q", "qwen2.5:3b", 4))
    metrics = caught.value.metrics
    assert caught.value.code == "ollama_unavailable"
    assert "ConnectError" not in caught.value.message
    assert metrics["partial"] is True
    assert metrics["llm"]["error_attempts"] == 1
    assert metrics["llm_attempts"][0]["error_category"] == "connection"
    assert metrics["timings_ms"]["preflight"] is not None
    assert metrics["timings_ms"]["mcp_startup"] is not None
    serialized = json.dumps(metrics)
    assert "10.0.0.1" not in serialized and "ConnectError" not in serialized
    InvestigationMetrics.model_validate(metrics)


def test_failed_mcp_startup_is_still_timed(wiring):
    clock = iter([10.0, 10.0, 10.0, 10.25]).__next__  # preflight start/end, startup start/end
    recorder = InvestigationRecorder(clock=clock)
    wiring.session_error = OSError("cannot spawn")
    with pytest.raises(InvestigationError) as caught:
        asyncio.run(run_agent(make_settings(), "q", "qwen2.5:3b", 4, recorder))
    assert caught.value.code == "mcp_unavailable"
    assert recorder.phases["mcp_startup"] == 250.0


def test_cancelled_mcp_startup_is_still_timed(wiring, monkeypatch):
    entered = threading.Event()

    @asynccontextmanager
    async def hanging_session(settings):
        entered.set()
        await asyncio.Event().wait()
        yield

    monkeypatch.setattr(service_module, "mcp_session", hanging_session)
    recorder = InvestigationRecorder()

    async def scenario():
        task = asyncio.create_task(run_agent(make_settings(), "q", "qwen2.5:3b", 4, recorder))
        while not entered.is_set():
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert recorder.phases["mcp_startup"] >= 0


def test_service_adds_queue_wait_and_total_without_mutating_the_result():
    original = {**RESULT, "metrics": InvestigationRecorder().snapshot()}

    async def agent(question, model, max_rounds):
        return original

    result = asyncio.run(InvestigationService(make_settings(), agent).run("q", "m", 1))
    timings = result["metrics"]["timings_ms"]
    assert 0 <= timings["queue_wait"] <= timings["total"]
    assert original["metrics"]["timings_ms"]["total"] is None
    assert {k: v for k, v in result.items() if k != "metrics"} == RESULT


class _StallingModel:
    """One tool call, then a model call that never returns."""

    def __init__(self):
        self.replies = [AIMessage(content="", tool_calls=[
            {"name": "search_accounts", "args": {"query": "N"}, "id": "a"}])]

    def bind_tools(self, specs):
        return self

    async def ainvoke(self, messages):
        if self.replies:
            return self.replies.pop(0)
        await asyncio.Event().wait()


class _AccountsMCP:
    async def call_tool(self, name, args):
        return SimpleNamespace(isError=False, content=[],
                               structuredContent={"result": [{"id": 1, "name": "N"}]})


_TOOLS = [SimpleNamespace(name="search_accounts", description="",
                          inputSchema={"type": "object",
                                       "properties": {"query": {"type": "string"}}})]


def test_timeout_keeps_partial_metrics_from_the_default_runner(monkeypatch):
    # The service's own timeout scope is expired by the model once it stalls, so the test
    # does not depend on how fast the first round runs.
    long_timeout = 600.0
    scopes = []
    real_timeout = asyncio.timeout

    def capturing_timeout(delay):
        scope = real_timeout(delay)
        if delay is not None and delay > long_timeout / 2:
            scopes.append(scope)
        return scope

    class ExpiringModel(_StallingModel):
        async def ainvoke(self, messages):
            if self.replies:
                return self.replies.pop(0)
            scopes[0].reschedule(asyncio.get_running_loop().time())
            await asyncio.Event().wait()

    async def stalling_run_agent(settings, question, model, max_rounds, recorder):
        return await investigate(question, ExpiringModel(), _AccountsMCP(), _TOOLS,
                                 recorder=recorder)

    monkeypatch.setattr(asyncio, "timeout", capturing_timeout)
    monkeypatch.setattr(service_module, "run_agent", stalling_run_agent)

    async def scenario():
        service = InvestigationService(
            make_settings(investigation_timeout_seconds=long_timeout))
        with pytest.raises(InvestigationError) as caught:
            await service.run("Find", "m", 2)
        assert not service._slots.locked()
        return caught.value

    error = asyncio.run(scenario())
    assert len(scopes) == 1
    assert error.code == "timeout"
    assert "Try a narrower question" in error.message
    metrics = InvestigationMetrics.model_validate(error.metrics)
    assert metrics.partial is True
    assert [a.outcome for a in metrics.llm_attempts] == ["ok", "cancelled"]
    assert metrics.llm.cancelled_attempts == 1 and metrics.llm.error_attempts == 0
    assert metrics.tools.by_outcome == {"executed_success": 1}
    assert metrics.timings_ms.graph is not None
    assert metrics.timings_ms.total >= metrics.timings_ms.queue_wait >= 0


class _ScriptedModel:
    def __init__(self, replies):
        self.replies = list(replies)

    def bind_tools(self, specs):
        return self

    async def ainvoke(self, messages):
        return self.replies.pop(0)


def _searches(count):
    return [AIMessage(content="", tool_calls=[
        {"name": "search_accounts", "args": {"query": f"q{n}"}, "id": f"c{n}"}])
        for n in range(count)]


def test_concurrent_investigations_keep_separate_metrics(monkeypatch):
    together = threading.Barrier(2, timeout=5)

    async def run_agent_stub(settings, question, model, max_rounds, recorder):
        await asyncio.to_thread(together.wait)  # both investigations are in flight at once
        searches = int(question)
        replies = [*_searches(searches), AIMessage(content="Done."), AIMessage(content="Done.")]
        return await investigate("Find", _ScriptedModel(replies), _AccountsMCP(), _TOOLS,
                                 max_rounds=4, recorder=recorder)

    monkeypatch.setattr(service_module, "run_agent", run_agent_stub)

    async def scenario():
        service = InvestigationService(make_settings(investigation_max_concurrency=2))
        return await asyncio.gather(service.run("1", "m", 4), service.run("3", "m", 4))

    one, three = asyncio.run(scenario())
    for result, searches in ((one, 1), (three, 3)):
        metrics = InvestigationMetrics.model_validate(result["metrics"])
        assert metrics.tools.executed == searches == len(result["tool_trace"])
        assert metrics.llm.attempts == searches + 2
        assert metrics.timings_ms.queue_wait is not None


def test_queue_wait_measures_time_spent_waiting_for_a_slot(monkeypatch):
    holding, release = threading.Event(), threading.Event()

    async def run_agent_stub(settings, question, model, max_rounds, recorder):
        if question == "first":
            holding.set()
            await asyncio.to_thread(release.wait, 5)
        return {**RESULT, "metrics": recorder.snapshot()}

    monkeypatch.setattr(service_module, "run_agent", run_agent_stub)

    async def scenario():
        service = InvestigationService(make_settings(investigation_max_concurrency=1))
        first = asyncio.create_task(service.run("first", "m", 1))
        assert await asyncio.to_thread(holding.wait, 5)
        second = asyncio.create_task(service.run("second", "m", 1))
        await asyncio.sleep(0)  # the second request starts and blocks on the slot
        assert service._slots.locked() and not second.done()
        release.set()
        return await first, await second

    first, second = asyncio.run(scenario())
    first_wait = first["metrics"]["timings_ms"]["queue_wait"]
    second_timings = second["metrics"]["timings_ms"]
    assert second_timings["queue_wait"] > first_wait >= 0
    assert second_timings["total"] >= second_timings["queue_wait"]


def test_timeout_of_a_custom_agent_has_no_metrics():
    async def hanging(question, model, max_rounds):
        await asyncio.Event().wait()

    service = InvestigationService(make_settings(investigation_timeout_seconds=0.1), hanging)
    with pytest.raises(InvestigationError) as caught:
        asyncio.run(service.run("q", "m", 1))
    assert caught.value.code == "timeout"
    assert caught.value.metrics is None


def test_cancelling_the_default_runner_frees_the_slot(monkeypatch):
    started, cleaned_up = threading.Event(), threading.Event()

    async def hanging_run_agent(settings, question, model, max_rounds, recorder):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned_up.set()

    monkeypatch.setattr(service_module, "run_agent", hanging_run_agent)

    async def scenario():
        service = InvestigationService(make_settings())
        request = asyncio.create_task(service.run("q", "m", 1))
        assert await asyncio.to_thread(started.wait, 5)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert await asyncio.to_thread(cleaned_up.wait, 5)
        # Resolves only once the worker thread has finished and released its slot.
        await asyncio.wait_for(service._slots.acquire(), 5)
        service._slots.release()

    asyncio.run(scenario())
