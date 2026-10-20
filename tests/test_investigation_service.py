"""Service tests: timeouts, cancellation, concurrency and error mapping without Ollama or MCP."""

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest

from enterprise_ai_workflow_agent.agent import service as service_module
from enterprise_ai_workflow_agent.agent.service import (
    InvestigationError,
    InvestigationService,
    check_ollama,
    run_agent,
)
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

    async def fake_investigate(question, model, client, tools, max_rounds):
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
