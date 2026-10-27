"""API tests with an injected fake runner: no Ollama, MCP subprocess or database needed."""

import asyncio
import json
import re
import threading
import time
import tomllib
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from enterprise_ai_workflow_agent import __version__
from enterprise_ai_workflow_agent.agent.service import InvestigationError, InvestigationService
from enterprise_ai_workflow_agent.api.investigation_schemas import InvestigationRequest
from enterprise_ai_workflow_agent.api.routes import investigations as routes
from enterprise_ai_workflow_agent.api.routes.investigations import get_investigation_service
from enterprise_ai_workflow_agent.core.config import Settings, settings
from enterprise_ai_workflow_agent.main import create_app

ROOT = Path(__file__).resolve().parents[1]

TRACE = [
    {"tool": "search_accounts", "arguments": {"query": "Northstar"},
     "normalized_arguments": {"query": "Northstar"}, "success": True, "record_count": 1},
    {"tool": "get_account_tickets", "arguments": {"account_id": 999},
     "normalized_arguments": {"account_id": 999}, "success": False, "record_count": 0,
     "error": "Unverified record ID: account_id=999 has not been established"},
]
EVIDENCE = [
    {"type": "account", "id": 1, "title": "Northstar Labs", "description": None,
     "fields": {"industry": "Software", "plan": "Enterprise", "status": "active"},
     "relevance": "context"},
    {"type": "document", "id": 3, "title": "SSO troubleshooting",
     "description": "Check token expiry", "fields": {"product_id": 7, "score": 4},
     "relevance": "relevant"},
]
ANALYSIS = {
    "findings": [],
    "guidance": [{"text": "General guidance.", "evidence": [{"type": "document", "id": 3}]}],
    "unsupported": [],
    "unconfirmed": [{"text": "No retrieved record establishes a root cause.", "evidence": []}],
}


class FakeService:
    def __init__(self, result=None, error=None):
        self.result = result if result is not None else {
            "answer": "Northstar Labs is account 1.\n\nVerified evidence:\nAccount 1",
            "narrative": "Northstar Labs is account 1.",
            "tool_rounds": 2,
            "tool_trace": TRACE,
            "evidence": EVIDENCE,
            "analysis": ANALYSIS,
            "limited": False,
        }
        self.error = error
        self.calls = []

    async def run(self, question, model, max_rounds):
        self.calls.append((question, model, max_rounds))
        if self.error:
            raise self.error
        return self.result


def make_client(service):
    app = create_app()
    app.dependency_overrides[get_investigation_service] = lambda: service
    return TestClient(app, raise_server_exceptions=False)


def post(service, body):
    with make_client(service) as client:
        return client.post("/api/v1/investigations", json=body)


def test_model_proposals_are_exposed_as_diagnostics():
    proposals = [{"turn": 1, "round": 2, "tool": "get_account_tickets",
                  "arguments": {"account_id": 999}, "outcome": "rejected",
                  "reason": "Unverified record ID", "id_verification": "unverified"}]
    service = FakeService()
    turn_log = [{"turn": 1, "response": "text", "tool_calls": 0, "invalid_tool_calls": 0,
                 "discarded": True}]
    service.result = {**service.result, "model_proposals": proposals, "model_turns": 1,
                      "model_turn_log": turn_log}
    body = post(service, {"question": "Investigate SSO for Northstar Labs."}).json()
    assert body["model_proposals"] == proposals
    assert (body["model_turns"], body["model_turn_log"]) == (1, turn_log)
    assert body["tool_calls"] == len(TRACE)
    legacy = post(FakeService(), {"question": "Hi there."}).json()
    assert (legacy["model_proposals"], legacy["model_turns"], legacy["model_turn_log"]) == (
        [], None, [])


def test_successful_investigation_serialization():
    service = FakeService()
    response = post(service, {"question": "Investigate SSO for Northstar Labs.",
                              "model": "qwen2.5:3b", "max_rounds": 3})
    assert response.status_code == 200
    body = response.json()
    assert str(uuid.UUID(body["id"])) == body["id"]
    assert body["question"] == "Investigate SSO for Northstar Labs."
    assert body["model"] == "qwen2.5:3b"
    assert body["status"] == "completed"
    assert body["answer"] == "Northstar Labs is account 1.\n\nVerified evidence:\nAccount 1"
    assert body["narrative"] == "Northstar Labs is account 1."
    assert body["tool_rounds"] == 2
    assert body["tool_calls"] == 2
    assert isinstance(body["duration_seconds"], float)
    assert service.calls == [("Investigate SSO for Northstar Labs.", "qwen2.5:3b", 3)]


def test_trace_and_evidence_are_preserved():
    body = post(FakeService(), {"question": "Investigate"}).json()
    defaults = {"error": None, "origin": "model", "reason": None, "adjustments": [],
                "executed": None}
    assert body["tool_trace"] == [{**defaults, **entry} for entry in TRACE]
    assert body["tool_trace"][1]["error"].startswith("Unverified record ID")
    assert body["evidence"] == [{"relevance_basis": None, **record} for record in EVIDENCE]
    assert body["analysis"] == {"context": [], **ANALYSIS}


def test_defaults_come_from_settings():
    service = FakeService()
    assert post(service, {"question": "  Investigate  "}).status_code == 200
    assert service.calls == [("Investigate", settings.ollama_model,
                              settings.investigation_max_rounds)]


def test_round_limit_alone_is_completed_with_round_limit_flag():
    service = FakeService({"answer": "Stopped at the tool-round limit.", "tool_rounds": 1,
                          "tool_trace": TRACE[:1], "evidence": EVIDENCE, "limited": False,
                          "round_limit_reached": True})
    response = post(service, {"question": "Investigate", "max_rounds": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["round_limit_reached"] is True
    assert body["limited_reason"] is None


def test_limited_status_and_reason_are_passed_through():
    reason = "Required evidence is missing. The tool-round limit (1) was reached."
    service = FakeService({"answer": "Stopped.", "tool_rounds": 1, "tool_trace": TRACE[:1],
                           "evidence": [], "limited": True, "limited_reason": reason,
                           "round_limit_reached": True})
    response = post(service, {"question": "Investigate", "max_rounds": 1})
    assert response.status_code == 200
    assert response.json()["status"] == "limited"
    assert response.json()["limited_reason"] == reason


def test_incomplete_evidence_reason_and_orchestrator_trace_are_serialized():
    trace = [*TRACE[:1], {
        "tool": "get_account_tickets", "origin": "orchestrator",
        "reason": "Tickets for verified account 1 had not been retrieved.",
        "arguments": {"account_id": 1}, "normalized_arguments": {"account_id": 1},
        "success": True, "record_count": 2,
    }, {
        "tool": "search_accounts", "arguments": {"query": "N", "limit": 1000},
        "normalized_arguments": {"query": "N", "limit": 20},
        "adjustments": ["limit=1000 is outside the allowed range 1..20; used 20"],
        "success": True, "record_count": 1,
    }]
    reason = "Required evidence is missing. Tickets for account 2 had not been retrieved."
    service = FakeService({"answer": "Partial.", "tool_rounds": 3, "tool_trace": trace,
                           "evidence": [], "limited": True, "limited_reason": reason})
    body = post(service, {"question": "Investigate"}).json()
    assert body["status"] == "limited"
    assert body["limited_reason"] == reason
    assert body["tool_trace"][1]["origin"] == "orchestrator"
    assert body["tool_trace"][1]["reason"].startswith("Tickets for verified account 1")
    assert body["tool_trace"][2]["adjustments"] == [
        "limit=1000 is outside the allowed range 1..20; used 20"]


@pytest.mark.parametrize("body, field", [
    ({"question": ""}, "question"),
    ({"question": "   \n\t "}, "question"),
    ({"question": "x" * 4001}, "question"),
    ({"question": "ok", "max_rounds": 0}, "max_rounds"),
    ({"question": "ok", "max_rounds": 9}, "max_rounds"),
    ({"question": "ok", "max_rounds": "many"}, "max_rounds"),
    ({"question": "ok", "model": "bad model; rm -rf"}, "model"),
    ({"question": "ok", "unexpected": 1}, "unexpected"),
    ({}, "question"),
])
def test_invalid_requests_return_structured_422(body, field):
    service = FakeService()
    response = post(service, body)
    assert response.status_code == 422
    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "invalid_request"
    assert field in {item["field"] for item in payload["error"]["details"]}
    assert service.calls == []


def test_malformed_json_is_reported_against_the_body():
    service = FakeService()
    with make_client(service) as client:
        response = client.post("/api/v1/investigations", content=b'{"question":',
                               headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert response.json()["error"]["details"] == [
        {"field": "body", "message": "JSON decode error"}]
    assert service.calls == []


def test_question_at_maximum_length_is_accepted():
    assert post(FakeService(), {"question": "x" * 4000}).status_code == 200


@pytest.mark.parametrize("code, status", [
    ("ollama_unavailable", 503),
    ("ollama_integration_missing", 503),
    ("mcp_unavailable", 503),
    ("model_inference_failed", 503),
    ("model_unavailable", 404),
    ("timeout", 504),
])
def test_service_errors_map_to_http_status(code, status):
    service = FakeService(error=InvestigationError(code, "Readable message."))
    response = post(service, {"question": "Investigate"})
    assert response.status_code == status
    body = response.json()
    assert body["status"] == "failed"
    assert body["question"] == "Investigate"
    assert body["error"] == {"code": code, "message": "Readable message.", "details": None}
    assert uuid.UUID(body["id"])


def test_unexpected_failure_hides_internals():
    secret = "postgresql://user:hunter2@db/prod"
    response = post(FakeService(error=RuntimeError(f"boom {secret}")), {"question": "x"})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "hunter2" not in response.text
    assert "Traceback" not in response.text


def test_config_endpoint_exposes_defaults_only():
    with make_client(FakeService()) as client:
        response = client.get("/api/v1/investigations/config")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == settings.ollama_model
    assert body["max_rounds_limit"] == 8
    assert "ollama_base_url" not in body
    assert "database_url" not in body


def test_existing_endpoints_are_unchanged():
    with make_client(FakeService()) as client:
        assert client.get("/api/v1/health").json() == {"status": "ok"}
        paths = client.get("/openapi.json").json()["paths"]
        # Non-investigation validation errors keep FastAPI's default shape.
        invalid = client.get("/api/v1/accounts", params={"limit": 0})
    for path in ("/api/v1/health", "/api/v1/accounts", "/api/v1/tickets",
                 "/api/v1/investigations", "/api/v1/investigations/config"):
        assert path in paths
    assert invalid.status_code == 422
    assert "detail" in invalid.json()


def test_app_version_matches_pyproject():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert "version" not in project  # single source: the package __version__
    assert project["dynamic"] == ["version"]
    assert create_app().version == __version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


# --- Client disconnects -----------------------------------------------------------------

class FakeRequest:
    """Reports a disconnect after `connected_checks` checks (never, if None)."""

    def __init__(self, connected_checks=None, error=None):
        self.remaining = connected_checks
        self.error = error
        self.checks = 0

    async def is_disconnected(self):
        self.checks += 1
        if self.error:
            raise self.error
        if self.remaining is None:
            return False
        self.remaining -= 1
        return self.remaining < 0


@pytest.fixture
def fast_poll(monkeypatch):
    monkeypatch.setattr(routes, "DISCONNECT_POLL_SECONDS", 0.01)


def test_disconnect_cancels_the_running_investigation(fast_poll):
    cleaned_up = asyncio.Event()

    async def hanging():
        try:
            await asyncio.sleep(30)
        finally:
            cleaned_up.set()

    async def scenario():
        started = time.monotonic()
        with pytest.raises(routes.ClientDisconnected):
            await routes.run_unless_disconnected(FakeRequest(connected_checks=2), hanging())
        assert cleaned_up.is_set()
        assert time.monotonic() - started < 5

    asyncio.run(scenario())


def test_connected_client_gets_the_result(fast_poll):
    async def work():
        await asyncio.sleep(0.05)
        return "done"

    request = FakeRequest()
    assert asyncio.run(routes.run_unless_disconnected(request, work())) == "done"
    assert request.checks >= 1


def test_failing_disconnect_check_does_not_abort_the_investigation(fast_poll):
    async def work():
        await asyncio.sleep(0.05)
        return "done"

    request = FakeRequest(error=RuntimeError("receive failed"))
    assert asyncio.run(routes.run_unless_disconnected(request, work())) == "done"


def test_server_cancellation_is_not_reported_as_a_disconnect(fast_poll):
    cleaned_up = asyncio.Event()

    async def hanging():
        try:
            await asyncio.sleep(30)
        finally:
            cleaned_up.set()

    async def scenario():
        handler = asyncio.create_task(routes.run_unless_disconnected(FakeRequest(), hanging()))
        await asyncio.sleep(0.05)
        handler.cancel()
        with pytest.raises(asyncio.CancelledError):
            await handler
        await asyncio.sleep(0)
        assert cleaned_up.is_set()

    asyncio.run(scenario())


def test_disconnect_frees_the_service_slot_for_the_next_investigation(fast_poll):
    """End to end through the endpoint and the real service, with a fake agent."""
    agent_cleaned_up = threading.Event()

    async def hanging_agent(question, model, max_rounds):
        try:
            await asyncio.sleep(30)
        finally:
            agent_cleaned_up.set()  # stands in for MCP session / subprocess teardown

    async def scenario():
        service = InvestigationService(
            Settings(_env_file=None, investigation_max_concurrency=1), hanging_agent)
        response = await routes.create_investigation(
            InvestigationRequest(question="Investigate"), service, FakeRequest(connected_checks=3))
        assert response.status_code == routes.CLIENT_CLOSED_REQUEST
        assert json.loads(response.body)["error"]["code"] == "cancelled"
        for _ in range(300):
            if agent_cleaned_up.is_set() and not service._slots.locked():
                break
            await asyncio.sleep(0.01)
        assert agent_cleaned_up.is_set()
        assert not service._slots.locked()

    asyncio.run(scenario())
