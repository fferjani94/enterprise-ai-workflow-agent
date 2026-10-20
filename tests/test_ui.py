"""Static UI delivery and the contract between the frontend and the backend."""

import re
import shutil
import subprocess
import typing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from enterprise_ai_workflow_agent.api.investigation_schemas import (
    EvidenceRecord,
    InvestigationAnalysis,
    InvestigationResponse,
    ToolTraceEntry,
)
from enterprise_ai_workflow_agent.api.routes.investigations import ERROR_STATUS
from enterprise_ai_workflow_agent.core.config import MAX_QUESTION_LENGTH
from enterprise_ai_workflow_agent.main import app
from enterprise_ai_workflow_agent.web import WEB_DIR

HTML = (WEB_DIR / "index.html").read_text(encoding="utf-8")
JS = (WEB_DIR / "static" / "app.js").read_text(encoding="utf-8")
SEED = (Path(__file__).parents[1] / "src" / "enterprise_ai_workflow_agent" / "db"
        / "seed.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_index_is_served_with_security_headers(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "OpsPilot" in response.text
    assert "/static/app.css" in response.text and "/static/app.js" in response.text
    csp = response.headers["content-security-policy"]
    assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("path, content_type", [
    ("/static/app.css", "text/css"),
    ("/static/app.js", "javascript"),
    ("/static/favicon.svg", "image/svg+xml"),
])
def test_static_assets_load(client, path, content_type):
    response = client.get(path)
    assert response.status_code == 200
    assert content_type in response.headers["content-type"]
    assert response.content


def test_missing_static_asset_is_404(client):
    assert client.get("/static/missing.js").status_code == 404
    assert client.get("/static/../main.py").status_code in (400, 404)


def test_api_and_docs_are_unaffected(client):
    assert client.get("/api/v1/health").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/api/v1/investigations" in schema["paths"]
    assert "/" not in schema["paths"]


def test_ui_has_no_external_resources():
    assert not re.search(r"""(?:src|href)=["'](?:https?:)?//""", HTML)
    assert "@import" not in (WEB_DIR / "static" / "app.css").read_text(encoding="utf-8")
    assert not re.search(r"https?://", JS)
    assert "<style" not in HTML and not re.search(r"\sstyle=", HTML)


def test_javascript_never_parses_model_output_as_markup():
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write",
                      "eval(", "new Function"):
        assert forbidden not in JS


def test_every_element_id_used_by_javascript_exists_in_html():
    used = set(re.findall(r"\$\('([\w-]+)'\)", JS))
    used |= set(re.findall(r"getElementById\('([\w-]+)'\)", JS))
    defined = set(re.findall(r'\sid="([\w-]+)"', HTML))
    assert used
    assert used <= defined, used - defined


def test_javascript_uses_the_real_api_contract():
    assert "const API = '/api/v1/investigations'" in JS
    assert "`${API}/config`" in JS
    response_fields = set(InvestigationResponse.model_fields)
    read_by_js = {"id", "question", "status", "answer", "tool_rounds", "tool_calls",
                  "duration_seconds", "tool_trace", "evidence", "narrative", "analysis"}
    assert read_by_js <= response_fields
    for field in read_by_js:
        assert re.search(rf"\b{field}\b", JS), field


def test_javascript_handles_every_backend_error_code():
    titles = set(re.findall(r"^\s+(\w+): '", JS.split("ERROR_TITLES = {")[1].split("};")[0], re.M))
    assert set(ERROR_STATUS) | {"invalid_request", "internal_error"} <= titles


def test_evidence_groups_match_backend_record_types():
    backend = set(typing.get_args(EvidenceRecord.model_fields["type"].annotation))
    groups = set(re.findall(r"\['(\w+)', '[\w ]+'\]", JS.split("EVIDENCE_GROUPS = [")[1]
                            .split("];")[0]))
    assert groups == backend


def test_question_limit_in_html_matches_backend():
    assert f'maxlength="{MAX_QUESTION_LENGTH}"' in HTML


def test_example_prompts_use_seeded_data_only():
    prompts = re.findall(r'data-prompt="([^"]+)"', HTML)
    assert len(prompts) >= 3
    seeded = {"Northstar Labs", "Bluebird Commerce"}
    assert all(name in SEED for name in seeded)
    for prompt in prompts:
        assert any(name in prompt for name in seeded), prompt


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_javascript_has_valid_syntax():
    result = subprocess.run(["node", "--check", str(WEB_DIR / "static" / "app.js")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_relevance_labels_match_backend_values():
    backend = set(typing.get_args(EvidenceRecord.model_fields["relevance"].annotation))
    labels = set(re.findall(r"^\s+(\w+): \['", JS.split("const RELEVANCE = {")[1]
                            .split("};")[0], re.M))
    assert labels == backend


def test_analysis_sections_match_backend_fields():
    sections = set(re.findall(r"\['(\w+)', '[^']+'\]", JS.split("ANALYSIS_SECTIONS = [")[1]
                              .split("];")[0]))
    assert sections == set(InvestigationAnalysis.model_fields)


def test_links_are_limited_to_web_urls_and_synthetic_hosts_are_not_links():
    assert "url.protocol !== 'https:' && url.protocol !== 'http:'" in JS
    assert "noopener noreferrer" in JS
    assert "Example URL, not a live link" in JS
    assert "invalid|example|test|localhost" in JS


def test_answer_is_rendered_from_the_narrative_without_the_appended_evidence_list():
    assert "data.narrative" in JS
    assert "Verified evidence:" not in JS


def test_trace_entries_without_arguments_are_not_rendered_as_null():
    # Trace entries for calls over the per-round limit have no arguments; the API sends null.
    limited = {"tool": "search_accounts", "success": False, "error": "Tool-call limit exceeded"}
    assert ToolTraceEntry(**limited).model_dump(mode="json")["arguments"] is None
    assert "entry.arguments == null" in JS
    assert "entry.arguments === undefined" not in JS
