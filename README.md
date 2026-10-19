# Enterprise AI Workflow Agent

A portfolio project for evidence-backed, approval-gated enterprise AI workflows.

> Status: Milestone 1 — FastAPI foundation. LLM orchestration, MCP tools, business data, and approvals are planned but not implemented yet.

## Local development

Requires Python 3.11+.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
uvicorn enterprise_ai_workflow_agent.main:app --reload
```

Open http://127.0.0.1:8000/docs or visit http://127.0.0.1:8000/api/v1/health.

## Quality checks

```powershell
ruff check .
pytest -q
```

## Docker

```powershell
docker build -t enterprise-ai-workflow-agent .
docker run --rm -p 8000:8000 enterprise-ai-workflow-agent
```

## Roadmap

- [x] API foundation, tests, linting, Docker, CI
- [ ] Synthetic enterprise data and PostgreSQL
- [ ] MCP tools and document retrieval
- [ ] LLM orchestration and workflow state
- [ ] Human approval and audit trail
- [ ] Frontend and agent evaluation

## Security

Use only synthetic/public data. Never commit `.env` or API credentials. No autonomous write actions are implemented in this milestone.
