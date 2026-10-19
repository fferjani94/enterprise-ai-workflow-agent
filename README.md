# Enterprise AI Workflow Agent (OpsPilot)

A portfolio project demonstrating enterprise AI workflows over fictional SaaS data.

## Milestone 3: Read-only MCP tools and knowledge search

Stack: FastAPI, SQLAlchemy 2.0, Alembic, SQLite (PostgreSQL-ready), official Python MCP SDK.

### Windows PowerShell setup

From the repository root, with an existing `.venv`:

```powershell
.venv\Scripts\python.exe -m pip install --no-cache-dir -e ".[dev]"
.venv\Scripts\python.exe -m alembic upgrade head
.venv\Scripts\python.exe -m enterprise_ai_workflow_agent.db.seed
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m alembic check
.venv\Scripts\python.exe scripts\smoke_mcp.py
```

Run FastAPI separately:

```powershell
.venv\Scripts\python.exe -m uvicorn enterprise_ai_workflow_agent.main:app --reload
```

Swagger: http://127.0.0.1:8000/docs

### MCP server

Run as a stdio MCP server:

```powershell
.venv\Scripts\python.exe -m enterprise_ai_workflow_agent.mcp_server.server
```

This command waits for JSON-RPC messages from an MCP client; it is **not** an HTTP server.
Do not type messages into it or expect a URL. Use `scripts/smoke_mcp.py` for an end-to-end
client/server smoke test.

Registered tools:
- `search_accounts(query, limit=10)`
- `get_account_tickets(account_id, limit=10)`
- `get_ticket_details(ticket_id)`
- `get_product_information(product_id)`
- `search_knowledge(query, product_id=None, limit=5)`

### Knowledge search limitations

Knowledge search is deterministic, case-insensitive token-overlap ranking with title
boosting. It is **not** semantic retrieval, embeddings, pgvector, or RAG. Documents
are returned with snippets and source metadata. Later milestones can add vector search.

### Safety boundaries

All tools are read-only. Tool parameters use SQLAlchemy bound expressions, not raw SQL.
There is no authentication, authorization, tenant isolation, or approval/audit layer yet.
**Use synthetic data only; do not deploy publicly.** Tool responses can contain untrusted
document content, so a future LLM agent must not treat retrieved text as instructions.

### CI

The existing `.github/workflows/ci.yml` runs Ruff, pytest (including MCP registration and
service tests), Alembic migration checks, and Docker image build on every push/PR.

### Database schema

Milestone 3 does not change the database schema; Alembic stays at revision `0001`.
