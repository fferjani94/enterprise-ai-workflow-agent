# Enterprise AI Workflow Agent (OpsPilot)

A portfolio project demonstrating reliable, approval-gated enterprise AI workflows.

## Milestone 2: Data layer

Python 3.11+, FastAPI, SQLAlchemy 2.0, Alembic, SQLite locally; PostgreSQL-compatible models.
The database contains fictional B2B SaaS customers and documentation. No proprietary data.

### Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
alembic upgrade head
python -m enterprise_ai_workflow_agent.db.seed
python -m uvicorn enterprise_ai_workflow_agent.main:app --reload
```

Swagger: http://127.0.0.1:8000/docs

### API endpoints

- `GET /api/v1/health`
- `GET /api/v1/accounts`
- `GET /api/v1/accounts/{account_id}`
- `GET /api/v1/accounts/{account_id}/contacts`
- `GET /api/v1/products`
- `GET /api/v1/tickets?account_id=1`
- `GET /api/v1/tickets/{ticket_id}`
- `GET /api/v1/knowledge?product_id=1`

### Tests

```powershell
pytest -q
ruff check .
alembic check
```

### Database migration

Use `alembic upgrade head` after pulling schema changes. To create a new migration:

```powershell
alembic revision --autogenerate -m "describe change"
```

### Architecture notes

- Routes use injected SQLAlchemy sessions; no raw SQL generated from user input.
- Migration scripts, not `create_all`, own the database schema.
- Read-only API; write operations and approval/audit controls will come later.
- The synthetic seed script skips insertion when accounts already exist.
- Authentication, authorization and tenant isolation are not yet implemented; do not expose this API publicly.
