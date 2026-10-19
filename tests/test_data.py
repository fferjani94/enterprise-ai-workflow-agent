from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from enterprise_ai_workflow_agent.db.models import Account, Product, Ticket
from enterprise_ai_workflow_agent.db.session import build_engine, get_db
from enterprise_ai_workflow_agent.main import create_app


@pytest.fixture
def client(tmp_path: Path):
    url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    engine = build_engine(url)
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", url)
    # Migration env.py reads Settings, so use the environment override.
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"DATABASE_URL": url}):
        # env.py imports settings already instantiated; override the value temporarily.
        from enterprise_ai_workflow_agent.core.config import settings

        old_url = settings.database_url
        settings.database_url = url
        try:
            command.upgrade(config, "head")
        finally:
            settings.database_url = old_url

    maker = sessionmaker(bind=engine, expire_on_commit=False)
    with maker.begin() as db:
        account = Account(name="Test Account", industry="Tech", plan="Pro", status="active")
        product = Product(name="Test Product", version="1.0", description="Demo")
        db.add_all([account, product])
        db.flush()
        db.add(Ticket(account_id=account.id, product_id=product.id,
                      subject="Cannot sign in", description="Login error",
                      status="open", priority="high"))

    app = create_app()

    def override_db():
        with maker() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    engine.dispose()


def test_accounts(client):
    response = client.get("/api/v1/accounts")
    assert response.status_code == 200
    assert response.json()[0]["name"] == "Test Account"
    assert client.get("/api/v1/accounts/999").status_code == 404


def test_tickets_filter_and_validation(client):
    response = client.get("/api/v1/tickets", params={"account_id": 1})
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert client.get("/api/v1/tickets", params={"account_id": 999}).json() == []
    assert client.get("/api/v1/tickets", params={"limit": 101}).status_code == 422
    assert client.get("/api/v1/tickets/999").status_code == 404


def test_contacts_and_knowledge(client):
    assert client.get("/api/v1/accounts/1/contacts").json() == []
    assert client.get("/api/v1/knowledge").json() == []
    assert client.get("/api/v1/products").json()[0]["name"] == "Test Product"
