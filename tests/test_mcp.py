import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from enterprise_ai_workflow_agent.db.base import Base
from enterprise_ai_workflow_agent.db.models import (
    Account,
    KnowledgeDocument,
    Product,
    Ticket,
)
from enterprise_ai_workflow_agent.mcp_server import services
from enterprise_ai_workflow_agent.mcp_server.server import mcp


@pytest.fixture
def db():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        account = Account(name="Northstar Labs", industry="Software",
                          plan="Enterprise", status="active")
        product = Product(name="CloudDesk", version="2.4",
                          description="Support workspace")
        session.add_all([account, product])
        session.flush()
        session.add_all([
            Ticket(account_id=account.id, product_id=product.id,
                   subject="SSO login fails", description="Token expired",
                   status="open", priority="high"),
            KnowledgeDocument(product_id=product.id, title="SSO troubleshooting",
                              content="Check identity provider and token expiry.",
                              source_url="https://docs.example.invalid/sso"),
        ])
        session.commit()
        yield session
    engine.dispose()


def test_account_search(db):
    assert services.search_accounts(db, "north")[0]["name"] == "Northstar Labs"
    assert services.search_accounts(db, "missing") == []
    with pytest.raises(ValueError):
        services.search_accounts(db, " ")


def test_ticket_tools(db):
    tickets = services.get_account_tickets(db, 1)
    assert len(tickets) == 1
    assert services.get_ticket_details(db, tickets[0]["id"])["priority"] == "high"
    assert services.get_ticket_details(db, 999)["found"] is False
    assert services.get_product_information(db, 1)["version"] == "2.4"


def test_knowledge_search(db):
    results = services.search_knowledge(db, "SSO token expiry")
    assert len(results) == 1
    assert results[0]["title"] == "SSO troubleshooting"
    assert results[0]["score"] > 0
    assert services.search_knowledge(db, "webhook") == []
    assert services.search_knowledge(db, "SSO", product_id=999) == []


def test_limits(db):
    with pytest.raises(ValueError):
        services.get_account_tickets(db, 1, limit=21)
    with pytest.raises(ValueError):
        services.search_knowledge(db, "SSO", limit=0)


def test_mcp_tools_registered():
    names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert names == {
        "search_accounts", "get_account_tickets", "get_ticket_details",
        "get_product_information", "search_knowledge",
    }
