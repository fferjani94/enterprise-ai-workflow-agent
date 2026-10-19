from sqlalchemy import select

from enterprise_ai_workflow_agent.db.models import (
    Account,
    Contact,
    KnowledgeDocument,
    Product,
    Ticket,
)
from enterprise_ai_workflow_agent.db.session import SessionLocal


def seed() -> None:
    with SessionLocal.begin() as db:
        if db.scalar(select(Account.id).limit(1)) is not None:
            print("Database already contains accounts; skipping seed.")
            return

        account = Account(
            name="Northstar Labs", industry="Software", plan="Enterprise", status="active"
        )
        second = Account(
            name="Bluebird Commerce", industry="Retail", plan="Business", status="active"
        )
        product = Product(
            name="CloudDesk", version="2.4", description="Customer support workspace",
        )
        other = Product(name="FlowSync", version="1.8", description="Workflow integrations")
        db.add_all([account, second, product, other])
        db.flush()
        db.add_all([
            Contact(
                account_id=account.id, name="Avery Stone",
                email="avery@northstar.example", role="Admin",
            ),
            Contact(
                account_id=second.id, name="Morgan Lee",
                email="morgan@bluebird.example", role="Support Lead",
            ),
            Ticket(
                account_id=account.id, product_id=product.id,
                   subject="Intermittent login failures",
                description="SSO login fails for some users.",
                   status="open", priority="high"),
            Ticket(
                account_id=account.id, product_id=product.id,
                   subject="Dashboard loading slowly",
                description="Dashboard sometimes takes 20 seconds.",
                   status="resolved", priority="medium"),
            Ticket(
                account_id=second.id, product_id=other.id,
                   subject="Webhook retry behavior",
                description="Retries produce duplicate events.",
                   status="open", priority="medium"),
            KnowledgeDocument(
                product_id=product.id, title="SSO troubleshooting",
                              content="Check identity provider settings and token expiry.",
                              source_url="https://docs.example.invalid/clouddesk/sso"),
            KnowledgeDocument(
                product_id=other.id, title="Webhook delivery",
                              content="Webhook deliveries use exponential backoff on failures.",
                              source_url="https://docs.example.invalid/flowsync/webhooks"),
        ])
    print("Seeded synthetic accounts, contacts, products, tickets, and documents.")


if __name__ == "__main__":
    seed()
