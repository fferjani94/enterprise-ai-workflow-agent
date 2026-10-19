"""Initial enterprise data schema.

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "accounts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("industry", sa.String(80), nullable=False),
        sa.Column("plan", sa.String(40), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
    )
    op.create_index("ix_accounts_name", "accounts", ["name"], unique=True)
    op.create_table(
        "products",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False, unique=True),
        sa.Column("version", sa.String(40), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
    )
    op.create_table(
        "contacts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("account_id", sa.Integer(), sa.ForeignKey("accounts.id"), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("role", sa.String(80), nullable=False),
        sa.UniqueConstraint("email", name="uq_contacts_email"),
    )
    op.create_index("ix_contacts_account_id", "contacts", ["account_id"])
    op.create_table(
        "tickets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("account_id", sa.Integer(), sa.ForeignKey("accounts.id"), nullable=False),
        sa.Column("product_id", sa.Integer(), sa.ForeignKey("products.id"), nullable=False),
        sa.Column("subject", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("priority", sa.String(30), nullable=False),
    )
    op.create_index("ix_tickets_account_id", "tickets", ["account_id"])
    op.create_index("ix_tickets_product_id", "tickets", ["product_id"])
    op.create_table(
        "knowledge_documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("product_id", sa.Integer(), sa.ForeignKey("products.id"), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source_url", sa.String(500), nullable=True),
    )
    op.create_index("ix_knowledge_documents_product_id", "knowledge_documents", ["product_id"])


def downgrade() -> None:
    op.drop_table("knowledge_documents")
    op.drop_table("tickets")
    op.drop_table("contacts")
    op.drop_table("products")
    op.drop_table("accounts")
