from __future__ import annotations

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from enterprise_ai_workflow_agent.db.base import Base


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    industry: Mapped[str] = mapped_column(String(80))
    plan: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(30), default="active")

    contacts: Mapped[list[Contact]] = relationship(back_populates="account")
    tickets: Mapped[list[Ticket]] = relationship(back_populates="account")


class Contact(Base):
    __tablename__ = "contacts"
    __table_args__ = (UniqueConstraint("email", name="uq_contacts_email"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(80))

    account: Mapped[Account] = relationship(back_populates="contacts")


class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    version: Mapped[str] = mapped_column(String(40))
    description: Mapped[str] = mapped_column(Text)

    tickets: Mapped[list[Ticket]] = relationship(back_populates="product")
    knowledge_documents: Mapped[list[KnowledgeDocument]] = relationship(back_populates="product")


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    subject: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30))
    priority: Mapped[str] = mapped_column(String(30))

    account: Mapped[Account] = relationship(back_populates="tickets")
    product: Mapped[Product] = relationship(back_populates="tickets")


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    product: Mapped[Product] = relationship(back_populates="knowledge_documents")
