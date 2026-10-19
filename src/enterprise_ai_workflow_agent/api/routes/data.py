from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from enterprise_ai_workflow_agent.api.schemas import (
    AccountOut,
    ContactOut,
    KnowledgeDocumentOut,
    ProductOut,
    TicketOut,
)
from enterprise_ai_workflow_agent.db.models import (
    Account,
    Contact,
    KnowledgeDocument,
    Product,
    Ticket,
)
from enterprise_ai_workflow_agent.db.session import get_db

router = APIRouter(tags=["Enterprise data"])
Db = Annotated[Session, Depends(get_db)]
Limit = Annotated[int, Query(ge=1, le=100)]


@router.get("/accounts", response_model=list[AccountOut])
def list_accounts(db: Db, limit: Limit = 50):
    return db.scalars(select(Account).order_by(Account.id).limit(limit)).all()


@router.get("/accounts/{account_id}", response_model=AccountOut)
def get_account(account_id: int, db: Db):
    account = db.get(Account, account_id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")
    return account


@router.get("/accounts/{account_id}/contacts", response_model=list[ContactOut])
def list_contacts(account_id: int, db: Db):
    if db.get(Account, account_id) is None:
        raise HTTPException(status_code=404, detail="Account not found")
    return db.scalars(select(Contact).where(Contact.account_id == account_id)).all()


@router.get("/products", response_model=list[ProductOut])
def list_products(db: Db, limit: Limit = 50):
    return db.scalars(select(Product).order_by(Product.id).limit(limit)).all()


@router.get("/tickets", response_model=list[TicketOut])
def list_tickets(db: Db, account_id: int | None = None, limit: Limit = 50):
    statement = select(Ticket)
    if account_id is not None:
        statement = statement.where(Ticket.account_id == account_id)
    return db.scalars(statement.order_by(Ticket.id).limit(limit)).all()


@router.get("/tickets/{ticket_id}", response_model=TicketOut)
def get_ticket(ticket_id: int, db: Db):
    ticket = db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@router.get("/knowledge", response_model=list[KnowledgeDocumentOut])
def list_knowledge(db: Db, product_id: int | None = None, limit: Limit = 50):
    statement = select(KnowledgeDocument)
    if product_id is not None:
        statement = statement.where(KnowledgeDocument.product_id == product_id)
    return db.scalars(statement.order_by(KnowledgeDocument.id).limit(limit)).all()
