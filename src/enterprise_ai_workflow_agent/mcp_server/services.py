import re
from collections.abc import Iterable
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from enterprise_ai_workflow_agent.db.models import (
    Account,
    KnowledgeDocument,
    Product,
    Ticket,
)

MAX_RESULTS = 20
TICKET_MAX_RESULTS = 50
# Requested status -> stored statuses it selects. "unresolved" means not resolved/closed.
TICKET_STATUS_VALUES = {
    "open": ("open",), "resolved": ("closed", "resolved"),
    "closed": ("closed", "resolved"), "unresolved": ("unresolved",),
}
FINISHED_STATUSES = ("closed", "resolved")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {"a", "an", "and", "are", "for", "how", "in", "is", "of",
             "on", "or", "the", "to", "what", "with"}


def _limit(limit: int) -> int:
    if not 1 <= limit <= MAX_RESULTS:
        raise ValueError(f"limit must be between 1 and {MAX_RESULTS}")
    return limit


def _tokens(text: str) -> set[str]:
    return {token for token in TOKEN_PATTERN.findall(text.lower())
            if token not in STOPWORDS and len(token) > 1}


def search_accounts(db: Session, query: str, limit: int = 10) -> list[dict]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if len(query) > 200:
        raise ValueError("query must be at most 200 characters")
    rows = db.scalars(
        select(Account)
        .where(Account.name.ilike(f"%{query.strip()}%", escape="\\"))
        .order_by(Account.name)
        .limit(_limit(limit))
    ).all()
    return [{"id": a.id, "name": a.name, "industry": a.industry,
             "plan": a.plan, "status": a.status} for a in rows]


def ticket_status_filter(status: str | Iterable[str] | None) -> list[str]:
    """Canonical, sorted status filter for a requested status (or list); [] means all."""
    if status is None:
        return []
    requested = [status] if isinstance(status, str) else list(status)
    values: set[str] = set()
    for item in requested:
        key = str(item).strip().lower()
        if key not in TICKET_STATUS_VALUES:
            raise ValueError("status must be one of: " + ", ".join(TICKET_STATUS_VALUES))
        values.update(TICKET_STATUS_VALUES[key])
    return sorted(values)


def ticket_status_matches(status: Any, status_filter: Iterable[str]) -> bool:
    """Whether a stored ticket status is selected by a canonical status filter."""
    wanted = set(status_filter)
    if not wanted:
        return True
    value = str(status or "").strip().lower()
    return value in wanted or ("unresolved" in wanted and value not in FINISHED_STATUSES)


def get_account_tickets(
    db: Session, account_id: int, status: str | list[str] | None = None,
    limit: int = 10, offset: int = 0,
) -> dict[str, Any]:
    """One page of an account's tickets, ordered by ticket ID, with the SQL total."""
    if not 1 <= limit <= TICKET_MAX_RESULTS:
        raise ValueError(f"limit must be between 1 and {TICKET_MAX_RESULTS}")
    if offset < 0:
        raise ValueError("offset must be 0 or greater")
    status_filter = ticket_status_filter(status)
    predicates = [Ticket.account_id == account_id]
    if status_filter:
        stored = func.lower(func.trim(Ticket.status))
        selected = [v for v in status_filter if v != "unresolved"]
        clauses = [stored.in_(selected)] if selected else []
        if "unresolved" in status_filter:
            clauses.append(stored.not_in(FINISHED_STATUSES))
        predicates.append(or_(*clauses))
    total = db.scalar(select(func.count()).select_from(Ticket).where(*predicates)) or 0
    rows = db.scalars(
        select(Ticket).where(*predicates).order_by(Ticket.id).offset(offset).limit(limit)
    ).all()
    has_more = offset + len(rows) < total
    # Metadata precedes the records so it survives truncation of the serialized result.
    return {
        "account_id": account_id, "status_filter": status_filter, "order": "ticket_id",
        "total": total, "offset": offset, "limit": limit, "returned": len(rows),
        "has_more": has_more, "next_offset": offset + len(rows) if has_more else None,
        "tickets": [
            {"id": t.id, "account_id": t.account_id, "product_id": t.product_id,
             "subject": t.subject, "status": t.status, "priority": t.priority}
            for t in rows
        ],
    }


def get_ticket_details(db: Session, ticket_id: int) -> dict:
    ticket = db.get(Ticket, ticket_id)
    if ticket is None:
        return {"found": False, "ticket_id": ticket_id}
    return {"found": True, "id": ticket.id, "account_id": ticket.account_id,
            "product_id": ticket.product_id, "subject": ticket.subject,
            "description": ticket.description, "status": ticket.status,
            "priority": ticket.priority}


def get_product_information(db: Session, product_id: int) -> dict:
    product = db.get(Product, product_id)
    if product is None:
        return {"found": False, "product_id": product_id}
    return {"found": True, "id": product.id, "name": product.name,
            "version": product.version, "description": product.description}


def search_knowledge(
    db: Session, query: str, product_id: int | None = None, limit: int = 5
) -> list[dict]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if len(query) > 200:
        raise ValueError("query must be at most 200 characters")
    limit = _limit(limit)
    terms = _tokens(query)
    if not terms:
        return []
    statement = select(KnowledgeDocument).order_by(KnowledgeDocument.id)
    if product_id is not None:
        statement = statement.where(KnowledgeDocument.product_id == product_id)
    documents = db.scalars(statement).all()
    ranked = []
    for doc in documents:
        title_terms = _tokens(doc.title)
        body_terms = _tokens(doc.content)
        score = 3 * len(terms & title_terms) + len(terms & body_terms)
        if score:
            ranked.append((score, doc))
    ranked.sort(key=lambda item: (-item[0], item[1].id))
    return [
        {"id": doc.id, "product_id": doc.product_id, "title": doc.title,
         "snippet": doc.content[:350], "source_url": doc.source_url,
         "score": score}
        for score, doc in ranked[:limit]
    ]
