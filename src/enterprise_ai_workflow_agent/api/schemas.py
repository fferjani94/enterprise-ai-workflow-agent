from pydantic import BaseModel, ConfigDict


class ORMResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class AccountOut(ORMResponse):
    id: int
    name: str
    industry: str
    plan: str
    status: str


class ContactOut(ORMResponse):
    id: int
    account_id: int
    name: str
    email: str
    role: str


class ProductOut(ORMResponse):
    id: int
    name: str
    version: str
    description: str


class TicketOut(ORMResponse):
    id: int
    account_id: int
    product_id: int
    subject: str
    description: str
    status: str
    priority: str


class KnowledgeDocumentOut(ORMResponse):
    id: int
    product_id: int
    title: str
    content: str
    source_url: str | None
