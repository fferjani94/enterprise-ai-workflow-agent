import logging
from collections.abc import Callable
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import Field

from enterprise_ai_workflow_agent.db.session import SessionLocal
from enterprise_ai_workflow_agent.mcp_server import services

logger = logging.getLogger(__name__)

mcp = FastMCP("OpsPilot Enterprise Data")

# Published in the tool schema (minimum/maximum) so models can see the allowed range.
# services._limit still enforces it as defense in depth.
Limit = Annotated[
    int,
    Field(ge=1, le=services.MAX_RESULTS,
          description=f"Maximum number of results, 1-{services.MAX_RESULTS}."),
]
TicketLimit = Annotated[
    int,
    Field(ge=1, le=services.TICKET_MAX_RESULTS,
          description=f"Page size, 1-{services.TICKET_MAX_RESULTS}."),
]
Offset = Annotated[int, Field(ge=0, description="Number of matching tickets to skip.")]
TicketStatus = Literal["open", "resolved", "closed", "unresolved"]
AccountId = Annotated[
    int, Field(description="Account ID from a successful search_accounts result.")
]


def _run(operation: Callable[..., Any], *args: Any) -> Any:
    """Run a read-only lookup; expose validation messages, never internal exceptions."""
    try:
        with SessionLocal() as db:
            return operation(db, *args)
    except ValueError as exc:
        # Raised by services for invalid arguments; the messages are written for callers.
        raise ToolError(str(exc)) from None
    except Exception:
        logger.exception("Tool %s failed", operation.__name__)
        raise ToolError(
            "Internal error while reading data; the lookup was not completed."
        ) from None


@mcp.tool()
def search_accounts(query: str, limit: Limit = 10) -> list[dict]:
    """Search fictional SaaS customer accounts by name (read-only).

    Matches a case-insensitive substring of the name; use one account name per query."""
    return _run(services.search_accounts, query, limit)


@mcp.tool()
def get_account_tickets(
    account_id: AccountId,
    status: Annotated[
        TicketStatus | list[TicketStatus] | None,
        Field(description="Only tickets with this status (or any of these); omit for all."),
    ] = None,
    limit: TicketLimit = 10,
    offset: Offset = 0,
) -> dict[str, Any]:
    """List one page of support tickets for a customer account, ordered by ticket ID
    (not by date), with the total number of matching tickets (read-only)."""
    return _run(services.get_account_tickets, account_id, status, limit, offset)


@mcp.tool()
def get_ticket_details(ticket_id: int) -> dict:
    """Get the description, status and priority of a support ticket."""
    return _run(services.get_ticket_details, ticket_id)


@mcp.tool()
def get_product_information(product_id: int) -> dict:
    """Get the version and description of a SaaS product."""
    return _run(services.get_product_information, product_id)


@mcp.tool()
def search_knowledge(
    query: str, product_id: int | None = None, limit: Limit = 5
) -> list[dict]:
    """Search internal product documents using deterministic lexical ranking."""
    return _run(services.search_knowledge, query, product_id, limit)


def main() -> None:
    # stdio MCP transport: stdout is reserved for protocol messages.
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
