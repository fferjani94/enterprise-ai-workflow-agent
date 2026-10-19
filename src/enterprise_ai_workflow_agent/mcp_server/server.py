from mcp.server.fastmcp import FastMCP

from enterprise_ai_workflow_agent.db.session import SessionLocal
from enterprise_ai_workflow_agent.mcp_server import services

mcp = FastMCP("OpsPilot Enterprise Data")


@mcp.tool()
def search_accounts(query: str, limit: int = 10) -> list[dict]:
    """Search fictional SaaS customer accounts by name (read-only)."""
    with SessionLocal() as db:
        return services.search_accounts(db, query, limit)


@mcp.tool()
def get_account_tickets(account_id: int, limit: int = 10) -> list[dict]:
    """List support tickets for a customer account (read-only)."""
    with SessionLocal() as db:
        return services.get_account_tickets(db, account_id, limit)


@mcp.tool()
def get_ticket_details(ticket_id: int) -> dict:
    """Get the description, status and priority of a support ticket."""
    with SessionLocal() as db:
        return services.get_ticket_details(db, ticket_id)


@mcp.tool()
def get_product_information(product_id: int) -> dict:
    """Get the version and description of a SaaS product."""
    with SessionLocal() as db:
        return services.get_product_information(db, product_id)


@mcp.tool()
def search_knowledge(
    query: str, product_id: int | None = None, limit: int = 5
) -> list[dict]:
    """Search internal product documents using deterministic lexical ranking."""
    with SessionLocal() as db:
        return services.search_knowledge(db, query, product_id, limit)


def main() -> None:
    # stdio MCP transport: stdout is reserved for protocol messages.
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
