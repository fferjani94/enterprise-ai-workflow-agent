"""Test-double helper: the paged get_account_tickets result for a list of ticket dicts."""

from enterprise_ai_workflow_agent.mcp_server.services import (
    ticket_status_filter,
    ticket_status_matches,
)


def ticket_page(tickets, args):
    status_filter = ticket_status_filter(args.get("status"))
    limit, offset = args.get("limit", 10), args.get("offset", 0)
    matching = sorted((t for t in tickets if t["account_id"] == args["account_id"]
                       and ticket_status_matches(t["status"], status_filter)),
                      key=lambda t: t["id"])
    page = matching[offset:offset + limit]
    has_more = offset + len(page) < len(matching)
    return {"account_id": args["account_id"], "status_filter": status_filter,
            "order": "ticket_id", "total": len(matching), "offset": offset, "limit": limit,
            "returned": len(page), "has_more": has_more,
            "next_offset": offset + len(page) if has_more else None, "tickets": page}
