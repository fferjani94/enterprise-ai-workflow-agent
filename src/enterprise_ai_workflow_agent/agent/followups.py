"""Deterministic, intent-aware evidence planning. No LLM calls.

The model chooses which tools to call. When it stops, these rules check whether evidence
that the question explicitly asks for is still missing and, if so, which read-only lookups
would retrieve it. Lookups only use record IDs that earlier tool results returned.

Intent comes from explicit terms in the question:
- tickets:   support-issue terms (ticket, incident, issue, error, failure, ...). Tickets are
             retrieved for the verified account(s) that the question names.
- knowledge: guidance terms (troubleshoot, documentation, how to, next steps, ...). A
             knowledge search is run with the question's topic terms.
- account:   account-attribute terms (account, plan, industry, subscription, tier).
An account-only question ("What plan is X on?") never triggers a ticket lookup.

Account discovery: the requested account names are the single name read from the question
(a run of capitalised words, see account_references) or, when there are several, the names
listed together ("X and Y", "X, Y"). Each requested name is tracked independently, from
the trace and the verified evidence (see account_status):
- resolved:   one verified account contains the name (or equals it)
- ambiguous:  several verified accounts contain it; none is chosen
- not found:  a successful search whose query covers the name matched no account with it
- failed:     the orchestrator's own search for the name failed; it is not retried
- unsearched: no successful covering search yet; search_accounts is planned with the name
Account IDs still come only from MCP results. Several unlisted capitalised phrases are never
guessed between.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

from enterprise_ai_workflow_agent.agent.grounding import (
    COMPLETE,
    PARTIAL,
    UNRETRIEVED,
    TicketCoverage,
    requested_ticket_statuses,
    status_label,
    topic_terms,
)
from enterprise_ai_workflow_agent.agent.grounding import FAILED as RETRIEVAL_FAILED
from enterprise_ai_workflow_agent.mcp_server.services import (
    TICKET_MAX_RESULTS,
    ticket_status_filter,
    ticket_status_matches,
)

MAX_FOLLOWUP_ACCOUNTS = 3
MAX_KNOWLEDGE_QUERY_TERMS = 8
# Ticket pages the orchestrator requests per account and status filter.
MAX_TICKET_PAGES = 2
TICKET_PAGE_SIZE = TICKET_MAX_RESULTS
# Recency needs a creation date, which tickets do not have; ticket IDs are not one.
RECENCY_TERMS = re.compile(
    r"\b(?:most\s+recent|recent|latest|newest|oldest)\s+(?:[a-z-]+\s+){0,2}?"
    r"(?:tickets?|incidents?|cases?)\b"
    r"|\btickets?\b.{0,40}\b(?:chronolog\w*|by date|by creation|sorted by (?:date|time))",
    re.IGNORECASE,
)

TICKET_TERMS = re.compile(
    r"\b(?:tickets?|incidents?|issues?|problems?|support|escalat\w*|outages?|complain\w*"
    r"|errors?|fail\w*|bugs?|broken|crash\w*|unable|cannot|can[’']t|not working"
    r"|timeouts?|downtime)\b",
    re.IGNORECASE,
)
# A subset of TICKET_TERMS that asks for ticket records themselves, not just symptoms.
# Tickets can only be listed per account, so such a question needs an account lookup.
EXPLICIT_TICKET_TERMS = re.compile(
    r"\b(?:tickets?|incidents?|escalations?|support (?:cases?|requests?))\b", re.IGNORECASE
)
KNOWLEDGE_TERMS = re.compile(
    r"\b(?:troubleshoot\w*|fix|fixes|workarounds?|guidance|guides?|documentation"
    r"|documents?|docs?|knowledge|articles?|runbooks?|how (?:to|do|can|should)"
    r"|next steps?|steps to|recommend\w*|suggest\w*|best practices?|remediat\w*"
    r"|mitigat\w*)\b",
    re.IGNORECASE,
)
# Words that ask for guidance but do not describe the topic, so they are left out of the
# knowledge query.
_QUERY_NOISE = frozenset({"next", "step", "steps"})
_TOKEN = re.compile(r"[a-z0-9]+")

ACCOUNT_TERMS = re.compile(
    r"\b(?:accounts?|plans?|industry|subscriptions?|tiers?)\b", re.IGNORECASE
)
# Words that topic_terms() treats as generic but that ask for more than the ticket records.
EXPLORATORY_TERMS = re.compile(
    r"\b(?:why|investigat\w*|troubleshoot\w*|fix\w*|guid\w*|recommend\w*|suggest\w*|help"
    r"|resolve|solve|steps?|problems?|issues?|affect\w*|related|relevant|evidence|cite)\b",
    re.IGNORECASE,
)
# Trace error of a model ticket page skipped because its list was already complete.
REDUNDANT_PAGE_ERROR = "Ticket list already complete"
# Capitalised words that start or shape sentences rather than name an account.
_NOT_NAMES = frozenset("""
a an and any are can check could did do does explain find for from give has have help how
i if in investigate is it list look may me my of on or our please search see show should
summarize tell the their them they this those what when where which who why will with would
you your we users user customer customers account accounts ticket tickets support
""".split())
# Product and protocol words that are capitalised but are not account names.
_TOPIC_WORDS = frozenset({"sso", "saml", "oidc", "okta", "idp", "api", "mfa", "ui", "url",
                          "http", "https", "json", "csv"})
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9&'’.-]*")
_SENTENCE_END = re.compile(r"[.!?]\s+|\n")


_POSSESSIVE = re.compile(r"(?:'s|’s|'|’)$")


def _is_name_word(word: str) -> bool:
    core = _POSSESSIVE.sub("", word).rstrip(".") or word
    if not core[:1].isupper() or core.lower() in _NOT_NAMES or core.lower() in _TOPIC_WORDS:
        return False
    return not (core.isupper() and len(core) > 1)  # acronyms such as SSO are not names


def account_references(question: str) -> list[str]:
    """Distinct candidate account names: runs of capitalised words in the question.

    A single capitalised word at the start of a sentence is ignored, since grammar, not a
    name, may capitalise it. Matching is deliberately conservative: when in doubt, no
    candidate (or several) is returned, and no lookup is made.
    """
    found: list[str] = []
    for sentence in _SENTENCE_END.split(question):
        matches = list(_WORD.finditer(sentence))
        run: list[int] = []  # indices into matches
        for index in range(len(matches) + 1):
            word = matches[index].group(0) if index < len(matches) else ""
            gap = (sentence[matches[index - 1].end():matches[index].start()].strip()
                   if 0 < index < len(matches) else "")
            # Punctuation ends a name ("Northstar Labs, Bluebird Commerce"); "&" after a
            # single word joins it ("Procter & Gamble"), otherwise it separates names.
            joins = not gap or (gap == "&" and len(run) == 1)
            if word and _is_name_word(word) and (not run or joins):
                run.append(index)
                continue
            if run and not (run[0] == 0 and len(run) == 1):
                text = sentence[matches[run[0]].start():matches[run[-1]].end()]
                name = _POSSESSIVE.sub("", " ".join(text.split())).rstrip(".")
                if name.casefold() not in {f.casefold() for f in found}:
                    found.append(name)
            run = [index] if word and _is_name_word(word) else []
    return found


@dataclass(frozen=True)
class Intent:
    tickets: bool
    knowledge: bool
    explicit_tickets: bool = False
    account: bool = False


@dataclass(frozen=True)
class FollowUp:
    tool: str
    arguments: dict[str, Any]
    reason: str


@dataclass(frozen=True)
class EvidencePlan:
    calls: list[FollowUp]  # lookups that would fill a gap
    gaps: list[str]  # gaps that no automatic lookup can fill


def classify_intent(question: str) -> Intent:
    return Intent(
        tickets=TICKET_TERMS.search(question) is not None,
        knowledge=KNOWLEDGE_TERMS.search(question) is not None,
        explicit_tickets=EXPLICIT_TICKET_TERMS.search(question) is not None,
        account=ACCOUNT_TERMS.search(question) is not None,
    )


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _succeeded(trace: list[dict[str, Any]], tool: str, **arguments: Any) -> bool:
    return any(
        entry.get("success") and entry.get("tool") == tool
        and all((entry.get("normalized_arguments") or {}).get(k) == v
                for k, v in arguments.items())
        for entry in trace
    )


def _target_accounts(
    question: str, accounts: dict[int, dict[str, Any]]
) -> tuple[list[int], bool]:
    """Verified accounts the question refers to, and whether the choice is ambiguous."""
    padded = f" {' '.join(_tokens(question))} "
    named = [
        account_id for account_id, account in accounts.items()
        if (name := " ".join(_tokens(str(account.get("name") or ""))))
        and f" {name} " in padded
    ]
    if named:
        return named[:MAX_FOLLOWUP_ACCOUNTS], False
    if len(accounts) == 1:
        return list(accounts), False
    return [], len(accounts) > 1


def knowledge_query(question: str, evidence: dict[str, Any]) -> str:
    """Topic terms of the question, in question order, without entity names."""
    entities = [
        {"type": kind, "title": record.get("name")}
        for kind, category in (("account", "accounts"), ("product", "products"))
        for record in evidence[category].values()
    ]
    terms = topic_terms(question, entities)
    ordered: list[str] = []
    for token in _tokens(question):
        if (token in terms and token not in ordered and token not in _QUERY_NOISE
                and not KNOWLEDGE_TERMS.fullmatch(token)):
            ordered.append(token)
    return " ".join(ordered[:MAX_KNOWLEDGE_QUERY_TERMS])


RESOLVED, AMBIGUOUS, NOT_FOUND, FAILED, UNSEARCHED = (
    "resolved", "ambiguous", "not_found", "failed", "unsearched")


def _folded(text: Any) -> str:
    return " ".join(str(text or "").split()).casefold()


def search_covers(query: Any, name: str) -> bool:
    """Whether an account search with this query would return every account named `name`.

    search_accounts matches names by case-insensitive substring, so a query covers a name
    when it is the name or a part of it ("Globex" covers "Globex Corporation"); a query that
    adds other words ("Northstar Labs Bluebird Commerce") does not.
    """
    folded = _folded(query)
    return bool(folded) and folded in _folded(name)


def account_status(
    name: str, evidence: dict[str, Any], trace: list[dict[str, Any]]
) -> tuple[str, list[int]]:
    """Resolution state of one requested account name, and the verified IDs it matches."""
    matches = [account_id for account_id, account in evidence["accounts"].items()
               if _folded(name) in _folded(account.get("name"))]
    exact = [account_id for account_id in matches
             if _folded(evidence["accounts"][account_id].get("name")) == _folded(name)]
    searches = [entry for entry in trace if entry.get("tool") == "search_accounts"
                and search_covers((entry.get("normalized_arguments") or {}).get("query"),
                                  name)]
    covered = any(entry.get("success") for entry in searches)
    if len(exact) == 1 or (covered and len(matches) == 1):
        return RESOLVED, exact or matches
    if len(matches) > 1:
        return AMBIGUOUS, matches
    if covered:
        return NOT_FOUND, []
    own_search = {"query": name}
    if any(entry.get("normalized_arguments") == own_search for entry in searches):
        return FAILED, []
    return UNSEARCHED, []


def _listed_references(question: str, references: list[str]) -> list[str]:
    """Names that the question lists together with another name ("X and Y", "X, Y")."""
    listed: list[str] = []
    for first, second in zip(references, references[1:]):
        pattern = (re.escape(first) + r"(?:'s|’s|'|’)?\s*(?:,\s*(?:and\s+|or\s+)?|\s+(?:and"
                   r"|or|&|vs\.?|versus)\s+)(?:the\s+)?" + re.escape(second))
        if re.search(pattern, question, re.IGNORECASE):
            listed.extend(name for name in (first, second) if name not in listed)
    return listed


def requested_names(question: str) -> list[str]:
    """The one account name in the question, or the names it lists together."""
    references = account_references(question)
    return references if len(references) == 1 else _listed_references(question, references)


def unresolved_listed_accounts(
    question: str, evidence: dict[str, Any], trace: list[dict[str, Any]]
) -> tuple[str, ...]:
    """Listed account names left unresolved while another name in the list was verified.

    Listed alongside a verified account, such a name is an account, not a topic.
    """
    listed = _listed_references(question, account_references(question))
    states = {name: account_status(name, evidence, trace)[0] for name in listed}
    if RESOLVED not in states.values():
        return ()
    return tuple(name for name, state in states.items() if state != RESOLVED)


def plan_evidence(
    question: str, evidence: dict[str, Any], trace: list[dict[str, Any]]
) -> EvidencePlan:
    """Evidence the question asks for that has not been retrieved yet."""
    intent = classify_intent(question)
    calls: list[FollowUp] = []
    gaps: list[str] = []
    if intent.tickets and evidence["accounts"]:
        targets, ambiguous = ticket_targets(question, evidence, trace)
        if ambiguous:
            ids = ", ".join(str(i) for i in sorted(evidence["accounts"]))
            gaps.append(
                f"Several accounts matched ({ids}) and the question does not name one of "
                "them exactly, so their tickets were not retrieved automatically."
            )
        statuses = requested_ticket_statuses(question)
        for account_id in targets:
            _plan_ticket_page(ticket_coverage(account_id, statuses, evidence, trace),
                              calls, gaps)
    if intent.tickets and RECENCY_TERMS.search(question):
        gaps.append(
            "The question asks for tickets by recency, but tickets have no creation date. "
            "Tickets are listed by ticket ID, which is not a chronological order."
        )
    if intent.knowledge and not _succeeded(trace, "search_knowledge"):
        query = knowledge_query(question, evidence)
        if query:
            calls.append(FollowUp(
                "search_knowledge", {"query": query},
                "The question asks for guidance, but no knowledge search had succeeded.",
            ))
    _plan_account_lookup(question, intent, evidence, trace, calls, gaps)
    return EvidencePlan(calls, gaps)


def ticket_targets(
    question: str, evidence: dict[str, Any], trace: list[dict[str, Any]]
) -> tuple[list[int], bool]:
    """Verified accounts whose tickets the question asks for, and whether that is ambiguous."""
    targets, ambiguous = _target_accounts(question, evidence["accounts"])
    # A shortened name ("Northstar") that resolved to one verified account is a target too.
    for name in requested_names(question):
        status, ids = account_status(name, evidence, trace)
        if status == RESOLVED:
            targets.extend(i for i in ids if i not in targets)
    targets = targets[:MAX_FOLLOWUP_ACCOUNTS]
    return targets, ambiguous and not targets


def parse_ticket_page(payload: Any, account_id: Any = None) -> dict[str, Any] | None:
    """Pagination metadata of a get_account_tickets result; None for legacy or bad shapes.

    With `account_id`, a page for another account, or holding another account's
    tickets, is rejected.
    """
    if isinstance(payload, str):
        try:
            return parse_ticket_page(json.loads(payload), account_id)
        except (ValueError, TypeError):
            return None
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], str):
        return parse_ticket_page(payload[0], account_id)  # MCP text-content fallback
    if not isinstance(payload, dict):
        return None
    if "tickets" not in payload and "result" in payload:
        return parse_ticket_page(payload["result"], account_id)
    tickets, status_filter = payload.get("tickets"), payload.get("status_filter")
    numbers = [payload.get(k) for k in ("account_id", "total", "offset", "returned")]
    if not (isinstance(tickets, list) and isinstance(status_filter, list)
            and all(type(n) is int and n >= 0 for n in numbers)
            and all(isinstance(s, str) for s in status_filter)):
        return None
    ids = [t["id"] for t in tickets if isinstance(t, dict) and type(t.get("id")) is int]
    page_account, total, offset, returned = numbers
    if returned != len(tickets) or len(ids) != len(tickets) or offset + returned > total:
        return None
    if account_id is not None and (page_account != account_id or any(
            t.get("account_id") != account_id for t in tickets)):
        return None
    return {"account_id": page_account, "status_filter": sorted(status_filter), "total": total,
            "offset": offset, "returned": returned, "ticket_ids": ids}


def _status_key(status: Any) -> tuple[str, ...] | None:
    try:
        return tuple(ticket_status_filter(status))
    except (ValueError, TypeError):
        return None


def _entry_filter(entry: dict[str, Any]) -> tuple[str, ...] | None:
    page = entry.get("page")
    if page:
        return tuple(page["status_filter"])
    return _status_key((entry.get("normalized_arguments") or {}).get("status"))


def _page_span(pages: list[dict[str, Any]]) -> tuple[bool, int, set[int], int]:
    """(complete, first offset not covered from 0, unique IDs, latest total) of pages.

    Completeness needs consistent totals, contiguous pages from offset 0 up to the total,
    and as many unique tickets as the total: has_more alone cannot show skipped pages.
    """
    ids: set[int] = set().union(*(page["ticket_ids"] for page in pages))
    total = pages[-1]["total"]
    covered = 0
    for page in sorted(pages, key=lambda p: p["offset"]):
        if page["offset"] > covered:
            break
        covered = max(covered, page["offset"] + page["returned"])
    consistent = len({page["total"] for page in pages}) == 1
    return consistent and covered >= total and len(ids) == total, covered, ids, total


def ticket_coverage(
    account_id: int, statuses: frozenset[str] | tuple[str, ...], evidence: dict[str, Any],
    trace: list[dict[str, Any]],
) -> TicketCoverage:
    """Coverage of one account's tickets matching a status filter, from the trace alone.

    Only paged results with metadata can make it complete; a bare list (legacy result) or
    a page for another filter shows some tickets, never that all were seen. A complete
    unfiltered retrieval covers every status filter.
    """
    wanted = tuple(sorted(statuses))
    entries = [e for e in trace if e.get("tool") == "get_account_tickets"
               and (e.get("normalized_arguments") or {}).get("account_id") == account_id
               and not str(e.get("error") or "").startswith(
                   ("Unverified record ID", REDUNDANT_PAGE_ERROR))]
    succeeded = [e for e in entries if e.get("success")]
    # Pages must belong to this account (also checked when the result is parsed).
    owned = [e["page"] for e in succeeded
             if e.get("page") and e["page"]["account_id"] == account_id]
    pages = [p for p in owned if tuple(p["status_filter"]) == wanted]
    used = sum(1 for e in succeeded if e.get("origin") == "orchestrator"
               and _entry_filter(e) == wanted)
    matching = {ticket_id for ticket_id, ticket in evidence["tickets"].items()
                if ticket.get("account_id") == account_id
                and ticket_status_matches(ticket.get("status"), wanted)}
    name = (evidence["accounts"].get(account_id) or {}).get("name")

    def result(state: str, total: int | None, retrieved: int,
               next_offset: int | None = None) -> TicketCoverage:
        return TicketCoverage(account_id, name, wanted, state, total, retrieved,
                              next_offset, used)

    if pages:
        complete, covered, ids, total = _page_span(pages)
        if complete:
            return result(COMPLETE, total, total)
    if wanted:
        broad = [p for p in owned if not p["status_filter"]]
        broad_complete, _, broad_ids, _ = _page_span(broad) if broad else (False, 0, set(), 0)
        if broad_complete:
            count = sum(1 for i in broad_ids if ticket_status_matches(
                (evidence["tickets"].get(i) or {}).get("status"), wanted))
            return result(COMPLETE, count, count)
    if pages:
        return result(PARTIAL, total, len(ids | matching), covered)
    # Lookups for another status filter retrieved none of the wanted tickets.
    if any(_entry_filter(e) in {wanted, ()} for e in succeeded):
        return result(PARTIAL, None, len(matching), 0)
    if any(_entry_filter(e) in {wanted, (), None} for e in entries):
        return result(RETRIEVAL_FAILED, None, len(matching))
    return result(UNRETRIEVED, None, len(matching))


def _plan_ticket_page(
    coverage: TicketCoverage, calls: list[FollowUp], gaps: list[str]
) -> None:
    """The next page an incomplete account needs, or a gap once its page budget is spent."""
    account_id, label = coverage.account_id, status_label(coverage.status_filter)
    if coverage.state == COMPLETE:
        return
    seen = (f"{coverage.retrieved} of "
            f"{coverage.total if coverage.total is not None else 'an unknown number of'} "
            f"{label}tickets")
    if coverage.pages >= MAX_TICKET_PAGES:
        gaps.append(
            f"Tickets for verified account {account_id} were only partially retrieved "
            f"({seen}); the page limit ({MAX_TICKET_PAGES} per account) was reached."
        )
        return
    arguments: dict[str, Any] = {"account_id": account_id, "limit": TICKET_PAGE_SIZE}
    if coverage.status_filter:
        arguments["status"] = list(coverage.status_filter)
    if coverage.next_offset:
        arguments["offset"] = coverage.next_offset
    if coverage.state == PARTIAL:
        reason = (f"Tickets for verified account {account_id} were only partially retrieved "
                  f"({seen}); the next page starts at offset {coverage.next_offset or 0}.")
    elif coverage.state == RETRIEVAL_FAILED:
        reason = (f"The question asks about support issues, but tickets for verified account "
                  f"{account_id} could not be retrieved (the lookup failed).")
    else:
        reason = (f"The question asks about support issues, but tickets for verified account "
                  f"{account_id} had not been retrieved.")
    calls.append(FollowUp("get_account_tickets", arguments, reason))


def ticket_call_covers(arguments: dict[str, Any], planned: dict[str, Any]) -> bool:
    """Whether a requested ticket lookup can stand in for a planned page.

    It must start at the same offset, for the same status filter or (for a first page)
    without one. Its page size may be smaller: coverage then stays partial and the next
    page is planned from where it ended.
    """
    offset = planned.get("offset", 0)
    status = _status_key(arguments.get("status"))
    return (arguments.get("account_id") == planned["account_id"]
            and (status == _status_key(planned.get("status")) or (status == () and not offset))
            and arguments.get("offset", 0) == offset)


def ticket_coverage_report(
    question: str, evidence: dict[str, Any], trace: list[dict[str, Any]]
) -> list[TicketCoverage]:
    """Coverage for every account whose tickets the question asks for."""
    if not (classify_intent(question).tickets and evidence["accounts"]):
        return []
    statuses = requested_ticket_statuses(question)
    return [ticket_coverage(account_id, statuses, evidence, trace)
            for account_id in ticket_targets(question, evidence, trace)[0]]


def list_question_satisfied(
    question: str, evidence: dict[str, Any], trace: list[dict[str, Any]]
) -> bool:
    """Whether a simple ticket-list question already has every record it asks for.

    Conservative: topic, guidance or exploratory wording, or any requested account that
    is not resolved to exactly one verified account, keeps the model in the loop.
    """
    intent = classify_intent(question)
    if (not intent.explicit_tickets or intent.knowledge
            or EXPLORATORY_TERMS.search(question) or RECENCY_TERMS.search(question)):
        return False
    names = requested_names(question)
    if not names or len(names) > MAX_FOLLOWUP_ACCOUNTS:
        return False
    states = [account_status(name, evidence, trace) for name in names]
    if any(state != RESOLVED for state, _ in states):
        return False
    entities = [{"type": kind, "title": record.get("name")}
                for kind, category in (("account", "accounts"), ("product", "products"))
                for record in evidence[category].values()]
    if topic_terms(question, entities, tuple(names)):
        return False
    plan = plan_evidence(question, evidence, trace)
    if plan.calls or plan.gaps:
        return False
    coverage = {c.account_id: c.state
                for c in ticket_coverage_report(question, evidence, trace)}
    resolved = {account_id for _, ids in states for account_id in ids}
    return (bool(coverage) and resolved <= set(coverage)
            and all(state == COMPLETE for state in coverage.values()))


def _plan_account_lookup(
    question: str, intent: Intent, evidence: dict[str, Any], trace: list[dict[str, Any]],
    calls: list[FollowUp], gaps: list[str],
) -> None:
    """Resolve each requested account name, or explain why an account is unknown."""
    if not (intent.tickets or intent.account):
        return
    references = account_references(question)
    requested = requested_names(question)
    if evidence["accounts"] and len(requested) < 2:
        return  # one account: the verified accounts are matched by _target_accounts
    if len(references) > 1 and not requested:
        names = ", ".join(f'"{name}"' for name in references)
        gaps.append(
            f"The question mentions several possible account names ({names}), so no "
            "account was looked up automatically. Name one account to investigate."
        )
        return
    if len(requested) > MAX_FOLLOWUP_ACCOUNTS:
        gaps.append(
            f"The question names {len(requested)} accounts; at most {MAX_FOLLOWUP_ACCOUNTS} "
            "are investigated automatically. Name fewer accounts."
        )
        return
    for name in requested:
        status, ids = account_status(name, evidence, trace)
        if status == UNSEARCHED:
            calls.append(FollowUp(
                "search_accounts", {"query": name},
                f"The question names the account \"{name}\", but no account search covering "
                "that name had succeeded.",
            ))
        elif status == FAILED:
            gaps.append(f"The account search for \"{name}\" failed, so the account could "
                        "not be verified.")
        elif status == AMBIGUOUS:
            gaps.append(
                f"The name \"{name}\" matches several verified accounts "
                f"({', '.join(str(i) for i in sorted(ids))}); none was chosen automatically."
            )
    if not references and (intent.explicit_tickets or intent.account or not trace):
        # A search the model invented (the question names no account) verifies nothing.
        gaps.append(
            "The question asks about support tickets or account details, but it names no "
            "identifiable account, so no account could be verified."
        )
