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

import re
from dataclasses import dataclass
from typing import Any

from enterprise_ai_workflow_agent.agent.grounding import topic_terms

MAX_FOLLOWUP_ACCOUNTS = 3
MAX_KNOWLEDGE_QUERY_TERMS = 8

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
        targets, ambiguous = _target_accounts(question, evidence["accounts"])
        # A shortened name ("Northstar") that resolved to one verified account is a target too.
        for name in requested_names(question):
            status, ids = account_status(name, evidence, trace)
            if status == RESOLVED:
                targets.extend(i for i in ids if i not in targets)
        targets = targets[:MAX_FOLLOWUP_ACCOUNTS]
        ambiguous = ambiguous and not targets
        if ambiguous:
            ids = ", ".join(str(i) for i in sorted(evidence["accounts"]))
            gaps.append(
                f"Several accounts matched ({ids}) and the question does not name one of "
                "them exactly, so their tickets were not retrieved automatically."
            )
        for account_id in targets:
            if not _succeeded(trace, "get_account_tickets", account_id=account_id):
                calls.append(FollowUp(
                    "get_account_tickets", {"account_id": account_id},
                    f"The question asks about support issues, but tickets for verified "
                    f"account {account_id} had not been retrieved.",
                ))
    if intent.knowledge and not _succeeded(trace, "search_knowledge"):
        query = knowledge_query(question, evidence)
        if query:
            calls.append(FollowUp(
                "search_knowledge", {"query": query},
                "The question asks for guidance, but no knowledge search had succeeded.",
            ))
    _plan_account_lookup(question, intent, evidence, trace, calls, gaps)
    return EvidencePlan(calls, gaps)


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
