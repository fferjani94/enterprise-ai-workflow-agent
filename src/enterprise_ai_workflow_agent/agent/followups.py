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

Account discovery: when a ticket or account question was answered without any account
lookup, and exactly one account name can be read from the question (a run of capitalised
words, see account_references), search_accounts is called with that text. Account IDs
still come only from the MCP result. Several candidate names are never guessed between.
If an account search matched nothing but its query did not correspond to a name in the
question (for example two names combined in one query), each unsearched name is searched
separately; finding nothing is a negative result only for names that were searched.
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
        words = [m.group(0) for m in _WORD.finditer(sentence)]
        run: list[str] = []
        for index, word in enumerate([*words, ""]):
            if word and _is_name_word(word):
                run.append(word)
                continue
            starts_sentence = index - len(run) == 0
            if run and not (starts_sentence and len(run) == 1):
                name = _POSSESSIVE.sub("", " ".join(run)).rstrip(".")
                if name.casefold() not in {f.casefold() for f in found}:
                    found.append(name)
            run = []
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


def _unsearched_references(references: list[str], trace: list[dict[str, Any]]) -> list[str]:
    """Names from the question that no successful account search covered.

    A search covers a name when its query is the name or a part of it ("Globex" covers
    "Globex Corporation"); a query that adds other words ("Northstar Labs Bluebird
    Commerce") does not, since finding nothing for it says nothing about either name.
    """
    queries = [
        " ".join(_tokens(str((entry.get("normalized_arguments") or {}).get("query") or "")))
        for entry in trace
        if entry.get("tool") == "search_accounts" and entry.get("success")
    ]
    return [
        name for name in references
        if not any(query and f" {query} " in f" {' '.join(_tokens(name))} "
                   for query in queries)
    ]


def plan_evidence(
    question: str, evidence: dict[str, Any], trace: list[dict[str, Any]]
) -> EvidencePlan:
    """Evidence the question asks for that has not been retrieved yet."""
    intent = classify_intent(question)
    calls: list[FollowUp] = []
    gaps: list[str] = []
    if intent.tickets and evidence["accounts"]:
        targets, ambiguous = _target_accounts(question, evidence["accounts"])
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
    """Recover a skipped account lookup, or explain why the account is unknown."""
    if not (intent.tickets or intent.account) or evidence["accounts"]:
        return
    attempted = any(entry.get("tool") == "search_accounts" for entry in trace)
    searched = _succeeded(trace, "search_accounts")
    references = account_references(question)
    if not attempted and len(references) == 1:
        calls.append(FollowUp(
            "search_accounts", {"query": references[0]},
            f"The question is about the account \"{references[0]}\", but no account lookup "
            "had been made.",
        ))
        return
    if references and searched:
        # Matching nothing is a negative result only for the names actually searched.
        # Observed with qwen2.5:3b: "Northstar Labs Bluebird Commerce" in a single query.
        unsearched = _unsearched_references(references, trace)
        if not unsearched:
            return
        if len(unsearched) <= MAX_FOLLOWUP_ACCOUNTS:
            calls.extend(FollowUp(
                "search_accounts", {"query": name},
                f"The question names the account \"{name}\", but no account search had used "
                "that name.",
            ) for name in unsearched)
            return
    if len(references) > 1:
        names = ", ".join(f'"{name}"' for name in references)
        gaps.append(
            f"The question mentions several possible account names ({names}), so no "
            "account was looked up automatically. Name one account to investigate."
        )
    elif references:
        gaps.append("The question asks about support tickets or account details, but no "
                    "account lookup succeeded.")
    elif intent.explicit_tickets or intent.account or not trace:
        # A search the model invented (the question names no account) verifies nothing.
        gaps.append(
            "The question asks about support tickets or account details, but it names no "
            "identifiable account, so no account could be verified."
        )
