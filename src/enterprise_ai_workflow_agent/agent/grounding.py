"""Deterministic relevance labelling and answer checks for retrieved evidence.

No LLM calls: a record is "relevant" only when its retrieved title or description shares a
term with the topic of the question. Matching is lexical and deliberately conservative.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from enterprise_ai_workflow_agent.mcp_server.services import (
    TICKET_STATUS_VALUES,
    ticket_status_matches,
)

TOKEN = re.compile(r"[a-z0-9]+")

# Words that describe how to investigate, not what is being investigated.
GENERIC_TERMS = frozenset("""
about account accounts across affect affected affecting all also and any are been before both
but can cite check closed could customer customers data did does docs document documentation
documents each every evidence find for from get give has have how into investigate investigation
include includes including
issue issues its latest list may more most not open our please priority problem problems recent
related relevant report reported reports resolved results retrieve search see show some status
support tell that the their them then there these they this those ticket tickets tool tools use
used using was were what when where which who why will with would you your
troubleshoot troubleshooting fix fixes fixing guidance guide guides step steps next suggest
recommend help resolve solve
""".split())

HEDGE = re.compile(
    r"\b(?:unrelated|not related|no evidence|no relation|not (?:clearly |directly )?"
    r"(?:linked|connected|associated)|does not|doesn't|is not|isn't|not an?)\b",
    re.IGNORECASE,
)
CAUSAL = re.compile(
    r"\b(?:root cause|caused by|is caused|are caused|was caused|the cause|due to|because of"
    r"|resulting from|results from|attributed to|the (?:issue|problem) (?:is|was))\b",
    re.IGNORECASE,
)

CAUSAL_NOTE = (
    "\n\nNote: no retrieved record confirms a root cause. Any cause or diagnosis above is "
    "an unconfirmed hypothesis."
)

# Symptom words that describe *that* something is wrong, not *what*. When the question also
# names a specific topic, a record that shares only these words is not a topic match
# ("Webhook ... on failures" is not about "SSO login failures").
WEAK_TERMS = frozenset({
    "fail", "fails", "failed", "failing", "failure", "failures", "error", "errors", "broken",
    "unable", "intermittent", "outage", "outages", "down",
})

# Small, explainable groups of terms that name the same support topic. A question term in a
# group lets records match the group's other terms, labelled as a *related* match. Stems are
# fine: matching is by prefix (see _same_term). A shared topic does not establish causation.
RELATED_TERMS: tuple[tuple[str, frozenset[str]], ...] = (
    ("sign-in / authentication", frozenset({
        "sso", "saml", "oidc", "okta", "idp", "login", "logon", "signin", "authenticat",
    })),
    ("performance", frozenset({"slow", "latency", "performance", "loading", "lag"})),
    ("billing", frozenset({"billing", "invoice", "payment", "refund"})),
)

# A line that only introduces a section, e.g. "**Unrelated records**", "## Findings",
# "Unrelated or unconfirmed records:".
_HEADING = re.compile(r"^\s*(?:#{1,6}\s+.+|\*\*[^*]+\*\*:?|__[^_]+__:?|[^.!?:]{1,60}:)\s*$")
_HEDGED_SECTION = re.compile(r"\b(?:unrelated|unconfirmed|not related|no match)\b",
                             re.IGNORECASE)
_UNKNOWN_CAUSE = re.compile(
    r"\b(?:unknown|undetermined|cannot be determined|not (?:yet )?(?:known|established"
    r"|identified|determined)|no (?:retrieved )?records? (?:states?|establish(?:es)?"
    r"|confirms?|identif(?:y|ies))|no (?:other |further )?(?:evidence|records?) (?:points?"
    r"|indicates?|shows?|suggests?))\b",
    re.IGNORECASE,
)
_ASSERTED_LINK = re.compile(
    r"\b(?:related to|linked to|connected to|associated with|caus\w*|due to|because"
    r"|explains?|contribut\w*)\b",
    re.IGNORECASE,
)

Record = dict[str, Any]


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def _same_term(a: str, b: str) -> bool:
    # Prefix matching covers plurals and simple inflections (failure/failures, load/loading).
    return a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)))


def topic_terms(question: str, records: list[Record], names: tuple[str, ...] = ()) -> set[str]:
    """Terms naming the investigated topic, excluding generic words and entity names.

    `names` are further entity names that no retrieved record carries (an account the
    question lists but that could not be verified).
    """
    entity_names = {
        token
        for record in records
        if record["type"] in {"account", "product"}
        for token in _tokens(str(record.get("title") or ""))
    } | {token for name in names for token in _tokens(name)}
    return {
        token for token in _tokens(question)
        if len(token) >= 3 and not token.isdigit()
        and token not in GENERIC_TERMS and token not in entity_names
    }


def related_terms(terms: set[str]) -> dict[str, tuple[str, set[str]]]:
    """Related terms implied by the question: term -> (topic, question terms behind it)."""
    related: dict[str, tuple[str, set[str]]] = {}
    for topic, group in RELATED_TERMS:
        sources = {t for t in terms if any(_same_term(t, g) for g in group)}
        if not sources:
            continue
        for member in group:
            if not any(_same_term(member, t) for t in terms):
                related.setdefault(member, (topic, sources))
    return related


def classify_relevance(
    question: str, records: list[Record], names: tuple[str, ...] = ()
) -> list[Record]:
    """Add 'relevance' and 'relevance_basis' labels to each evidence record (in place).

    context     account or product that identifies who/what was investigated
    relevant    retrieved title/description matches the question's topic, directly or via a
                related term (RELATED_TERMS); the basis says which
    no_match    retrieved, but nothing in its retrieved fields matches the topic
    unassessed  the question names no specific topic
    """
    terms = topic_terms(question, records, names)
    # With a specific topic in the question, symptom words alone are not enough.
    strong = {t for t in terms if t not in WEAK_TERMS}
    match_terms = strong or terms
    related = related_terms(strong or terms)
    for record in records:
        record["relevance_basis"] = None
        if record["type"] in {"account", "product"}:
            record["relevance"] = "context"
            continue
        if not terms:
            record["relevance"] = "unassessed"
            continue
        words = set(_tokens(" ".join(
            str(part) for part in (record.get("title"), record.get("description")) if part
        )))
        if any(_same_term(t, w) for t in match_terms for w in words):
            direct = sorted({w for w in words for t in terms if _same_term(t, w)})
            record["relevance"] = "relevant"
            record["relevance_basis"] = "Shares the question term(s): " + ", ".join(direct)
            continue
        linked = sorted({
            (w, related[member][0], ", ".join(sorted(related[member][1])))
            for w in words for member in related if _same_term(member, w)
        })
        if linked:
            record["relevance"] = "relevant"
            record["relevance_basis"] = "Related topic: " + "; ".join(
                f"'{word}' ({topic}, like '{source}')" for word, topic, source in linked
            ) + ". A shared topic does not establish a cause."
        else:
            record["relevance"] = "no_match"
    return records


def _ref(record: Record) -> dict[str, Any]:
    return {"type": record["type"], "id": record["id"]}


def _statement(text: str, *records: Record) -> dict[str, Any]:
    return {"text": text, "evidence": [_ref(record) for record in records]}


DOCS_UNAVAILABLE = ("No relevant documentation was retrieved, so documentation-backed "
                    "troubleshooting guidance is unavailable.")

# Answer sections rendered from build_analysis, in order: (title, analysis keys).
ANSWER_SECTIONS = (
    ("Relevant support tickets", ("findings",)),
    ("Relevant knowledge documents", ("guidance",)),
    ("Other retrieved records", ("context", "unsupported")),
    ("Unknown or unconfirmed", ("unconfirmed",)),
)


def _detail(record: Record, keys: tuple[str, ...]) -> str:
    fields = record.get("fields") or {}
    detail = ", ".join(str(fields[k]) for k in keys if fields.get(k) not in (None, ""))
    return f" ({detail})" if detail else ""


# A status word directly qualifying the ticket noun ("open support tickets") is a filter.
_STATUS_FILTER = re.compile(
    r"\b(open|resolved|closed|unresolved)\s+(?:support\s+)?(?:tickets?|incidents?|issues?"
    r"|cases?)\b",
    re.IGNORECASE,
)


def requested_ticket_statuses(question: str) -> frozenset[str]:
    """Ticket statuses the question explicitly asks for; empty when it sets no filter."""
    return frozenset(value for word in _STATUS_FILTER.findall(question)
                     for value in TICKET_STATUS_VALUES[word.lower()])


COMPLETE, PARTIAL, FAILED, UNRETRIEVED = "complete", "partial", "failed", "unretrieved"


@dataclass(frozen=True)
class TicketCoverage:
    """How completely one account's tickets matching one status filter were retrieved."""
    account_id: int
    account_name: str | None
    status_filter: tuple[str, ...]  # canonical statuses; empty means all statuses
    state: str  # COMPLETE, PARTIAL, FAILED or UNRETRIEVED
    total: int | None  # SQL total of matching tickets, when known
    retrieved: int  # unique matching tickets retrieved
    next_offset: int | None = None  # first offset not yet retrieved, while incomplete
    pages: int = 0  # successful orchestrator page requests for this filter


def status_label(status_filter: Sequence[str]) -> str:
    """'open ', 'resolved or closed ', ... for use before the word 'tickets'."""
    values = set(status_filter)
    if not values:
        return ""
    if values == {"closed", "resolved"}:
        return "resolved or closed "
    return " or ".join(sorted(values)) + " "


def _tickets(count: int) -> str:
    return "ticket" if count == 1 else "tickets"


def coverage_statement(coverage: TicketCoverage) -> str:
    who = (f"{coverage.account_name} (Account {coverage.account_id})"
           if coverage.account_name else f"Account {coverage.account_id}")
    label = status_label(coverage.status_filter)
    if coverage.state == COMPLETE:
        total = coverage.total or 0
        if not total:
            return f"{who}: 0 {label}tickets (complete)."
        return f"{who}: {total} {label}{_tickets(total)} retrieved (complete)."
    if coverage.state == PARTIAL:
        if coverage.total is not None:
            return (f"{who}: {coverage.retrieved} of {coverage.total} {label}"
                    f"{_tickets(coverage.total)} retrieved; detailed list incomplete.")
        return (f"{who}: {coverage.retrieved} {label}{_tickets(coverage.retrieved)} "
                "retrieved; total unknown, detailed list incomplete.")
    if coverage.state == FAILED:
        return f"{who}: ticket retrieval failed; ticket count unknown."
    return f"{who}: tickets not retrieved; ticket count unknown."


def build_analysis(
    records: list[Record],
    *,
    expects_tickets: bool = False,
    issue_question: bool = False,
    notes: tuple[str, ...] = (),
    ticket_statuses: frozenset[str] = frozenset(),
    coverage: Sequence[TicketCoverage] = (),
) -> dict[str, list[dict[str, Any]]]:
    """Deterministic answer sections built only from verified, labelled records.

    findings     per-account ticket coverage, then relevant support tickets (or all tickets
                 when relevance is unassessed)
    guidance     relevant knowledge documents: general documentation, not a diagnosis
    context      accounts and products that identify what was investigated, and tickets
                 outside the statuses the question asks for (ticket_statuses)
    unsupported  retrieved records with no topic match: not evidence of this issue
    unconfirmed  what the records do not establish, plus caller-supplied notes
    No model output is used, so record lists and counts here are authoritative.
    """
    retrieved = [r for r in records if r["type"] == "ticket"]
    other_status = [r for r in retrieved if not ticket_status_matches(
        (r.get("fields") or {}).get("status"), ticket_statuses)]
    tickets = [r for r in retrieved if r not in other_status]
    documents = [r for r in records if r["type"] == "document"]
    findings, guidance, context, unsupported, unconfirmed = [], [], [], [], []

    accounts = {r["id"]: r for r in records if r["type"] == "account"}
    for item in coverage:
        account = accounts.get(item.account_id)
        findings.append(_statement(coverage_statement(item), *([account] if account else [])))

    matching = [r for r in tickets if r["relevance"] == "relevant"]
    for ticket in matching:
        related = str(ticket.get("relevance_basis") or "").startswith("Related topic")
        findings.append(_statement(
            f"Ticket {ticket['id']}, \"{ticket.get('title') or 'untitled'}\""
            f"{_detail(ticket, ('status', 'priority'))}, "
            + ("is on a topic related to the question. "
               f"{ticket['relevance_basis']}" if related
               else "matches the topic of the question."),
            ticket,
        ))
    unassessed = [r for r in tickets if r["relevance"] == "unassessed"]
    for ticket in unassessed:
        findings.append(_statement(
            f"Ticket {ticket['id']}, \"{ticket.get('title') or 'untitled'}\""
            f"{_detail(ticket, ('status', 'priority'))}.", ticket,
        ))
    if unassessed:
        findings.append(_statement(
            "The question names no specific topic, so ticket relevance was not assessed.",
            *unassessed,
        ))
    elif tickets and not matching:
        findings.append(_statement(
            "No retrieved ticket matches the topic of the question.", *tickets
        ))
    elif not tickets and other_status:
        findings.append(_statement(
            "No retrieved ticket has the requested status ("
            + ", ".join(sorted(ticket_statuses)) + ").", *other_status,
        ))
    elif not tickets and expects_tickets:
        findings.append(_statement("No support tickets were retrieved."))

    for doc in (r for r in documents if r["relevance"] in {"relevant", "unassessed"}):
        guidance.append(_statement(
            f"Document {doc['id']}, \"{doc.get('title') or 'untitled'}\": "
            f"{doc.get('description') or 'no excerpt'} "
            "This is general documentation, not a confirmed diagnosis for this customer.", doc,
        ))
    if not guidance and issue_question:
        guidance.append(_statement(DOCS_UNAVAILABLE))

    for record in (r for r in records if r["relevance"] == "context"):
        title = record.get("title")
        context.append(_statement(
            f"{record['type'].capitalize()} {record['id']}"
            + (f", \"{title}\"" if title else " (details not retrieved)")
            + _detail(record, ("plan", "industry", "status", "version"))
            + ": context for the investigation, not evidence of the issue.", record,
        ))
    for ticket in other_status:
        context.append(_statement(
            f"Ticket {ticket['id']}, \"{ticket.get('title') or 'untitled'}\""
            f"{_detail(ticket, ('status', 'priority'))}, was retrieved but does not have "
            "the status the question asks for.", ticket,
        ))
    for ticket in (r for r in tickets if r["relevance"] == "no_match"):
        unsupported.append(_statement(
            f"Ticket {ticket['id']}, \"{ticket.get('title') or 'untitled'}\", belongs to the "
            "same account, but nothing in its retrieved fields links it to this question. "
            "It is not evidence of this issue.", ticket,
        ))
    for doc in (r for r in documents if r["relevance"] == "no_match"):
        unsupported.append(_statement(
            f"Knowledge document {doc['id']}, \"{doc.get('title') or 'untitled'}\", was "
            "retrieved but does not match the topic of the question.", doc,
        ))

    if records and issue_question:
        unconfirmed.append(_statement("No retrieved record establishes a root cause."))
    if matching and guidance and documents:
        unconfirmed.append(_statement(
            "Whether the documented guidance applies to the matching ticket(s) is not "
            "confirmed.", *matching,
        ))
    unconfirmed.extend(_statement(note) for note in notes)
    return {
        "findings": findings, "guidance": guidance, "context": context,
        "unsupported": unsupported, "unconfirmed": unconfirmed,
    }


def render_sections(analysis: dict[str, list[dict[str, Any]]]) -> str:
    """Markdown for the deterministic answer sections; empty sections are omitted."""
    blocks = []
    for title, keys in ANSWER_SECTIONS:
        items = [item["text"] for key in keys for item in analysis.get(key, [])]
        if items:
            blocks.append(f"**{title}**\n" + "\n".join(f"- {item}" for item in items))
    return "\n\n".join(blocks)


# A sentence that tells someone what to do (an instruction or a recommendation).
_MODAL_ADVICE = (
    r"\b(?:you|we|they|users?|the customer|admins?|administrators?|the team)\s+"
    r"(?:should|must|need to|may need to|might need to|can try|could try|may want to)\b"
)
_ADVICE = re.compile(
    r"^[\s>*_#-]*(?:\d+[.)]\s*)?(?:\*\*)?(?:please\s+)?(?:check|verify|ensure|review|confirm"
    r"|configure|reconfigure|reset|clear|restart|reboot|update|upgrade|contact|try|test"
    r"|inspect|investigate|consider|rotate|renew|enable|disable|validate|monitor|escalate"
    r"|re-?authenticate|re-?enter|look into|make sure)\b"
    rf"|{_MODAL_ADVICE}"
    r"|\b(?:recommend\w*|next steps?|workarounds?|to (?:fix|resolve|troubleshoot|mitigate))\b",
    re.IGNORECASE,
)
UNCITED_GUIDANCE_NOTE = (
    "_Troubleshooting suggestions that no retrieved document supports were removed._"
)
# A sentence that opens by stating that something is absent ("No further steps to resolve
# are available") is not an instruction, even when it mentions steps, unless it also tells
# someone what to do ("No steps are documented, so you might need to ...").
_STATES_ABSENCE = re.compile(r"^[\s>*_#-]*(?:there (?:is|are|were|was) no|no|none)\b",
                             re.IGNORECASE)
_MODAL_ADVICE_RE = re.compile(_MODAL_ADVICE, re.IGNORECASE)


def _is_advice(sentence: str) -> bool:
    if not _ADVICE.search(sentence):
        return False
    return not (_STATES_ABSENCE.match(sentence) and not _MODAL_ADVICE_RE.search(sentence))


def _cites_document(sentence: str, documents: list[Record]) -> bool:
    folded = sentence.casefold()
    return any(
        re.search(rf"\b(?:knowledge\s+)?(?:document|doc)\s*(?:id|#)?\s*:?\s*{doc['id']}\b",
                  sentence, re.IGNORECASE)
        or (doc.get("title") and str(doc["title"]).casefold() in folded)
        for doc in documents
    )


def remove_uncited_guidance(text: str, records: list[Record]) -> tuple[str, int]:
    """Drop instructions that do not cite a relevant retrieved document.

    Applies to every model-written narrative, so no troubleshooting step is presented as
    document-backed without a retrieved document behind it. Returns the text and the
    number of sentences removed.
    """
    documents = [r for r in records
                 if r["type"] == "document" and r["relevance"] in {"relevant", "unassessed"}]
    removed = 0
    kept_lines = []
    for line in text.splitlines():
        sentences = re.split(r"(?<=[.!?])\s+", line)
        kept = [s for s in sentences
                if not (s.strip() and _is_advice(s) and not _cites_document(s, documents))]
        removed += len(sentences) - len(kept)
        if kept and any(s.strip() for s in kept) or not line.strip():
            kept_lines.append(" ".join(kept))
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()
    return cleaned, removed


# Claims that an action definitely fixes the customer's problem ("re-uploading the
# certificate resolves the issue", "this will fix SSO login"). Only present and future
# tense is matched: ticket status wording ("Ticket 12 was resolved", "is resolved", "has
# been resolved") reports a retrieved fact and is left alone.
_CERTAINTY = r"(?:definitely|certainly|surely|always|fully|completely)\s+"
_FIX_OBJECT = (
    r"(?=\s+(?:the|this|that|these|those|your|their|its|our|any|all|such|it|them"
    r"|[a-z]+(?:'s|’s)|sso|saml|login|sign-?in|authentication|access|errors?|issues?"
    r"|problems?|failures?|outages?)\b)"
)
_FIX_CLAIM = re.compile(
    rf"\b(?:(?:will|shall|should|would)\s+(?:{_CERTAINTY})?(?P<modal>fix|resolve|solve|eliminate)"
    rf"|(?:is|are)\s+(?:guaranteed|certain|sure)\s+to\s+(?P<sure>fix|resolve|solve|eliminate)"
    rf"|(?:{_CERTAINTY})?(?P<present>fixes|resolves|solves|eliminates)){_FIX_OBJECT}"
    rf"|\bwill\s+be\s+(?:{_CERTAINTY})?(?P<passive>fixed|resolved|solved)\b",
    re.IGNORECASE,
)
_NEGATED = re.compile(r"\b(?:not|never|no longer)\s+$|n't\s+$", re.IGNORECASE)
# The sentence already says the effect is not established.
_EPISTEMIC = re.compile(
    r"\b(?:no (?:retrieved )?(?:record|evidence|document)s?|not (?:confirmed|established|known)"
    r"|unconfirmed|unknown|unclear|whether)\b",
    re.IGNORECASE,
)
_SENTENCE_START = re.compile(r".*(?:[.!?]\s+|\n)", re.DOTALL)
FIX_CLAIM_NOTE = (
    "_Statements that a step fixes or resolves the problem were softened: no retrieved "
    "record confirms that any action resolves this customer's issue._"
)


def qualify_fix_claims(text: str) -> tuple[str, int]:
    """Soften unsupported claims that an action fixes the problem to "may help ...".

    The records are tickets and general documentation; none confirms that a step resolves
    this customer's issue. Quoted text (record titles) and negated claims are not changed.
    Returns the text and the number of claims softened.
    """
    changed = 0

    def soften(match: re.Match[str]) -> str:
        nonlocal changed
        before = text[:match.start()]
        line_start = before.rfind("\n") + 1
        boundary = _SENTENCE_START.match(before)
        sentence = before[boundary.end():] if boundary else before
        if (_NEGATED.search(before) or _EPISTEMIC.search(sentence)
                or before[line_start:].count('"') % 2
                or before.count("“") > before.count("”")):
            return match.group(0)
        if verb := match.group("passive"):
            replacement = f"may be {verb.lower()}"
        else:
            verb = (match.group("modal") or match.group("sure") or match.group("present"))
            base = verb.lower()
            if match.group("present"):
                base = "fix" if base == "fixes" else base.removesuffix("s")
            replacement = f"may help {base}"
        changed += 1
        return replacement[0].upper() + replacement[1:] if match.group(0)[0].isupper() \
            else replacement

    return _FIX_CLAIM.sub(soften, text), changed


_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_COUNT_CLAIM = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?:(?!of\b|the\b|these\b|those\b|and\b|or\b)[a-z-]+\s+){0,2}(tickets?|documents?)\b",
    re.IGNORECASE,
)
# "Ticket 11 ..." is an ID, not a count.
_ID_PREFIX = re.compile(r"(?:ticket|document|account|product|id|number|#)\s*[:#]?\s*$",
                        re.IGNORECASE)
_RELEVANCE_CLAIM = re.compile(
    r"\b(?:related|relevant|match\w*|concern\w*|about|affect\w*|linked|regarding)\b",
    re.IGNORECASE,
)


# Wording attached to a count that presents it as what was retrieved, not what exists:
# "20 tickets (were) retrieved" after the count, or "the first 20 tickets" before it.
_RETRIEVED_AFTER = re.compile(  # used with .match(text, pos), which anchors at pos
    r"\s*(?:(?:were|was|have been|has been|are|is)\s+)?"
    r"(?:retrieved|shown|listed|returned|displayed|fetched)\b",
    re.IGNORECASE)
_RETRIEVED_BEFORE = re.compile(r"\bfirst\s+$", re.IGNORECASE)
# Names listed together ("X and Y", "X, Y", "X & Y") form one combined subject.
_NAME_JOIN = re.compile(r"^\s*(?:,|,?\s*and|&)\s*$", re.IGNORECASE)


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = max(text.rfind(mark, 0, start) for mark in (". ", "! ", "? ", "\n")) + 1
    right = min((i for i in (text.find(mark, end) for mark in (". ", "! ", "? ", "\n"))
                 if i != -1), default=len(text))
    return left, right


def _sentence_at(text: str, start: int, end: int) -> str:
    left, right = _sentence_bounds(text, start, end)
    return text[left:right]


def _account_mentions(
    sentence: str, coverage: Sequence[TicketCoverage]
) -> list[tuple[int, int, TicketCoverage]]:
    mentions = []
    for item in coverage:
        patterns = [rf"\baccount\s*(?:id\s*)?[:#]?\s*{item.account_id}\b"]
        if item.account_name:
            patterns.append(r"\b" + r"\s+".join(map(re.escape, item.account_name.split()))
                            + r"\b")
        for pattern in patterns:
            mentions += [(m.start(), m.end(), item)
                         for m in re.finditer(pattern, sentence, re.IGNORECASE)]
    return sorted(mentions, key=lambda m: m[0])


def _count_subject(
    text: str, start: int, end: int, coverage: Sequence[TicketCoverage]
) -> list[TicketCoverage]:
    """The account(s) a count is about: the nearest name before it (with names listed
    together with it), else the first name after it, else every investigated account."""
    left, right = _sentence_bounds(text, start, end)
    sentence = text[left:right]
    mentions = _account_mentions(sentence, coverage)
    before = [m for m in mentions if m[1] <= start - left][::-1]
    after = [m for m in mentions if m[0] >= end - left]
    chain = before or after
    if not chain:
        return list(coverage)
    group = [chain[0]]
    for mention in chain[1:]:
        first, second = (mention, group[-1]) if before else (group[-1], mention)
        if first[1] <= second[0] and _NAME_JOIN.match(sentence[first[1]:second[0]]):
            group.append(mention)
        elif mention[2] is not group[-1][2]:
            break
    return list({id(m[2]): m[2] for m in group}.values())


def _group_counts(records: list[Record]) -> set[int]:
    """Counts supported by some grouping of the records (all, relevance, status, priority)."""
    return {len(records)} | {
        sum(1 for r in records if r["relevance"] == label)
        for label in ("relevant", "no_match", "unassessed")
    } | {
        sum(1 for r in records if (r.get("fields") or {}).get(field) == value)
        for r0 in records for field, value in (r0.get("fields") or {}).items()
        if field in {"status", "priority"}
    }


def _ticket_counts(
    text: str, match: re.Match[str], tickets: list[Record],
    coverage: Sequence[TicketCoverage],
) -> set[int]:
    """Ticket counts a claim may state, scoped to the account(s) it is about.

    Several accounts listed together only support their combined total. Retrieved
    counts are valid when the accounts' tickets are complete or the count is worded as
    retrieved ("20 tickets retrieved", "the first 20 tickets").
    """
    about = _count_subject(text, match.start(), match.end(), coverage)
    totals = [c.total for c in about]
    allowed = set() if None in totals else {sum(totals)}
    qualified = (_RETRIEVED_AFTER.match(text, match.end())
                 or _RETRIEVED_BEFORE.search(text[:match.start()]))
    if qualified or all(c.state == COMPLETE for c in about):
        ids = {c.account_id for c in about}
        scoped = (tickets if len(about) == len(coverage) else
                  [r for r in tickets if (r.get("fields") or {}).get("account_id") in ids])
        allowed |= _group_counts(scoped)
    return allowed


def miscounted_records(
    text: str, records: list[Record], coverage: Sequence[TicketCoverage] = ()
) -> list[str]:
    """Count claims ("two support tickets ...") that no grouping of the records supports.

    With ticket coverage, ticket counts are checked per account against SQL totals and,
    where valid, the retrieved records (see _ticket_counts).
    """
    problems = []
    for match in _COUNT_CLAIM.finditer(text):
        if _ID_PREFIX.search(text[:match.start()]):
            continue
        number, noun = match.group(1), match.group(2)
        count = int(number) if number.isdigit() else _NUMBER_WORDS[number.lower()]
        kind = "ticket" if noun.lower().startswith("ticket") else "document"
        same = [r for r in records if r["type"] == kind]
        relevant = sum(1 for r in same if r["relevance"] == "relevant")
        clause = re.split(r"[.;,:!?]", text[match.start():match.end() + 60])[0]
        if _RELEVANCE_CLAIM.search(clause):
            # "two tickets related to SSO" must match the relevant records exactly.
            if count != relevant:
                problems.append(f"{number} {noun}")
            continue
        if kind == "ticket" and coverage:
            allowed = _ticket_counts(text, match, same, coverage)
        else:
            allowed = {relevant} | _group_counts(same)
        if count not in allowed:
            problems.append(f"{number} {noun}")
    return problems


_COMPLETENESS_CLAIM = re.compile(
    r"\b(?:these|those|here|this|that) (?:are|is) (?:all|every)\b"
    r"|\ball (?:of )?(?:the |their |its |his |her )?(?:[a-z-]+ ){0,2}tickets\b"
    r"|\bevery (?:[a-z-]+ ){0,2}ticket\b"
    r"|\b(?:complete|full|entire|exhaustive) (?:ticket )?(?:list|set|picture|overview)\b"
    r"|\bfully (?:investigated|retrieved|reviewed|covered|checked)\b"
    r"|\b(?:no|not any) (?:other|additional|further|more|remaining) (?:[a-z-]+ ){0,2}tickets\b"
    r"|\binvestigation (?:is|was|has been) (?:now )?(?:complete|completed|finished|done)\b"
    r"|\b(?:all|both|each|every) (?:of the )?(?:requested |named |listed )?accounts? "
    r"(?:were|was|have been|has been|are|is) (?:all )?(?:fully )?"
    r"(?:investigated|checked|reviewed|covered|retrieved)\b",
    re.IGNORECASE,
)
# A negation directly before the claim ("not a complete list", "Not all tickets").
_NEGATED_CLAIM = re.compile(
    r"(?:\bnot|n't|\bnever)\s+(?:(?:a|an|the|yet|quite|necessarily)\s+)?$", re.IGNORECASE)
_NO_TICKETS = re.compile(
    r"\b(?:no|not any|zero)\b(?!\s+(?:other|additional|further|more)\b)"
    r".{0,65}\b(?:tickets|support tickets)\b"
    r"|\b(?:tickets|support tickets)\b.{0,65}\b(?:none|not found)\b",
    re.IGNORECASE,
)
_STATUS_WORD = re.compile(r"\b(open|resolved|closed|unresolved)\b", re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    return [s for line in text.splitlines() for s in re.split(r"(?<=[.!?])\s+", line)
            if s.strip()]


def _named(text: str, coverage: Sequence[TicketCoverage]) -> list[TicketCoverage]:
    folded = " ".join(text.split()).casefold()
    return [c for c in coverage
            if (c.account_name and " ".join(c.account_name.split()).casefold() in folded)
            or re.search(rf"\baccount\s*(?:id\s*)?[:#]?\s*{c.account_id}\b", text, re.I)]


def unsupported_completeness_claims(
    text: str, coverage: Sequence[TicketCoverage], outstanding: bool = False
) -> list[str]:
    """Claims that the tickets or the investigation are complete when they are not.

    Checked while any account's tickets are incomplete, or while the investigation has
    outstanding gaps or lookups (`outstanding`). Only a negation directly before the
    claim ("not a complete list") disarms it.
    """
    if not outstanding and all(c.state == COMPLETE for c in coverage):
        return []
    return [match.group(0) for sentence in _sentences(text)
            for match in _COMPLETENESS_CLAIM.finditer(sentence)
            if not _NEGATED_CLAIM.search(sentence[:match.start()])]


def _supports_no_tickets(coverage: TicketCoverage, claimed: frozenset[str]) -> bool:
    """Whether a complete zero result covers every status the claim is about.

    A claim naming no status ("no tickets") needs an unfiltered zero total.
    """
    if coverage.state != COMPLETE or coverage.total:
        return False
    return not coverage.status_filter or (
        bool(claimed) and claimed <= set(coverage.status_filter))


def unsupported_no_ticket_claims(
    text: str, coverage: Sequence[TicketCoverage]
) -> list[str]:
    """"No tickets" claims about an account whose zero total is not established.

    A claim is checked against the accounts its clause names, else those its sentence
    names, else every investigated account. Only a complete retrieval with a total of
    zero for every status the claim covers supports it.
    """
    problems = []
    for sentence in _sentences(text):
        for clause in re.split(r"[,;]|\b(?:while|but|whereas|although)\b", sentence):
            if not _NO_TICKETS.search(clause):
                continue
            claimed = frozenset(value for word in _STATUS_WORD.findall(clause)
                                for value in TICKET_STATUS_VALUES[word.lower()])
            about = _named(clause, coverage) or _named(sentence, coverage) or list(coverage)
            if not all(_supports_no_tickets(c, claimed) for c in about):
                problems.append(clause.strip())
    return problems


def unrelated_mentions(answer: str, records: list[Record]) -> list[Record]:
    """Records without a topic match that the answer mentions without a disclaimer.

    A mention is disclaimed by hedging words in the same sentence, or by appearing under a
    section heading that labels its content unrelated or unconfirmed.
    """
    clauses: list[tuple[str, bool]] = []
    hedged_section = False
    for line in answer.splitlines():
        if _HEADING.match(line):
            hedged_section = bool(_HEDGED_SECTION.search(line))
            continue
        if not line.strip():
            continue
        # Inline "Heading: content" lines carry their own label.
        label, _, rest = line.partition(":")
        inline = bool(rest) and len(label) <= 60 and bool(_HEDGED_SECTION.search(label))
        clauses.extend((clause, hedged_section or inline)
                       for clause in re.split(r"(?<=[.!?])\s+", line))
    flagged = []
    for record in records:
        if record["type"] not in {"ticket", "document"} or record["relevance"] != "no_match":
            continue
        by_id = re.compile(
            rf"\b{record['type']}\s*(?:id|#|number)?\s*(?:(?:is|equals)\s+|[:#=]\s*)?"
            rf"{record['id']}\b", re.IGNORECASE)
        title = str(record.get("title") or "").casefold()
        for clause, hedged in clauses:
            mentioned = by_id.search(clause) or (title and title in clause.casefold())
            if not mentioned or HEDGE.search(clause):
                continue
            # A section heading does not excuse a sentence that asserts a link.
            if not hedged or _ASSERTED_LINK.search(clause):
                flagged.append(record)
                break
    return flagged


def has_causal_claim(answer: str) -> bool:
    """True if a sentence asserts a cause. Headings and explicit unknowns do not count."""
    for line in answer.splitlines():
        if _HEADING.match(line):
            continue
        for clause in re.split(r"(?<=[.!?])\s+", line):
            if CAUSAL.search(clause) and not _UNKNOWN_CAUSE.search(clause):
                return True
    return False
