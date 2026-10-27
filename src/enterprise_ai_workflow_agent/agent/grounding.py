"""Deterministic relevance labelling and answer checks for retrieved evidence.

No LLM calls: a record is "relevant" only when its retrieved title or description shares a
term with the topic of the question. Matching is lexical and deliberately conservative.
"""

import re
from typing import Any

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
    r"\b(open|resolved|closed)\s+(?:support\s+)?(?:tickets?|incidents?|issues?|cases?)\b",
    re.IGNORECASE,
)
_STATUS_VALUES = {"open": {"open"}, "resolved": {"resolved", "closed"},
                  "closed": {"closed", "resolved"}}


def requested_ticket_statuses(question: str) -> frozenset[str]:
    """Ticket statuses the question explicitly asks for; empty when it sets no filter."""
    return frozenset(value for word in _STATUS_FILTER.findall(question)
                     for value in _STATUS_VALUES[word.lower()])


def build_analysis(
    records: list[Record],
    *,
    expects_tickets: bool = False,
    issue_question: bool = False,
    notes: tuple[str, ...] = (),
    ticket_statuses: frozenset[str] = frozenset(),
) -> dict[str, list[dict[str, Any]]]:
    """Deterministic answer sections built only from verified, labelled records.

    findings     relevant support tickets (or all tickets when relevance is unassessed)
    guidance     relevant knowledge documents: general documentation, not a diagnosis
    context      accounts and products that identify what was investigated, and tickets
                 outside the statuses the question asks for (ticket_statuses)
    unsupported  retrieved records with no topic match: not evidence of this issue
    unconfirmed  what the records do not establish, plus caller-supplied notes
    No model output is used, so record lists and counts here are authoritative.
    """
    retrieved = [r for r in records if r["type"] == "ticket"]
    other_status = [r for r in retrieved if ticket_statuses and str(
        (r.get("fields") or {}).get("status") or "").casefold() not in ticket_statuses]
    tickets = [r for r in retrieved if r not in other_status]
    documents = [r for r in records if r["type"] == "document"]
    findings, guidance, context, unsupported, unconfirmed = [], [], [], [], []

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


def miscounted_records(text: str, records: list[Record]) -> list[str]:
    """Count claims ("two support tickets ...") that no grouping of the records supports."""
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
        allowed = {len(same), relevant} | {
            sum(1 for r in same if r["relevance"] == label)
            for label in ("relevant", "no_match", "unassessed")
        } | {
            sum(1 for r in same if (r.get("fields") or {}).get(field) == value)
            for r0 in same for field, value in (r0.get("fields") or {}).items()
            if field in {"status", "priority"}
        }
        if count not in allowed:
            problems.append(f"{number} {noun}")
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
