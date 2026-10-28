"""Scoring of one investigation result against a benchmark case, and report aggregation.

Four separate dimensions:
- answer_correctness: expected facts present, forbidden claims absent, expected answer source
- evidence_grounding: citation validity (cited IDs were retrieved), scored separately for
  model-written text and for the whole answer; evidence retrieval (required records
  retrieved, forbidden ones not); and the agent's own unsupported-claim detectors applied to
  model-written text
- task_completion: completed/limited/failed status, limitation reason, ticket coverage
- tool_efficiency: required/forbidden tools, call and round budgets, redundant/failed calls

All checks are lexical. They catch the specific claims a case names; they are not a general
hallucination detector, and the unsupported-claim detectors are the same ones the agent uses,
so they guard against regressions rather than providing an independent judgement.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from enterprise_ai_workflow_agent.agent.grounding import (
    TicketCoverage,
    miscounted_records,
    unrelated_mentions,
    unsupported_completeness_claims,
    unsupported_no_ticket_claims,
)
from enterprise_ai_workflow_agent.evaluation.dataset import (
    CATEGORIES,
    Benchmark,
    Case,
    Pattern,
    benchmark_digest,
)

ANSWER, GROUNDING, COMPLETION, EFFICIENCY = (
    "answer_correctness", "evidence_grounding", "task_completion", "tool_efficiency")
DIMENSIONS = (ANSWER, GROUNDING, COMPLETION, EFFICIENCY)
MODEL_SOURCES = ("model", "synthesis")
# Rejection reasons recorded by the agent, by the prefix of the proposal's reason.
REJECTION_REASONS = {"unverified_id": "Unverified record ID",
                     "duplicate": "Duplicate call skipped"}

# A typed reference and every ID listed after it: "Ticket 101", "tickets #101, #102 and 103",
# "Ticket IDs: 101/999", "Document ID 201".
_TYPED_REF = re.compile(
    r"\b(account|ticket|product|document)s?\b\s*(?:(?:ids?|numbers?)\b\s*)?[:#=]?\s*"
    r"(#?\d+\b(?:\s*(?:,\s*(?:(?:and|or)\s+)?|(?:and|or|&|/)\s*)#?\d+\b)*)",
    re.IGNORECASE)
# "#101" with no record type.
_SHORTHAND_REF = re.compile(r"(?<![\w#])#(\d+)\b")


@dataclass(frozen=True)
class Run:
    """An investigation result plus how the scripted model was used (offline runs only)."""
    result: dict[str, Any]
    unused_replies: dict[str, int] = field(default_factory=dict)
    exhausted: list[str] = field(default_factory=list)
    error: str | None = None  # the run raised instead of returning a result


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    expected: Any
    observed: Any


@dataclass(frozen=True)
class DimensionScore:
    checks: list[Check]
    measures: dict[str, Any]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    category: str
    dimensions: dict[str, DimensionScore]
    integrity: list[Check]  # whether the run exercised the case as scripted
    observed: dict[str, Any]
    trace: list[str]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.integrity) and all(
            d.passed for d in self.dimensions.values())

    def failures(self) -> list[str]:
        groups = [("harness_integrity", self.integrity),
                  *((name, d.checks) for name, d in self.dimensions.items())]
        return [f"{group}/{c.name}: expected {c.expected!r}, observed {c.observed!r}"
                for group, checks in groups for c in checks if not c.passed]

    def explain(self) -> str:
        """Actionable failure text: case, failed checks, observed outcome and tool trace."""
        lines = [f"Case {self.case_id} ({self.category}) "
                 + ("passed." if self.passed else "failed:")]
        lines += [f"  - {failure}" for failure in self.failures()]
        lines += [f"  {key}: {value!r}" for key, value in self.observed.items()]
        lines.append("  tool trace:" if self.trace else "  tool trace: (no tool calls)")
        lines += [f"    {line}" for line in self.trace]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id, "category": self.category, "passed": self.passed,
            "failures": self.failures(),
            "dimensions": {name: {"passed": d.passed, "measures": d.measures,
                                  "checks": [vars(c) for c in d.checks]}
                           for name, d in self.dimensions.items()},
            "integrity": [vars(c) for c in self.integrity],
            "observed": self.observed, "trace": self.trace,
        }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def observed_status(result: dict[str, Any]) -> str:
    if not result:
        return "error"
    if result.get("model_error"):
        return "failed"
    return "limited" if result.get("limited") else "completed"


def retrieved_records(result: dict[str, Any]) -> set[tuple[str, int]]:
    return {(record["type"], record["id"]) for record in result.get("evidence", [])}


def cited_records(
    text: str, retrieved: set[tuple[str, int]] = frozenset()
) -> tuple[set[tuple[str, int]], list[str]]:
    """(cited records, ambiguous shorthand references) in the text.

    A typeless "#N" takes the type of the only retrieved record with ID N. With no such
    record it cites ("record", N), which is never valid; with several it is ambiguous and
    reported, not guessed.
    """
    cited: set[tuple[str, int]] = set()
    spans = []
    for match in _TYPED_REF.finditer(text):
        spans.append(match.span())
        kind = match.group(1).lower()
        cited |= {(kind, int(number)) for number in re.findall(r"\d+", match.group(2))}
    ambiguous = []
    for match in _SHORTHAND_REF.finditer(text):
        if any(start <= match.start() < end for start, end in spans):
            continue
        number = int(match.group(1))
        kinds = sorted({kind for kind, record_id in retrieved if record_id == number})
        if len(kinds) > 1:
            ambiguous.append(f"#{number} ({' or '.join(kinds)})")
        else:
            cited.add((kinds[0] if kinds else "record", number))
    return cited, ambiguous


def model_written(result: dict[str, Any]) -> bool:
    return not result.get("model_error") and result.get("answer_source") in MODEL_SOURCES


def _scoped(pattern: Pattern, result: dict[str, Any]) -> str:
    return str(result.get("narrative" if pattern.scope == "narrative" else "answer", ""))


def _coverage(result: dict[str, Any]) -> list[TicketCoverage]:
    return [TicketCoverage(**{**item, "status_filter": tuple(item["status_filter"])})
            for item in result.get("ticket_coverage", [])]


def unsupported_claims(result: dict[str, Any]) -> list[str]:
    """Unsupported counts, completeness, "no tickets" and unrelated-record claims in the
    model-written narrative. Deterministic evidence summaries contain no model claims."""
    if not model_written(result):
        return []
    text, records, coverage = result.get("narrative", ""), result.get("evidence", []), \
        _coverage(result)
    return ([f"count: {claim}" for claim in miscounted_records(text, records, coverage)]
            + [f"completeness: {claim}" for claim in unsupported_completeness_claims(
                text, coverage, outstanding=bool(result.get("limited")))]
            + [f"no tickets: {claim}" for claim in unsupported_no_ticket_claims(text, coverage)]
            + [f"unrelated: {r['type']} {r['id']}" for r in unrelated_mentions(text, records)])


def score_answer(case: Case, result: dict[str, Any]) -> DimensionScore:
    expect = case.expect.answer
    checks = []
    if expect.source_in:
        checks.append(Check("answer_source", result.get("answer_source") in expect.source_in,
                            list(expect.source_in), result.get("answer_source")))
    matched = 0
    for fact in expect.required_facts:
        found = fact.search(_scoped(fact, result))
        matched += found is not None
        checks.append(Check(f"required_fact: {fact.description}", found is not None,
                            {"scope": fact.scope, "any_of": fact.any_of},
                            found or f"not found in {fact.scope}"))
    found_claims = 0
    for claim in expect.forbidden_claims:
        found = claim.search(_scoped(claim, result))
        found_claims += found is not None
        checks.append(Check(f"forbidden_claim: {claim.description}", found is None,
                            f"absent from {claim.scope}", found or "absent"))
    return DimensionScore(checks, {
        "facts_required": len(expect.required_facts), "facts_matched": matched,
        "fact_recall": _ratio(matched, len(expect.required_facts)),
        "forbidden_claims_checked": len(expect.forbidden_claims),
        "forbidden_claims_found": found_claims,
    })


def score_grounding(case: Case, result: dict[str, Any]) -> DimensionScore:
    expect = case.expect.grounding
    retrieved = retrieved_records(result)
    # Model-written text and the whole answer (deterministic sections included) separately.
    model_text = str(result.get("narrative", "")) if model_written(result) else ""
    model_cited, model_ambiguous = cited_records(model_text, retrieved)
    cited, ambiguous = cited_records(str(result.get("answer", "")), retrieved)
    model_invalid, invalid = sorted(model_cited - retrieved), sorted(cited - retrieved)
    checks = [
        Check("model_citation_validity", not model_invalid,
              "every ID cited by the model was retrieved",
              [f"{kind} {number}" for kind, number in model_invalid] or "all valid"),
        Check("answer_citation_validity", not invalid,
              "every ID cited in the answer was retrieved",
              [f"{kind} {number}" for kind, number in invalid] or "all valid"),
    ]
    found = 0
    for ref in expect.required_evidence:
        ok = (ref.type, ref.id) in retrieved
        found += ok
        checks.append(Check(f"required_evidence: {ref}", ok, "retrieved",
                            "retrieved" if ok else "not retrieved"))
    leaked = 0
    for ref in expect.forbidden_evidence:
        ok = (ref.type, ref.id) not in retrieved
        leaked += not ok
        checks.append(Check(f"forbidden_evidence: {ref}", ok, "not retrieved",
                            "not retrieved" if ok else "retrieved"))
    problems = unsupported_claims(result)
    checks.append(Check("unsupported_claims", not problems, [], problems))
    return DimensionScore(checks, {
        "model_citations": len(model_cited), "model_invalid_citations": len(model_invalid),
        "model_citation_validity": _ratio(len(model_cited) - len(model_invalid),
                                          len(model_cited)),
        "citations": len(cited), "invalid_citations": len(invalid),
        "citation_validity": _ratio(len(cited) - len(invalid), len(cited)),
        "ambiguous_citations": sorted(set(model_ambiguous + ambiguous)),
        "evidence_required": len(expect.required_evidence), "evidence_retrieved": found,
        "evidence_recall": _ratio(found, len(expect.required_evidence)),
        "forbidden_evidence_retrieved": leaked, "unsupported_claims": len(problems),
    })


def score_completion(case: Case, result: dict[str, Any]) -> DimensionScore:
    expect = case.expect.completion
    status = observed_status(result)
    checks = [Check("status", status == expect.status, expect.status, status)]
    if expect.limited_reason:
        reason = str(result.get("limited_reason") or "")
        found = expect.limited_reason.search(reason)
        checks.append(Check(f"limited_reason: {expect.limited_reason.description}",
                            found is not None, expect.limited_reason.any_of,
                            reason or "no limitation reported"))
    states = {item["account_id"]: item["state"] for item in result.get("ticket_coverage", [])}
    matched = 0
    for item in expect.coverage:
        state = states.get(item.account_id, "not assessed")
        matched += state == item.state
        checks.append(Check(f"coverage: account {item.account_id}", state == item.state,
                            item.state, state))
    return DimensionScore(checks, {
        "status": status, "status_correct": status == expect.status,
        "coverage_expected": len(expect.coverage), "coverage_matched": matched,
        "round_limit_reached": bool(result.get("round_limit_reached")),
    })


def score_efficiency(case: Case, result: dict[str, Any]) -> DimensionScore:
    expect = case.expect.efficiency
    trace = result.get("tool_trace", [])
    executed = [entry for entry in trace if entry.get("executed")]
    failed = [entry for entry in executed if not entry.get("success")]
    keys = [json.dumps([e["tool"], e.get("normalized_arguments")], sort_keys=True, default=str)
            for e in executed]
    redundant = len(keys) - len(set(keys))
    used = {entry["tool"] for entry in executed}
    rounds = int(result.get("tool_rounds", 0))
    checks = [Check(f"required_tool: {tool}", tool in used, "executed",
                    "executed" if tool in used else "not executed")
              for tool in expect.required_tools]
    checks += [Check(f"forbidden_tool: {tool}", tool not in used, "not executed",
                     "executed" if tool in used else "not executed")
               for tool in expect.forbidden_tools]
    if expect.max_executed_calls is not None:
        checks.append(Check("max_executed_calls", len(executed) <= expect.max_executed_calls,
                            f"<= {expect.max_executed_calls}", len(executed)))
    if expect.max_rounds is not None:
        checks.append(Check("max_rounds", rounds <= expect.max_rounds,
                            f"<= {expect.max_rounds}", rounds))
    checks.append(Check("max_failed_calls", len(failed) <= expect.max_failed_calls,
                        f"<= {expect.max_failed_calls}", len(failed)))
    checks.append(Check("max_redundant_calls", redundant <= expect.max_redundant_calls,
                        f"<= {expect.max_redundant_calls}", redundant))
    rejected = [p for p in result.get("model_proposals", []) if p.get("outcome") == "rejected"]
    observed_rejections = sorted({
        f"{p['tool']} ({kind})" for p in rejected for kind, prefix in REJECTION_REASONS.items()
        if str(p.get("reason") or "").startswith(prefix)})
    for item in expect.required_rejections:
        wanted = f"{item.tool} ({item.reason})"
        checks.append(Check(f"required_rejection: {wanted}", wanted in observed_rejections,
                            "rejected", observed_rejections or "no rejections"))
    if expect.max_rejected_proposals is not None:
        checks.append(Check("max_rejected_proposals",
                            len(rejected) <= expect.max_rejected_proposals,
                            f"<= {expect.max_rejected_proposals}", len(rejected)))
    orchestrated = sum(1 for e in executed if e.get("origin") == "orchestrator")
    if expect.min_orchestrator_calls:
        checks.append(Check("min_orchestrator_calls",
                            orchestrated >= expect.min_orchestrator_calls,
                            f">= {expect.min_orchestrator_calls}", orchestrated))
    return DimensionScore(checks, {
        "executed_calls": len(executed),
        "model_calls": sum(1 for e in executed if e.get("origin") == "model"),
        "orchestrator_calls": orchestrated,
        "failed_calls": len(failed), "redundant_calls": redundant,
        "rejected_proposals": len(rejected),
        "tool_rounds": rounds,
    })


def trace_lines(result: dict[str, Any]) -> list[str]:
    lines = []
    for number, entry in enumerate(result.get("tool_trace", []), start=1):
        args = json.dumps(entry.get("normalized_arguments", entry.get("arguments")),
                          sort_keys=True, default=str)
        if entry.get("success"):
            outcome = f"ok, {entry.get('record_count', 0)} record(s)"
        else:
            outcome = ("failed" if entry.get("executed") else "not executed") + \
                f": {entry.get('error')}"
        lines.append(f"{number}. [{entry.get('origin')}] {entry.get('tool')} {args} -> {outcome}")
    return lines


def _excerpt(text: Any, limit: int = 300) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


def score_case(case: Case, run: Run) -> CaseScore:
    result = run.result
    integrity = [
        Check("run_completed", run.error is None, "no exception", run.error or "completed"),
        Check("script_not_exhausted", not run.exhausted, [], run.exhausted),
        Check("script_fully_used", not run.unused_replies, {}, run.unused_replies),
    ]
    return CaseScore(
        case_id=case.id, category=case.category,
        dimensions={ANSWER: score_answer(case, result), GROUNDING: score_grounding(case, result),
                    COMPLETION: score_completion(case, result),
                    EFFICIENCY: score_efficiency(case, result)},
        integrity=integrity,
        observed={"status": observed_status(result),
                  "answer_source": result.get("answer_source"),
                  "answer_notice": result.get("answer_notice"),
                  "limited_reason": result.get("limited_reason"),
                  "model_error": result.get("model_error"),
                  "narrative": _excerpt(result.get("narrative"))},
        trace=trace_lines(result),
    )


# --- Aggregation and reports ---------------------------------------------------------------

def _sum(scores: list[CaseScore], dimension: str, measure: str) -> int:
    return sum(int(s.dimensions[dimension].measures[measure] or 0) for s in scores)


def aggregate(scores: list[CaseScore]) -> dict[str, Any]:
    def tally(subset: list[CaseScore]) -> dict[str, Any]:
        passed = sum(s.passed for s in subset)
        return {"cases": len(subset), "passed": passed, "pass_rate": _ratio(passed, len(subset))}

    unsafe = [s for s in scores if s.category == "unsafe_conclusion"]
    blocked = sum(s.dimensions[ANSWER].passed and s.dimensions[GROUNDING].passed
                  for s in unsafe)
    executed = _sum(scores, EFFICIENCY, "executed_calls")
    return {
        "summary": tally(scores),
        "by_category": {c: tally([s for s in scores if s.category == c]) for c in CATEGORIES},
        "dimensions": {d: tally_dimension(scores, d) for d in DIMENSIONS},
        "metrics": {
            ANSWER: {
                "fact_recall": _ratio(_sum(scores, ANSWER, "facts_matched"),
                                      _sum(scores, ANSWER, "facts_required")),
                "forbidden_claims_found": _sum(scores, ANSWER, "forbidden_claims_found"),
                "unsafe_conclusions_blocked": _ratio(blocked, len(unsafe)),
            },
            GROUNDING: {
                "model_citation_validity": _ratio(
                    _sum(scores, GROUNDING, "model_citations")
                    - _sum(scores, GROUNDING, "model_invalid_citations"),
                    _sum(scores, GROUNDING, "model_citations")),
                "answer_citation_validity": _ratio(
                    _sum(scores, GROUNDING, "citations")
                    - _sum(scores, GROUNDING, "invalid_citations"),
                    _sum(scores, GROUNDING, "citations")),
                "ambiguous_citations": sum(
                    len(s.dimensions[GROUNDING].measures["ambiguous_citations"])
                    for s in scores),
                "evidence_recall": _ratio(_sum(scores, GROUNDING, "evidence_retrieved"),
                                          _sum(scores, GROUNDING, "evidence_required")),
                "forbidden_evidence_retrieved": _sum(scores, GROUNDING,
                                                     "forbidden_evidence_retrieved"),
                "unsupported_claims": _sum(scores, GROUNDING, "unsupported_claims"),
            },
            COMPLETION: {
                "status_accuracy": _ratio(_sum(scores, COMPLETION, "status_correct"),
                                          len(scores)),
                "coverage_accuracy": _ratio(_sum(scores, COMPLETION, "coverage_matched"),
                                            _sum(scores, COMPLETION, "coverage_expected")),
            },
            EFFICIENCY: {
                "executed_calls": executed,
                "mean_calls_per_case": _ratio(executed, len(scores)),
                "failed_calls": _sum(scores, EFFICIENCY, "failed_calls"),
                "redundant_calls": _sum(scores, EFFICIENCY, "redundant_calls"),
                "rejected_proposals": _sum(scores, EFFICIENCY, "rejected_proposals"),
                "mean_rounds_per_case": _ratio(_sum(scores, EFFICIENCY, "tool_rounds"),
                                               len(scores)),
            },
        },
    }


def tally_dimension(scores: list[CaseScore], dimension: str) -> dict[str, Any]:
    passed = sum(s.dimensions[dimension].passed for s in scores)
    return {"cases": len(scores), "passed": passed, "pass_rate": _ratio(passed, len(scores))}


def build_report(benchmark: Benchmark, scores: list[CaseScore], mode: str) -> dict[str, Any]:
    """A deterministic report: no timestamps or durations, so identical runs compare equal."""
    run_ids = [score.case_id for score in scores]
    full = run_ids == [case.id for case in benchmark.cases]
    return {
        "benchmark": {"name": benchmark.name, "schema_version": benchmark.schema_version,
                      "sha256": benchmark_digest(benchmark),
                      "cases_total": len(benchmark.cases),
                      "selection": "full" if full else run_ids},
        "mode": mode,
        **aggregate(scores),
        "cases": [score.to_dict() for score in scores],
    }


_RATES = ("rate", "recall", "validity", "accuracy", "blocked")


def _fmt(value: Any, key: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}" if key.endswith(_RATES) else str(value)


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    identity = report["benchmark"]
    selection = identity["selection"]
    lines = [
        f"# {identity['name']} ({report['mode']})", "",
        f"Benchmark sha256 `{identity['sha256']}`; "
        + ("all " + str(identity["cases_total"]) + " cases." if selection == "full" else
           f"**subset** of {identity['cases_total']} cases: {', '.join(selection)}."), "",
        f"**{summary['passed']}/{summary['cases']} cases passed** "
        f"({_fmt(summary['pass_rate'], 'pass_rate')}).", "",
        "| Category | Passed | Cases |", "|---|---|---|",
        *(f"| {name} | {t['passed']} | {t['cases']} |"
          for name, t in report["by_category"].items()),
        "", "| Dimension | Cases passed | Metrics |", "|---|---|---|",
        *(f"| {name} | {report['dimensions'][name]['passed']}/"
          f"{report['dimensions'][name]['cases']} | "
          + ", ".join(f"{key} {_fmt(value, key)}"
                      for key, value in report["metrics"][name].items())
          + " |" for name in DIMENSIONS),
        "", "| Case | Category | Result | Status | Answer source | Calls | Rounds |",
        "|---|---|---|---|---|---|---|",
    ]
    for case in report["cases"]:
        efficiency = case["dimensions"][EFFICIENCY]["measures"]
        lines.append(f"| {case['id']} | {case['category']} | "
                     f"{'pass' if case['passed'] else 'FAIL'} | {case['observed']['status']} | "
                     f"{case['observed']['answer_source']} | {efficiency['executed_calls']} | "
                     f"{efficiency['tool_rounds']} |")
    failed = [case for case in report["cases"] if not case["passed"]]
    if failed:
        lines += ["", "## Failures"]
        for case in failed:
            lines += ["", f"### {case['id']} ({case['category']})",
                      *(f"- {failure}" for failure in case["failures"]),
                      f"- observed: `{json.dumps(case['observed'], default=str)}`",
                      "- tool trace:", *(f"  - `{line}`" for line in case["trace"])]
    return "\n".join(lines) + "\n"
