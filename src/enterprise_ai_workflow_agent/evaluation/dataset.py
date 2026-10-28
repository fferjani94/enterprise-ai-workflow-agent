"""Versioned benchmark cases: question, scripted model replies and expected outcomes."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = 1
# Valid for a source checkout; installed packages must pass the path explicitly.
DEFAULT_BENCHMARK = Path(__file__).resolve().parents[3] / "evals" / "benchmark_v1.json"

Category = Literal["success", "missing_account", "incomplete_evidence", "unsafe_conclusion",
                   "tool_discipline"]
CATEGORIES: tuple[str, ...] = Category.__args__
Status = Literal["completed", "limited", "failed"]
RecordType = Literal["account", "ticket", "product", "document"]
AnswerSource = Literal["model", "synthesis", "evidence_summary"]
CoverageState = Literal["complete", "partial", "failed", "unretrieved"]
ToolName = Literal["search_accounts", "get_account_tickets", "get_ticket_details",
                   "get_product_information", "search_knowledge"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScriptedToolCall(_Strict):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ScriptedReply(_Strict):
    """One model reply: text and/or tool calls, or a raised (non-transient) error."""
    text: str = ""
    tool_calls: list[ScriptedToolCall] = Field(default_factory=list)
    error: Literal["ConnectionError", "TimeoutError"] | None = None


class Script(_Strict):
    """Offline model replies: the tool-calling loop and the tool-less synthesis call."""
    tool_loop: list[ScriptedReply] = Field(default_factory=list)
    synthesis: list[ScriptedReply] = Field(default_factory=list)


class Pattern(_Strict):
    """A claim matched by any of several case-insensitive regexes (equivalent wordings).

    scope "narrative" matches only the narrative (model-written unless the answer source is
    evidence_summary); "answer" also matches the deterministic record sections.
    """
    description: str
    any_of: list[str] = Field(min_length=1)
    scope: Literal["answer", "narrative"] = "answer"

    @field_validator("any_of")
    @classmethod
    def _compiles(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regex {pattern!r}: {exc}") from None
        return patterns

    def search(self, text: str) -> str | None:
        """The matched text, or None. Whitespace is normalised before matching."""
        flat = " ".join(text.split())
        for pattern in self.any_of:
            if match := re.search(pattern, flat, re.IGNORECASE):
                return match.group(0)
        return None


class RecordRef(_Strict):
    type: RecordType
    id: int

    def __str__(self) -> str:
        return f"{self.type} {self.id}"


class CoverageExpectation(_Strict):
    account_id: int
    state: CoverageState


class AnswerExpectation(_Strict):
    """Answer correctness: what the user-visible answer must and must not state."""
    source_in: list[AnswerSource] = Field(default_factory=list)  # empty: any source
    required_facts: list[Pattern] = Field(default_factory=list)
    forbidden_claims: list[Pattern] = Field(default_factory=list)


class GroundingExpectation(_Strict):
    """Evidence retrieval targets; citation validity is checked for every case."""
    required_evidence: list[RecordRef] = Field(default_factory=list)
    forbidden_evidence: list[RecordRef] = Field(default_factory=list)


class CompletionExpectation(_Strict):
    status: Status
    limited_reason: Pattern | None = None
    coverage: list[CoverageExpectation] = Field(default_factory=list)


class RejectionExpectation(_Strict):
    """A model tool proposal that must be rejected before reaching MCP, and why."""
    tool: ToolName
    reason: Literal["unverified_id", "duplicate"]


class EfficiencyExpectation(_Strict):
    """Bounds rather than exact sequences, so equivalent tool plans still pass."""
    required_tools: list[ToolName] = Field(default_factory=list)
    forbidden_tools: list[ToolName] = Field(default_factory=list)
    max_executed_calls: int | None = Field(default=None, ge=0)
    max_failed_calls: int = Field(default=0, ge=0)
    max_redundant_calls: int = Field(default=0, ge=0)
    max_rounds: int | None = Field(default=None, ge=0)
    required_rejections: list[RejectionExpectation] = Field(default_factory=list)
    max_rejected_proposals: int | None = Field(default=None, ge=0)
    min_orchestrator_calls: int = Field(default=0, ge=0)


class Expectation(_Strict):
    answer: AnswerExpectation = Field(default_factory=AnswerExpectation)
    grounding: GroundingExpectation = Field(default_factory=GroundingExpectation)
    completion: CompletionExpectation
    efficiency: EfficiencyExpectation = Field(default_factory=EfficiencyExpectation)


class Case(_Strict):
    id: str = Field(pattern=r"^[A-Z][0-9]+$")
    category: Category
    description: str
    question: str = Field(min_length=1, max_length=4000)
    fixture: str = "default"
    max_rounds: int = Field(default=4, ge=1, le=8)
    pre_resolve: bool = True
    # MCP tools whose service function raises, to exercise failed retrievals.
    failing_tools: list[ToolName] = Field(default_factory=list)
    script: Script
    expect: Expectation


class Benchmark(_Strict):
    schema_version: Literal[1]
    name: str
    description: str
    cases: list[Case] = Field(min_length=1)

    @field_validator("cases")
    @classmethod
    def _unique_ids(cls, cases: list[Case]) -> list[Case]:
        ids = [case.id for case in cases]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate case IDs: {', '.join(duplicates)}")
        return cases


def load_benchmark(path: Path | str = DEFAULT_BENCHMARK) -> Benchmark:
    return Benchmark.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def benchmark_digest(benchmark: Benchmark) -> str:
    """SHA-256 of the canonical case content; independent of file formatting and line endings."""
    canonical = json.dumps(benchmark.model_dump(mode="json"), sort_keys=True,
                           separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
