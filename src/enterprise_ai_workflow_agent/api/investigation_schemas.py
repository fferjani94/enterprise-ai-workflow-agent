from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from enterprise_ai_workflow_agent.core.config import MAX_ALLOWED_ROUNDS, MAX_QUESTION_LENGTH

ToolOutcome = Literal[
    "executed_success", "executed_error", "invalid_arguments", "unknown_tool",
    "unverified_id_rejected", "duplicate_rejected", "redundant_page_rejected",
    "call_limit_rejected",
]
ErrorCategory = Literal["timeout", "connection", "transient", "other"]


class InvestigationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    model: str | None = Field(
        default=None, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
        description="Ollama model tag. Defaults to OLLAMA_MODEL.",
    )
    max_rounds: int | None = Field(
        default=None, ge=1, le=MAX_ALLOWED_ROUNDS,
        description="Maximum tool rounds. Defaults to INVESTIGATION_MAX_ROUNDS.",
    )


class ToolTraceEntry(BaseModel):
    tool: str
    origin: Literal["model", "orchestrator"] = Field(
        default="model",
        description="'orchestrator' marks deterministic lookups added when the model stopped "
        "before retrieving evidence the question asks for.",
    )
    reason: str | None = Field(default=None, description="Why the orchestrator made the call.")
    arguments: Any = None
    normalized_arguments: Any = None
    adjustments: list[str] = Field(
        default_factory=list, description="Arguments clamped to the tool schema's bounds."
    )
    success: bool
    executed: bool | None = Field(
        default=None, description="Whether the MCP tool was actually called. False for "
        "calls rejected or skipped before execution.")
    record_count: int = 0
    error: str | None = None
    outcome: ToolOutcome | None = Field(
        default=None, description="What happened to the call. 'executed_*' calls reached the "
        "MCP tool; '*_rejected', 'unknown_tool' and unexecuted 'invalid_arguments' were "
        "skipped before execution. Independent of origin. Absent in older results.")
    trigger: Literal["pre_resolution", "followup"] | None = Field(
        default=None, description="For orchestrator calls: searches for named accounts before "
        "the first model turn, or follow-ups for missing evidence.")
    latency_ms: float | None = Field(
        default=None, description="MCP round-trip time, for executed calls only.")
    error_category: ErrorCategory | None = Field(
        default=None, description="Kind of failure when the tool call raised.")


class ModelProposalEntry(BaseModel):
    """Diagnostics: a tool call the model proposed, and what became of it."""

    turn: int = Field(description="Model planning turn that proposed the call.")
    round: int | None = Field(
        default=None, description="Tool round that executed, rejected or superseded it.")
    tool: str
    arguments: Any = None
    outcome: Literal["executed", "rejected", "superseded"]
    reason: str | None = None
    id_verification: Literal["before_batch", "same_batch", "unverified"] | None = Field(
        default=None, description="For ID-based calls: whether the ID was verified before "
        "the model's batch, only within that batch, or never.")


class ModelTurnEntry(BaseModel):
    """Diagnostics: the kind of reply one tool-loop model turn produced."""

    turn: int
    response: Literal["tool_calls", "text", "empty", "error"]
    tool_calls: int = 0
    invalid_tool_calls: int = Field(
        default=0, description="Tool calls whose arguments could not be parsed.")
    discarded: bool = Field(
        default=False, description="Removed from the conversation by an orchestrator "
        "follow-up.")


class EvidenceRecord(BaseModel):
    type: Literal["account", "ticket", "product", "document"]
    id: int
    title: str | None = None
    description: str | None = None
    fields: dict[str, Any] = Field(default_factory=dict)
    relevance: Literal["context", "relevant", "no_match", "unassessed"] = "unassessed"
    relevance_basis: str | None = Field(
        default=None, description="Why the record was labelled relevant (shared or related "
        "topic terms). A shared topic does not establish a cause.",
    )


class EvidenceRef(BaseModel):
    type: Literal["account", "ticket", "product", "document"]
    id: int


class AnalysisStatement(BaseModel):
    text: str
    evidence: list[EvidenceRef] = Field(default_factory=list)


class InvestigationAnalysis(BaseModel):
    """Deterministic answer sections built from the verified records, not by the model."""

    findings: list[AnalysisStatement] = Field(
        default_factory=list, description="Relevant support tickets.")
    guidance: list[AnalysisStatement] = Field(
        default_factory=list, description="Relevant knowledge documents (general guidance).")
    context: list[AnalysisStatement] = Field(
        default_factory=list, description="Accounts and products that identify what was "
        "investigated.")
    unsupported: list[AnalysisStatement] = Field(
        default_factory=list, description="Retrieved records with no topic match.")
    unconfirmed: list[AnalysisStatement] = Field(
        default_factory=list, description="What the records do not establish.")


class PhaseTimings(BaseModel):
    """Milliseconds; None when the phase was not measured or not reached.

    Stage timings include retries and the elapsed time of calls cancelled in flight.
    """

    total: float | None = Field(default=None, description="Service time, queue wait included.")
    queue_wait: float | None = None
    preflight: float | None = None
    mcp_startup: float | None = Field(
        default=None, description="Includes a startup that failed or was cancelled.")
    graph: float | None = None
    llm_tool_loop: float | None = None
    llm_synthesis: float | None = None
    tool_execution: float | None = Field(
        default=None, description="All MCP calls, including any cancelled in flight.")
    orchestration_overhead: float | None = Field(
        default=None, description="Graph time spent outside model and tool calls.")


class LlmMetrics(BaseModel):
    calls: int = Field(default=0, description="Logical model calls; retries excluded.")
    attempts: int = 0
    retries: int = 0
    ok_attempts: int = 0
    error_attempts: int = Field(default=0, description="Non-retryable model failures.")
    transient_error_attempts: int = Field(
        default=0, description="Retryable provider errors.")
    empty_attempts: int = Field(
        default=0, description="Replies with no text or tool call; retried, not failures.")
    cancelled_attempts: int = Field(
        default=0, description="In flight when the investigation was cancelled.")


class TokenCounts(BaseModel):
    """Provider-reported tokens summed over attempts; None when never reported.

    Failed and empty attempts are included. The conversation is re-sent every turn, so
    input tokens count processed prompt tokens.
    """

    input: int | None = None
    output: int | None = None
    total: int | None = Field(
        default=None, description="Reported totals, else input + output per attempt.")
    reported_attempts: int = 0
    unreported_attempts: int = Field(
        default=0, description="Attempts that produced a reply without usage data.")


class TokenUsage(TokenCounts):
    by_stage: dict[Literal["tool_loop", "synthesis"], TokenCounts] = Field(
        default_factory=dict, description="The same counts per model stage.")


class LlmAttemptMetrics(BaseModel):
    stage: Literal["tool_loop", "synthesis"]
    call: int = Field(description="Logical model call; retries share the number.")
    turn: int | None = None
    attempt: int
    prompt_variant: Literal["with_system", "without_system"] | None = None
    outcome: Literal["ok", "empty", "transient_error", "error", "cancelled"]
    error_category: ErrorCategory | None = None
    latency_ms: float
    tool_calls: int = 0
    invalid_tool_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    provider_load_ms: float | None = None
    provider_prompt_eval_ms: float | None = None
    provider_eval_ms: float | None = None
    provider_total_ms: float | None = None
    done_reason: Literal["stop", "length", "load", "unknown"] | None = None


class SynthesisMetrics(BaseModel):
    status: Literal["completed", "failed", "skipped", "cancelled", "not_reached"] = \
        "not_reached"
    latency_ms: float | None = None
    attempts: int = 0


class LatencyStats(BaseModel):
    count: int = 0
    total_ms: float = 0.0
    max_ms: float | None = None
    mean_ms: float | None = None


class ToolMetrics(BaseModel):
    trace_entries: int = Field(default=0, description="Equals the response's tool_calls.")
    executed: int = Field(default=0, description="Completed MCP calls; cancelled excluded.")
    cancelled: int = Field(
        default=0, description="MCP calls in flight when the investigation was cancelled.")
    by_outcome: dict[str, int] = Field(default_factory=dict)
    by_origin: dict[str, int] = Field(default_factory=dict)
    by_trigger: dict[str, int] = Field(default_factory=dict)
    rejected_proposals: int | None = None
    superseded_proposals: int | None = None
    latency: LatencyStats = Field(
        default_factory=LatencyStats, description="Completed MCP calls only.")
    latency_by_tool: dict[str, LatencyStats] = Field(default_factory=dict)


class InvestigationMetrics(BaseModel):
    """Diagnostics: timings, token usage and tool outcomes. No prompts or tool data."""

    schema_version: Literal[1] = 1
    partial: bool = Field(
        default=False, description="Collected from an investigation that did not finish.")
    timings_ms: PhaseTimings = Field(default_factory=PhaseTimings)
    llm: LlmMetrics = Field(default_factory=LlmMetrics)
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    llm_attempts: list[LlmAttemptMetrics] = Field(default_factory=list)
    synthesis: SynthesisMetrics = Field(default_factory=SynthesisMetrics)
    tools: ToolMetrics = Field(default_factory=ToolMetrics)


class InvestigationResponse(BaseModel):
    id: str
    question: str
    model: str
    status: Literal["completed", "limited"]
    limited_reason: str | None = Field(
        default=None,
        description="Why a 'limited' investigation is incomplete: required evidence was "
        "never retrieved.",
    )
    answer: str
    narrative: str | None = Field(
        default=None, description="The answer without the appended evidence list."
    )
    answer_source: Literal["model", "synthesis", "evidence_summary"] = Field(
        default="model",
        description="'model': the tool-calling model's answer; 'synthesis': written by the "
        "tool-less synthesis step from verified records; 'evidence_summary': deterministic "
        "list of verified records with no model-generated conclusion.",
    )
    answer_notice: str | None = Field(
        default=None, description="Why a deterministic evidence summary was returned."
    )
    round_limit_reached: bool = Field(
        default=False, description="The tool-round budget was used up. This alone does not "
        "make an investigation 'limited'.",
    )
    tool_rounds: int
    tool_calls: int
    duration_seconds: float
    tool_trace: list[ToolTraceEntry]
    model_proposals: list[ModelProposalEntry] = Field(
        default_factory=list, description="Diagnostics: every tool call the model proposed, "
        "including calls rejected or replaced by orchestrator lookups. Not evidence.")
    model_turns: int | None = Field(
        default=None, description="Diagnostics: tool-loop model turns (synthesis excluded).")
    model_turn_log: list[ModelTurnEntry] = Field(
        default_factory=list, description="Diagnostics: one entry per tool-loop model turn.")
    evidence: list[EvidenceRecord]
    analysis: InvestigationAnalysis = Field(default_factory=InvestigationAnalysis)
    metrics: InvestigationMetrics | None = Field(
        default=None, description="Diagnostics: timings, token usage and tool outcomes.")


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: list[dict[str, str]] | None = None


class InvestigationErrorResponse(BaseModel):
    id: str
    question: str | None = None
    status: Literal["failed"] = "failed"
    error: ErrorDetail
    duration_seconds: float = 0.0
    metrics: InvestigationMetrics | None = Field(
        default=None, description="Partial diagnostics for timeouts and model failures.")


class InvestigationConfig(BaseModel):
    model: str
    max_rounds: int
    max_rounds_limit: int
    max_question_length: int
    timeout_seconds: float
