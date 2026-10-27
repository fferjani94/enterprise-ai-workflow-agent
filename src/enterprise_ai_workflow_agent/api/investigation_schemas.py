from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from enterprise_ai_workflow_agent.core.config import MAX_ALLOWED_ROUNDS, MAX_QUESTION_LENGTH


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


class InvestigationConfig(BaseModel):
    model: str
    max_rounds: int
    max_rounds_limit: int
    max_question_length: int
    timeout_seconds: float
