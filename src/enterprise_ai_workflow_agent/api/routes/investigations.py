import asyncio
import contextlib
import logging
import time
import uuid
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from enterprise_ai_workflow_agent.agent.service import InvestigationError, InvestigationService
from enterprise_ai_workflow_agent.api.investigation_schemas import (
    ErrorDetail,
    InvestigationConfig,
    InvestigationErrorResponse,
    InvestigationRequest,
    InvestigationResponse,
)
from enterprise_ai_workflow_agent.core.config import (
    MAX_ALLOWED_ROUNDS,
    MAX_QUESTION_LENGTH,
    settings,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/investigations", tags=["Investigations"])

ERROR_STATUS = {
    "ollama_unavailable": 503,
    "ollama_integration_missing": 503,
    "mcp_unavailable": 503,
    "model_inference_failed": 503,
    "model_unavailable": 404,
    "timeout": 504,
}

# How often the endpoint checks whether the client has gone away.
DISCONNECT_POLL_SECONDS = 0.5
# Non-standard "client closed request" status; the client never receives it.
CLIENT_CLOSED_REQUEST = 499


class ClientDisconnected(Exception):
    """The client closed the connection before the investigation finished."""


async def _wait_for_disconnect(request: Request) -> None:
    while not await request.is_disconnected():
        await asyncio.sleep(DISCONNECT_POLL_SECONDS)


async def run_unless_disconnected(request: Request, work):
    """Await `work`, cancelling it if the client disconnects first.

    Uvicorn does not cancel a request handler when the client goes away, so without this an
    abandoned investigation keeps its MCP subprocess and concurrency slot until it finishes
    or times out. Cancellation propagates into the service, which unwinds the run.
    """
    task = asyncio.ensure_future(work)
    watcher = asyncio.ensure_future(_wait_for_disconnect(request))
    try:
        done, _ = await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        task.cancel()
        watcher.cancel()
        raise
    if task in done:
        watcher.cancel()
        return task.result()
    if watcher.exception() is not None:
        logger.warning("Client disconnect check failed: %r", watcher.exception())
        return await task
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task
    raise ClientDisconnected


@lru_cache
def get_investigation_service() -> InvestigationService:
    return InvestigationService(settings)


Service = Annotated[InvestigationService, Depends(get_investigation_service)]


def _error_response(
    status_code: int,
    investigation_id: str,
    error: ErrorDetail,
    question: str | None = None,
    started: float | None = None,
) -> JSONResponse:
    body = InvestigationErrorResponse(
        id=investigation_id,
        question=question,
        error=error,
        duration_seconds=round(time.perf_counter() - started, 3) if started else 0.0,
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def _error_field(err: dict) -> str:
    # A JSON syntax error's location is a character offset, not a field.
    if err.get("type") == "json_invalid":
        return "body"
    return ".".join(str(part) for part in err["loc"][1:]) or "body"


def invalid_request_response(exc: RequestValidationError) -> JSONResponse:
    details = [{"field": _error_field(err), "message": err["msg"]} for err in exc.errors()]
    return _error_response(
        422,
        str(uuid.uuid4()),
        ErrorDetail(
            code="invalid_request",
            message="The investigation request is invalid.",
            details=details,
        ),
    )


@router.get("/config", response_model=InvestigationConfig)
def investigation_config() -> InvestigationConfig:
    return InvestigationConfig(
        model=settings.ollama_model,
        max_rounds=settings.investigation_max_rounds,
        max_rounds_limit=MAX_ALLOWED_ROUNDS,
        max_question_length=MAX_QUESTION_LENGTH,
        timeout_seconds=settings.investigation_timeout_seconds,
    )


@router.post(
    "",
    response_model=InvestigationResponse,
    responses={
        404: {"model": InvestigationErrorResponse, "description": "Model not installed"},
        422: {"model": InvestigationErrorResponse, "description": "Invalid request"},
        500: {"model": InvestigationErrorResponse, "description": "Unexpected failure"},
        503: {"model": InvestigationErrorResponse, "description": "Ollama or tools unavailable"},
        504: {"model": InvestigationErrorResponse, "description": "Investigation timed out"},
    },
)
async def create_investigation(payload: InvestigationRequest, service: Service, request: Request):
    investigation_id = str(uuid.uuid4())
    model = payload.model or settings.ollama_model
    max_rounds = payload.max_rounds or settings.investigation_max_rounds
    started = time.perf_counter()
    try:
        result = await run_unless_disconnected(
            request, service.run(payload.question, model, max_rounds))
    except ClientDisconnected:
        logger.info("Investigation %s cancelled: the client disconnected after %.2fs",
                    investigation_id, time.perf_counter() - started)
        return _error_response(
            CLIENT_CLOSED_REQUEST,
            investigation_id,
            ErrorDetail(code="cancelled", message="The client closed the request."),
            payload.question,
            started,
        )
    except InvestigationError as exc:
        logger.warning("Investigation %s failed: %s", investigation_id, exc.code)
        return _error_response(
            ERROR_STATUS.get(exc.code, 500),
            investigation_id,
            ErrorDetail(code=exc.code, message=exc.message),
            payload.question,
            started,
        )
    except Exception:
        logger.exception("Investigation %s failed unexpectedly", investigation_id)
        return _error_response(
            500,
            investigation_id,
            ErrorDetail(
                code="internal_error",
                message="The investigation failed unexpectedly. See the server logs.",
            ),
            payload.question,
            started,
        )
    duration = round(time.perf_counter() - started, 3)
    trace = result["tool_trace"]
    logger.info("Investigation %s finished in %.2fs", investigation_id, duration)
    return InvestigationResponse(
        id=investigation_id,
        question=payload.question,
        model=model,
        status="limited" if result.get("limited") else "completed",
        limited_reason=result.get("limited_reason"),
        answer=result["answer"],
        narrative=result.get("narrative", result["answer"]),
        answer_source=result.get("answer_source", "model"),
        answer_notice=result.get("answer_notice"),
        round_limit_reached=result.get("round_limit_reached", False),
        tool_rounds=result["tool_rounds"],
        tool_calls=len(trace),
        duration_seconds=duration,
        tool_trace=trace,
        evidence=result.get("evidence", []),
        analysis=result.get("analysis", {}),
    )
