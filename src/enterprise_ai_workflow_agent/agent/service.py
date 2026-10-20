"""Runs the existing LangGraph investigation against Ollama and a stdio MCP server."""

import asyncio
import contextlib
import functools
import logging
import sys
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from enterprise_ai_workflow_agent.agent.graph import investigate
from enterprise_ai_workflow_agent.core.config import Settings

logger = logging.getLogger(__name__)

# Exception class names reported by the graph when Ollama cannot be reached mid-run.
CONNECTION_ERROR_TYPES = {
    "ConnectError", "ConnectTimeout", "ConnectionError", "ConnectionRefusedError",
    "ReadTimeout", "RemoteProtocolError", "ReadError",
}

AgentRunner = Callable[[str, str, int], Awaitable[dict[str, Any]]]


class InvestigationError(Exception):
    """An expected failure with a client-safe code and message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def build_chat_model(settings: Settings, model_name: str) -> Any:
    from langchain_ollama import ChatOllama  # optional "agent" extra

    return ChatOllama(
        model=model_name,
        base_url=settings.ollama_base_url,
        temperature=settings.ollama_temperature,
        num_predict=settings.ollama_num_predict,
        num_ctx=settings.ollama_num_ctx,
    )


def mcp_server_params(settings: Settings) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "enterprise_ai_workflow_agent.mcp_server.server"],
        # The child gets a minimal environment, so pass the database explicitly.
        env={"DATABASE_URL": settings.database_url},
    )


@contextlib.asynccontextmanager
async def mcp_session(settings: Settings) -> AsyncIterator[ClientSession]:
    async with stdio_client(mcp_server_params(settings)) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            yield client


async def check_ollama(
    base_url: str,
    model_name: str,
    timeout: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    """Fail fast, with a precise error, if Ollama is down or the model is not installed."""
    unavailable = InvestigationError(
        "ollama_unavailable",
        "Cannot reach the Ollama service. Start it with 'ollama serve' and check "
        "OLLAMA_BASE_URL.",
    )
    try:
        async with httpx.AsyncClient(
            base_url=base_url, timeout=timeout, transport=transport
        ) as http:
            response = await http.get("/api/tags")
        response.raise_for_status()
        installed = {
            name
            for item in response.json().get("models", [])
            for name in (item.get("name"), item.get("model"))
            if name
        }
    except (httpx.HTTPError, httpx.InvalidURL, ValueError, AttributeError) as exc:
        logger.warning("Ollama preflight failed: %s", type(exc).__name__)
        raise unavailable from None
    if model_name not in installed and f"{model_name}:latest" not in installed:
        raise InvestigationError(
            "model_unavailable",
            f"Model '{model_name}' is not installed in Ollama. "
            f"Install it with: ollama pull {model_name}",
        )


async def run_agent(
    settings: Settings, question: str, model_name: str, max_rounds: int
) -> dict[str, Any]:
    """One investigation: one chat model and one MCP session, both closed on exit."""
    try:
        model = build_chat_model(settings, model_name)
    except ImportError:
        raise InvestigationError(
            "ollama_integration_missing",
            'The Ollama integration is not installed. Run: pip install -e ".[agent]"',
        ) from None
    await check_ollama(
        settings.ollama_base_url, model_name, settings.ollama_preflight_timeout_seconds
    )
    stage = "mcp"
    try:
        async with mcp_session(settings) as client:
            tools = (await client.list_tools()).tools
            stage = "agent"
            result = await investigate(question, model, client, tools, max_rounds=max_rounds)
    except Exception as exc:
        if stage != "mcp":
            raise
        logger.exception("MCP server failed to start")
        raise InvestigationError(
            "mcp_unavailable", "The tool server could not be started."
        ) from exc
    if result["model_error"]:
        code = (
            "ollama_unavailable"
            if result["model_error"] in CONNECTION_ERROR_TYPES
            else "model_inference_failed"
        )
        raise InvestigationError(
            code,
            "The local model stopped responding after several attempts. Small local "
            "models fail intermittently, so try again. If it keeps failing, check the "
            "server logs and that Ollama can generate responses.",
        )
    return result


class _Job:
    """Lets the event loop cancel the agent task running in a worker thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._cancelled = False

    def attach(self, loop: asyncio.AbstractEventLoop, task: asyncio.Task) -> bool:
        with self._lock:
            self._loop, self._task = loop, task
            return not self._cancelled

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            if self._loop is not None and self._task is not None:
                with contextlib.suppress(RuntimeError):  # loop already closed
                    self._loop.call_soon_threadsafe(self._task.cancel)


class InvestigationService:
    """Bounded-concurrency, time-limited investigation runner.

    Each investigation runs in a worker thread with its own event loop, so the
    FastAPI loop is never blocked and (on Windows) the loop always supports
    subprocesses. The timeout is enforced inside that loop, so MCP sessions and
    subprocesses are unwound by normal cancellation.
    """

    def __init__(self, settings: Settings, agent: AgentRunner | None = None) -> None:
        self.settings = settings
        self._agent: AgentRunner = agent or functools.partial(run_agent, settings)
        self._slots = asyncio.Semaphore(settings.investigation_max_concurrency)

    async def run(self, question: str, model: str, max_rounds: int) -> dict[str, Any]:
        timeout = self.settings.investigation_timeout_seconds
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout)
        except TimeoutError:
            raise InvestigationError(
                "timeout",
                f"The service stayed busy for {timeout:g} seconds. Try again shortly.",
            ) from None
        remaining = deadline - loop.time()
        if remaining <= 0:
            self._slots.release()
            raise InvestigationError(
                "timeout", f"The investigation did not start within {timeout:g} seconds."
            )
        job = _Job()
        work = asyncio.ensure_future(
            asyncio.to_thread(self._run_in_thread, question, model, max_rounds, remaining, job)
        )
        # The slot is released only when the thread has really finished.
        work.add_done_callback(self._finish)
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            job.cancel()
            raise

    def _finish(self, work: asyncio.Future) -> None:
        self._slots.release()
        if not work.cancelled():
            work.exception()  # mark as retrieved; the caller already saw it

    def _run_in_thread(
        self, question: str, model: str, max_rounds: int, timeout: float, job: _Job
    ) -> dict[str, Any]:
        async def main() -> dict[str, Any]:
            if not job.attach(asyncio.get_running_loop(), asyncio.current_task()):
                raise asyncio.CancelledError
            try:
                async with asyncio.timeout(timeout) as scope:
                    return await self._agent(question, model, max_rounds)
            except TimeoutError:
                if not scope.expired():
                    raise
                raise InvestigationError(
                    "timeout",
                    f"The investigation did not finish within "
                    f"{self.settings.investigation_timeout_seconds:g} seconds. "
                    "Try a narrower question or fewer tool rounds.",
                ) from None

        factory = asyncio.ProactorEventLoop if sys.platform == "win32" else asyncio.new_event_loop
        with asyncio.Runner(loop_factory=factory) as runner:
            return runner.run(main())
