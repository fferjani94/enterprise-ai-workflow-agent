"""Offline runs: scripted model, the real FastMCP server in memory, isolated SQLite data.

No Ollama, network or subprocess is used. Runs patch module-level MCP server state, so
they must not execute concurrently.
"""

import asyncio
import contextlib
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

from langchain_core.messages import AIMessage
from mcp.shared.memory import create_connected_server_and_client_session
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from enterprise_ai_workflow_agent.agent.graph import investigate
from enterprise_ai_workflow_agent.db.base import Base
from enterprise_ai_workflow_agent.db.models import Account, KnowledgeDocument, Product, Ticket
from enterprise_ai_workflow_agent.evaluation.dataset import Case, Script, ScriptedReply
from enterprise_ai_workflow_agent.evaluation.metrics import Run
from enterprise_ai_workflow_agent.mcp_server import server, services

# --- Fixtures: explicit primary keys, independent of production seeding --------------------

Fixture = dict[str, list[dict[str, Any]]]

_DEFAULT: Fixture = {
    "accounts": [
        {"id": 1, "name": "Northstar Labs", "industry": "Software", "plan": "Enterprise",
         "status": "active"},
        {"id": 2, "name": "Bluebird Commerce", "industry": "Retail", "plan": "Business",
         "status": "active"},
    ],
    "products": [
        {"id": 11, "name": "CloudDesk", "version": "2.4",
         "description": "Customer support workspace"},
        {"id": 12, "name": "FlowSync", "version": "1.8", "description": "Workflow integrations"},
    ],
    "tickets": [
        {"id": 101, "account_id": 1, "product_id": 11, "subject": "Intermittent login failures",
         "description": "SSO login fails for some users.", "status": "open", "priority": "high"},
        {"id": 102, "account_id": 1, "product_id": 11, "subject": "Dashboard loading slowly",
         "description": "Dashboard sometimes takes 20 seconds.", "status": "resolved",
         "priority": "medium"},
        {"id": 103, "account_id": 2, "product_id": 12, "subject": "Webhook retry behavior",
         "description": "Retries produce duplicate events.", "status": "open",
         "priority": "medium"},
    ],
    "documents": [
        {"id": 201, "product_id": 11, "title": "SSO troubleshooting",
         "content": "Check identity provider settings and token expiry.",
         "source_url": "https://docs.example.invalid/clouddesk/sso"},
        {"id": 202, "product_id": 12, "title": "Webhook delivery",
         "content": "Webhook deliveries use exponential backoff on failures.",
         "source_url": "https://docs.example.invalid/flowsync/webhooks"},
    ],
}


def _extend(base: Fixture, **extra: list[dict[str, Any]]) -> Fixture:
    return {table: [*rows, *extra.get(table, [])] for table, rows in base.items()}


FIXTURES: dict[str, Fixture] = {
    "default": _DEFAULT,
    # A second account containing "Northstar", so the bare name is ambiguous.
    "ambiguous": _extend(_DEFAULT, accounts=[
        {"id": 3, "name": "Northstar Logistics", "industry": "Logistics", "plan": "Business",
         "status": "active"},
    ], tickets=[
        {"id": 104, "account_id": 3, "product_id": 12, "subject": "Label printing errors",
         "description": "Shipping labels fail to print.", "status": "open", "priority": "low"},
    ]),
    # 61 open Northstar tickets: more than one 50-ticket page.
    "bulk": _extend(_DEFAULT, tickets=[
        {"id": 1000 + n, "account_id": 1, "product_id": 11, "subject": f"Audit export {n}",
         "description": "Scheduled audit export request.", "status": "open", "priority": "low"}
        for n in range(1, 61)
    ]),
}

_MODELS = (("accounts", Account), ("products", Product), ("tickets", Ticket),
           ("documents", KnowledgeDocument))


def fixture_ids(name: str) -> set[tuple[str, int]]:
    """(record type, ID) of every record in a fixture."""
    kinds = {"accounts": "account", "products": "product", "tickets": "ticket",
             "documents": "document"}
    return {(kinds[table], row["id"]) for table, rows in FIXTURES[name].items() for row in rows}


def build_database(name: str) -> sessionmaker[Session]:
    """A fresh in-memory SQLite database holding one fixture."""
    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    try:
        Base.metadata.create_all(engine)
        sessions = sessionmaker(engine)
        with sessions.begin() as db:
            for table, model in _MODELS:  # parents first, for foreign keys
                db.add_all(model(**row) for row in FIXTURES[name][table])
                db.flush()
    except BaseException:
        engine.dispose()
        raise
    return sessions


# --- Scripted model ------------------------------------------------------------------------

class ScriptExhausted(RuntimeError):
    """The graph asked for more model replies than the case scripts: the script is stale."""


_ERRORS: dict[str, type[Exception]] = {"ConnectionError": ConnectionError,
                                       "TimeoutError": TimeoutError}


class _ReplyQueue:
    def __init__(self, stage: str, replies: list[ScriptedReply]) -> None:
        self.stage = stage
        self.replies = replies
        self.used = 0
        self.exhausted = False

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        if self.used >= len(self.replies):
            self.exhausted = True
            raise ScriptExhausted(f"no scripted {self.stage} reply left "
                                  f"({len(self.replies)} scripted)")
        reply = self.replies[self.used]
        self.used += 1
        if reply.error:
            raise _ERRORS[reply.error](f"scripted {reply.error}")
        return AIMessage(content=reply.text, tool_calls=[
            {"name": call.tool, "args": dict(call.args), "id": f"{self.stage}-{self.used}-{i}",
             "type": "tool_call"}
            for i, call in enumerate(reply.tool_calls)
        ])


class ScriptedModel:
    """Chat-model stand-in: the bound model replays the tool loop, the raw model synthesis."""

    def __init__(self, script: Script) -> None:
        self.tool_loop = _ReplyQueue("tool_loop", list(script.tool_loop))
        self.synthesis = _ReplyQueue("synthesis", list(script.synthesis))

    def bind_tools(self, specs: list[dict[str, Any]]) -> _ReplyQueue:
        return self.tool_loop

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        return await self.synthesis.ainvoke(messages)

    def unused(self) -> dict[str, int]:
        return {queue.stage: len(queue.replies) - queue.used
                for queue in (self.tool_loop, self.synthesis)
                if len(queue.replies) > queue.used}

    def exhausted(self) -> list[str]:
        return [queue.stage for queue in (self.tool_loop, self.synthesis) if queue.exhausted]


# --- Runner --------------------------------------------------------------------------------

def _failing(tool: str) -> Callable[..., Any]:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"injected {tool} failure")
    return fail


async def run_case_async(case: Case) -> Run:
    sessions = build_database(case.fixture)
    model = ScriptedModel(case.script)
    try:
        with contextlib.ExitStack() as patches:
            patches.enter_context(mock.patch.object(server, "SessionLocal", sessions))
            for tool in case.failing_tools:
                patches.enter_context(mock.patch.object(services, tool, _failing(tool)))
            async with create_connected_server_and_client_session(
                    server.mcp._mcp_server) as client:
                tools = (await client.list_tools()).tools
                result = await investigate(case.question, model, client, tools,
                                           max_rounds=case.max_rounds,
                                           pre_resolve=case.pre_resolve)
    except BaseExceptionGroup as group:
        # The in-memory MCP session's task group wraps the agent's own exception.
        leaf = _single_leaf(group)
        if leaf is group:
            raise
        raise leaf from group
    finally:
        sessions.kw["bind"].dispose()
    return Run(result=result, unused_replies=model.unused(), exhausted=model.exhausted())


def _single_leaf(exc: BaseException) -> BaseException:
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


def run_case(case: Case) -> Run:
    return asyncio.run(run_case_async(case))


def run_case_safely(case: Case) -> Run:
    """Like run_case, but an exception becomes a failed Run so later cases still run."""
    try:
        return run_case(case)
    except Exception as exc:
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        where = f"{Path(frame.filename).name}:{frame.lineno} in {frame.name}"
        message = "".join(traceback.format_exception_only(exc)).strip()
        return Run(result={}, error=f"{message} (at {where})")
