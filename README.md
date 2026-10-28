# OpsPilot — Enterprise AI Workflow Agent

Fictional SaaS customer support investigation agent, using FastAPI, SQLite/SQLAlchemy,
Alembic, MCP, LangGraph, optional local Ollama inference, and a dependency-free web
dashboard.

**Status:** Milestone 5 complete (version 0.5.0), plus the Milestone 6A offline evaluation
benchmark ([Offline evaluation](#offline-evaluation-milestone-6a)). The dashboard,
investigation API, CLI and MCP tools work end to end **on a local machine** with a local
Ollama model and the seeded SQLite database. The project is **not production-ready**: see
[Production deployment limitations](#production-deployment-limitations).

## Run locally on Windows (PowerShell)

From the repository root:

```powershell
.venv\Scripts\python.exe -m pip install --no-cache-dir -e ".[dev,agent]"
.venv\Scripts\python.exe -m alembic upgrade head
.venv\Scripts\python.exe -m enterprise_ai_workflow_agent.db.seed
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m alembic check
.venv\Scripts\python.exe scripts\smoke_mcp.py
```

Install Ollama separately from https://ollama.com/download and pull a model:

```powershell
ollama pull qwen2.5:3b
ollama list
```

Start the Ollama service if it is not already running (`ollama serve` in a separate terminal).

### Dashboard

Start the server **from the repository root** (the default `DATABASE_URL`,
`sqlite:///./opspilot.db`, is relative to the working directory, and the MCP tool server
inherits it):

```powershell
.venv\Scripts\python.exe -m uvicorn enterprise_ai_workflow_agent.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/. Pick an example or type a question, choose the number of tool
rounds and select **Investigate** (or press Ctrl+Enter). The result shows:

- **Answer**: a short narrative and where it came from (model, tool-less synthesis, or a
  deterministic evidence summary), with a warning banner for limited investigations.
- **What the evidence supports**: rule-generated sections (relevant tickets, relevant
  documents, other records, unknowns). Record lists and counts come only from here.
- **Retrieved records**: every record returned by the tools, with its relevance label.
- **Tool execution timeline**: each call with its arguments, adjustments, errors, and
  calls added by the orchestrator.

**Stop waiting** cancels the request. The server notices the closed connection within about
a second and cancels the investigation, which shuts down its MCP subprocess and frees the
concurrency slot. API docs are at http://127.0.0.1:8000/docs.

### Investigation API

`POST /api/v1/investigations` runs one investigation synchronously and returns the answer,
evidence, analysis sections and tool trace:

```powershell
$body = '{"question": "Investigate SSO issues for Northstar Labs. Check tickets and documentation.", "max_rounds": 4}'
Invoke-RestMethod http://127.0.0.1:8000/api/v1/investigations -Method Post -ContentType application/json -Body $body
```

- Request: `question` (1-4000 characters, required), optional `model` (an installed Ollama
  tag) and `max_rounds` (1-8). Unknown fields are rejected.
- `200`: `status` is `completed` or `limited` (with `limited_reason`). Other fields:
  `answer`, `narrative`, `answer_source`, `answer_notice`, `round_limit_reached`,
  `tool_rounds`, `tool_calls`, `duration_seconds`, `tool_trace`, `evidence`, `analysis`.
- Errors use one shape, `{"id", "question", "status": "failed", "error": {"code",
  "message", "details"}, "duration_seconds"}`: `422 invalid_request`, `404
  model_unavailable`, `503 ollama_unavailable` / `ollama_integration_missing` /
  `mcp_unavailable` / `model_inference_failed`, `504 timeout`, `500 internal_error`.
  Messages never include configuration values, paths or exception details.
- `GET /api/v1/investigations/config` returns the default model, round limits, question
  length limit and timeout (no URLs or credentials).

Runtime settings (environment variables or `.env`, see `.env.example`): `OLLAMA_BASE_URL`,
`OLLAMA_MODEL`, `INVESTIGATION_TIMEOUT_SECONDS` (default 300, covers queueing too),
`INVESTIGATION_MAX_ROUNDS` (default 4), `INVESTIGATION_MAX_CONCURRENCY` (default 1;
further requests queue).

### CLI

```powershell
.venv\Scripts\python.exe -m enterprise_ai_workflow_agent.agent.cli "Investigate SSO issues for Northstar Labs. Check support tickets and documentation. Cite the evidence." --model qwen2.5:3b
```

`qwen2.5:3b` is a small starting model; tool-calling accuracy and response quality vary.
If resources allow, try a larger tool-capable model. Ollama model downloads can take
multiple GB of disk space. Confirm free space before pulling weights.

## Offline evaluation (Milestone 6A)

```powershell
.venv\Scripts\python.exe -m enterprise_ai_workflow_agent.evaluation              # all cases
.venv\Scripts\python.exe -m enterprise_ai_workflow_agent.evaluation --case U3     # one case
```

Runs the versioned benchmark [`evals/benchmark_v1.json`](evals/benchmark_v1.json) (25 cases:
6 successful investigations, 4 missing accounts, 5 incomplete evidence, 7 unsafe
conclusions, 3 tool-discipline cases) and writes `evals/results/benchmark_v1-offline.{json,md}`
(git-ignored). Runs filtered with `--case` write `...-offline-subset-<IDs>.{json,md}` and
never overwrite the full-suite report. Reports record the benchmark's content SHA-256 and
which cases ran. The exit code is 1 if any case fails; failures print the case ID,
expected vs observed values and the tool trace. A case that raises is reported as a
failure (`run_completed`) and the remaining cases still run. The same cases run in
`pytest` (`tests/test_evaluation.py`), and every case must pass.

Each case runs the real LangGraph agent against the **real MCP server in memory** on its
own SQLite fixture with explicit record IDs (`evaluation/harness.py`; production seeding is
not used). The model is **scripted**: tool-loop and synthesis replies come from the case.
No Ollama, network or subprocess is needed, and reports contain no timings, so repeated
runs are identical. A script that runs out of replies, or has replies left over, fails the
case, so a case cannot pass on a graph path that never requested its scripted text.

The benchmark covers pre-resolved and model-first (`pre_resolve: false`) runs, multi-page
ticket retrieval, rejected guessed IDs and skipped duplicate calls, failed lookups, the
round limit, tool-loop and synthesis failures, and seven unsupported-conclusion guards.
`tests/test_evaluation.py` disables each of those guards in turn (mutation tests) and
checks that the corresponding case fails, and checks that replacing a synthesis with
generic text fails every case whose answer must come from synthesis.

Metrics, per case and aggregated per category (`evaluation/metrics.py`):

| Dimension | Checks and metrics |
|---|---|
| Answer correctness | required facts present (fact recall), forbidden claims absent, expected answer source; share of unsafe cases blocked. Facts are scoped to the **narrative** (model-written text, or the deterministic notice when the answer was withheld) or to the whole **answer** including the rule-generated record sections |
| Evidence grounding | **citation validity**, scored separately for model-written text and for the whole answer: every cited ID was retrieved with the same record type (`Ticket 101`, `Tickets 101 and 103`, `Ticket IDs: 101, 103`, or `#101` when exactly one retrieved record has that ID); **evidence retrieval** (required records retrieved, forbidden ones not); unsupported-claim count in model-written text |
| Task completion | `completed`/`limited`/`failed` status, limitation reason, per-account ticket coverage |
| Tool efficiency | required/forbidden tools, executed-call and round budgets, failed and redundant (identical) calls, rejected proposals by tool and reason, minimum orchestrator follow-ups |

Expectations accept equivalent wording (each fact lists alternative regexes) and use upper
bounds rather than exact call sequences. Limitations:

- Scripted replies test orchestration, retrieval and answer guardrails, **not model
  quality**. Live-model evaluation (repeated runs against Ollama) is future work.
- All checks are lexical. They detect the specific claims each case names; they are **not
  general hallucination detection**. A wrong statement that no pattern covers goes
  unnoticed, and a narrative fact only shows that the expected wording is present, not
  that the rest of the narrative is true.
- Citation parsing is heuristic: a typeless `#N` that matches records of several types is
  reported as ambiguous and not scored; IDs written without `#` or a record type ("see
  101") are not recognised; a number listed after an ID ("Ticket 101, 2 users") is read as
  a second ID.
- The unsupported-claim metric reuses the agent's own grounding detectors, so it guards
  against regressions but is not an independent judge.
- Runs patch module-level MCP server state (restored on exit, including after exceptions)
  and must not run concurrently.

Add a case by appending it to `evals/benchmark_v1.json` (the schema is in
`evaluation/dataset.py`). Use `--verbose` to see agent logs. Changing the meaning of
existing cases calls for a new benchmark version rather than an edit.

## Architecture

Browser dashboard (or CLI) -> FastAPI `POST /api/v1/investigations` ->
`InvestigationService` (concurrency limit, timeout, cancellation; one worker thread and
event loop per run) -> LangGraph StateGraph with an Ollama chat model -> MCP stdio client ->
MCP server subprocess (one per investigation) with five read-only tools -> SQLAlchemy ->
SQLite. The result passes deterministic grounding checks before it is returned. The data
endpoints (`/api/v1/accounts`, `/api/v1/tickets`, ...) read the database directly.

Tool calls are limited to 4 rounds by default, with at most 5 tool calls per round;
tool output is truncated to 12,000 characters per call. Unknown tool names are
rejected. Errors become tool messages for the agent to interpret. The final response
includes a tool-call trace. There is no persistent memory across runs.

Each investigation's MCP subprocess is closed when the run ends, fails, times out or is
cancelled (on Windows through a job object, so the venv launcher's child process exits
too). Uvicorn does not cancel a request handler when the client disconnects, so the
endpoint polls for a closed connection and cancels the run itself.

### Reliability safeguards for small models

- **Bounded arguments.** Tool schemas publish `minimum`/`maximum` for `limit` (1-20).
  Out-of-range numbers are clamped to the schema bounds before the MCP call; the trace
  keeps the original and sent arguments plus an `adjustments` note. Record IDs are never
  changed. The MCP server still validates every call.
- **Clean tool errors.** Validation errors reach the model as short, actionable text
  (for example `Invalid arguments. limit: Input should be less than or equal to 20`).
  Unexpected server exceptions are logged and returned as a generic message.
- **Intent-aware follow-ups.** When the model stops, deterministic rules (no extra LLM
  call) check whether evidence the question explicitly asks for is missing:
  support-issue terms (ticket, incident, error, failure, ...) require tickets for the
  verified account the question names; guidance terms (troubleshoot, documentation,
  how to, next steps, ...) require a knowledge search. Missing lookups run through the
  same validated tool path, within `max_rounds`, and appear in the trace with
  `origin: "orchestrator"` and a `reason`. Dependent lookups chain (an account search,
  then its tickets) while rounds remain. Account-only questions get no ticket lookups, and
  several matching accounts are never guessed between.
- **Account discovery.** If a ticket or account question ends with no account lookup at
  all (in testing, `qwen2.5:3b` sometimes returns an empty first reply), and exactly one
  account name can be read from the question (a run of capitalised words that are not
  question words or acronyms such as SSO), `search_accounts` is called with that text.
  Account IDs still come only from the MCP result. Several candidate names, or none, make
  the investigation `limited` with an explanation; nothing is guessed. If the model's own
  account search matched nothing but did not search a name from the question on its own
  (for example both names in one query), each such name is searched separately; an empty
  result is a negative finding only for names that were actually searched.
- **Round budget.** The last tool round goes to required lookups first; the model's own
  calls in that round run alongside them within the per-round call limit. `max_rounds` is
  never raised.
- **Repeated lookups.** An identical call that already succeeded is not re-executed. If a
  model turn only repeats earlier lookups while required evidence is missing, the
  orchestrator uses that round for the missing lookup instead.
- **Transient model failures.** Ollama "token repeat" errors, empty streams and empty
  replies (no text and no tool call) are retried within a fixed budget of 4 attempts.
  Persistent empty replies are treated as the model stopping, not as a failure.
- **Answer = model narrative + deterministic sections.** Record lists and counts never
  come from the model. Every answer ends with sections rendered from the labelled records:
  *Relevant support tickets*, *Relevant knowledge documents* (or an explicit statement that
  documentation-backed guidance is unavailable), *Other retrieved records* and *Unknown or
  unconfirmed*. The dashboard shows the same sections. The model writes only a short
  explanatory narrative.
- **Final synthesis.** Evidence collection and narrative writing are separate. Once the
  required evidence is in hand and the model stops (or the orchestrator completed the
  evidence, or the tool-round limit is reached), a single synthesis call, made with the
  model *without tools*, writes the narrative. It sees only the question and the verified
  records, grouped by relevance with their IDs. It also replaces a tool-loop answer that
  fails the grounding checks. Synthesis runs at most once (transient errors are retried
  within the attempt budget), stays inside the investigation timeout, and is skipped when
  required evidence is missing.
- **Grounding for every narrative.** The tool-loop answer, a synthesis, and a synthesis
  after the round limit all pass the same checks: cited IDs must exist *with the same
  record type*; records without a topic match must not be presented as related; record
  counts must match the records (a count of "related" tickets must equal the relevant
  ones); troubleshooting instructions must cite a relevant retrieved document, otherwise
  they are removed with a visible note; causal claims get an "unconfirmed" note. A
  narrative that fails is replaced by a deterministic one (`answer_source:
  "evidence_summary"` with an `answer_notice`), never an invented conclusion.
- **Fix claims.** No record confirms that a step resolves a customer's problem, so
  present- or future-tense claims that one does ("checking the settings resolves the
  issue", "this will fix the login failures") are softened to "may help resolve/fix", with
  a visible note. Ticket status wording ("Ticket 2 was resolved", "is resolved"), hedged,
  negated and quoted text are left unchanged. The synthesis prompt also asks the model not
  to make such claims.
- **Status.** `completed`: the required evidence was retrieved and a grounded answer
  was returned. `limited`: required evidence is still missing (lookup failed, ambiguous or
  unidentifiable account, or the round budget ran out first), explained in
  `limited_reason`. Reaching the tool-round limit alone is reported as
  `round_limit_reached`, not as `limited`. Retrieving records that do not match the
  question, or finding no matching account, is a complete, negative result (the answer
  states which search matched nothing). Infrastructure failures in the tool loop are
  `failed` errors.
- **Relevance.** Records are labelled by lexical overlap with the question's topic, plus
  a few explainable groups of related terms (for example *sso*, *saml* and *login* for
  sign-in problems). Generic symptom words (failure, error, ...) alone do not make a record
  relevant when the question names a topic. Each relevant record carries a
  `relevance_basis`. A related topic does not establish a cause.

## Production deployment limitations

The application is built and tested for local, single-user use. Before any shared or
production deployment it would still need:

- **Security:** no authentication, authorization, tenant isolation or rate limiting. Bind
  the server to `127.0.0.1` (as above) and do not expose it publicly.
- **Data:** synthetic seed data only; tools are read-only and there is no approval system.
- **Scaling:** the concurrency limit and request queue live in one server process; several
  Uvicorn workers would each apply their own limit.
- **Operations:** no observability backend (the trace is per run), and the application's
  INFO logs are not shown under Uvicorn's default logging configuration.
- **Testing:** CI uses fake LLM/MCP clients (no Ollama or model download). Real-model and
  browser checks are run manually.

### Docker (current limitations)

The `Dockerfile` builds and starts the API and dashboard, and CI builds the image, but the
image is not yet a working investigation environment (planned for Milestone 8):

- It installs the package without the `agent` extra, so investigations return `503
  ollama_integration_missing`.
- `OLLAMA_BASE_URL` defaults to `127.0.0.1`, which inside a container is the container
  itself. Ollama on the host needs `host.docker.internal` or a shared network.
- The image contains neither `migrations/`/`alembic.ini` nor a database, and nothing creates
  tables at startup, so the data endpoints and tools fail unless a prepared SQLite file is
  mounted and `DATABASE_URL` points to it.
- There is no compose file, health check or Ollama service definition.

## Answer accuracy limitations

- Evidence grounding is rule-based and prompted, **not formally verified**. Check
  citations against the tool output. There is no guarantee of factual correctness or of
  immunity to prompt injection.
- Knowledge search uses lexical token overlap, not embeddings/vector retrieval.
- The guards check record IDs and types, relevance claims, record counts, uncited
  troubleshooting instructions and causal wording. They do not check the model's
  interpretation: a narrative may still overstate (for example, calling a resolved
  ticket "the solution"). The deterministic sections below the narrative are the
  authoritative record list.
- Troubleshooting advice is detected by phrasing (imperatives, "you should", "next
  steps"...). Softer wording such as "it is important to address this" is not removed.
- Fix claims are detected by phrasing too: "X resolves/fixes/will fix ..." is softened, but
  paraphrases such as "is the solution" or "addresses the root problem", and past-tense
  claims ("this resolved the issue"), are not changed.
- Account discovery reads capitalised names. Lowercase names, or a single-word name at
  the start of a sentence, are not recovered. A capitalised product name next to an
  account name makes the question ambiguous, and the run ends `limited` instead.
- Relevance is lexical, with a few hand-written groups of related terms; it is not
  semantic search.
- `qwen2.5:3b` often spends tool rounds on invented IDs (blocked) or repeated calls. Real
  investigations took about 12-160 s on a laptop (most 30-90 s); a question with no
  identifiable account can use the whole round budget before it ends `limited`.
- Reliability figures come from small manual batches (8 real runs each), not a benchmark.

## Existing MCP server

```powershell
.venv\Scripts\python.exe scripts\smoke_mcp.py
```

The server supports `search_accounts`, `get_account_tickets`, `get_ticket_details`,
`get_product_information`, and `search_knowledge`.
