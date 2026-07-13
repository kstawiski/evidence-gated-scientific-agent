# Evidence Bench

Evidence Bench is a self-hosted computational-science agent with a browser UI,
isolated workspaces, sandboxed Python and R, MCP research tools, complete run
provenance, and an A2A 1.0 interface. It can run as a standalone Docker service or
as the local-scientist backend for a delegation skill.

- Qwen3.6-27B is the primary planner, tool user, analyst, and report writer.
- Gemma 4 12B independently plans and audits the master plan and report.
- Deterministic Python code controls routing, tool policy, plan linting,
  claim–source validation, repair limits, and provenance.
- Agreement between models never overrides a failed deterministic check.

Each browser or A2A workspace keeps its inputs, run history, generated tables and
figures, claim ledger, source records, model review, logs, and SHA-256 manifest
together. Workspaces cannot read or modify one another. Python/R see their input
workspace read-only and can write only to a bounded per-call output directory.

The web workbench also provides reviewed workflow starters, a “reuse protocol”
action, immutable input and environment manifests, and a downloadable provenance
bundle. See [lessons adopted from Open Science Desktop](docs/UPSTREAM_LESSONS.md)
for the design boundary and explicit non-goals.

## Run the web service with Docker

Requirements: Docker Engine with Compose, two OpenAI-compatible model endpoints,
and a Linux host on which the nested bubblewrap workers can create mount and
process namespaces.

```bash
cp .env.example .env
# Edit .env: set independent WEB_PASSWORD and A2A_TOKEN values and model URLs.
docker compose up --build -d
curl http://127.0.0.1:8080/healthz
```

Open <http://127.0.0.1:8080> and sign in with `WEB_USERNAME` / `WEB_PASSWORD`.
The safe Compose default publishes only on loopback. Put the service behind TLS
before changing `WEB_BIND_ADDRESS` to a LAN or public interface.

The service exposes:

- `GET /.well-known/agent-card.json` — public A2A 1.0 Agent Card;
- `POST /a2a` — JSON-RPC A2A endpoint using `Authorization: Bearer <A2A_TOKEN>`;
- `/api/docs` — authenticated OpenAPI documentation;
- `/healthz` — unauthenticated container health check.

See [`docs/WEB_AND_A2A.md`](docs/WEB_AND_A2A.md) for A2A examples, deployment
details, workspace semantics, and threat boundaries.

The current milestone supports evidence retrieval, real computation, and
workspace-scoped package installation. It
can retrieve current sources through Brave Search, retrieve library documentation
through Context7, optionally inspect public web pages through a shared Chrome
DevTools service, and read bounded files inside one assigned workspace. With the
explicit `--enable-code` flag, Qwen can run complete Python and R analysis scripts
through typed tools in an offline bubblewrap sandbox. Inputs are mounted read-only;
only a per-call output directory is writable, and generated files are hashed and
linked to computed claims. Later calls can read earlier outputs at
`/prior/<execution-id>/output`; repair attempts can read prior-attempt outputs
under `/history/attempt-N/<execution-id>/output` without mutating them.

When a required library is absent, typed tools can install validated requirements
from canonical PyPI, CRAN, or Bioconductor entry points into an immutable
per-workspace environment generation. Installer hooks run in a separate networked
builder that has neither research data nor application secrets. Analysis mounts
the selected generation read-only and remains offline. Direct URLs, VCS/path
requirements, package-manager flags, and arbitrary shell/package sources are
rejected. Exact versions, an installed-tree hash, repository, generation, and lock
hash used by each computation are retained in provenance.

The model still cannot invoke an arbitrary host shell, mutate or delete raw input
data, use Git, install operating-system packages, or contact the network from
Python/R. Some packages with uncommon native system dependencies may therefore
fail explicitly until the public builder/runtime image adds those libraries.

## Why ADK 2

The planning stage is a real ADK 2 graph: task normalization fans out to blinded
Qwen and Gemma planners, joins their typed plans, deterministically lints them,
then passes them through Qwen synthesis and a separate Gemma audit. Schema-only
graph nodes call the local servers' native strict JSON-schema interface because a
live ADK 2.3 test found that the graph's third-party LLM wrapper could surface
Gemma output as raw `Content` instead of its validated schema.

Research is deliberately two-stage. An unconstrained-output ADK agent uses
`McpToolset` and the typed analysis tools under deterministic callbacks to gather
evidence; a separate strict JSON-schema call converts that bounded research packet
into the typed report. A
live compatibility test found that combining `output_schema` and MCP tools caused
this local Qwen/ADK stack to skip tool selection. The controller rejects supported
claims unless their source URL and retrieval date occur in controller-recorded MCP
evidence. Computed claims similarly fail validation unless they cite the exact
path of an artifact produced by a successful sandbox run. This preserves ADK tool
routing without trusting source-shaped model JSON.

The local Qwen tool parser can occasionally emit malformed arguments after earlier
tool calls have already succeeded. The controller fails normally when no evidence
was produced, but for computation runs it can recover from a malformed *trailing*
call by rebuilding the research packet solely from successful execution records,
captured logs, and generated output files. The recovery is explicit in the event
ledger; it never converts a failed computation into evidence.

Relevant upstream documentation:

- [ADK graph workflows](https://adk.dev/graphs/)
- [ADK MCP tools](https://adk.dev/tools-custom/mcp-tools/)
- [ADK LiteLLM connector](https://adk.dev/agents/models/litellm/)
- [Brave Search MCP](https://github.com/brave/brave-search-mcp-server)
- [Chrome DevTools MCP](https://github.com/ChromeDevTools/chrome-devtools-mcp)
- [Context7](https://github.com/upstash/context7)

Python dependencies are pinned to the versions tested on the fleet. In
particular, this project does not permit the compromised LiteLLM 1.82.7 or
1.82.8 releases described in ADK's 2026 security advisory.

Role-specific output budgets are intentionally smaller than the model serving
ceilings. A live fleet test showed that thinking-enabled Gemma could consume an
entire 1.8k-token budget before emitting JSON. Narrow Gemma planner/auditor calls
therefore disable hidden thinking and use strict schemas; Qwen planning and
synthesis retain thinking. Tool research and report formatting have separate
budgets so retrieval cannot be hidden inside a schema-only response.

## Local development

```bash
uv sync --extra dev --extra analysis
npm ci --ignore-scripts
```

On an NFS workspace, keep the Python environment on local storage:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/evidence-gated-agent-venv \
  uv sync --extra dev --link-mode=copy
```

MCP keys can be supplied as environment variables or read as data (never sourced)
from the owner-only file `~/.config/mcp-services.env`:

```text
CONTEXT7_API_KEY=...
BRAVE_API_KEY=...
```

The loader rejects symlinks and permissive local files. It also accepts a
root-owned, read-only Docker secret.

## Verify

```bash
uv run pytest -m 'not live'
uv run scientific-agent preflight --mcp context7,brave-search,chrome-devtools --enable-code
uv run pytest -m live tests/test_live.py
uv run pytest tests/test_execution.py
```

The first command is entirely offline. Preflight
starts the pinned MCP servers, discovers their tool schemas, checks both model
catalogues without printing credentials, and runs fixed Python/R sandbox probes.
Three model/MCP tests exercise all MCP schemas, the complete dual-model planning
graph, and an actual Qwen→ADK→Context7 tool call. Two local-runtime tests exercise
Python, R, output capture, network and environment isolation, read-only inputs,
timeouts, symlink rejection, and call budgets.
The deployed, case-specific scientific gates are documented in
[`evals/README.md`](evals/README.md).
The v0.3.0 release candidate passed all three deployed gates: 11/11 on a planted
Python/R effect analysis, 9/9 on corrupted-input handling, and 9/9 on an
authenticated A2A retrieval-and-grounding task.

## Run the CLI

The default enables Context7 and Brave Search. Chrome is explicit because the
fleet browser is shared external state.

```bash
uv run scientific-agent run \
  --prompt-file /private/task.txt

uv run scientific-agent run \
  --mcp context7,brave-search,chrome-devtools \
  - < /private/task.txt

uv run scientific-agent run \
  --mode simple --enable-code --mcp '' \
  "Analyze data/cohort.csv in Python, independently check group summaries in R, and save result tables"
```

`--mode simple` is the default for bounded retrieval, calculation, and evidence
extraction: one lean Qwen plan/execution path, deterministic validation, and one
final Gemma audit. It does not run dual plans or retry merely because the critic
disagrees. Use `--mode full` for genuinely multi-stage scientific design.

The base Python runtime exposes NumPy, pandas, SciPy, statsmodels, scikit-learn,
and matplotlib. The base R runtime exposes ggplot2, dplyr, survival, data.table,
and jsonlite. Additional PyPI/CRAN/Bioconductor packages are installed on demand
per workspace. Scripts read inputs at `/workspace`, earlier calls in the current
attempt at `/prior`, earlier repair attempts at `/history`, and
must write outputs below `/output`. Code-enabled preflight
imports the full Python/R analysis set inside the sandbox and fails before model
execution if the host installation is incomplete. Each call is isolated, offline,
resource-bounded, and capped by a per-attempt call budget.
Full mode allows 12 calls per attempt so ordinary Python/R corrections remain
usable; simple mode caps itself at four calls and 120 seconds per call. Tool
responses expose the remaining count and exhaustion fails closed.

Each run creates a mode-0700 directory under `runs/` containing the typed plan,
lint result, report, Gemma audit, run configuration, private size-bounded MCP
evidence artifacts, source scripts and logs, generated computation artifacts,
append-only tool event log, retrieval/computation records, and SHA-256 manifest.
Rejected drafts are retained under `attempts/attempt-N` together with the exact
deterministic findings and Gemma review that triggered repair; successful evidence
is carried forward so a citation-only repair does not force redundant analysis.
Partial files from failed scripts are retained under `rejected_output` for audit,
but they are not exposed at the normal `/prior/.../output` evidence path and cannot
support claims.
The CLI exits `0` only for supported results, `3` for a scientifically unresolved
result, and `1` for infrastructure or schema failure.

## Current boundary

This is an evidence-gated research tool, not an oracle or an unattended clinical
decision system. Human review remains required for consequential scientific and
medical decisions. See
[`docs/MVP_SCOPE.md`](docs/MVP_SCOPE.md),
[`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md), and
[`docs/DELEGATE_INTEGRATION.md`](docs/DELEGATE_INTEGRATION.md).

## Public release

The project is Apache-2.0 licensed and includes generic CI, multi-architecture
GHCR release automation for both runtime and package-builder images,
contribution/security policies, an A2A 1.0 Agent Card, and no secret or
deployment-specific endpoint in tracked configuration. Tagging `v0.3.0` in a
public GitHub repository builds and publishes the corresponding containers. A
ready-to-adapt upstream contribution dossier lives in
[`docs/A2A_ECOSYSTEM_SUBMISSION.md`](docs/A2A_ECOSYSTEM_SUBMISSION.md).
