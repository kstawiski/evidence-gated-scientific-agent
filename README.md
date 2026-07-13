# Evidence-gated scientific agent

This repository implements the dual-model local scientific worker used by the
explicit `delegate --agent local-scientist` lane. It is not selected by automatic
routing until the promotion benchmark in `docs/MVP_SCOPE.md` is complete.

- Qwen3.6-27B is the primary planner, tool user, analyst, and report writer.
- Gemma 4 12B independently plans and audits the master plan and report.
- Deterministic Python code controls routing, tool policy, plan linting,
  claim–source validation, repair limits, and provenance.
- Agreement between models never overrides a failed deterministic check.

The current milestone supports both evidence retrieval and real computation. It
can retrieve current sources through Brave Search, retrieve library documentation
through Context7, optionally inspect public web pages through a shared Chrome
DevTools service, and read bounded files inside one assigned workspace. With the
explicit `--enable-code` flag, Qwen can run complete Python and R analysis scripts
through typed tools in an offline bubblewrap sandbox. Inputs are mounted read-only;
only a per-call output directory is writable, and generated files are hashed and
linked to computed claims.

The model still cannot invoke an arbitrary host shell, install packages, mutate
the project, delete input data, use Git, or contact the network from Python/R.
Those are separate capabilities, not side effects of enabling analysis.

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

## Setup

```bash
uv sync --extra dev
npm ci --ignore-scripts
```

On an NFS workspace, keep the Python environment on local storage:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/evidence-gated-agent-venv \
  uv sync --extra dev --link-mode=copy
```

MCP keys are read as data (never sourced) from the owner-only file
`~/.config/mcp-services.env`:

```text
CONTEXT7_API_KEY=...
BRAVE_API_KEY=...
```

The loader rejects symlinks, foreign ownership, and any mode other than `0600`.

## Verify

```bash
uv run pytest -m 'not live'
uv run scientific-agent preflight --include-chrome --enable-code
uv run pytest -m live tests/test_live.py
uv run pytest tests/test_execution.py
```

The first command is entirely offline (29 tests at this milestone). Preflight
starts the pinned MCP servers, discovers their tool schemas, checks both model
catalogues without printing credentials, and runs fixed Python/R sandbox probes.
Three model/MCP tests exercise all MCP schemas, the complete dual-model planning
graph, and an actual Qwen→ADK→Context7 tool call. Two local-runtime tests exercise
Python, R, output capture, network and environment isolation, read-only inputs,
timeouts, symlink rejection, and call budgets.

## Run

The default enables Context7 and Brave Search. Chrome is explicit because the
fleet browser is shared external state.

```bash
uv run scientific-agent run \
  --prompt-file /private/task.txt

uv run scientific-agent run \
  --mcp context7,brave-search,chrome-devtools \
  - < /private/task.txt

uv run scientific-agent run \
  --enable-code --mcp '' \
  "Analyze data/cohort.csv in Python, independently check group summaries in R, and save result tables"
```

The sandboxed Python runtime currently exposes NumPy, pandas, SciPy, statsmodels,
scikit-learn, and matplotlib. The R runtime exposes the installed base packages
plus ggplot2, dplyr, survival, and data.table. Scripts read the repository at
`/workspace` and must write outputs below `/output`. Each call is isolated, offline,
resource-bounded, and capped by a per-attempt call budget.

Each run creates a mode-0700 directory under `runs/` containing the typed plan,
lint result, report, Gemma audit, run configuration, private size-bounded MCP
evidence artifacts, source scripts and logs, generated computation artifacts,
append-only tool event log, retrieval/computation records, and SHA-256 manifest.
The CLI exits `0` only for supported results, `3` for a scientifically unresolved
result, and `1` for infrastructure or schema failure.

## Current boundary

This is an MVP, not a production autonomous scientist. See
[`docs/MVP_SCOPE.md`](docs/MVP_SCOPE.md),
[`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md), and
[`docs/DELEGATE_INTEGRATION.md`](docs/DELEGATE_INTEGRATION.md).
