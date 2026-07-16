# Evidence Bench

Evidence Bench is a self-hosted computational-science agent with a browser UI,
isolated workspaces, sandboxed Python and R, MCP research tools, complete run
provenance, and an A2A 1.0 interface. It can run as a standalone Docker service or
as the explicit `umed-task` backend for a delegation skill. The historical
`local-scientist` route is a deprecated compatibility alias.

- Qwen3.6-27B is the primary planner, tool user, analyst, and report writer.
- A separately served Gemma 4 instruct model independently plans and audits the
  master plan and report. Its endpoint and exact model are deployment settings
  (`GEMMA_BASE_URL` and `GEMMA_MODEL`), so the critic need not run on the
  application host or use the example model from `.env.example`.
- Deterministic Python code controls routing, tool policy, plan linting,
  claim–source validation, repair limits, and provenance.
- Agreement between models never overrides a failed deterministic check.

Gemma is the sole image-understanding model. Generated PNG, JPEG, and WebP figures
are supplied only to Gemma in bounded batches; Qwen never receives raster images
or makes visual-quality judgments. A confined sandbox worker also extracts OCR
text and word geometry, while deterministic code compares rendered labels with
machine-readable results. Gemma audits the actual rasters together with those
records and bounded table previews; missing OCR is recorded but does not skip the
multimodal review.

When a task asks about uploaded figures, scans, TIFFs, visual-proof PDFs, Office
documents, or archive members, deterministic controller code converts a bounded
set of pages/frames to PNG without interpreting them. Qwen may also create
lossless review rasters under `/output/visual-review`, but it never receives the
pixels. Gemma's structured observations, limitations, and unreviewed-page list
are stored in `gemma_input_visual_review.json`, shown in the workspace explorer,
and supplied to Qwen as evidence for the article. The audit is cached by exact
image hashes across repair rounds.

Each browser or A2A workspace keeps its inputs, run history, generated tables and
figures, claim ledger, source records, model review, logs, and SHA-256 manifest
together. Workspaces cannot read or modify one another. Python/R see their input
workspace read-only and can write only to a bounded per-call output directory.
A workspace allows only one queued or active run at a time, so its inputs and
protocol stay locked — and immutable — for the duration of that run.

While a run is active, visible non-thought output from Qwen and Gemma,
controller artifacts, and registered figures/tables become available as soon
as they are written. A Server-Sent Events stream reports phase changes, bounded
tool-request summaries, outcomes, and artifact links with polling as fallback.
The provenance rail includes a readable live tail plus a near-full-screen
console. The workspace explorer previews input files and every generated UTF-8
text artifact in the browser while runs are active; complete-file downloads
remain available.
Events and live artifacts never expose model reasoning/chain-of-thought,
prompts, raw code, raw tool-response evidence, unapproved arguments, or
credentials. A run can be cancelled
cooperatively at any time, which preserves partial artifacts as explicitly
incomplete rather than reporting a false result. Once a run has a completed,
audited report, a follow-up request starts a new Qwen→Gemma audited child
revision against it; the parent report and its evidence are immutable and are
never overwritten. The UI leaves Python/R disabled for each new follow-up by
default, which is appropriate for writing, caption, and interpretation changes;
users explicitly enable code only when a reanalysis or sensitivity analysis
requires new computation. Cancelling a revision preserves it as a separate,
immutable partial record and does not alter its parent. See
[`docs/WEB_AND_A2A.md`](docs/WEB_AND_A2A.md#run-lifecycle-live-observability-cancellation-and-revisions)
for the events/cancel/follow-up/display APIs.

Every report is a standards-derived **exploratory** scientific article —
Abstract, Introduction, Methods, Results, Discussion, and Conclusions, with
figures/tables registered only from exact successful computation artifacts —
never a claim of peer review, science lock, manuscript readiness, or
submission readiness. See
[`docs/REPORTING_STANDARD.md`](docs/REPORTING_STANDARD.md) for the article
and evidence-gating contract.
The critic is constrained to the stated task contract: it does not invent journal
word-count scopes, merge manuscript and supplement counts, or reject authorized
placeholders as though it were an autonomous submission-readiness service.

The web workbench also provides reviewed workflow starters, a “reuse protocol”
action, immutable input and environment manifests, and a downloadable provenance
bundle. See [lessons adopted from Open Science Desktop](docs/UPSTREAM_LESSONS.md)
for the design boundary and explicit non-goals.

Compose includes its own persistent Chromium service; it never connects to a
user's personal browser or CDP endpoint. Qwen reaches its unpublished DevTools
port through a fixed-target gateway, while lab users can open the same live browser inside the workbench to
clear bot checks or complete publisher interactions. Only the passwordless
noVNC view is published, so bind it exclusively to a trusted LAN/Tailnet. The
browser is isolated from application, worker, and model networks and has only
public-web egress through a private-address-denying proxy. Its profile and downloads persist below `EVIDENCE_BENCH_BROWSER_PATH`, and
the application receives the downloads directory read-only at
`/browser-downloads`. See
[`docs/WEB_AND_A2A.md`](docs/WEB_AND_A2A.md#managed-interactive-research-browser).

Biomedical and health-science runs also have a typed PubMed quality gate. Qwen
must record a PubMed search, acquire relevant PMID records, and cite locally
stored Markdown evidence; legitimate open-access PDFs are verified and stored
when available, while missing PDFs remain explicit. Users can manually obtain
otherwise accessible papers in the managed browser and ask the agent to verify
and import the exact download. See
[`docs/LITERATURE_ACQUISITION.md`](docs/LITERATURE_ACQUISITION.md).

## Run the web service with Docker

Requirements: Docker Engine with Compose, two OpenAI-compatible model endpoints,
and a Linux host on which the nested bubblewrap workers can create mount and
process namespaces.

```bash
cp .env.example .env
# Edit .env: set independent WEB_PASSWORD and A2A_TOKEN values and model URLs.
# For a trusted private network with no browser login, set WEB_AUTH_ENABLED=false
# and remove WEB_USERNAME / WEB_PASSWORD.
# To share the managed browser on that same trusted network, set
# BROWSER_BIND_ADDRESS and optionally BROWSER_NOVNC_PORT.
# A host serving both a private LAN and Tailnet may bind WEB_BIND_ADDRESS and
# BROWSER_BIND_ADDRESS to 0.0.0.0 only when its firewall exposes those ports
# exclusively on the intended trusted interfaces. Keep BROWSER_PUBLIC_URL empty
# so the embedded browser follows the hostname used to open the workbench.
docker compose up --build -d
curl http://127.0.0.1:8080/healthz
```

Open <http://127.0.0.1:8080>. By default, sign in with `WEB_USERNAME` /
`WEB_PASSWORD`. When `WEB_AUTH_ENABLED=false`, the browser and REST API require
no login; A2A and internal worker tokens remain enforced.
The safe Compose default publishes only on loopback. Bind directly only on a
trusted private LAN/Tailnet; use an authenticated TLS reverse proxy for any
publicly reachable interface.

Workspace intake accepts any number of sequential files and streams each upload
to private persistent storage with browser-visible progress. The default per-file
ceiling is 4 GiB (`WEB_MAX_UPLOAD_BYTES=4294967296`). For directories or hundreds
of related inputs, upload one ZIP so the controller can inventory member names and
sizes before planning; archive contents remain untrusted data and are opened only
inside bounded inspection/execution paths. The WebUI always creates the scientific
article and can additionally require an editable PPTX presentation, reproducible
`.ipynb`, and/or machine-readable result ZIP. Missing or structurally invalid
requested artifacts fail the deterministic gate, and every generated file appears
in the live artifact browser and final provenance bundle.

The service exposes:

- `GET /.well-known/agent-card.json` — public A2A 1.0 Agent Card;
- `POST /a2a` — JSON-RPC A2A endpoint using `Authorization: Bearer <A2A_TOKEN>`;
- `POST /api/workspaces/{id}/runs`, `GET /api/runs/{run_id}` — start and poll a run;
- `PUT /api/workspaces/{id}/files/{filename}` — atomic raw-body streaming upload
  for large inputs without multipart temp-file duplication;
- `GET`, `POST`, `PATCH`, `DELETE /api/knowledge...` — manage, preview,
  version, enable/disable, reindex, and test-search the instance-local knowledge
  library, including exact chunk and verified run-acquisition history; run
  submission snapshots the selected immutable generations;
- `GET /api/knowledge/jobs`, `.../jobs/{id}/events`, `POST
  .../jobs/{id}/{cancel|retry}` — inspect and control persistent background Qwen
  text/Gemma visual indexing without making the current published generation
  unavailable;
- `POST /api/knowledge/search/visuals`, `GET
  /api/knowledge/{document_id}/visuals` — search and preview hash-verified images
  that Gemma indexed from actual uploaded or deterministically extracted pixels;
- `GET /api/runs/{run_id}/knowledge/passages/{passage_id}` and
  `.../knowledge/documents/{document_id}/{text|original}`, plus
  `.../knowledge/visuals/{knowledge_visual_id}` — inspect the exact cited
  passage, full extracted text, immutable original, or hash-verified raster
  preserved in that run;
- `GET /api/runs/{run_id}/events`, `.../events/stream` — cursor-based event log
  plus an SSE live stream with polling fallback;
- `GET /api/workspaces/{id}/file-preview?filename=...` — bounded UTF-8 preview
  for a path-confined workspace input;
- `GET /api/runs/{run_id}/artifact-preview?path=...` — bounded UTF-8 preview
  for live or final text artifacts (up to 512 KiB, with explicit head/tail
  truncation for larger files);
- `POST /api/runs/{run_id}/cancel` — cooperative cancellation;
- `POST /api/runs/{run_id}/follow-ups` — start an audited Qwen→Gemma child
  revision against a completed report, with a per-revision `enable_code`
  override;
- `GET`, `POST /api/runs/{run_id}/discussion` — read or continue an
  evidence-bounded Gemma explanation thread for a completed report; a reply may
  propose, but never automatically starts, an audited Qwen→Gemma revision;
- `GET /api/integrations` — filenames, sizes, setup hints, and SHA-256 digests
  for the WebUI's agent downloads;
- `GET /api/integrations/skill`, `GET /api/integrations/a2a` — deterministic,
  credential-free Codex/Claude skill and A2A 1.0 starter archives;
- `GET /api/runs/{run_id}/artifacts`, `.../displays/{id}/image`,
  `.../displays/{id}/table`, `.../references/{source_id}/pdf`, `.../bundle` —
  live and final artifact/display access plus inline preview of verified cited
  papers;
- `/api/docs` — authenticated OpenAPI documentation;
- `/healthz` — unauthenticated container health check.

See [`docs/WEB_AND_A2A.md`](docs/WEB_AND_A2A.md) for A2A examples, deployment
details, workspace semantics, run-lifecycle APIs, and threat boundaries, and
[`docs/REPORTING_STANDARD.md`](docs/REPORTING_STANDARD.md) for the report
contract.

## Lab agent skill

[`skills/evidence-bench`](skills/evidence-bench/SKILL.md) is a portable skill for
Claude and Codex agents on the UMED lab LAN/Tailnet. Its dependency-free client
creates isolated workspaces, uploads inputs, submits work with Python/R and all
three research MCPs enabled by default, streams controller events, cancels bad
runs, lets s8-Gemma explain a completed report and draft a revision brief, starts
audited follow-up revisions, and downloads provenance bundles. It
uses the lab production service at `http://10.20.102.122` and requires no browser
username or password.

The lab production deployment is isolated from the owner's private task
service: it has separate containers, workspaces, package environments, browser
profile, worker tokens, and persistent storage. Port 8070 is intentionally not
the lab skill's default.

The WebUI's **Skill + A2A downloads** control provides the same installable
skill as a versioned ZIP and a dependency-free A2A starter containing the exact
deployment URLs. Both downloads expose their archive SHA-256 and include an
inner `SHA256SUMS`; the A2A bearer token is intentionally never bundled.

Install the folder as `evidence-bench` under either agent's skill directory:

```bash
# Claude Code
cp -R skills/evidence-bench ~/.claude/skills/evidence-bench

# Codex
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R skills/evidence-bench "${CODEX_HOME:-$HOME/.codex}/skills/evidence-bench"
```

Then ask the agent to use `$evidence-bench`, or run the bundled client directly:

```bash
python3 ~/.claude/skills/evidence-bench/scripts/evidence_bench.py run \
  --workspace-name "Welch analysis" \
  --objective "Analyze the uploaded data and produce an evidence-linked report." \
  --file data.csv --wait --download-dir ./evidence-bench-result
```

Add `--requested-output pptx_presentation` (repeatable with
`analysis_notebook` or `data_bundle`) to make those files required outputs.

This optional lab profile is intentionally deployment-specific. The container,
A2A protocol, application configuration, and public integration sample remain
generic and configurable for other installations.

The current milestone supports evidence retrieval, real computation, and
workspace-scoped package installation. It
can retrieve current sources through Brave Search, retrieve library documentation
through Context7, inspect relevant public web pages through the service-owned
Chrome DevTools endpoint, and read bounded files inside one assigned workspace.
All three research connections are enabled by default when configured; the
researcher prompt directs Qwen to use the connection relevant to the question
rather than calling tools mechanically. With the
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

Immutable generations are governed by separate per-generation, cumulative
per-workspace, and cumulative deployment-wide logical-byte quotas. Quota
admission is serialized; an update is rejected before copying its predecessor
when capacity is already insufficient. During installation, the worker polls the
staging package tree and terminates the installer process group when the strictest
generation, workspace, or deployment-wide allowance is observed to be exceeded;
failed staging generations are removed. Polling can permit transient overshoot
during one 100 ms polling interval plus directory-scan and process-termination
latency, so the residual overshoot is time-bounded rather than a fixed byte count.
Deleting a workspace also invokes the package worker's authenticated cleanup
endpoint and removes all of that workspace's Python and R generations without
touching retained workspaces.

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
The normalized task includes virtual `/workspace/...` names and hashes from the
immutable input manifest. Plan linting blocks any file-like input name that is
neither in that manifest nor declared as a plan output, so an invented dataset
filename cannot survive merely because a model critic overlooks it.

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

Thinking is enabled for both model roles by the recommended serving aliases.
`*_ENABLE_THINKING=inherit` omits backend-specific template controls and relies
on that stable endpoint contract; direct backends can explicitly use `true` or
`false`. `QWEN_MAX_TOKENS=0` and
`GEMMA_MAX_TOKENS=0` omit an explicit client output ceiling, allowing compatible
proxies to use the context remaining after the prompt, tools, and final answer.
Positive values impose deployment-specific ceilings. Native JSON-schema decoding
is used when it is compatible with the endpoint's reasoning mode. If a server
cannot combine the two, set that role's `*_NATIVE_JSON_SCHEMA=false`: Evidence
Bench then asks for JSON in the ordinary final channel and applies the same local
Pydantic validation and bounded repair policy. It never stores or streams the
provider's `reasoning_content` field. While a request is live, it examines only a
bounded in-memory suffix for sustained contiguous sentence/fragment repetition.
A match must cover at least 2 KiB and recur at two consecutive byte-based
checkpoints; this avoids treating ordinary repeated schema fields as a loop and
does not depend on provider chunk sizes. A detected loop closes the model stream
and uses the single bounded schema-repair retry. Emitting a second complete
top-level JSON value is also a deterministic structured-output loop: exactly one
value is permitted, so the stream is closed without waiting for further copies.
Likewise, one syntactically complete but schema-invalid value is closed
immediately and sent to the bounded repair path; appending a correction would
itself violate the one-value contract.
Gemma also has a 192 kB private-reasoning no-final-progress safeguard. It does
not send a token ceiling to the model: generation remains unrestricted, but a
sample that emits roughly tens of thousands of private reasoning tokens without
starting its final channel is terminated and independently retried once. This
prevents a continuously varying Gemma reasoning loop from occupying the single
local slot for hours while preserving a large reasoning allowance.
The raw reasoning suffix is immediately discarded and never becomes scientific
evidence.
If the repair sample also repeats, the planning transition is recorded as
`inconclusive` with a `plan-critic-unavailable` finding and research does not
begin. Critic failure can never be converted into approval.
Maximum reasoning remains operationally bounded by cooperative cancellation,
no-progress detection, and per-call wall time. The example deployment keeps an
admitted model request attached for up to 21,600 seconds through
`QWEN_REQUEST_TIMEOUT_SECONDS` and `GEMMA_REQUEST_TIMEOUT_SECONDS`. Backend queue
time is part of that wall time: a busy local model is waited for, not treated as
scientific disagreement. This six-hour safety limit is not a token or reasoning
budget, and the UI/A2A cancellation path remains responsive throughout. Plan
review itself uses a fixed five-criterion contract so unrestricted reasoning does
not become an open-ended review scope.

Model transport distinguishes capacity from invalid work. Connection failures
and HTTP 429, 502, 503, and 504 responses received before any model output wait
with exponential backoff for up to `QWEN_CAPACITY_WAIT_SECONDS` or
`GEMMA_CAPACITY_WAIT_SECONDS` (21,600 seconds by default). The wait is visible in
the live stream and immediately cancellable. HTTP 500 retains three bounded
restart attempts; HTTP 400 is never retried as capacity. A streamed request is
retried only before it has emitted any answer or reasoning chunk, preventing a
partly observed response from being silently replayed. Exhausted capacity waits
fail the current scientific transition closed; they do not manufacture critic
approval while an independently hosted model is restarting.
When a streaming gateway omits or delays its terminal `[DONE]` event, Evidence
Bench closes the stream as soon as the final channel contains exactly one value
that passes the required Pydantic schema. This happens only after the model has
emitted its answer and does not cap the preceding reasoning budget.

Tool observations are independently context-bounded and do not reduce the model
reasoning allowance. A single model-visible result is capped at 64 KiB and all
tool observations in one research attempt at 256 KiB. When a result would cross
either boundary, the complete permitted response is preserved as a hashed run
artifact and the model receives compact metadata plus a bounded text preview.
Browser screenshot base64 is never injected as text. This prevents Chrome or MCP
payloads from consuming the context intended for reasoning and the final report.
If an ADK research turn nevertheless ends after successful controller-recorded
retrieval or computation, a bounded evidence preview is passed to reporting and
all ordinary deterministic/critic gates still apply; a turn with no successful
evidence fails immediately.

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
The v0.4.0 deployed PubMed/full-text gate passed 17/17 checks, including typed
search/acquisition, verified local Markdown and PDF evidence, exact scientific
extraction, a repair triggered by the independent critic, and final
deterministic and Gemma passes. Live deployment probes also verified cooperative
cancellation, persistent service-owned Chromium with internal-only CDP and
passwordless trusted-network noVNC, private-target egress denial, and isolated
installation/loading of one package each from PyPI, CRAN, and Bioconductor. The
compact scored record is
[`v0.4.0-pubmed-fulltext.json`](evals/results/v0.4.0-pubmed-fulltext.json). The
final v0.4.0 planted-effect gate passed **18/18** on 2026-07-15 in workspace
`187c0fe5-2967-4bbd-a297-f7a9423274be`, using image digest
`sha256:4f055eb3a5515b49257fad69e701dd3d46ec07fdf28c430b09293e66c4a2021c`.
Parent run `5428105d-8979-4bf1-8dd1-76f9fedccee2` independently recovered and
reconciled the planted +5 effect in Python and R; Qwen and Gemma streamed, live
artifacts were accessible, and deterministic, Gemma report, and
OCR/geometry/table display checks passed. The accepted code-disabled report
revision, `b5bbf30c-15bd-42f2-bb65-b06519a94a9c`, produced no new result
outputs, preserved the parent immutably, and passed final manual caption/prose
inspection. An earlier nominally supported revision with inverted provenance
was rejected and was not counted. Its compact scored record is
[`v0.4.0-known-effect.json`](evals/results/v0.4.0-known-effect.json).

The v0.4.0 A2A live gate also passed on 2026-07-15 with functional image
`sha256:e95760b378f4923142e499899ebb481687c0a71012aee480556458a6d2a6f726`.
Task `44702ea7-72d8-4545-853e-82fd926e0831`, backed by run
`0c69fa4f-459b-419f-81a8-47737f732ce6`, streamed submitted, working, and
completed states plus `report.md` and `run-summary.json`, and finished
scientifically supported. Scientific MCP probe run
`fa5e58b9-92b4-4bfb-82f3-fd0e14dd279d` in workspace
`6e08b205-4d2b-49fb-a0ec-5bbbea735c4a` exercised Brave Search, Context7, and
the typed PubMed tools; a blocking PubMed-title mismatch was repaired from
stored acquisition evidence before Gemma passed the report. See the
[`v0.4.0 A2A live record`](evals/results/v0.4.0-a2a-live.json).

The final knowledge-grounding gate ran on clean commit `ca09622` on 2026-07-16.
Across 30 exact, synonym, and Polish queries, Qwen-enriched hybrid retrieval
achieved Recall@10 1.00 and nDCG@10 0.975, versus lexical Recall@10 0.733 and
nDCG@10 0.704. The synonym/Polish recall gain was +0.40 (seeded bootstrap 95%
CI 0.20–0.60), with no exact-query recall loss. Five text and three visual
no-answer queries returned no false positives; all 122 returned text passages
audited as exact immutable source slices rather than descriptor prose. Gemma
ranked each of six structure-dependent scientific figures first. See the
[`v0.4.0 knowledge-grounding record`](evals/results/v0.4.0-knowledge-grounding.json).

One protocol limitation remains explicit: v0.4.0 uses the SDK
`InMemoryTaskStore`, so A2A `GetTask` snapshots and subscription state do not
survive a web-process restart even though Evidence Bench workspaces, runs, and
provenance do. The issue-first contribution proposal is open in
[`a2aproject/a2a-samples#639`](https://github.com/a2aproject/a2a-samples/issues/639);
a maintainer invited a small A2A 1.0 contribution, now available as
[`a2aproject/a2a-samples#642`](https://github.com/a2aproject/a2a-samples/pull/642)
with its lint gate passing. See
[`evals/README.md`](evals/README.md) for the release-gate ledger.

## Run the CLI

The default enables Context7, Brave Search, and the service-owned Chrome DevTools
connection. Use `--mcp ''` to opt out of every MCP, or pass an explicit subset.
Browser navigation remains policy-limited to public HTTP(S) destinations.

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
extraction: one lean Qwen plan, deterministic plan lint, a fixed five-criterion
Gemma plan audit, and independent Gemma audits of the final article and displays,
followed by another audit after every deterministically admissible repair. An
objective validation failure returns directly to Qwen with exact findings before
Gemma is called. It does not ask Gemma to generate a redundant long-form plan,
but a concrete blocking critic finding enters the same bounded Qwen repair →
deterministic validation → Gemma re-audit loop;
unsupported disagreement alone is not a blocker. Use
`--mode full` for genuinely multi-stage scientific design.
Research model/tool budgets are cumulative across repair rounds. Exhausting a
budget stops further evidence gathering, but an existing report repair continues
in existing-evidence-only mode and remains subject to deterministic validation
and Gemma review. It never becomes a generic success or bypasses the gate.

The base Python runtime exposes NumPy, pandas, SciPy, statsmodels, scikit-learn,
and matplotlib. The base R runtime exposes ggplot2, dplyr, survival, data.table,
and jsonlite. Additional PyPI/CRAN/Bioconductor packages are installed on demand
per workspace. Scripts read inputs at `/workspace`, earlier calls in the current
attempt at `/prior`, earlier repair attempts at `/history`, and
must write outputs below `/output`. Code-enabled preflight
imports the full Python/R analysis set inside the sandbox and fails before model
execution if the host installation is incomplete. Each call is isolated, offline,
resource-bounded, and capped by a per-attempt call budget.
Full mode allows 12 calls for initial analysis so ordinary Python/R corrections
remain usable; simple mode caps initial analysis at four calls, while repair
rounds allow up to eight calls and 120 seconds per call. Display-only repair reuses prior numeric evidence and
must not repeat valid estimation, reconciliation, or controller provenance. Tool
responses expose the remaining count and exhaustion fails closed.

Cross-language reconciliation is controller-verified rather than trusted as model
prose. Its JSON must list at least one Python/R comparison, bind both values to
successful JSON artifacts by SHA-256 and dot-delimited JSON path, and record the
tolerance, absolute difference, and per-comparison verdict. The controller reloads
the hashed artifacts and recomputes every difference and the top-level verdict; a
bare `all_pass: true` is invalid. Exact generated-artifact paths mistakenly placed
in a claim's `evidence_refs` are converted to stable, inspectable `SourceRecord`
IDs only when the path exactly matches successful computation evidence. Unknown
paths still fail validation.
Generated JSON containing a t statistic, degrees of freedom, and p-value is also
checked against the Student-t distribution; an impossible tuple blocks the report.
If a critic asks for more display decimals for the same table where the controller
has found excessive precision, the raw critic response remains inspectable but the
contradictory blocker is discarded. OCR alone cannot overrule direct visual review:
a typo blocker is discarded only when Gemma gives two incompatible direct readings
of the same exact display element and OCR from that display corroborates the
proposed correction. Metadata-versus-raster disagreement remains a real defect.

Each run creates a mode-0700 directory under `runs/` containing the typed plan,
lint result, report, Gemma audit, run configuration, private size-bounded MCP
evidence artifacts, source scripts and logs, generated computation artifacts,
append-only tool event log, retrieval/computation records, and SHA-256 manifest.
Rejected drafts are retained under `attempts/attempt-N` together with the exact
deterministic findings and Gemma review that triggered repair; successful evidence
is carried forward so a citation-only repair does not force redundant analysis.
Full and simple modes run up to `MAX_REPAIR_ROUNDS` repair/re-audit cycles (four
by default; accepted range 0–8). Fixable report and display defects remain blocking. Properly stated
inherent design limitations may pass only as nonblocking comments; exhausting
the automatic budget writes `repair_exhausted.json` and returns
`requires_human_decision`, never a validated result.
Reader-facing tables are limited to four significant digits while exact values
remain in machine-readable artifacts. The report cannot infer observational,
randomized, experimental, synthetic, or representative design from filename,
balance, effect size, or data cleanliness; missing design metadata stays explicit.
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
GHCR release automation for runtime, package-builder, and managed-browser images,
contribution/security policies, an A2A 1.0 Agent Card, and no secrets in tracked
configuration. The optional lab skill documents its explicitly authorized
private deployment endpoint; runtime configuration remains portable. Tagging `v0.4.0` in a
public GitHub repository builds and publishes the corresponding containers. A
ready-to-adapt upstream contribution dossier lives in
[`docs/A2A_ECOSYSTEM_SUBMISSION.md`](docs/A2A_ECOSYSTEM_SUBMISSION.md).
