# Standalone web service and A2A integration

## WebUI agent downloads

The sidebar's **Skill + A2A downloads** control offers two versioned archives:

- `GET /api/integrations/skill` packages the installable `evidence-bench`
  Codex/Claude skill;
- `GET /api/integrations/a2a` packages a dependency-free A2A 1.0 client, setup
  guide, and deployment-specific `connection.json`.

`GET /api/integrations` returns each archive's fixed URL, filename, byte size,
SHA-256, and concise setup hint. Every ZIP is assembled in sorted order with
fixed timestamps from an explicit source-file allowlist. Archive paths are not
user-controlled, symlinks are rejected, and each archive contains an inner
`SHA256SUMS` covering its component files. The HTTP response also exposes the
complete digest in `X-Checksum-SHA256` and a SHA-256 `ETag`.

The A2A starter records only public connection metadata. It never contains the
`A2A_TOKEN`, model credentials, MCP credentials, worker tokens, or browser
credentials. Users obtain the independent A2A bearer token from the deployment
administrator and supply it through their local environment.

## Workspace model

A workspace is the unit of isolation and scientific provenance. Uploaded inputs
live below its private `files/` directory. Each run writes to a new directory
below `runs/`; the browser never accepts an arbitrary filesystem path. Filenames
are basenames only, symlinks are rejected, and paths are resolved before access.
Only one run may be queued or active in a workspace, so inputs cannot be changed
under an analysis.

The metadata database uses SQLite WAL mode. On service restart, unfinished runs
are marked `interrupted`; they are never silently reported as successful.

## Local model load and queue status

The WebUI refreshes `GET /api/model-status` every five seconds while visible
(less often in a background tab). The endpoint reports only the configured
executor/critic alias, role, reachability, active and queued request counts,
optional slot occupancy, and Qwen context-cache usage. It never returns a
backend address, response body, exception, credential, prompt, or model output.

Monitoring is optional and configured only by the service operator:

```dotenv
QWEN_STATUS_BASE_URL=http://host.docker.internal:9004
GEMMA_STATUS_BASE_URL=http://gemma-inference.internal:8080
MODEL_STATUS_TIMEOUT_SECONDS=2
MODEL_STATUS_CACHE_SECONDS=3
```

`QWEN_STATUS_BASE_URL` expects vLLM Prometheus metrics
(`vllm:num_requests_running`, `vllm:num_requests_waiting`, and optionally
`vllm:kv_cache_usage_perc`). `GEMMA_STATUS_BASE_URL` expects llama.cpp metrics
(`llamacpp:requests_processing`, `llamacpp:requests_deferred`) and optionally
`/slots` for total capacity. A deferred/waiting count is an authoritative
current queue; a fully occupied llama.cpp slot set means Gemma's next step may
wait. vLLM can continuously batch work, so an active Qwen request with zero
waiting requests is described as busy with a clear queue rather than saturated.

These values are trusted startup configuration, not request parameters. The
application appends only fixed `/metrics` and `/slots` paths, rejects credentials,
paths, queries, and fragments in configured origins, disables proxy inheritance
and redirects for probes, caps each response at 512 KiB, and bounds timeouts to
five seconds. Leave either origin empty when the backend does not expose the
expected interface; analysis still works and the UI labels monitoring as
unconfigured.

## Independent model endpoint and review boundary

The application does not require the critic model to be co-located with Qwen or
with the web service. `GEMMA_BASE_URL` and `GEMMA_MODEL` select any compatible
OpenAI-style Gemma endpoint in the private deployment environment; the example
value in `.env.example` is not a fixed architecture choice. Endpoint routes and
credentials remain outside Git.

Thinking is enabled independently for both roles. The `inherit` setting relies
on a gateway alias that already enables unrestricted thinking and omits
backend-specific template arguments; direct backends can explicitly select
`true` or `false`. A zero `*_MAX_TOKENS` value
means that Evidence Bench does not send an explicit client token ceiling; it
does not mean zero output. Endpoints that cannot combine thinking and native
JSON-schema decoding can disable `*_NATIVE_JSON_SCHEMA` for that role and use
prompt-constrained final JSON with the same local validation. Provider
`reasoning_content` is neither streamed to users nor retained as provenance. A
bounded in-memory suffix is inspected only for sustained contiguous no-progress
repetition covering at least 2 KiB. The same content-free signature must occur at
two consecutive byte-based checkpoints, so detection is independent of provider
chunking and ordinary repeated schema fields do not trigger it. On a match,
Evidence Bench closes the stream, records a safe retry notice in the live output,
and makes the one permitted structured repair attempt; the private text is
discarded. Two complete top-level JSON values in one response trigger the same
bounded repair immediately, even when their wording differs, because the
structured contract permits exactly one value.
One complete but schema-invalid value also closes immediately and enters the
bounded repair path, so a delayed or missing gateway terminator cannot strand an
otherwise repairable transition.
If that repair also repeats or is invalid, the controller writes an explicit
`plan-critic-unavailable` finding and terminates the plan as `inconclusive` before
research. It does not expose the private text, retry indefinitely, or treat a
missing critic verdict as agreement.
Every planner also receives the exact immutable input manifest through virtual
`/workspace/...` artifact references. Deterministic plan lint rejects file-like
input names absent from that manifest and absent from declared upstream outputs;
generic terms such as “uploaded dataset” remain valid when no exact name is needed.
Per-role request timeouts are wall-time and cancellation bounds, not reasoning
budgets. The default is 7,200 seconds so an uncapped model can finish a long
reasoning pass near a 150K context window; users can still cancel immediately.

Gemma is the sole image-understanding model. The controller sends generated PNG,
JPEG, and WebP figures only to Gemma in bounded batches; Qwen never receives
raster images or makes visual-quality judgments. The confined sandbox worker
also extracts bounded OCR text, word boxes, and dimensions, and deterministic
validators compare rendered labels with machine-readable results. Gemma reviews
the actual rasters with this supplementary evidence and bounded table previews.
Missing OCR is recorded but does not skip the multimodal review.

Source-image tasks use the same asymmetric boundary. The trusted controller
converts bounded TIFF frames, PDF pages, and supported image members from
ZIP/DOCX/PPTX/XLSX inputs to model-compatible PNG; Qwen may render additional selected
pages under `/output/visual-review` but never interprets or receives raster
bytes. Only Gemma receives those images. Its structured observations and explicit
coverage gaps are recorded in `gemma_input_visual_review.json`, streamed as a
live artifact, and passed to Qwen as text evidence. Exact input hashes prevent
unchanged visuals from being re-audited in every repair round.

Local model restarts are handled with a small, explicit retry budget. Transport
errors and HTTP 429, 500, 502, 503, and 504 responses receive at most three total
attempts with short backoff; streamed calls are retried only when neither reasoning
nor final-answer content has arrived. If the endpoint remains unavailable, the scientific transition fails
closed or its independent review stays explicitly inconclusive.
If a gateway delays `[DONE]` after emitting a complete schema-valid final JSON
value, the client closes that stream and accepts the validated value immediately;
all preceding maximum-thinking output has already been consumed.

### Workspace-wide input locking

A workspace allows at most one queued or active run at a time
(`WorkspaceStore.create_run`, enforced by a unique partial SQLite index). While
a run is `queued`, `running`, or `cancel_requested`:

- `POST /api/workspaces/{workspace_id}/files` and
  `DELETE /api/workspaces/{workspace_id}/files/{filename}` fail with `409`
  (`cannot upload files while a run is active` /
  `cannot delete files while a run is active`).
- `DELETE /api/workspaces/{workspace_id}` fails with `409`
  (`cannot delete a workspace with an active run`).

This guarantees a report's claims and computation can always be traced back to
the exact input set the run actually saw. Monitoring, artifact/bundle
downloads, run history, and cancellation remain available while a run is
locked.

## Run lifecycle: live observability, cancellation, and revisions

### Live observability

`GET /api/runs/{run_id}/events?after_id=<id>` returns the append-only event
log for a run (id, event type, run status/phase at that point, actor —
`Controller`, `Qwen`, `Gemma`, or `User` for a cancellation request — a short
message, and an optional relative artifact path), ordered by id, capped at
500 rows per call. `GET /api/runs/{run_id}/events/stream?after_id=<id>` provides
the same records as Server-Sent Events with keep-alives and a terminal
`stream_end`; cursor polling remains the fallback. Events report phase
transitions, safe tool-request summaries, outcomes, and when Qwen or Gemma begins
updating a visible-output artifact or a controller artifact becomes available.
PubMed identifiers and safe numeric limits may be shown. Search terms, workspace
paths, filenames, and package lists are represented only by byte/count and hash
summaries; URLs are reduced to their public origin. Raw code, arbitrary argument
objects, URL paths/query strings/userinfo/fragments, and credentials are never
emitted. The UI polls those bounded
text artifacts while the model is producing research, article, or audit output,
so the observable-output panel updates before the complete response exists.
Its **Open console** control expands the selected sanitized model stream to a
near-full-screen, 14 px monospace view and keeps it current while the file
grows. The compact tail remains available in the provenance rail.
Events never carry model chain-of-thought/reasoning traces, system prompts, raw
code, or MCP/worker credentials.

### Live artifact access

`GET /api/runs/{run_id}` (run detail) lists the run's `artifacts`. While a run
is still active this list is derived live from whatever has been written
under the run's provenance directory so far (each entry is flagged `"live":
true` and has no SHA-256 yet); once the run finishes, the list comes from the
SHA-256-hashed manifest instead. Either way,
`GET /api/runs/{run_id}/artifacts?path=<relative-path>` serves the named file
(the path is resolved and confined to the run's provenance directory). This is
how the UI's live activity log offers in-progress downloads, for example the
visible Qwen research packet or report draft flagged in an
`artifact_ready`/`model_output` event.

`GET /api/runs/{run_id}/artifact-preview?path=<relative-path>` decodes a
run-confined artifact as UTF-8 and returns a browser-safe preview. Files up to
512 KiB are returned completely. Larger files return a bounded head and tail
with an explicit middle-omitted marker, so previewing cannot force an
unbounded response. Head and tail cuts are repaired only at their UTF-8 codepoint
boundaries, so Greek symbols and accented names at a cut do not turn a valid
article into a binary-file error. NUL-containing or otherwise non-UTF-8 binary files are rejected and
remain available through the download endpoint. The UI links common
scientific text, source-code, notebook, configuration, log, and
extensionless-text artifacts to this preview, both during and after a run.
Raw `tool_call_log.jsonl` and the private `evidence/` tool-response tree remain
outside the Web explorer; their complete audit representation is retained in the
downloadable provenance bundle.

`GET /api/workspaces/{workspace_id}/file-preview?filename=<name>` applies the
same bounded UTF-8 rules to an immutable workspace input. The Web UI's workspace
explorer shows both inputs and the selected run's live/final artifact tree;
preview and download controls remain available while mutation is locked.

Once a report is registered, its figures and tables are also addressable
individually and by kind:

- `GET /api/runs/{run_id}/displays/{display_id}/image` — streams a
  registered figure (PNG/JPEG/WebP) inline, after re-verifying its SHA-256
  against `display_manifest.json`.
- `GET /api/runs/{run_id}/displays/{display_id}/table` — returns the
  registered table's bounded preview (columns, rows, totals, `truncated`
  flag) and metadata as JSON.
- `GET /api/runs/{run_id}/references/{source_id}/pdf` — streams a cited,
  controller-verified PDF inline after rechecking its SHA-256 against
  `reference_manifest.json`. The Sources tab opens each local article Markdown
  copy in the large text preview and retains a separate canonical PubMed link.
- `GET /api/runs/{run_id}/bundle` — streams a zip of every file under the
  run's provenance directory (report, evidence, logs, manifest, and
  artifacts).

See [`docs/REPORTING_STANDARD.md`](REPORTING_STANDARD.md#registered-displays-figures-and-tables)
for how a figure or table becomes a registered display.

### Cancellation

`POST /api/runs/{run_id}/cancel` requests cooperative cancellation. The run
moves to `cancel_requested`, the controller stops at the next bounded
checkpoint (including terminating active Python/R computation), and the run
is marked `cancelled`. Partial artifacts already written are preserved under
the run's provenance directory and flagged as incomplete
(`run_cancelled.json`) — they are never presented as a validated result.
For a cancelled child revision, that incomplete provenance remains an immutable
revision record: it never replaces or mutates the completed parent report, and a
later improvement request starts another child rather than resuming or rewriting
the cancelled one.

### Follow-up revisions

`POST /api/runs/{run_id}/follow-ups` starts a new Qwen→Gemma audited **child**
run against a completed parent report. The request accepts a per-revision code
override:

```json
{
  "request": "Run a prespecified sensitivity analysis and update the interpretation.",
  "enable_code": true
}
```

The web UI presents this as **Allow Python/R for this revision** and leaves it
unchecked for every new follow-up. Keep it off for writing, caption, figure-text,
or interpretation changes that can reuse the parent's audited evidence. Enable
it explicitly only when reanalysis, a sensitivity analysis, or another requested
change genuinely requires new computation. API clients may omit `enable_code`;
for backward compatibility, omission inherits the parent run's setting, whereas
an explicit `false` disables code for that child.

The parent run's `scientific_report.json` and provenance are immutable: the
controller loads them read-only, requires the workspace's current inputs to
exactly match the parent run's recorded input manifest, and writes a
`parent_lineage.json` recording the parent run id, a SHA-256 of the revision
request, and SHA-256s of the parent's report/evidence artifacts. The child
report is a new, separately audited record; it can reuse valid inherited
evidence but never overwrites the parent. A follow-up can only be started
once the parent run has a report and no run is currently active in the
workspace. A cancelled follow-up remains a separately addressable immutable
partial record and is never promoted to a completed report.

### Report discussion and revision briefs

`GET /api/runs/{run_id}/discussion` returns the persistent final-answer thread
for a completed report. `POST /api/runs/{run_id}/discussion` accepts
`{"message": "..."}` and asks the independently configured Gemma model to explain
or challenge only the immutable report, deterministic validation, Gemma audit,
registered sources/displays, run result, and bounded prior discussion. Discussion
is unavailable while a run is active and never changes the report or its status.

Only final structured answers, evidence references, unresolved uncertainties,
and an optional `suggested_revision_prompt` are stored. Hidden reasoning is
neither returned nor persisted. A suggested prompt is a user-reviewed draft: the
UI can copy it into the existing follow-up form, but cannot submit it
automatically. Starting that follow-up remains a distinct Qwen→deterministic
validation→Gemma child run with immutable parent lineage.

## Managed interactive research browser

Compose runs a dedicated `browser` service built from `browser/Dockerfile`.
This is Evidence Bench state: it does not connect to, proxy, or reuse a lab
member's personal browser, cookies, profile, IP address, or CDP endpoint.
Chromium, its X display, VNC server, and noVNC gateway all run in that one
container.

The browser has no direct Internet or application-network route. Its access paths
have deliberately different exposure:

- Public browsing is forced through `browser-egress`, a repository-built Squid
  service on both the internal `browser-control` network and the dedicated
  external-capable `public-egress` bridge, which no application service joins.
  It denies loopback, RFC1918, link-local, Tailnet, reserved,
  multicast, and local IPv6 destinations after DNS resolution; HTTP is limited
  to port 80 and CONNECT to port 443. Chromium cannot bypass it because
  `browser-control` is `internal:true`.
- CDP listens on port 9222 only on `browser-control`. The application is attached
  to a separate `browser-client` network and receives the fixed internal URL
  `http://browser-cdp-gateway:9222`. The capability-dropped gateway has both
  internal networks but listens only on 9222 and forwards only to
  `browser:9222`; Compose publishes neither CDP endpoint on the host.
- noVNC is the human interaction path. Compose publishes only container port
  6080 at `BROWSER_BIND_ADDRESS:BROWSER_NOVNC_PORT`. The workbench's **Open
  research browser** control embeds `vnc.html` in a near-full-screen dialog and
  remains available while an analysis is running, so a user can clear a bot
  check without stopping the evidence workflow. The same view can be opened in
  a separate tab.

When `BROWSER_PUBLIC_URL` is empty, the UI derives the noVNC URL from the
workbench page's current hostname and `BROWSER_NOVNC_PORT`. Set an explicit
HTTP(S) URL when a reverse proxy, nonmatching hostname, or TLS termination is
used. An HTTPS workbench cannot embed a plain-HTTP noVNC endpoint because of
browser mixed-content rules; terminate both behind HTTPS in that case.

The browser profile and downloads persist together under
`EVIDENCE_BENCH_BROWSER_PATH`:

```text
<EVIDENCE_BENCH_BROWSER_PATH>/
├── profile/     # Chromium cookies, local state, cache metadata
├── downloads/   # publisher PDFs and other manual downloads
└── home/        # small desktop/browser state
```

The application mounts only `downloads/`, read-only, at
`/browser-downloads`. This lets the acquisition workflow import a paper that a
human downloaded after clearing a challenge without giving the web process
write access to the browser profile or downloaded bytes. Stop Compose before
snapshotting the profile, and treat profile backups as credential-bearing even
when users follow the recommended no-personal-login rule.

noVNC is intentionally passwordless for the trusted-lab deployment. Anyone who
can reach its published port can see the screen, control Chromium, inspect its
history, and use its active publisher sessions. Never bind it to a public
interface. Prefer a private address or a TLS/OIDC reverse proxy when the network
is not fully trusted. Do not enter personal credentials in the shared browser.
The browser container mounts no research workspace, model/API secret, Docker
socket, SSH material, or host home directory.

When one host must serve both a private LAN and a Tailnet, it may set both
`WEB_BIND_ADDRESS=0.0.0.0` and `BROWSER_BIND_ADDRESS=0.0.0.0` only if host or
upstream firewall policy restricts ports 8080/6080 (or their configured
replacements) to those trusted interfaces. Leave `BROWSER_PUBLIC_URL` empty: the
UI then derives noVNC from the exact LAN or Tailnet hostname used by each user.
This is not an acceptable configuration on a publicly reachable unfiltered host.

## A2A 1.0

Evidence Bench publishes a standards-based Agent Card at:

```text
GET /.well-known/agent-card.json
```

The declared JSON-RPC interface is `POST /a2a`. The card is public for discovery;
execution requires the independent `A2A_TOKEN` bearer credential. Each A2A
`context_id` maps to one persistent isolated workspace, allowing a client to send
several tasks against the same inputs without exposing browser workspace IDs.

Example JSON-RPC request:

```bash
curl -sS http://127.0.0.1:8080/a2a \
  -H "Authorization: Bearer $A2A_TOKEN" \
  -H 'A2A-Version: 1.0' \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "analysis-1",
    "method": "SendMessage",
    "params": {
      "message": {
        "messageId": "message-1",
        "role": "ROLE_USER",
        "parts": [{"text": "Analyze the attached cohort and report uncertainty."}],
        "metadata": {
          "enable_code": true,
          "mcp_servers": ["context7", "brave-search", "chrome-devtools"]
        }
      },
      "configuration": {"returnImmediately": false}
    }
  }'
```

For least privilege, A2A defaults to `enable_code=false`. Context7, Brave Search,
and the service-owned Chrome DevTools connection are enabled by default when
available. Omitting `metadata.mcp_servers` selects those service defaults; an
explicit empty list opts out, and an explicit subset narrows the run. Raw file parts
are accepted when they include a filename; the service deliberately does not
fetch URL parts.

Successful A2A tasks return both a JSON run summary and `report.md` artifacts.
Scientifically inconclusive results are valid completed tasks whose summary keeps
the unresolved status visible. Infrastructure failures are A2A failed tasks.

The current implementation uses the SDK's `InMemoryTaskStore`. This means the
A2A task snapshots used by `GetTask` and task-subscription state live only for
the web process lifetime. Evidence Bench's underlying workspace, scientific run,
report, and provenance files remain durable, but an A2A client cannot rely on an
old task id being queryable or resubscribable after a container or process
restart. Production deployments that require protocol-layer task continuity
must replace this store with a durable A2A task-store implementation.

The deployed v0.4.0 transport and scientific MCP evidence is recorded in
[`evals/results/v0.4.0-a2a-live.json`](../evals/results/v0.4.0-a2a-live.json).

## Container boundary

The public web/A2A container is read-only except for `/data` and a size-bounded
`/tmp`, drops all Linux capabilities, enables `no-new-privileges`, limits
processes, memory, and CPUs, and does not mount the Docker socket. It shares no
network with either execution worker. Code reaches the token-authenticated
sandbox only through a fixed TCP 8090 gateway between `sandbox-client` and the
internal, no-egress `sandbox` network. Package requests similarly cross a fixed
TCP 8091 gateway between `package-client` and the internal `packages` network;
package hooks have public HTTP/HTTPS only through the private-address-denying
proxy and no direct route.

The managed browser is a separate public-web-capable workload with a persistent
profile and download directory. It shares no network with the application,
workers, or model endpoints: a public-only proxy provides egress and a narrow
CDP gateway bridges only TCP 9222 onto the application's internal
`browser-client` network. CDP remains unpublished; the application sees the
browser downloads at `/browser-downloads` read-only. The browser starts as root only
long enough to normalize bind-mount ownership, then drops to UID/GID 10002;
Chromium and the desktop are unprivileged. Its outer container drops all
capabilities except the ownership/UID transition set used during that bootstrap,
uses `no-new-privileges`, a read-only root filesystem, bounded tmpfs and shared
memory, and no host-sensitive mounts.

The worker receives only typed language, code, timeout, and path-confined
workspace/run identifiers. It needs a small namespace-oriented capability set, a
root controller process, a setuid bubblewrap binary, and unconfined outer
seccomp/AppArmor so bubblewrap can
construct the stricter inner sandbox. Separating it prevents the browser/API
process from holding those permissions. Worker artifacts are handed back to the
unprivileged web UID after each call. The worker and container runtime remain part
of the security boundary.

A third, non-published package worker is the only service with general outbound
network access. It accepts validated package names/version constraints, not shell
commands or URLs; mounts only one new environment generation and a temporary build
directory into each installer; drops installer hooks to UID 10001; and has no
research-data, application-source, Docker-socket, or credential mounts. PyPI,
CRAN, and Bioconductor are the permitted top-level registries. Package build hooks
can still make their own network requests, so deployments requiring domain-level
egress enforcement should place this service behind an allow-listing proxy.
Its bearer-authenticated cleanup endpoint is reachable only on the internal
package network. Per-generation, cumulative per-workspace, and cumulative global
logical-byte quotas bound retained environments. Admission and cleanup are
serialized against installation. The worker polls the active staging package tree
every 100 ms and terminates the installer process group when the strictest
generation, workspace, or global remaining allowance is exceeded; quota failures
remove the uncommitted staging generation. A fast writer can transiently overshoot
during the polling interval, directory scan, and process-group termination, so
that residual is bounded in time rather than by a fixed byte count. Workspace
deletion performs package cleanup while holding the SQLite deletion transaction,
so a cleanup failure rolls the metadata deletion back.

Inside each analysis call, bubblewrap:

- unshares all namespaces and has no network;
- mounts the workspace read-only at `/workspace`;
- mounts earlier calls read-only at `/prior` and one immutable package generation
  read-only when present;
- permits writes only below `/output` and temporary memory;
- clears the environment, hides credentials, and constrains CPU, address space,
  processes, files, output size, wall time, and call count;
- records code, logs, outputs, hashes, and status.

Do not deploy with `--privileged`, add capabilities, or mount the Docker socket,
host root, user home, SSH configuration, or cloud credentials.

## Shared lab deployment and persistence

For a private lab instance, bind the service only to a private LAN/Tailnet address
or place it behind an authenticated TLS reverse proxy. Browser Basic Auth is
enabled by default: `WEB_USERNAME` and `WEB_PASSWORD` protect the UI and REST API,
while A2A uses a separate token. A shared Basic Auth account is suitable for a
small trusted lab, but deployments requiring per-user attribution should terminate
OIDC/SSO at a reverse proxy.

On a trusted private network, set `WEB_AUTH_ENABLED=false` to remove the browser
login and omit `WEB_USERNAME` and `WEB_PASSWORD`. This opens the UI and REST API
to every client that can reach the bound address. It deliberately does not disable
the A2A bearer token, sandbox-worker token, or package-worker token: those protect
programmatic execution boundaries and do not create a login prompt for lab users.

MCP and browser observations have a separate model-context budget: 64 KiB per
result and 256 KiB cumulatively per research attempt. Complete allowed responses
that exceed the model-visible allowance are stored as hashed run artifacts, while
the model receives metadata and a bounded text preview. Screenshot base64 is
artifact-only. These limits preserve maximum thinking capacity; they do not set
`max_tokens` or shorten the reasoning budget.

Set `EVIDENCE_BENCH_DATA_PATH`, `EVIDENCE_BENCH_ENVIRONMENTS_PATH`, and
`EVIDENCE_BENCH_BROWSER_PATH` to durable host directories. The first contains SQLite metadata, uploads, reports, and run
provenance; the second contains immutable package generations; the third holds
the managed Chromium profile and downloads. Back up all three while
the Compose project is stopped, or snapshot the underlying filesystem. Do not
back up only SQLite: a report's computation and environment hashes refer to files
in the application and environment trees, and manually acquired paper artifacts
may originate in browser downloads.

Set `EVIDENCE_BENCH_DEPLOYMENT_ID` to a stable, unique value for each instance.
The knowledge directory is stamped with that identity and startup fails if a
private volume is accidentally mounted into the lab deployment or vice versa.
Knowledge documents are managed from the WebUI. Every new run snapshots the
selected immutable generations; verified PubMed article Markdown and available
PDFs cited by a deterministically passing run are automatically deduplicated and
added to that instance's library. The WebUI retains a per-document acquisition
history with the controller-verified source, run/workspace, identifiers, and
hashes, including later runs that reused already-known bytes. Search leads and
failed downloads are not promoted.

Where you point these three persistence paths is a deployment choice, not
something this project prescribes. Keep durable application state on backed-up
storage, monitor its capacity, and keep container layers and build temporary
files on suitable local scratch space when the durable volume is shared or
capacity constrained. Site-specific model routes and credentials always remain
in an owner-only `.env`, outside Git, regardless of where persistent data is
mounted.

## Public GitHub release checklist

1. Choose the public repository URL and add it to `CITATION.cff`.
2. Run `rg -n '(10\\.|100\\.64\\.|token|password)' --glob '!uv.lock'` and inspect
   every match for deployment-specific data.
3. Run `uv run pytest -m "not live"` and build the container.
4. Start Compose and run a Python, R, retrieval, and A2A smoke task.
5. Review `SECURITY.md`, the threat model, and scientific limitations.
6. Push the release tag; the release workflow publishes the
   multi-architecture image to `ghcr.io/<owner>/<repository>`.

An upstream A2A ecosystem contribution should point to the public repository and
describe the scientific-analysis skill, auth scheme, supported input/output modes,
and evidence artifacts. The implementation uses the official A2A Python SDK and
advertises protocol version 1.0. Follow the current upstream contribution guide:
open an issue/proposal and align the intended sample shape with maintainers before
significant implementation, then submit the agreed change as a pull request.

The release-ready contribution text and interoperability checklist are maintained
in [`A2A_ECOSYSTEM_SUBMISSION.md`](A2A_ECOSYSTEM_SUBMISSION.md).
