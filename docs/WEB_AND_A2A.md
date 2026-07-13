# Standalone web service and A2A integration

## Workspace model

A workspace is the unit of isolation and scientific provenance. Uploaded inputs
live below its private `files/` directory. Each run writes to a new directory
below `runs/`; the browser never accepts an arbitrary filesystem path. Filenames
are basenames only, symlinks are rejected, and paths are resolved before access.
Only one run may be queued or active in a workspace, so inputs cannot be changed
under an analysis.

The metadata database uses SQLite WAL mode. On service restart, unfinished runs
are marked `interrupted`; they are never silently reported as successful.

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
          "mcp_servers": ["brave-search", "context7"]
        }
      },
      "configuration": {"returnImmediately": false}
    }
  }'
```

For least privilege, A2A defaults to `enable_code=false` and no MCP servers. A
caller must explicitly request those capabilities in message metadata. Supported
MCP names are `context7`, `brave-search`, and `chrome-devtools`. Raw file parts
are accepted when they include a filename; the service deliberately does not
fetch URL parts.

Successful A2A tasks return both a JSON run summary and `report.md` artifacts.
Scientifically inconclusive results are valid completed tasks whose summary keeps
the unresolved status visible. Infrastructure failures are A2A failed tasks.

## Container boundary

The public web/A2A container is read-only except for `/data` and a size-bounded
`/tmp`, drops all Linux capabilities, enables `no-new-privileges`, limits
processes, memory, and CPUs, and does not mount the Docker socket. Code is sent
over a private internal Docker network to a separate token-authenticated sandbox
worker with no published port and no Internet route.

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
or place it behind an authenticated TLS reverse proxy. `WEB_USERNAME` and
`WEB_PASSWORD` protect the browser; A2A uses a separate token. A shared Basic Auth
account is suitable for a small trusted lab, but deployments requiring per-user
attribution should terminate OIDC/SSO at a reverse proxy.

Set `EVIDENCE_BENCH_DATA_PATH` and `EVIDENCE_BENCH_ENVIRONMENTS_PATH` to durable
host directories. The first contains SQLite metadata, uploads, reports, and run
provenance; the second contains immutable package generations. Back up both while
the Compose project is stopped, or snapshot the underlying filesystem. Do not
back up only SQLite: a report's computation and environment hashes refer to files
in both trees.

The UMED `s1` deployment keeps these bind mounts below `/data-onix/EvidenceBench`.
That NAS is persistent but shared and relatively full, so operators must monitor
capacity and must not use it as high-volume scratch. Container images and build
temporary files remain on local `/var`/tmpfs; only durable application state is
stored on Onix. Site-specific model routes and credentials remain in an owner-only
`.env`, outside Git.

## Public GitHub release checklist

1. Choose the public repository URL and add it to `CITATION.cff`.
2. Run `rg -n '(10\\.|100\\.64\\.|token|password)' --glob '!uv.lock'` and inspect
   every match for deployment-specific data.
3. Run `uv run pytest -m "not live"` and build the container.
4. Start Compose and run a Python, R, retrieval, and A2A smoke task.
5. Review `SECURITY.md`, the threat model, and scientific limitations.
6. Push the initial release and tag `v0.3.0`; the release workflow publishes the
   multi-architecture image to `ghcr.io/<owner>/<repository>`.

An upstream A2A ecosystem contribution should point to the public repository and
describe the scientific-analysis skill, auth scheme, supported input/output modes,
and evidence artifacts. The implementation uses the official A2A Python SDK and
advertises protocol version 1.0.

The release-ready contribution text and interoperability checklist are maintained
in [`A2A_ECOSYSTEM_SUBMISSION.md`](A2A_ECOSYSTEM_SUBMISSION.md).
