# Persistent deployment

This guide is for a long-running Evidence Bench instance with existing Qwen and
Gemma endpoints. For a laptop or workstation that should install Ollama and
choose models automatically, use the
[macOS, Linux, and WSL2 local setup tutorial](../docs/LOCAL_SETUP.md).

Evidence Bench has two Compose paths:

- `compose.yaml` is the base configuration and contains source-build targets.
- `compose.yaml` plus `compose.local.yaml` replaces those targets with versioned
  public GHCR images. This is the recommended release deployment.

Site routes, model credentials, and generated service credentials stay outside
the repository in an owner-only `.env` file.

## Persistent quick start

### 1. Prepare the host

Use a Linux host with Docker Engine and Docker Compose v2. The Qwen executor and
Gemma critic must expose OpenAI-compatible endpoints that are reachable from a
container. Keep the WebUI on loopback until TLS and access control are ready.

```bash
git clone https://github.com/kstawiski/evidence-gated-scientific-agent.git
cd evidence-gated-scientific-agent
docker info
docker compose version
```

Confined Python/R execution requires Linux namespaces and the capabilities
declared for the two worker containers. Do not remove those settings, run the
whole stack as privileged, or mount the Docker socket into a service.

### 2. Create the private configuration

The provisioning helper generates independent browser, A2A, sandbox-worker, and
package-worker credentials without printing them. Replace the example routes,
models, and contact email below. Export API keys too when an endpoint requires
one.

```bash
export SCIENTIFIC_AGENT_PUBLIC_URL=http://127.0.0.1:8080
export EVIDENCE_BENCH_DEPLOYMENT_ID=lab-prod
export QWEN_BASE_URL=http://host.docker.internal:8000/v1
export QWEN_MODEL=Qwen/Qwen3.6-27B
export GEMMA_BASE_URL=http://host.docker.internal:8001/v1
export GEMMA_MODEL=gemma4-12b-it
export SCIENTIFIC_AGENT_NCBI_EMAIL=your.name@institution.edu

python3 deploy/provision_env.py --output .env
unset QWEN_API_KEY GEMMA_API_KEY
```

Replace the illustrative NCBI address with a real monitored maintainer address.
PubMed/PMC retrieval requires `SCIENTIFIC_AGENT_NCBI_EMAIL`; Evidence Bench
checks basic email syntax but cannot verify deliverability. Review `.env.example`
for optional routes and resource limits. Keep `.env` mode 0600 and never copy it
into an image.

For durable host-managed storage, create three separate directories and export
their absolute paths before running `provision_env.py`:

```bash
export EVIDENCE_BENCH_DATA_PATH=/srv/evidence-bench/data
export EVIDENCE_BENCH_ENVIRONMENTS_PATH=/srv/evidence-bench/environments
export EVIDENCE_BENCH_BROWSER_PATH=/srv/evidence-bench/browser
```

The application container runs as UID 10001 and must be able to write the data
directory. The package worker manages the environments directory. The browser
path contains a credential-bearing shared Chromium profile; restrict host access
accordingly.

### 3. Start a released version

The overlay defaults to the repository's current version. To pin another
published version, add `EVIDENCE_BENCH_VERSION=<version>` to `.env` before the
pull. Always validate the merged configuration:

```bash
docker compose --env-file .env \
  -f compose.yaml -f compose.local.yaml config --quiet
docker compose --env-file .env \
  -f compose.yaml -f compose.local.yaml pull
docker compose --env-file .env \
  -f compose.yaml -f compose.local.yaml up -d --no-build
```

Wait for the service, then verify both application health and model access:

```bash
curl -fsS http://127.0.0.1:8080/healthz
docker compose --env-file .env \
  -f compose.yaml -f compose.local.yaml \
  exec -T evidence-bench scientific-agent preflight --mcp ""
grep -E '^(WEB_USERNAME|WEB_PASSWORD)=' .env
```

Open <http://127.0.0.1:8080> only on the host. Configure the authenticated TLS
route below before making the service reachable from another network.

To stop containers while preserving volumes and bind-mounted state:

```bash
docker compose --env-file .env \
  -f compose.yaml -f compose.local.yaml down
```

### Build the checked-out source instead

Developers can omit the release overlay and build the exact checkout. This is a
source build, not a pull of published images:

```bash
docker compose --env-file .env -f compose.yaml config --quiet
docker compose --env-file .env -f compose.yaml up --build -d
```

## Model routes

Qwen and Gemma are independent OpenAI-compatible endpoints. Configure the critic
with `GEMMA_BASE_URL` and `GEMMA_MODEL`; it may run on a separate, more capable
inference host while the Evidence Bench service remains on its application host.
Do not put a site IP, route, or model credential in tracked Compose or
documentation files.

Both recommended aliases enable thinking by default. Keep
`QWEN_ENABLE_THINKING=inherit` and `GEMMA_ENABLE_THINKING=inherit` for such a
gateway so the client does not send backend-specific template controls. Direct
backends can opt into an explicit `true` or `false`. Setting
`QWEN_MAX_TOKENS=0` and `GEMMA_MAX_TOKENS=0` omits the client token ceiling so a
compatible proxy can
allocate all remaining context to reasoning and the final answer. If a proxy
cannot combine Qwen thinking with native JSON-schema decoding, set
`QWEN_NATIVE_JSON_SCHEMA=false`; final-channel JSON remains locally validated
and bounded to one repair. `QWEN_REQUEST_TIMEOUT_SECONDS` and
`GEMMA_REQUEST_TIMEOUT_SECONDS` bound wall time without capping reasoning and
remain subject to cooperative cancellation.

Transient connection failures and HTTP 429, 500, 502, 503, and 504 responses
receive at most three total attempts with bounded backoff. Some compatible
gateways report a temporarily unreachable inference host as HTTP 500. A
streaming request is retried only before content has arrived. This tolerates a
short model-server restart but does not hide a prolonged outage: exhausted
critic calls leave review inconclusive rather than approving the result.

When a model endpoint is reachable only through a bastion, copy and rename the
example into the deploying user's systemd directory, then replace every uppercase
placeholder:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/evidence-bench-qwen-tunnel.service.example \
  ~/.config/systemd/user/evidence-bench-qwen-tunnel.service
```

Bind the local forward to the Docker bridge address rather than `0.0.0.0`.
Containers can then use `http://host.docker.internal:<LOCAL_PORT>/v1` without
exposing the tunnel on the LAN or Tailscale private network (Tailnet).

```bash
systemctl --user daemon-reload
systemctl --user enable --now evidence-bench-qwen-tunnel.service
```

For a boot-persistent release installation, copy and rename the unit, replace
`INSTALL_DIRECTORY` with the absolute checkout path, and enable the matching
service name:

```bash
sudo cp deploy/evidence-bench-compose.service.example \
  /etc/systemd/system/evidence-bench.service
# Edit /etc/systemd/system/evidence-bench.service before continuing.
sudo systemctl daemon-reload
sudo systemctl enable --now evidence-bench.service
```

The unit complements Compose's `restart: unless-stopped`: systemd reconciles the
whole project after boot, while Docker restarts individual failed containers.
Keep the owner-only `.env` alongside the checkout and never copy it into an image.

## TLS reverse proxy and remote browser access

For a remote lab deployment, keep `WEB_BIND_ADDRESS=127.0.0.1` and terminate TLS
at a reverse proxy on the same host. Preserve the built-in WebUI password, or use
OIDC/SSO at the proxy when individual attribution is required. Configure the
proxy for streaming responses and set `SCIENTIFIC_AGENT_PUBLIC_URL` to the final
`https://` origin.

The research browser uses passwordless noVNC (a browser-based VNC client). Do
not expose port 6080 directly to the public Internet. Either leave
`BROWSER_BIND_ADDRESS=127.0.0.1` for host-only access or put noVNC behind its own
authenticated HTTPS route and set `BROWSER_PUBLIC_URL` to that URL. An HTTPS
workbench cannot embed an HTTP noVNC endpoint because browsers block mixed
content. The proxy must support WebSocket upgrades for noVNC.

The [Web, REST, A2A, and browser contract](../docs/WEB_AND_A2A.md#shared-lab-deployment-and-persistence)
explains authentication behavior. The
[managed-browser section](../docs/WEB_AND_A2A.md#managed-interactive-research-browser)
describes the noVNC, Chrome DevTools Protocol (CDP), and mixed-content
boundaries. Never publish CDP port 9222; it is an internal application
capability.

## Backup, upgrade, and rollback

Use durable host paths for predictable backups. Before an upgrade, stop the
stack and take one filesystem snapshot or backup containing the data,
environments, and browser directories together. Treat the browser directory as
credential-bearing. Start the same version again after confirming the backup is
readable.

To upgrade released images:

1. Read the target release notes and record the current Git commit and
   `EVIDENCE_BENCH_VERSION`.
2. Stop the stack and make the coordinated backup described above.
3. Update the checkout so its Compose overlay matches the target release, then
   set the target `EVIDENCE_BENCH_VERSION` in `.env`.
4. Run the merged `config --quiet`, `pull`, and `up -d --no-build` commands from
   the persistent quick start.
5. Repeat the health check, model preflight, and browser boundary checks.

For an application rollback, restore the previous checkout and image version,
then recreate the stack. If the failed upgrade changed stored data, stop the
stack and restore the matching three-directory backup before starting the older
version. Do not mix an older application with data written by a newer release
unless that release explicitly documents compatibility.

## Multiple isolated instances on one host

Use a separate checkout, mode-0600 `.env`, Compose project name,
`EVIDENCE_BENCH_DEPLOYMENT_ID`, published web and noVNC ports, and all three
persistent paths for every instance. For example:

```bash
docker compose -p evidence-bench-prod --env-file .env \
  -f compose.yaml -f compose.local.yaml config --quiet
docker compose -p evidence-bench-prod --env-file .env \
  -f compose.yaml -f compose.local.yaml up -d --no-build
```

Do not point two projects at the same data, environment, or browser directory.
The A2A, sandbox-worker, and package-worker tokens must also be generated per
instance. A private instance can therefore be upgraded or stopped without
interrupting the shared lab service, and a lab workspace cannot appear in the
private workspace index.

Do not mount the Docker socket or a host home directory into either service.

For durable bind mounts, set these values in the private `.env`:

```text
EVIDENCE_BENCH_DATA_PATH=/durable/path/evidence-bench/data
EVIDENCE_BENCH_ENVIRONMENTS_PATH=/durable/path/evidence-bench/environments
EVIDENCE_BENCH_BROWSER_PATH=/durable/path/evidence-bench/browser
```

Package storage is bounded by parallel cumulative logical-byte and filesystem-
entry quotas. Each first limit covers one immutable generation, each second
limit covers all generations belonging to one workspace, and each third limit
covers the complete package-environment tree:

```text
SCIENTIFIC_AGENT_MAX_ENVIRONMENT_BYTES=21474836480
SCIENTIFIC_AGENT_MAX_WORKSPACE_ENVIRONMENT_BYTES=42949672960
SCIENTIFIC_AGENT_MAX_TOTAL_ENVIRONMENT_BYTES=214748364800
SCIENTIFIC_AGENT_MAX_ENVIRONMENT_ENTRIES=250000
SCIENTIFIC_AGENT_MAX_WORKSPACE_ENVIRONMENT_ENTRIES=500000
SCIENTIFIC_AGENT_MAX_TOTAL_ENVIRONMENT_ENTRIES=2500000
```

Choose the global ceiling from the durable volume's free capacity and expected
concurrency, monitor it, and raise it only deliberately. Deleting a workspace
calls the authenticated package worker cleanup before committing its metadata
deletion, removing all Python and R generations for that workspace; a cleanup
failure leaves the workspace and its run records intact.

The web UID (10001) must be able to write the data path. The package worker owns
the environment path through its confined root controller. On shared or
root-squashed storage, keep application source and Docker build cache on local
storage and reserve the durable mount for application data, immutable package
generations, and browser state.

The browser path contains a shared Chromium profile plus publisher downloads.
The application mounts only its `downloads/` child at `/browser-downloads:ro`.
Set `BROWSER_BIND_ADDRESS` to a private LAN/Tailnet address to make the
passwordless noVNC browser view available to lab members; leave it at
`127.0.0.1` if remote interaction is not needed. Port 9222 carries the Chrome
DevTools Protocol (CDP) used by the application and must remain internal-only
behind the fixed `browser-cdp-gateway`. Chromium itself is on an internal network
and reaches only public HTTP/HTTPS through the managed egress proxy; do not
attach it to `default`, `sandbox`, `packages`, or host model networks. Never
expose noVNC directly to the public Internet or use the shared profile for
personal credentials.

If one host must serve both a private LAN and Tailnet, bind
`WEB_BIND_ADDRESS=0.0.0.0` and `BROWSER_BIND_ADDRESS=0.0.0.0` only behind host or
upstream firewall rules that limit the published workbench and noVNC ports to
those trusted interfaces. Leave `BROWSER_PUBLIC_URL` empty so the UI derives the
browser endpoint from each user's current hostname. Never use this multi-interface
setting on an unfiltered publicly reachable host.

After each browser-image upgrade, verify the deployed boundary from the Compose
project directory. The first call must succeed; direct egress and each private
proxy target must fail, while CDP must succeed only through the gateway:

```bash
docker compose exec browser curl --fail --proxy http://browser-egress:3128 https://example.com/
! docker compose exec browser curl --fail --connect-timeout 3 https://example.com/
private_status=$(docker compose exec -T browser \
  curl -s -o /dev/null -w '%{http_code}' \
  --proxy http://browser-egress:3128 http://100.64.0.1/)
test "$private_status" = 403
docker compose exec browser-cdp-gateway curl --fail http://127.0.0.1:9222/json/version
```

Also confirm `docker compose exec browser getent hosts evidence-bench` and the
equivalent worker/model hostnames fail to resolve. The release CI performs the
same checks with an application-network probe container.

The application reaches the sandbox and package services only as
`sandbox-gateway:8090` and `environment-gateway:8091`; never add the application
to `sandbox` or `packages`. Package hooks run on the internal `packages` network
and receive `http://browser-egress:3128` as their only public-web route. After a
Compose change, repeat the public/proxy/private probes above from a temporary
container attached to `packages`, and confirm direct HTTPS fails there too. The
proxy alone joins `public-egress`; never attach `browser-egress` to `default`, and
verify `docker compose exec browser-egress getent hosts evidence-bench` fails.

On root-squashed NFS, pre-create the browser directories as the unprivileged
deployment user, set `BROWSER_RUNTIME_USER=<host-uid>:<host-gid>`, and set
`BROWSER_DOWNLOADS_MODE=0755`. The Chromium profile/home remain mode 0700;
only the paper inbox becomes world-readable, and the application still mounts
that inbox read-only. Do not use mode 0777.

Rerunning `provision_env.py` refreshes the owner-only Compose `.env` without
printing credentials. It preserves previously generated browser/A2A/worker
secrets, accepts endpoint settings through its process environment, and copies
only the allow-listed Context7 and Brave values from an optional owner-managed
Model Context Protocol (MCP) env file. When those keys and the managed browser
are available, all three research connections are selected by default. Users
can still opt out per run; the browser is confined to the service-owned CDP
boundary and public-site egress policy.
