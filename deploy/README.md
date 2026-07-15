# Deployment overlays

`compose.yaml` is the portable release surface. Site-specific routes and credentials
stay outside the repository in a mode-0600 `.env` file.

## Model routes

Qwen and Gemma are independent OpenAI-compatible endpoints. Configure the critic
with `GEMMA_BASE_URL` and `GEMMA_MODEL`; it may run on a separate, more capable
inference host while the Evidence Bench service remains on its application host.
Do not put a site IP, route, or model credential in tracked Compose or
documentation files.

Both recommended aliases enable thinking by default. Keep
`QWEN_ENABLE_THINKING=inherit` and `GEMMA_ENABLE_THINKING=inherit` for such a
gateway so the client does not send backend-specific template controls. Direct
backends can opt into an explicit `true` or `false`. Setting `QWEN_MAX_TOKENS=0` and
`GEMMA_MAX_TOKENS=0` omits the client token ceiling so a compatible proxy can
allocate all remaining context to reasoning and the final answer. If a proxy
cannot combine Qwen thinking with native JSON-schema decoding, set
`QWEN_NATIVE_JSON_SCHEMA=false`; final-channel JSON remains locally validated
and bounded to one repair. `QWEN_REQUEST_TIMEOUT_SECONDS` and
`GEMMA_REQUEST_TIMEOUT_SECONDS` bound wall time without capping reasoning and
remain subject to cooperative cancellation.

Transient connection failures and HTTP 429, 500, 502, 503, and 504 responses
receive at most three total attempts with bounded backoff. Some compatible
gateways report a temporarily unreachable inference host as HTTP 500. A streaming request is retried
only before content has arrived. This tolerates a short model-server restart but
does not hide a prolonged outage: exhausted critic calls leave review
inconclusive rather than approving the result.

When a model endpoint is reachable only through a bastion, copy
`evidence-bench-qwen-tunnel.service.example` into the deploying user's systemd
directory and replace the uppercase placeholders. Bind the local forward to the
Docker bridge address rather than `0.0.0.0`; containers can then use
`http://host.docker.internal:<LOCAL_PORT>/v1` without exposing the model tunnel on
the LAN or Tailnet.

```bash
systemctl --user daemon-reload
systemctl --user enable --now evidence-bench-qwen-tunnel.service
```

For a boot-persistent lab installation, copy
`evidence-bench-compose.service.example` into `/etc/systemd/system`, replace
`INSTALL_DIRECTORY` with the absolute checkout path, then enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now evidence-bench.service
```

The unit complements Compose's `restart: unless-stopped`: systemd reconciles the
whole project after boot, while Docker restarts individual failed containers.
Keep the owner-only `.env` alongside the checkout and never copy it into an image.

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
passwordless noVNC view available to lab members; leave it at `127.0.0.1` if
remote interaction is not needed. Port 9222 is CDP for the application and must
remain internal-only behind the fixed `browser-cdp-gateway`. Chromium itself is
on an internal network and reaches only public HTTP/HTTPS through the managed
egress proxy; do not attach it to `default`, `sandbox`, `packages`, or host model
networks. Never expose noVNC directly to the public Internet or use the shared
profile for personal credentials.

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
test "$(docker compose exec -T browser curl -s -o /dev/null -w '%{http_code}' --proxy http://browser-egress:3128 http://100.64.0.1/)" = 403
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

`provision_env.py` creates or refreshes an owner-only Compose `.env` without
printing credentials. It preserves previously generated browser/A2A/worker secrets,
accepts endpoint settings through its process environment, and copies only the
allow-listed Context7 and Brave values from an optional owner-managed MCP env file.
When those keys and the managed browser are available, all three research
connections are selected by default. Users can still opt out per run; the browser
is confined to the service-owned CDP boundary and public-site egress policy.
