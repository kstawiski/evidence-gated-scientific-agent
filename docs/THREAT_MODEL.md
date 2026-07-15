# Threat model

## Protected assets

- Raw scientific data and the assigned workspace.
- Fleet credentials and MCP API keys.
- Host files, services, and private Tailnet resources.
- Browser sessions, downloads, and cookies in the service-owned shared Chromium
  instance.
- Scientific integrity of the report, sources, claims, and provenance log.

## Trust boundaries

- Qwen and Gemma output is untrusted input.
- Retrieved web pages and MCP responses are untrusted input and may contain
  prompt injection.
- MCP subprocess packages are third-party code and are pinned exactly.
- The deterministic Python controller is the security and workflow authority.
- The service-owned Chromium profile and human-controlled page state are mutable
  and are not scientific evidence until acquired and registered by the controller.

## Current controls

- The models receive no arbitrary host shell, operating-system package manager,
  Git, deletion, or workspace-write tool. Scientific package installation is
  limited to typed, validated PyPI/CRAN/Bioconductor requests.
- Workspace tools resolve symlinks and reject any path outside the assigned root.
- Text reads are capped at 2 MiB and searches at 200 hits.
- Every tool name is checked against an allow-list before execution. Python/R are
  absent unless the human caller explicitly sets `--enable-code`.
- Python and R execute in fresh bubblewrap namespaces with `/workspace` read-only,
  prior outputs and an immutable package generation read-only, one per-call
  `/output` bind mount, an empty `/proc`, minimal `/dev`, a temporary home, and
  only selected language runtimes/libraries mounted.
- The computation namespace unshares networking and clears the inherited
  environment. Runtime probes and adversarial tests verify that outbound sockets,
  `/etc/passwd`, inherited test secrets, and workspace writes are unavailable.
- The application never shares a network with either execution worker. It reaches
  the sandbox only through a capability-dropped, fixed-destination TCP 8090
  gateway spanning `sandbox-client` and the internal `sandbox` network. The
  sandbox network has no egress route.
- `prlimit` plus in-sandbox process limits cap CPU, address space, processes,
  open files, core dumps, and output file size. The controller also caps wall time,
  source size, total output size, and calls per attempt.
- Automatic PMC/PubTator full text is rights-gated. A PMCID is insufficient: the
  controller requires a matching PMC Open Access-subset record, an explicit
  reusable license, and an allow-listed OA route before requesting or persisting
  full text. Otherwise only PubMed metadata and abstract text are retained with
  an explicit unavailable reason.
- The application never parses untrusted PDFs with Poppler. PDF text extraction
  is a token-authenticated sandbox-worker operation, and a missing worker fails
  closed. The worker accepts only a regular non-symlink PDF directly under
  `/data/workspaces/<uuid>/files/references/pdfs/`, rejects traversal and every
  broader/nested path, and runs `pdftotext` in a new no-network
  `bwrap --unshare-all` namespace for each request. Only that PDF is mounted
  read-only; one private temporary output, narrow runtime libraries/font data,
  and `/dev/null` are exposed under CPU, memory, process, file, and wall limits.
- Non-regular output files, including symlinks, are rejected. Successful outputs,
  source scripts, and captured logs receive SHA-256 hashes and private permissions.
- Computed claims must cite an exact generated output from a successful execution;
  source code and stdout/stderr are retained for audit but cannot satisfy this gate.
- Browser navigation accepts only public HTTP(S) destinations; localhost,
  private IPs, Tailnet addresses, link-local destinations, and file URLs are blocked.
- Chrome tools remain policy-disabled unless `chrome-devtools` is selected for
  the run. It is selected by default alongside Context7 and Brave Search; callers
  handling confidential tasks must explicitly opt out or choose a narrower set.
- Chromium has only the internal `browser-control` network. It does not share a
  network with the application, workers, or model endpoints and has no direct
  Internet route. A capability-dropped Squid service is its only egress path;
  that proxy resolves destinations itself, denies loopback, RFC1918, link-local,
  Tailnet, reserved, multicast, and local IPv6 ranges, permits plain HTTP only on
  port 80, and permits CONNECT only to port 443. Its external-capable
  `public-egress` bridge is dedicated to the proxy and is not the application's
  `default` network.
- Chromium CDP is not published and is not placed directly on the application
  network. A capability-dropped, fixed-destination TCP gateway spans
  `browser-control` and the separate internal `browser-client` network and
  proxies only TCP 9222 from the application to `browser:9222`. The passwordless
  noVNC port remains a separate human-only surface that deployments must bind to
  a trusted private address.
- The browser container has no research workspace, application/model secret,
  Docker socket, SSH material, or host-home mount. The application mounts only
  browser downloads read-only; it cannot modify the profile or a downloaded PDF.
- MCP children receive only basic process variables and their own required key.
- Secrets are loaded only from a regular, non-symlinked, current-user-owned
  mode-0600 file and are never written to logs.
- Tool results are logged by hash and byte count, not by copying potentially
  sensitive response bodies into the event ledger. Public retrieval bodies are
  preserved separately as mode-0600, size-bounded evidence artifacts inside the
  mode-0700 run directory.
- A supported claim's external source URL and retrieval date must match raw MCP
  evidence; changing the model-selected claim type cannot bypass this check.
- Repair loops are bounded.
- Sustained contiguous repetition in either final output or private reasoning
  closes the model stream and consumes the one bounded structured-repair retry.
  A match must cover at least 2 KiB and repeat at two consecutive byte-based
  checkpoints. Only a 12,000-character in-memory suffix is inspected; private
  reasoning is never logged or persisted.
- A second complete top-level JSON value is an immediate structured-output loop
  and consumes the same single repair allowance; objects need not be byte-identical.
- A second repetitive or invalid plan-critic sample produces an explicit
  `plan-critic-unavailable` finding and an `inconclusive` terminal plan. Research
  cannot start without a valid independent plan audit.
- Model-visible tool results are bounded to 64 KiB each and 256 KiB cumulatively
  per research attempt. Complete permitted oversized responses are hashed and
  retained as mode-0600 run artifacts; screenshot base64 is never injected into
  text context. This limits hostile or accidental context exhaustion without
  lowering the model's reasoning-token allowance.
- Every report attempt retains the draft, deterministic findings, critic report,
  and cumulative evidence. A critic transport/schema failure produces an explicit
  inconclusive review rather than implicit approval; unexpected run failures retain
  a hashed partial provenance bundle for diagnosis.
- Cumulative research-budget exhaustion stops additional tools. A repair may
  continue from already recorded evidence, but deterministic and independent
  review still decide its status; budget exhaustion cannot approve a report.
- Package installation is a separate token-authenticated service with no data or
  model/API secret mount. It does not share an application network: a
  capability-dropped fixed TCP 8091 gateway spans `package-client` and the
  internal `packages` network. Package subprocesses receive the fixed public-only
  proxy URL, and the worker network has no direct egress. Installer hooks see only
  one staging generation and tmpfs, run as UID 10001, and cannot access sibling
  workspace environments. Direct URLs,
  VCS/path requirements, flags, escaping symlinks, missing requested versions,
  and over-size environments fail closed.
- Package generations are immutable and atomically selected. Their exact package
  inventory, installed-tree hash, and lock file are copied into computation
  provenance before the resolved generation is mounted. The package worker also
  polls active staging bytes and terminates the installer process group on the
  first observed generation, workspace, or global quota breach, then removes the
  staging generation. Enforcement is polling-based: transient overshoot remains
  possible for the polling, directory-scan, and termination latency and has no
  fixed byte bound.

## Residual risks

- Anyone who can reach the passwordless noVNC port can view and control the
  shared browser and use active publisher sessions. Binding it publicly is an
  unsafe deployment. Users must not enter personal credentials into this shared
  profile.
- Chromium processes untrusted web content and runs with its in-browser sandbox
  disabled inside a separately constrained container because the outer Compose
  security profile blocks the namespaces required by Chromium's sandbox. A
  browser exploit could compromise that container, its profile, and downloads;
  it should not expose host-sensitive mounts or internal worker networks.
- Manually downloaded PDFs are untrusted input. A successful download or human
  bot-check clearance does not establish identity, integrity, or scientific
  relevance; the acquisition controller must validate and hash the imported file.
  These files remain `private_user_provided` artifacts with a terms warning; their
  presence does not grant Evidence Bench permission to redistribute publisher text.
- Browser egress depends on Squid's destination-address ACLs and the Compose
  internal-network guarantee. A proxy/runtime defect could weaken this boundary;
  releases therefore probe public HTTPS success, direct-egress failure, private
  destination denial, application-network DNS isolation, and CDP gateway health.
- Brave and Context7 send queries to external services. Tasks containing
  confidential data must not enable these tools.
- NPM and Python dependencies execute in the controller environment. Exact pins,
  lockfiles, package review, and isolated deployment are still required.
- The base sandbox uses the image's curated Python and R installations as read-only
  runtime mounts; a malicious native library in those trusted roots remains outside
  this boundary. Workspace additions are locked, but the base-image digest must
  also be retained by deployment tooling for full binary reproducibility.
- Canonical top-level package registries do not constrain secondary downloads made
  by package build hooks. The public-only proxy blocks private targets but permits
  arbitrary public port-80/443 destinations; stronger deployments need a registry
  hostname allow-list and a download-then-offline-build pipeline. Docker's embedded
  DNS also remains a potential low-bandwidth exfiltration channel for a malicious
  hook even though direct IP egress is unavailable.
- An arbitrary package may require a native library absent from the public image;
  installation or later import then fails explicitly.
- The outer fleet container does not permit bubblewrap's additional
  `--disable-userns` hardening. Bubblewrap still unshares all available namespaces,
  but deployment on a dedicated host should disable nested user namespaces and use
  a seccomp profile as an additional layer.
- `/proc` is intentionally empty because the outer container blocks a fresh proc
  mount. Analyses that require procfs will fail closed rather than receive host
  process information.
- LLM structured output can be schema-valid but scientifically wrong; retrieved
  sources, deterministic checks, and expert review remain necessary.
- URL presence proves retrieval, not claim entailment. Gemma audits entailment,
  and profile-specific deterministic entailment/source validators remain future work.
- Successful isolated execution does not prove scientific validity. Profile-specific
  validators and a controller-driven clean-environment rerun remain future gates.

## Fail-closed rules

- A deterministic validation failure cannot be overridden by either model.
- A failed MCP startup is an infrastructure failure, not permission to silently
  continue without requested evidence access.
- An unresolved Gemma blocking finding yields `requires_more_evidence` or
  `inconclusive`, never synthetic consensus.
- No model confidence score grants additional tool authority.
- A model cannot enable Python/R itself; without the caller's flag, those tool calls
  are policy-denied.
