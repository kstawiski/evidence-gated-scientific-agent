# Threat model

## Protected assets

- Raw scientific data and the assigned workspace.
- Fleet credentials and MCP API keys.
- Host files, services, and private Tailnet resources.
- Browser sessions and cookies in the shared Chrome instance.
- Scientific integrity of the report, sources, claims, and provenance log.

## Trust boundaries

- Qwen and Gemma output is untrusted input.
- Retrieved web pages and MCP responses are untrusted input and may contain
  prompt injection.
- MCP subprocess packages are third-party code and are pinned exactly.
- The deterministic Python controller is the security and workflow authority.
- The shared Chrome service is external mutable state.

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
- `prlimit` plus in-sandbox process limits cap CPU, address space, processes,
  open files, core dumps, and output file size. The controller also caps wall time,
  source size, total output size, and calls per attempt.
- Non-regular output files, including symlinks, are rejected. Successful outputs,
  source scripts, and captured logs receive SHA-256 hashes and private permissions.
- Computed claims must cite an exact generated output from a successful execution;
  source code and stdout/stderr are retained for audit but cannot satisfy this gate.
- Browser navigation accepts only public HTTP(S) destinations; localhost,
  private IPs, Tailnet addresses, link-local destinations, and file URLs are blocked.
- Chrome tools are disabled unless the caller explicitly includes
  `chrome-devtools` for the run.
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
- Every report attempt retains the draft, deterministic findings, critic report,
  and cumulative evidence. A critic transport/schema failure produces an explicit
  inconclusive review rather than implicit approval; unexpected run failures retain
  a hashed partial provenance bundle for diagnosis.
- Package installation is a separate token-authenticated internal service with no
  data or secret mount. Installer hooks see only one staging generation and tmpfs,
  run as UID 10001, and cannot access sibling workspace environments. Direct URLs,
  VCS/path requirements, flags, escaping symlinks, missing requested versions,
  and over-size environments fail closed.
- Package generations are immutable and atomically selected. Their exact package
  inventory, installed-tree hash, and lock file are copied into computation
  provenance before the resolved generation is mounted.

## Residual risks

- A public host can resolve or redirect to a private address after the policy
  check (DNS rebinding/redirect). Chrome isolation must be strengthened before
  using it with sensitive browser profiles.
- Brave and Context7 send queries to external services. Tasks containing
  confidential data must not enable these tools.
- NPM and Python dependencies execute in the controller environment. Exact pins,
  lockfiles, package review, and isolated deployment are still required.
- The base sandbox uses the image's curated Python and R installations as read-only
  runtime mounts; a malicious native library in those trusted roots remains outside
  this boundary. Workspace additions are locked, but the base-image digest must
  also be retained by deployment tooling for full binary reproducibility.
- Canonical top-level package registries do not constrain secondary downloads made
  by package build hooks. Stronger deployments need an egress allow-list/proxy and
  a download-then-offline-build pipeline.
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
