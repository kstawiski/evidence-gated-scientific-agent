# Delegate integration contract

The explicit `umed-task` lane uses this project after its offline, live,
sandbox, prompt-privacy, and adapter tests pass. `local-scientist` is a
deprecated compatibility alias. Automatic routing remains disabled until the
comparative promotion gates in `MVP_SCOPE.md` pass.

The integration is a thin adapter, not a second implementation:

```text
delegate.sh
  -> private prompt file/stdin
  -> scientific-agent preflight cache
  -> scientific-agent run [--enable-code] [--mcp <explicit subset or empty opt-out>]
  -> inspect run_result.json status
  -> return report.md plus provenance path
```

Required routing behavior:

- The canonical agent name is `umed-task`; do not overload `local-simple`,
  which is intentionally non-scientific.
- No paid-provider fallback.
- No premium verifier by default; Gemma is already the independent local critic,
  while deterministic validation remains the acceptance authority.
- Any unresolved status returns non-zero/provisional to the orchestrator.
- Research tools are explicit. The adapter passes `--mcp ''` for a file-only or
  no-network task, or an exact subset of Context7, Brave Search, and the
  service-owned Chrome DevTools connection when the objective requires them.
- Computation is a distinct route capability. The adapter may pass `--enable-code`
  only for an explicitly computational task and must report that authorization in
  its result metadata; it must never infer shell or package-install authority.
- Preserve the delegate skill's private prompt transport and clean environment.
- Log exact model IDs, endpoint catalogue checks, package lock hashes, result
  status, and provenance directory.
- Preserve the two-stage ADK research/native-schema boundary; do not collapse
  tool use and structured report output into one model call on this serving stack.
- Do not repeat an identical task after a schema omission. The controller permits
  up to four evidence-repair and independent re-audit rounds by default;
  unresolved output fails closed and returns nonzero.

Integration tests must prove that the adapter cannot enable arbitrary shell or
workspace-write tools,
cannot inject an MCP key into argv/logs, and cannot report success for
`requires_more_evidence`, `inconclusive`, or infrastructure failure.

## Installable lab client skill

The standalone browser/API deployment also ships a thin agent skill at
`skills/evidence-bench/`. Lab members can copy that folder to
`~/.claude/skills/evidence-bench` or `${CODEX_HOME:-~/.codex}/skills/evidence-bench`.
Its stdlib client targets `http://10.20.102.122:8070`, where the browser and REST
API are passwordless on the trusted network. It does not use or embed the A2A
bearer token.

The skill is distinct from automatic `/delegate` routing: invocation is explicit,
the service creates a persistent isolated workspace, and the calling agent polls
and downloads the resulting provenance bundle. Other deployments can set
`EVIDENCE_BENCH_URL` or pass `--base-url` without modifying the client.
