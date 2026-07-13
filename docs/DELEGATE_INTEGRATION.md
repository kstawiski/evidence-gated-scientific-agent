# Delegate integration contract

The explicit `local-scientist` lane may use this project after its offline,
live, sandbox, prompt-privacy, and adapter tests pass. Automatic routing remains
disabled until the comparative promotion gates in `MVP_SCOPE.md` pass.

The eventual integration should be a thin adapter, not a second implementation:

```text
delegate.sh
  -> private prompt file/stdin
  -> scientific-agent preflight cache
  -> scientific-agent run [--enable-code] --mcp <explicit set>
  -> inspect run_result.json status
  -> return report.md plus provenance path
```

Required routing behavior:

- The distinct agent name is `local-scientist`; do not overload
  `local-simple`, which is intentionally non-scientific.
- No paid-provider fallback.
- No premium verifier by default; Gemma is already the independent local critic,
  while deterministic validation remains the acceptance authority.
- Any unresolved status returns non-zero/provisional to the orchestrator.
- MCP is opt-in per task. Chrome remains separately explicit.
- Computation is a distinct route capability. The adapter may pass `--enable-code`
  only for an explicitly computational task and must report that authorization in
  its result metadata; it must never infer shell or package-install authority.
- Preserve the delegate skill's private prompt transport and clean environment.
- Log exact model IDs, endpoint catalogue checks, package lock hashes, result
  status, and provenance directory.
- Preserve the two-stage ADK research/native-schema boundary; do not collapse
  tool use and structured report output into one model call on this serving stack.
- Do not repeat an identical task after a schema omission. The controller permits
  one bounded evidence-repair round; unresolved output returns nonzero.

Integration tests must prove that the adapter cannot enable arbitrary shell or
workspace-write tools,
cannot inject an MCP key into argv/logs, and cannot report success for
`requires_more_evidence`, `inconclusive`, or infrastructure failure.
