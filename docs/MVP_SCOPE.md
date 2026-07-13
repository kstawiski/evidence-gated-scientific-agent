# MVP scope and acceptance gates

## Implemented

1. Typed task, plan, action, verification, source, claim, report, and run schemas.
2. Independent blinded Qwen and Gemma planning branches in an ADK 2 graph.
3. Deterministic plan linting before synthesis.
4. Qwen master-plan synthesis with explicit disagreement resolution records.
5. Independent Gemma plan audit and one bounded plan-repair attempt.
6. Two-stage Qwen execution: an ADK tool-using research/analysis pass followed by
   native strict-schema report assembly.
7. Context7, Brave Search, optional isolated-context Chrome DevTools, and bounded
   workspace inspection.
8. A deterministic callback on every tool call; non-allow-listed tools and
   private/non-HTTP browser targets fail closed.
9. Claim–source referential integrity plus URL/date matching against actual MCP
   output, independent of the claim type selected by Qwen.
10. Independent Gemma report audit with controller retrieval evidence and at
    most one repair round.
11. Append-only event log, private normalized MCP evidence artifacts, exact run
    configuration, and SHA-256 artifact manifest.
12. Explicit `--enable-code` authorization for typed Python and R tools backed by
    bubblewrap, `prlimit`, an unshared network namespace, clean environment,
    read-only workspace, and a dedicated writable output mount.
13. CPU, address-space, process, open-file, output-size, wall-time, source-size,
    and per-attempt call limits, with rejected symlink/device outputs.
14. Computation records containing source/log/output hashes, and deterministic
    rejection of computed claims that cite anything except a successful generated
    output artifact.

## Explicitly deferred

- Arbitrary host shell commands, Git mutation, package installation, workspace
  writes, and database calls.
- User-selectable environments or dependency installation inside a run.
- Domain validators for RNA-seq, variants, survival analysis, and figures.
- Clean-environment computational reruns.
- Visual Gemma audit of generated figures.
- Human approval UI for irreversible or decision-critical actions.
- Persistent services, authentication, or multi-user isolation.
- Automatic task routing into the local-scientist lane.

## Implemented execution-tool gate

- Python and R are separate typed tools; no model-supplied host command line is
  accepted.
- Each call uses a fresh bubblewrap namespace with no network, a read-only input
  mount, and one dedicated output directory.
- The controller enforces resource and call budgets outside model judgment.
- Tests cover symlink output, environment leakage, host-file visibility,
  workspace mutation, timeout, network access, and call-budget exhaustion.
- The tools remain absent from the allow-list unless the caller sets
  `--enable-code`; model agreement cannot grant that authority.

## Gate before automatic routing

- All offline and live tests pass from a fresh environment (currently 29 offline,
  3 model/MCP, and 2 local-sandbox gates).
- At least 30 representative tasks and 20 adversarial cases are recorded.
- Qwen-only, validators-only, review-every-operation, and evidence-gated modes
  are compared on error rate, completion, latency, tokens, and false blocking.
- Zero fabricated citations in the evaluation set.
- MCP package pins and Python lockfile are reviewed and reproducible.
- The delegate wrapper passes prompts through private stdin/files, uses no
  skill context in the worker, and labels unresolved output provisional.
