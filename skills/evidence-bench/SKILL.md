---
name: evidence-bench
description: Run evidence-gated scientific analyses on the lab Evidence Bench service at 10.20.102.122. Use when a Codex or Claude agent should delegate a literature review, statistical analysis, Python/R computation, scientific figure or table workflow, reproducible report, or audited follow-up revision to the shared local Qwen→deterministic-validation→Gemma system; upload task files, monitor live progress, cancel a bad run, and download its provenance bundle.
---

# Evidence Bench

Use the bundled stdlib-only client to submit work to the internal lab service. The service creates an isolated persistent workspace, lets Qwen retrieve evidence and run sandboxed Python/R, applies deterministic validation, then asks Gemma to audit the report. It returns an IMRaD-style scientific report plus sources, code, figures, tables, logs, and hashes.

For source-image tasks, Qwen receives no raster bytes. The controller converts a
bounded set of TIFF/PDF/Office/archive visuals and sends them only to Gemma; Qwen
receives Gemma's structured observations and explicit unreviewed-page list.

## Run a task

Write a precise objective. State endpoints, populations, contrasts, assumptions, required sensitivity analyses, and deliverables when known. Never invent missing scientific requirements.

```bash
python3 <skill-dir>/scripts/evidence_bench.py run \
  --workspace-name "short descriptive name" \
  --objective "Analyze the uploaded data, report effect sizes with uncertainty, and distinguish confirmatory from exploratory findings." \
  --file data.csv \
  --wait \
  --download-dir ./evidence-bench-result
```

The default service is `http://10.20.102.122:8070`. Override it only for an explicitly supplied deployment with `EVIDENCE_BENCH_URL` or `--base-url`.

Defaults are intentional:

- Python and R execution are enabled.
- Context7, Brave Search, and Chrome DevTools are enabled.
- The client streams controller events to stderr while keeping machine-readable JSON on stdout.
- `--download-dir` saves the complete ZIP provenance bundle when the run reaches a terminal state.

Use `--no-code` only for work that must not compute. Use `--no-research` when task text or inputs must not result in external searches. Context7 and Brave receive generated queries; do not submit PHI, credentials, or confidential identifiers to external MCP services.

## Inspect or control a run

```bash
python3 <skill-dir>/scripts/evidence_bench.py status --run-id <run-id>
python3 <skill-dir>/scripts/evidence_bench.py cancel --run-id <run-id>
python3 <skill-dir>/scripts/evidence_bench.py download --run-id <run-id> --output ./bundle.zip
```

Do not treat `running`, `failed`, `cancelled`, `inconclusive`, or `requires_more_evidence` as scientific success. Supported states still require the user to inspect limitations and the evidence bundle.

## Improve a report

Discuss a completed report with the configured s8-Gemma critic before deciding
whether to revise it:

```bash
python3 <skill-dir>/scripts/evidence_bench.py discuss \
  --run-id <completed-run-id> \
  --message "Explain the primary result and identify any claim that is stronger than its evidence."
```

The response may contain a `suggested_revision_prompt`. Review it; then pass the
approved text to `follow-up`. Follow-ups create an immutable child revision and
repeat the Qwen→Gemma evidence gate:

```bash
python3 <skill-dir>/scripts/evidence_bench.py follow-up \
  --run-id <completed-run-id> \
  --request "Correct the unsupported causal wording, add the prespecified sensitivity analysis, and revise only conclusions changed by the result." \
  --wait \
  --download-dir ./evidence-bench-revision
```

Enable computation for a follow-up with `--enable-code`; leave it off for writing-only changes.

## Agent operating rules

1. Prefer `run --wait --download-dir ...` so the user receives a finished record rather than only a job ID.
2. Report the workspace ID and run ID immediately after submission when the task may take time.
3. Monitor events. Cancel only when the user asks or when the workflow is clearly proceeding on the wrong task; a slow maximum-thinking phase is not by itself a failure.
4. If deterministic validation or Gemma raises a fixable issue, allow the bounded repair loop to run. Never rewrite a non-success state as consensus.
5. Return the final scientific status, report path, bundle path, unresolved limitations, and the browser URL `http://10.20.102.122:8070`.
6. Use the web interface for real-time artifact previews, model-visible output, cancellation, and manual access to the service-owned research browser.
7. Use `discuss` for explanation and critique. A Gemma discussion answer is advisory and cannot replace the audited follow-up workflow.

The service is reachable only from the lab LAN/Tailscale routes. It intentionally has no browser password; do not expose it to the public Internet.
