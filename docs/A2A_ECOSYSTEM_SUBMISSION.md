# A2A ecosystem contribution dossier

This document is ready to adapt for a contribution to the official
[`a2a-samples`](https://github.com/a2aproject/a2a-samples) repository after the
first public Evidence Bench release.

## Project

- **Name:** Evidence Bench
- **Source:** <https://github.com/kstawiski/evidence-gated-scientific-agent>
- **License:** Apache-2.0
- **Protocol:** A2A 1.0, implemented with the official Python SDK
- **Discovery:** `GET /.well-known/agent-card.json`
- **Transport:** JSON-RPC over HTTP at `POST /a2a`
- **Authentication:** bearer token for task execution; Agent Card is public

## Contribution summary

Evidence Bench demonstrates an A2A computational-science agent whose outputs are
evidence gated. A Qwen model performs planning and analysis, a separately served
deployment-configured Gemma model provides independent text-based methodological
review, and deterministic code controls policy, scientific validation, and
provenance. Figure review uses OCR text and word geometry extracted by a confined
sandbox worker; the dossier does not claim that Gemma inspected pixels or
performed a multimodal review. Tasks can use isolated
workspace files, offline Python/R execution, per-workspace PyPI/CRAN/Bioconductor
environments, and selected MCP research tools.

The advertised scientific-analysis skill accepts text tasks. A client can retain
an A2A `contextId` to continue work in the same isolated workspace. Successful
tasks return a structured summary and report artifacts; unsupported conclusions
remain explicitly inconclusive instead of being reported as successful consensus.

## Interoperability evidence

The v0.4.0 repository regression gate covers authenticated A2A 1.0
`SendMessage`, `SendStreamingMessage` (submitted → working → artifacts →
completed), and `CancelTask`, in addition to validating the public Agent Card.
The deployed v0.4.0 transport gate completed on 2026-07-15 with functional image
`sha256:e95760b378f4923142e499899ebb481687c0a71012aee480556458a6d2a6f726`.
A2A task `44702ea7-72d8-4545-853e-82fd926e0831`, backed by Evidence Bench run
`0c69fa4f-459b-419f-81a8-47737f732ce6`, emitted
`TASK_STATE_SUBMITTED`, `TASK_STATE_WORKING`, and `TASK_STATE_COMPLETED`, streamed
`report.md` and `run-summary.json`, returned the same artifacts through
`GetTask`, and finished scientifically `supported`.

A separate scientific MCP probe used workspace
`6e08b205-4d2b-49fb-a0ec-5bbbea735c4a` and run
`fa5e58b9-92b4-4bfb-82f3-fd0e14dd279d`. It exercised `brave-search` and
`context7`, with observed typed tools `resolve-library-id`, `query-docs`,
`brave_web_search`, `brave_llm_context`, `search_pubmed`, and
`acquire_pubmed_article`. A canonical PubMed-title mismatch blocked the initial
report; Qwen repaired the citation from stored acquisition evidence, after which
Gemma passed the report. The complete machine-readable record is
[`evals/results/v0.4.0-a2a-live.json`](../evals/results/v0.4.0-a2a-live.json).

The v0.4.0 server uses the SDK's `InMemoryTaskStore`. Evidence Bench workspace,
run, report, and provenance records remain durable, but the A2A task snapshots
used by `GetTask` and task-subscription state are process-local. After the web
process restarts, clients must not assume that an earlier A2A task id can still
be queried or resubscribed; the live gate did not claim restart durability for
that protocol-layer state.

A separate deployed v0.4.0 PubMed/full-text web-service gate passed 17/17 checks:
typed PubMed search/acquisition, hash-verified local Markdown and PDF evidence,
correct scientific extraction, independent text review, and a successful repair
round. This establishes the shared scientific workflow used behind A2A, but is
reported separately from the A2A transport result above.

The exact v0.4.0 non-A2A scores are backed by compact public records:
[`v0.4.0-pubmed-fulltext.json`](../evals/results/v0.4.0-pubmed-fulltext.json) and
[`v0.4.0-known-effect.json`](../evals/results/v0.4.0-known-effect.json). They
retain run identifiers, timestamps, evaluator checks, selected artifact hashes,
and the known-effect image digest; the missing PubMed image digest is disclosed
rather than reconstructed.

The v0.3.0 candidate previously passed a deployed, authenticated A2A 1.0
`SendMessage` run. That historical run used Brave Search and Context7, cited
retrieved official SciPy and R documentation, passed deterministic claim
validation, and passed independent Gemma review (9/9 case checks).

The issue-first step is complete in
[`a2aproject/a2a-samples#639`](https://github.com/a2aproject/a2a-samples/issues/639).
A small draft pull request modeled on the current `helloworld` sample is our
next contribution: the maintainer explicitly invited a draft PR with pytest and
recommended stable A2A 1.0 in
[the issue response](https://github.com/a2aproject/a2a-samples/issues/639#issuecomment-4969746684).
The reduced sample must include pytest coverage and an explicit
`AgentInterface(protocol_version="1.0")`, as Evidence Bench already does. It must also follow the
current
[`a2a-samples` contribution guide](https://github.com/a2aproject/a2a-samples/blob/main/CONTRIBUTING.md):
Python 3.12+, `uv`, formatting, static type checks, and pytest. Record the public
release's immutable container digests in the Evidence Bench release notes before
opening the requested draft pull request.

The official sample repository accepts new samples and documentation improvements;
its Python examples use the official SDK. Evidence Bench pins `a2a-sdk[fastapi]`
1.1.0 and advertises A2A specification 1.0 over JSON-RPC.

This dossier deliberately contains no private deployment URL or credential.
