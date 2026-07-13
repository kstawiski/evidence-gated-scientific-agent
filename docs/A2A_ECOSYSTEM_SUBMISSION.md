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
Gemma model provides independent methodological review, and deterministic code
controls policy, scientific validation, and provenance. Tasks can use isolated
workspace files, offline Python/R execution, per-workspace PyPI/CRAN/Bioconductor
environments, and selected MCP research tools.

The advertised scientific-analysis skill accepts text tasks. A client can retain
an A2A `contextId` to continue work in the same isolated workspace. Successful
tasks return a structured summary and report artifacts; unsupported conclusions
remain explicitly inconclusive instead of being reported as successful consensus.

## Interoperability evidence

The v0.3.0 candidate passed the repository A2A API test and a deployed,
authenticated A2A 1.0 `SendMessage` run. That live run used Brave Search and
Context7, cited retrieved official SciPy and R documentation, passed deterministic
claim validation, and passed independent Gemma review (9/9 case checks). The
release includes the threat model, protocol semantics, and provenance export.

After the public tag is published, record the immutable container digests in the
release notes and open a proposal issue before significant sample work, as
required by the destination contribution guide. Follow the maintainers' preferred
shape—an external integration entry or a reduced in-tree sample—for the pull
request.

The official sample repository accepts new samples and documentation improvements;
its Python examples use the official SDK. Evidence Bench pins `a2a-sdk[fastapi]`
1.1.0 and advertises A2A specification 1.0 over JSON-RPC.

This dossier deliberately contains no private deployment URL or credential.
