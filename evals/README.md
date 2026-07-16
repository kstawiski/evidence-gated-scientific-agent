# Deployed scientific evaluations

These cases exercise the running web service, model endpoints, sandbox workers,
and provenance downloads rather than mocking an agent response.

- `known-effect` requires both Python and R to recover the planted +5 mean-change
  difference, agree on the test statistic and Hedges g, emit a passing JSON
  reconciliation record, and generate and register a figure plus a report-ready
  CSV table.
- `corrupted-input` requires both languages to identify duplicate `C10`, missing
  `T05`, and extreme `T08` without silently modifying the data; the report must
  withhold decision readiness and label any calculation sensitivity-only.
- `retrieval-grounding` requires successful MCP retrieval, observed source URLs,
  official SciPy and R documentation domains, and the two unequal-variance API
  switches.
- `pubmed-fulltext` requires the agent to search, acquire, and search within one
  known open-access PubMed article; preserve its identifiers and evidentiary
  limitations; and expose hash-verified local Markdown and PDF copies through
  the portable report bundle.

Run one case from the deployed checkout. The owner-only `.env` supplies browser
authentication without placing credentials on the command line:

```bash
python3 evals/run_deployed_eval.py known-effect \
  --env-file .env \
  --output /durable/evaluations/known-effect.json
```

The command exits nonzero when any workflow, provenance, independent-review, or
case-specific scientific check fails. Failed results are retained for diagnosis.

Exercise the public A2A 1.0 streaming surface with the reusable live gate. Keep
the bearer token in the environment so it is neither passed in process arguments
nor printed by the evaluator:

```bash
A2A_TOKEN="$(< /secure/path/a2a-token)" \
A2A_BASE_URL=https://evidence-bench.example.org \
python3 evals/run_a2a_live_gate.py
```

The gate selects Context7, Brave Search, and Chrome DevTools by default. Optional
`--mcp-server` flags select a narrower set (an explicit empty value opts out), and
`--enable-code` authorizes sandboxed Python/R. The gate validates the Agent Card, server-assigned
task ID, submitted/working/completed SSE states, streamed report artifacts,
`GetTask`, scientific terminal status, and the report's Introduction, Methods,
Results, Discussion, and Conclusions sections. It emits only non-secret JSON
evidence and exits nonzero on a failed assertion.

## v0.4.0 release validation

The candidate was exercised against a persistent deployed Compose stack starting
on 2026-07-14, with final v0.4.0 gates completed on 2026-07-15, not against
mocked model responses. Model routes are private deployment configuration; the
repository does not embed them.

| Case or boundary | Result | What the gate established |
| --- | ---: | --- |
| PubMed/full-text | **17/17** | A biomedical run performed typed PubMed search and acquisition, imported a browser-obtained open-access PDF, verified and stored local Markdown/PDF copies, reported the exact PMID/PMCID/DOI, cohort count and survival estimates, constrained the prognostic interpretation, repaired a missing search artifact and DOI after independent review, and finished with deterministic and deployment-configured Gemma review passes. |
| Known planted effect | **18/18** | On 2026-07-15, workspace `187c0fe5-2967-4bbd-a297-f7a9423274be` used image `sha256:4f055eb3a5515b49257fad69e701dd3d46ec07fdf28c430b09293e66c4a2021c`. Parent run `5428105d-8979-4bf1-8dd1-76f9fedccee2` independently recovered the planted +5 effect in Python and R and reconciled the results; Qwen and Gemma streamed and live artifacts were accessed. Accepted code-disabled revision `b5bbf30c-15bd-42f2-bb65-b06519a94a9c` passed the evaluator, deterministic validation, Gemma report review, and OCR/geometry/table display review; it preserved the parent immutably, generated no result outputs, and passed final manual caption/prose inspection. An earlier nominally supported revision with inverted provenance was rejected and excluded from the score. |
| A2A 1.0 live interoperability | **PASS** | On 2026-07-15, functional image `sha256:e95760b378f4923142e499899ebb481687c0a71012aee480556458a6d2a6f726` served task `44702ea7-72d8-4545-853e-82fd926e0831` backed by run `0c69fa4f-459b-419f-81a8-47737f732ce6`. Streaming emitted submitted, working, and completed states plus `report.md` and `run-summary.json`; `GetTask` returned the completed artifacts and the scientific status was `supported`. MCP probe run `fa5e58b9-92b4-4bfb-82f3-fd0e14dd279d` in workspace `6e08b205-4d2b-49fb-a0ec-5bbbea735c4a` observed Brave Search, Context7, `resolve-library-id`, `query-docs`, `brave_web_search`, `brave_llm_context`, `search_pubmed`, and `acquire_pubmed_article`; it repaired a blocking canonical PubMed-title mismatch before Gemma passed. |
| Package lifecycle | **PASS** | The isolated package worker installed and the offline analysis sandbox loaded PyPI `emoji` 2.15.0, CRAN `moments` 0.14.1, and Bioconductor `BiocGenerics` 0.44.0. Deleting the workspace removed its immutable environment generations. |
| Cancellation | **PASS** | A live in-flight analysis was cooperatively cancelled and durably remained `cancelled`; its partial state was not presented as a report. |
| Managed browser boundary | **PASS** | The service-owned Chromium CDP session and passwordless trusted-network noVNC view were reachable through their intended gateways; CDP stayed unpublished, direct Internet and private proxy targets were denied, downloads/profile data survived restart, and the application saw downloads read-only. |
| Manuscript-package stress run (pre-fix) | **FAILED MERIT GATE** | Run `18b7b94f-6d3a-4c0c-941c-cc6b96ce0aeb` correctly stopped at `requires_human_decision`, but consumed all four repairs. It recovered bounded Python failures and improved scientific scope/PubMed anchoring; however, the critic invented manuscript-plus-supplement word-count arithmetic, treated permitted placeholders as blockers, and later lost article review to invalid TIFF display preparation. This retained failure directly motivated task-scope critic constraints, controller TIFF/PDF/archive rendering, Gemma-only source-image review, and preservation of article audits when display preparation fails. |
| Fresh planted-effect run (pre visual-clearance fix) | **FAILED MERIT GATE** | Run `4d143e16-3616-4f7b-8b38-7a9c1ea4f3fe` recovered and reconciled the exact Python/R statistics, but incorrectly labeled the prespecified task exploratory and accepted a visibly overlapping figure after a bare Gemma pass. The run is rejected despite its terminal `supported` status. It directly motivated controller-enforced task risk/method locks, exact per-display Gemma clearance attestations, and evaluator support for the valid nested result shapes. |

The PubMed gate's scored JSON and full provenance bundle are retained with the
deployment evaluation artifacts. The local paper copies are hash verified and
report citations link to the portable Markdown/PDF artifacts, not to an
unverified filename. The current display critic receives registered raster images
only on the Gemma endpoint, together with sandbox-extracted OCR text/geometry and
deterministic table previews. Qwen receives zero image inputs. The earlier
release-candidate records predate this multimodal boundary and must not be
reinterpreted as proof that their figures received pixel-level review.

Public compact records for the exact scores, run identifiers, timestamps, and
selected artifact hashes are retained as
[`results/v0.4.0-pubmed-fulltext.json`](results/v0.4.0-pubmed-fulltext.json) and
[`results/v0.4.0-known-effect.json`](results/v0.4.0-known-effect.json). The
PubMed evaluator did not record its image digest, and the public record states
that gap instead of inferring one retrospectively.
The retained manuscript stress failure is summarized in
[`results/v0.4.0-manuscript-stress-pre-fix.json`](results/v0.4.0-manuscript-stress-pre-fix.json).
The rejected fresh planted-effect run and its implemented repairs are summarized in
[`results/v0.4.0-known-effect-fresh-pre-visual-gate.json`](results/v0.4.0-known-effect-fresh-pre-visual-gate.json).

The A2A evidence is retained as
[`results/v0.4.0-a2a-live.json`](results/v0.4.0-a2a-live.json). The server uses
the SDK `InMemoryTaskStore`: durable Evidence Bench runs and provenance survive
service restart, but A2A `GetTask` snapshots and task-subscription state do not.
An issue-first proposal is open in
[`a2aproject/a2a-samples#639`](https://github.com/a2aproject/a2a-samples/issues/639),
and a small draft pull request is the proposed next action pending maintainer
direction.

These are narrow release gates, not an estimate of performance on arbitrary
scientific work. Failed and repaired attempts remain part of the durable audit
trail.

## Historical v0.3.0 validation

The release candidate was exercised against the permanently deployed Compose
stack on 2026-07-13, not against mocked model responses:

| Case | Result | What the gate established |
| --- | ---: | --- |
| Known planted effect | 11/11 | Python and R recovered the +5 mean-change contrast, agreed on Welch statistics and Hedges g within `1e-6`, generated a figure, cited successful artifacts, and preserved the input hash. |
| Corrupted input | 9/9 | Both languages found the seeded duplicate, missing value, and extreme record; the report withheld decision readiness and labeled calculations sensitivity-only and inconclusive. |
| A2A retrieval grounding | 9/9 | An authenticated A2A 1.0 `SendMessage` run retrieved official SciPy and R documentation through configured MCP services, used only observed URLs, passed deterministic claim checks, and passed independent Gemma review. |

The live deployment also installed and loaded a pinned PyPI package, a CRAN
package, and a Bioconductor package through the isolated package worker. Separate
container probes confirmed that failed-script outputs are quarantined from later
analysis calls and that successful later outputs remain reusable.

These are narrow release gates, not an estimate of performance on arbitrary
scientific work. The durable deployment retains the scored JSON, reports, source
scripts, rejected attempts, hashes, and manifests so failures and repairs remain
auditable.
