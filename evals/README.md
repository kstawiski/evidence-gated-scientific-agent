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
| Knowledge grounding | **PASS** | The clean-commit live gate on `ca09622` used 20 scientific source/distractor records, 30 exact/synonym/Polish queries, five adversarial/no-answer text queries, six scientific figures, and three visual no-answer queries. Hybrid Recall@10 was 1.00 with nDCG@10 0.975; the synonym/Polish recall gain was +0.40 (seeded bootstrap 95% CI 0.20–0.60), with no exact-query loss. All 122 retrieved passages were exact immutable source slices, no descriptor prose became evidence, every text/visual no-answer query stayed empty, and Gemma ranked all six figures first. Semantic answer IDs were absent from the indexed corpus. |
| Package lifecycle | **PASS** | The isolated package worker installed and the offline analysis sandbox loaded PyPI `emoji` 2.15.0, CRAN `moments` 0.14.1, and Bioconductor `BiocGenerics` 0.44.0. Deleting the workspace removed its immutable environment generations. |
| Cancellation | **PASS** | A live in-flight analysis was cooperatively cancelled and durably remained `cancelled`; its partial state was not presented as a report. |
| Managed browser boundary | **PASS** | The service-owned Chromium CDP session and passwordless trusted-network noVNC view were reachable through their intended gateways; CDP stayed unpublished, direct Internet and private proxy targets were denied, downloads/profile data survived restart, and the application saw downloads read-only. |
| Manuscript-package stress run (pre-fix) | **FAILED MERIT GATE** | Run `18b7b94f-6d3a-4c0c-941c-cc6b96ce0aeb` correctly stopped at `requires_human_decision`, but consumed all four repairs. It recovered bounded Python failures and improved scientific scope/PubMed anchoring; however, the critic invented manuscript-plus-supplement word-count arithmetic, treated permitted placeholders as blockers, and later lost article review to invalid TIFF display preparation. This retained failure directly motivated task-scope critic constraints, controller TIFF/PDF/archive rendering, Gemma-only source-image review, and preservation of article audits when display preparation fails. |
| Fresh planted-effect run (pre visual-clearance fix) | **FAILED MERIT GATE** | Run `4d143e16-3616-4f7b-8b38-7a9c1ea4f3fe` recovered and reconciled the exact Python/R statistics, but incorrectly labeled the prespecified task exploratory and accepted a visibly overlapping figure after a bare Gemma pass. The run is rejected despite its terminal `supported` status. It directly motivated controller-enforced task risk/method locks, exact per-display Gemma clearance attestations, and evaluator support for the valid nested result shapes. |
| Fresh planted-effect run (pre narrative-number fix) | **FAILED MERIT GATE** | Run `3d186baf-7a30-437a-900a-9fbc03e89b66` repaired the computation, passed deterministic and Gemma artifact/display review, and rescored 18/18 after provenance-aware evaluator selection. Manual review still rejected it because Results retained an obsolete `df≈564.06` from the superseded Python execution while both corrected implementations reported `df=38`. This motivated deterministic report-to-artifact degrees-of-freedom checking and explicit critic instructions to use the latest successful machine result. |
| v0.4.1 planted effect (pre transposition gate) | **FAILED MERIT GATE** | Run `1aaa0ccf-2f0d-4355-a4aa-20e6e9c060ae` recovered and reconciled the correct effect, but the effect estimate was plotted on the wrong coordinate while the axis claimed to show the mean difference. The repetitive critic response failed closed. This motivated static and post-execution estimand/interval dataflow checks. |
| v0.4.1 planted effect (pre category gate) | **CANCELLED / FAILED MERIT GATE** | Run `26b27ec4-8c6e-4a06-beb5-f381b79fd43c` corrected the effect panel but overlapped both raw-data groups at the same categorical position and emitted length-one R scalar arrays. The operator cancelled it during repair 1. This motivated unique-category validation and scalar R JSON requirements. |
| v0.4.1 planted effect (pre critic normalization) | **FAILED MERIT GATE** | Run `7511eb4c-659f-4d84-9980-9fdc767cc578` reconciled the correct statistics, passed deterministic validation, linked a local PubMed article, and auto-published it to knowledge. It was still rejected because the figure ambiguously labeled a raw mean difference as `d`, while Gemma returned valid objections in an incomplete schema that failed closed without becoming repair actions. This motivated raw-effect label checks and conservative preservation of explicit failed-review objections. |
| v0.4.1 planted effect (pre monotonic critic gate) | **TERMINAL SUPPORTED / FAILED MERIT GATE** | Run `6d3098ca-d2c4-4318-b403-d4900bea40d0` recovered the exact effect, corrected table precision, used the required Results structure, and rescored 18/18 after the evaluator learned its exact Welch schema. It remains rejected: Gemma first detected hidden point overplotting, then a format-repair call erased the fail with a bare pass; the article also understated AI involvement and overstated protocol timing and robustness. This motivated monotonic fail preservation and deterministic prose-overclaim gates. |
| v0.4.1 planted effect (pre plan-convergence gate) | **REQUIRES MORE EVIDENCE / FAILED MERIT GATE** | Run `4d6f55e3-f7b1-4b53-9a86-4b2e9c572c15` correctly failed closed before computation, but four repairs did not converge because each round received only the latest audit. Earlier Hedges-formula, semantic-arm, site-role, and raster-specification requirements were not preserved together. This motivated cumulative repair findings and explicit monotonic preservation across rounds. |
| v0.4.1 planted effect (pre assumption-reassurance gate) | **TERMINAL SUPPORTED / FAILED MERIT GATE** | Run `3f49337d-b3ce-4a62-84a7-6624b988c510` proved the cumulative planning fix live, recovered and reconciled the exact statistics, generated an unambiguous figure, and repaired table precision. It remains rejected because the report called the contrast robust and used balance/symmetry to reassure an untested normality assumption, while Gemma passed both. The evaluator's 16/18 was separately traced to valid `primary.ci_lower`/`primary.ci_upper` fields and rescored 18/18. |
| v0.4.1 planted effect (pre static-dataflow and uniformity gates) | **TERMINAL SUPPORTED / FAILED MERIT GATE** | Run `f9a82678-13d4-4d5d-b574-72b1fb30cc96` recovered and reconciled the exact statistics and rescored 18/18 after an evaluator field fix. It remains rejected because the static checker repeatedly denied provably zero-inclusive limits and valid f-string labels, runtime guidance missed two ErrorbarContainer type errors, and the final report converted a group mean into a uniform individual response while again using balance as normality reassurance. The final display also retained point-label overlap and a false caption description; deterministic validation and both Gemma reviews passed. |
| v0.4.1 planted effect (pre JSON-audit and display-contract gates) | **TERMINAL SUPPORTED / FAILED MERIT GATE** | Run `8d708cd9-d940-4892-b342-518a5cb808af` recovered the exact Python/R statistics and repaired table precision and interval clipping. It remains rejected: the final raster placed both raw groups at the control center and removed all numeric effect-axis ticks, Results called the untested result robust, and reconciliation omitted the task-required `reconciliation_passed` field. Deterministic validation and both Gemma reviews passed. The immutable run rescored 17/18 after the evaluator learned the valid `effect_size.hedges_correction_J` path; the remaining failed check is the genuine missing verdict. |
| v0.4.1 planted effect (pre layout and reader-table output gates) | **REQUIRES HUMAN DECISION / FAILED MERIT GATE** | Run `279f162e-024d-42e6-8ee9-b82179968f4c` recovered and reconciled the exact Python/R statistics, but two renders left 65.4% and 78.9% internal blank bands, report alt text falsely said the CI crossed zero, a 73%-confidence OCR hallucination consumed a repair, and the final CSV was nonrectangular. The immutable run rescored 14/18 after the evaluator learned the deployed `primary.hedges_correction_J` path; the other four failures are genuine consequences of failed deterministic validation and deferred display/independent review. This motivated plotting-warning variants, a deterministic blank-band gate, machine-bound interval-description checks, high-confidence overlap blocking, and strict post-execution CSV/TSV parsing. |
| v0.4.1 planted effect (pre structural-report grounding gates) | **TERMINAL SUPPORTED / FAILED MERIT GATE** | Run `290eef90-e1f0-445b-82fa-9604097b0e0f` recovered and reconciled the exact statistics, produced a compact figure and rectangular table, passed deterministic and Gemma report/display review, and immutably rescored 18/18 after the evaluator learned its fully qualified reconciliation names. Manual review still rejected it: the report called a two-site input single-site, invented an observational design, promoted agreement to implementation accuracy, and cited a diagnostics artifact that did not contain its explicit row/column/sample counts. This motivated task-bound report design validation, controller-profile site checks, reconciliation-scope language, and direct structural-count grounding. |

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
The live grounding metrics, thresholds, per-query ranks, clean Git state, model
routing, fixture hash, and nonvacuous evidence audit are retained in
[`results/v0.4.0-knowledge-grounding.json`](results/v0.4.0-knowledge-grounding.json).
The retained manuscript stress failure is summarized in
[`results/v0.4.0-manuscript-stress-pre-fix.json`](results/v0.4.0-manuscript-stress-pre-fix.json).
The rejected fresh planted-effect run and its implemented repairs are summarized in
[`results/v0.4.0-known-effect-fresh-pre-visual-gate.json`](results/v0.4.0-known-effect-fresh-pre-visual-gate.json).
The later artifact-passing but narrative-invalid planted-effect run is summarized in
[`results/v0.4.0-known-effect-pre-narrative-gate.json`](results/v0.4.0-known-effect-pre-narrative-gate.json).
The two rejected v0.4.1 runs that directly exercised effect-axis and categorical-axis
failure modes are retained as
[`results/v0.4.1-known-effect-pre-transposition-gate.json`](results/v0.4.1-known-effect-pre-transposition-gate.json)
and
[`results/v0.4.1-known-effect-pre-category-gate.json`](results/v0.4.1-known-effect-pre-category-gate.json).
The later computation-valid but critic-incomplete run is retained as
[`results/v0.4.1-known-effect-pre-critic-normalization.json`](results/v0.4.1-known-effect-pre-critic-normalization.json).
The mechanically passing but scientifically rejected critic-regression run is retained as
[`results/v0.4.1-known-effect-pre-monotonic-critic-gate.json`](results/v0.4.1-known-effect-pre-monotonic-critic-gate.json).
The fail-closed but non-convergent planning run is retained as
[`results/v0.4.1-known-effect-pre-plan-convergence.json`](results/v0.4.1-known-effect-pre-plan-convergence.json).
The planning-convergent but prose-invalid run is retained as
[`results/v0.4.1-known-effect-pre-assumption-reassurance-gate.json`](results/v0.4.1-known-effect-pre-assumption-reassurance-gate.json).
The later computation-valid but static-dataflow- and prose-invalid run is retained as
[`results/v0.4.1-known-effect-pre-static-dataflow-and-uniformity-gates.json`](results/v0.4.1-known-effect-pre-static-dataflow-and-uniformity-gates.json).
The subsequent exact-computation but audit-context- and display-invalid run is retained as
[`results/v0.4.1-known-effect-pre-json-audit-and-display-contracts.json`](results/v0.4.1-known-effect-pre-json-audit-and-display-contracts.json).

The A2A evidence is retained as
[`results/v0.4.0-a2a-live.json`](results/v0.4.0-a2a-live.json). The server uses
the SDK `InMemoryTaskStore`: durable Evidence Bench runs and provenance survive
service restart, but A2A `GetTask` snapshots and task-subscription state do not.
An issue-first proposal is open in
[`a2aproject/a2a-samples#639`](https://github.com/a2aproject/a2a-samples/issues/639),
and a small draft pull request is the proposed next action pending maintainer
direction: [`a2aproject/a2a-samples#642`](https://github.com/a2aproject/a2a-samples/pull/642)
is open and its lint check passes.

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
