# Deployed scientific evaluations

These cases exercise the running web service, model endpoints, sandbox workers,
and provenance downloads rather than mocking an agent response.

- `known-effect` requires both Python and R to recover the planted +5 mean-change
  difference, agree on the test statistic and Hedges g, emit a passing JSON
  reconciliation record, and generate a figure.
- `corrupted-input` requires both languages to identify duplicate `C10`, missing
  `T05`, and extreme `T08` without silently modifying the data; the report must
  withhold decision readiness and label any calculation sensitivity-only.
- `retrieval-grounding` requires successful MCP retrieval, observed source URLs,
  official SciPy and R documentation domains, and the two unequal-variance API
  switches.

Run one case from the deployed checkout. The owner-only `.env` supplies browser
authentication without placing credentials on the command line:

```bash
python3 evals/run_deployed_eval.py known-effect \
  --env-file .env \
  --output /durable/evaluations/known-effect.json
```

The command exits nonzero when any workflow, provenance, independent-review, or
case-specific scientific check fails. Failed results are retained for diagnosis.

## v0.3.0 release validation

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
