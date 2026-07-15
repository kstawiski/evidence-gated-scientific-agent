# Reporting standard

This describes the structure and boundary of the `ScientificReport` every
Evidence Bench run produces, as defined by
[`scientific_agent/schemas.py`](../scientific_agent/schemas.py),
[`scientific_agent/prompts.py`](../scientific_agent/prompts.py) (the
`SCIENTIFIC_REPORT_CONTRACT`), and rendered deterministically by
[`scientific_agent/reporting.py`](../scientific_agent/reporting.py).

## Exploratory, not manuscript-ready

Every report is a **standards-derived exploratory scientific report**. The
report-writing and audit prompts explicitly forbid claiming peer review,
science lock, manuscript readiness, or submission readiness. Article text
that overruns this boundary is a defect the independent Gemma audit
(`REPORT_AUDITOR`) is required to catch and the repair prompt (`REPAIRER`) is
required to fix. Human review remains required before any consequential
scientific or clinical decision.

## Article sections

`render_report_markdown()` renders one portable Markdown article with a
fixed, controller-owned heading order — the model cannot reorder or omit a
section:

1. **Abstract** (`executive_summary`)
2. **Introduction** — problem, knowledge gap, and objective or prespecified
   hypothesis; must not reveal observed results.
3. **Methods** (`methods`, a list of method statements) — setting/data,
   eligibility and analysis unit, endpoints and variables, missingness,
   statistical methods, effect uncertainty, multiplicity, sensitivity
   analyses, software/versions, and prespecified vs. exploratory status.
4. **Results** — the primary question first regardless of direction, with
   absolute denominators, effect sizes, and uncertainty; null, negative,
   discordant, and sensitivity findings are retained rather than dropped.
5. **Discussion** — the main answer first, then prior evidence, scientific
   or clinical meaning, competing explanations, generalizability, and
   limitations (`limitations` renders as a `### Limitations` subsection when
   present).
6. **Conclusions** — interpretation, never a restatement of Results, and
   never claims beyond the design, estimates, uncertainty, or external
   validation.
7. **Evidence ledger** — every `ClaimRecord` with its evidence status
   (`supported`, `partially_supported`, `contradicted`, `unsupported`, or
   `inconclusive`) and linked evidence references.
8. **Sources** — every `SourceRecord`; a URL source renders as a link, an
   artifact-backed source renders as a non-clickable label naming the
   evidence artifact file.
9. **Unresolved issues** (optional).

The `ScientificReport.narrative` field is a legacy pre-v0.4 free-text field
retained only so older reports still load; new reports use the typed article
sections above instead.

## Registered displays: figures and tables

A figure or table only appears in the article if it is registered as a
`ReportDisplay` (`scientific_agent/schemas.py`) and passes deterministic
validation in `linting.py` and `reporting.py` — the model cannot embed
arbitrary images or inline data.

- `artifact_path` must resolve to the **exact path of a file produced by a
  successful sandbox computation** (`resolve_display_artifact`); an
  artifact whose hash no longer matches the recorded computation evidence,
  or that isn't a successful-call output, is rejected.
- **Figures** must be PNG, JPEG, or WebP (`FIGURE_MEDIA_TYPES`), 1 byte–20 MB,
  with dimensions between 240×160 and 20,000×20,000, and the file extension
  must match the actual encoded format (`inspect_figure`). When raster DPI
  metadata is present it must report at least 300 DPI; lower reported values are
  rejected deterministically.
- **Tables** must be strict, UTF-8, rectangular CSV or TSV with a nonempty,
  unique header row, 1 byte–20 MB (`read_table_preview`). The rendered
  preview is capped at 50 rows and 20 columns; a larger table is marked
  `truncated` and the article links to the complete artifact instead of
  inlining it. Numeric reader-facing cells are limited to four significant
  digits; full computational precision belongs in a separate JSON/data
  artifact.
- Each display carries a `placement` of `methods`, `results`, or
  `discussion`, a short title, a self-contained caption (cohort/denominator,
  units, statistical test/model, and prespecified/exploratory status where
  applicable), optional `claim_ids`/`evidence_refs`, and figure `alt_text`
  that states the question, chart type, axes/groups, main pattern, and
  uncertainty.
- The model must not prefix its own title or caption with a figure/table
  number (`caption_has_number_prefix` is a lint check) — the controller
  assigns independent `Figure N` / `Table N` sequences from report display
  order, and verifies that each is mentioned in its declared `placement`
  section, when it materializes the display manifest
  (`materialize_displays`).

Registered displays are copied into a path-confined `displays/` directory
under the run's provenance root and recorded in `display_manifest.json`
(`version`, and one entry per display with its assigned number, relative
path, SHA-256, and byte size). The web API exposes each registered display
individually — see
[`docs/WEB_AND_A2A.md`](WEB_AND_A2A.md#live-artifact-access) — and
`render_report_markdown()` embeds the same manifest paths into `report.md`.

## Evidence gating

Claims are typed as `observed`, `computed`, `literature_supported`,
`inference`, or `hypothesis` and must cite `evidence_refs`. The deterministic
controller — not model self-report — decides whether a claim's cited
evidence actually exists: a `computed` claim must cite an exact successful
computation artifact path, and a `literature_supported`/`observed` claim
citing retrieval must cite a URL and retrieval date that occur in
controller-recorded MCP evidence. Agreement between Qwen and Gemma never
overrides a failed deterministic check.

## Repair and re-audit

A concrete deterministic or Gemma blocking finding reopens the affected report,
computation, or display surface. Qwen receives the exact findings and may read
prior-attempt artifacts only through `/history`; a real figure/table defect must
be regenerated and cannot be repaired by changing its caption alone. The
controller then reruns deterministic validation and gives the changed report and
actual displays to Gemma again. This repeats up to `MAX_REPAIR_ROUNDS` (four by
default; accepted range 0–8).

Display-only repair rounds reuse prior machine-readable results and are limited
to eight sandbox calls. They must not repeat valid Python/R estimation,
cross-language reconciliation, or controller provenance; one display-generation
call plus one direct retry is the expected path.

Correctable typos, labels, clutter, overlaps, false captions, inconsistent
numbers, and excessive display precision remain blocking. An inherent study
limitation may become nonblocking only when it cannot be resolved with the
available data or authorized methods, is stated explicitly, and every claim is
constrained accordingly. If the automatic budget ends with a blocking finding,
the run writes `repair_exhausted.json` and returns
`requires_human_decision`; it is never labeled validated.

After completion, users may open a separate Gemma discussion thread to ask what
a result means, challenge its support, or request a proposed revision brief. That
explanation lane is evidence bounded and read-only. Its optional brief must be
reviewed by the user and launched as a new audited child run; discussion alone
cannot alter the report, validation, audit, or scientific status.

The report may not infer that data are observational, randomized, experimental,
synthetic, or representative from filenames, balance, effect magnitude, or
cleanliness. If allocation or sampling metadata are missing, design is reported
as unspecified and causal/generalizability claims remain below that evidence
ceiling.
