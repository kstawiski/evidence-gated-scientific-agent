# Lessons adopted from Open Science Desktop

Evidence Bench and [Open Science Desktop](https://github.com/ai4s-research/open-science)
have different execution models: this project is a server-side, evidence-gated agent;
Open Science Desktop is a broad local-first desktop workbench. The following product
patterns are useful in both:

- human-reviewed workflow starters instead of blank-prompt onboarding;
- immutable input manifests and visible environment identity;
- one provenance bundle containing code, outputs, reviews, and hashes;
- a “reuse protocol” action that fills the task form but never auto-executes;
- artifact-first reporting and explicit run history.

The patterns are informed by Open Science Desktop's
[`WorkflowStarters.tsx`](https://github.com/ai4s-research/open-science/blob/main/apps/desktop/src/components/thread/WorkflowStarters.tsx),
[`provenance.rs`](https://github.com/ai4s-research/open-science/blob/main/apps/desktop/src-tauri/src/provenance.rs),
and [`ProvenancePanel.tsx`](https://github.com/ai4s-research/open-science/blob/main/apps/desktop/src/components/inspector/ProvenancePanel.tsx).
No upstream source code is copied.

We intentionally do not adopt unrestricted shell mode, agent-managed dependency
installation, or notebook state as the definitive record. Evidence Bench keeps raw
inputs read-only, executes only typed Python/R actions in a separate internal worker,
and treats files plus structured provenance as the reproducible source of truth.
