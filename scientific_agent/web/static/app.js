"use strict";

const state = {
  config: null,
  workspaces: [],
  workspace: null,
  selectedRun: null,
  activeRun: null,
  runEvents: [],
  eventCursor: 0,
  eventRunId: null,
  tab: "article",
  pollTimer: null,
  pollBackoff: 1500,
  pollErrorShown: false,
  eventSource: null,
  eventSourceRunId: null,
  cancelPending: false,
  workspaceToken: 0,
  modelOutputRequest: 0,
  modelOutputArtifact: null,
  previewRequest: 0,
  previewArtifact: null,
  browserUrl: null,
  discussion: [],
  discussionRunId: null,
  discussionLoading: false,
  discussionSending: false,
  integrations: null,
  modelStatus: null,
  modelStatusTimer: null,
  modelStatusLoading: false,
  knowledge: { stats: {}, documents: [], jobs: [] },
  knowledgeSelection: new Set(),
  knowledgeKnownIds: new Set(),
  knowledgeSelectionInitialized: false,
  knowledgeJobEvents: new Map(),
  knowledgePollTimer: null,
  knowledgePollLoading: false,
  knowledgeVisuals: [],
  knowledgeVisualTitle: "",
};

const el = (id) => document.getElementById(id);
const activeStates = new Set(["queued", "running", "cancel_requested"]);
const supportedStates = new Set(["supported", "supported_with_comments"]);
const warningStates = new Set(["inconclusive", "requires_more_evidence", "requires_human_decision", "contradicted"]);
const terminalStates = new Set(["supported", "supported_with_comments", "contradicted", "inconclusive", "requires_more_evidence", "requires_human_decision", "failed", "interrupted", "cancelled"]);
const textArtifactExtensions = new Set(["bib", "c", "cfg", "conf", "cpp", "css", "csv", "env", "go", "h", "html", "ini", "ipynb", "java", "js", "json", "jsonl", "log", "md", "py", "qmd", "r", "rmd", "rs", "rst", "sh", "sql", "stan", "svg", "tex", "toml", "ts", "tsv", "txt", "xml", "yaml", "yml"]);
const phaseOrder = ["input-intake", "planning", "research", "validation", "scientific-review", "repair", "finalizing"];
const phaseActors = {
  "input-intake": "Controller + Gemma",
  planning: "Qwen + Gemma",
  "plan-review": "Qwen + Gemma",
  research: "Qwen",
  reporting: "Qwen",
  validation: "Controller",
  "scientific-review": "Gemma",
  repair: "Qwen",
  finalizing: "Controller",
  canceling: "Controller",
};
const workflowTemplates = {
  analyze: "Analyze the uploaded dataset end to end. First inspect schema, missingness, duplicates, units, and plausible outliers. Lock the primary analysis before inspecting its result; report effect sizes with uncertainty and sensitivity analyses; save machine-readable tables and a publication-ready raster figure. Embed every final table and figure in an Introduction, Methods, Results, Discussion, and Conclusions report with a self-contained caption. Do not claim causality. Every numerical claim must cite an exact computation artifact.",
  crosscheck: "Independently analyze the uploaded dataset in both Python and R using the same prespecified estimand and statistical method. Save machine-readable results from each language, reconcile numerical differences against an explicit tolerance, run assumption and sensitivity checks, and report effect sizes with uncertainty. Embed the validated table and raster figure in an article-shaped report. Every numerical claim must cite an exact computation artifact.",
  audit: "Audit the scientific claim described below using retrieved primary or authoritative sources. Separate observed evidence, computation, inference, and unresolved uncertainty. Verify identifiers and supporting passages, look for corrections or contradictory evidence, and do not include any citation that was not retrieved by a configured tool. Present the result as Introduction, Methods, Results, Discussion, and Conclusions. Claim to audit: ",
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).detail || message; } catch (_) { /* not JSON */ }
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.json();
}

function toast(message, kind = "info") {
  const item = document.createElement("div");
  item.className = `toast ${kind}`;
  item.textContent = message;
  el("toast-region").append(item);
  window.setTimeout(() => item.remove(), 4500);
}

function formatBytes(bytes = 0) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) { value /= 1024; index += 1; }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[index]}`;
}

function formatDate(value) {
  if (!value) return "—";
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}

function displayStatus(status = "") { return status.replaceAll("_", " "); }

function statusClass(status) {
  if (supportedStates.has(status)) return "supported";
  if (warningStates.has(status)) return "warning";
  if (["failed", "interrupted", "cancelled"].includes(status)) return "failed";
  if (activeStates.has(status)) return "running";
  return "";
}

function workspaceLocked() { return Boolean(state.activeRun && activeStates.has(state.activeRun.status)); }

function activeSummary(workspace = state.workspace) {
  return workspace?.runs.find((run) => activeStates.has(run.status)) || null;
}

async function loadConfig() {
  [state.config, state.integrations, state.knowledge] = await Promise.all([
    api("/api/config"),
    api("/api/integrations"),
    api("/api/knowledge"),
  ]);
  initializeKnowledgeSelection();
  el("app-version").textContent = `v${state.config.version}`;
  el("executor-model").textContent = state.config.models.executor.split("/").pop();
  el("critic-model").textContent = state.config.models.critic.split("/").pop();
  document.querySelectorAll("#mcp-options input").forEach((input) => {
    input.checked = state.config.default_mcp_servers.includes(input.value) && state.config.mcp[input.value];
  });
  el("upload-guidance").textContent = `Select many files, or ZIP a large directory · ${formatBytes(state.config.max_upload_bytes)} per file`;
  configureResearchBrowser();
  configureIntegrationDownloads();
  renderKnowledgeSelection();
  setWorkspaceLocked(workspaceLocked());
}

function initializeKnowledgeSelection(selectNew = false) {
  const current = state.knowledge.documents.filter((item) => item.status === "ready");
  const currentIds = new Set(current.map((item) => item.id));
  for (const id of [...state.knowledgeSelection]) {
    if (!currentIds.has(id)) state.knowledgeSelection.delete(id);
  }
  if (!state.knowledgeSelectionInitialized) {
    for (const item of current) if (item.enabled) state.knowledgeSelection.add(item.id);
    state.knowledgeSelectionInitialized = true;
  } else if (selectNew) {
    for (const item of current) {
      if (item.enabled && !state.knowledgeKnownIds.has(item.id)) state.knowledgeSelection.add(item.id);
    }
  }
  state.knowledgeKnownIds = currentIds;
}

async function loadKnowledge(includeRetired = false, selectNew = false) {
  const catalog = await api(`/api/knowledge?include_retired=${includeRetired ? "true" : "false"}`);
  const eventJobs = (catalog.jobs || []).filter((job) => ["queued", "running", "cancel_requested", "failed", "cancelled"].includes(job.status)).slice(0, 12);
  const eventEntries = await Promise.all(eventJobs.map(async (job) => {
    try { return [job.id, await api(`/api/knowledge/jobs/${encodeURIComponent(job.id)}/events`)]; }
    catch (_) { return [job.id, []]; }
  }));
  state.knowledge = catalog;
  state.knowledgeJobEvents = new Map(eventEntries);
  initializeKnowledgeSelection(selectNew);
  renderKnowledgeSelection();
  renderKnowledgeCatalog();
  renderKnowledgeJobs();
}

function stopKnowledgePolling() {
  if (state.knowledgePollTimer) window.clearTimeout(state.knowledgePollTimer);
  state.knowledgePollTimer = null;
}

function scheduleKnowledgePolling(delay = 1800) {
  stopKnowledgePolling();
  if (!el("knowledge-library-dialog").open) return;
  state.knowledgePollTimer = window.setTimeout(async () => {
    if (state.knowledgePollLoading || !el("knowledge-library-dialog").open) return scheduleKnowledgePolling();
    state.knowledgePollLoading = true;
    try { await loadKnowledge(el("knowledge-show-retired").checked, true); }
    catch (_) { /* Keep the last visible catalog during transient polling failures. */ }
    finally { state.knowledgePollLoading = false; scheduleKnowledgePolling(); }
  }, delay);
}

function renderKnowledgeSelection() {
  const ready = state.knowledge.documents.filter((item) => item.status === "ready");
  const enabled = ready.filter((item) => item.enabled);
  const selected = enabled.filter((item) => state.knowledgeSelection.has(item.id));
  const jobs = state.knowledge.jobs || [];
  const waiting = jobs.filter((job) => ["queued", "running", "cancel_requested"].includes(job.status));
  const failed = jobs.filter((job) => job.status === "failed");
  el("knowledge-selection-count").textContent = `${selected.length} of ${enabled.length} enabled document${enabled.length === 1 ? "" : "s"} selected`;
  el("knowledge-stats").textContent = `${state.knowledge.stats?.documents || ready.length} active generations · ${state.knowledge.stats?.chunks || 0} exact passages · ${formatBytes(state.knowledge.stats?.bytes || 0)}`;
  el("knowledge-index-summary").textContent = `${waiting.length} waiting or active · ${failed.length} failed · ${ready.length} published generations available now`;
}

function knowledgeLabel(text, control) {
  const label = document.createElement("label");
  label.append(document.createTextNode(text), control);
  return label;
}

function knowledgeDocumentCard(documentRecord) {
  const card = document.createElement("article");
  const indexJob = (state.knowledge.jobs || []).find((job) => job.document_id === documentRecord.id);
  const effectiveStatus = documentRecord.status === "indexing" && indexJob?.status === "cancelled" ? "cancelled" : documentRecord.status;
  card.className = `knowledge-document ${documentRecord.enabled ? "enabled" : "disabled"} ${effectiveStatus}`;
  const header = document.createElement("header");
  const selection = document.createElement("input");
  selection.type = "checkbox";
  selection.checked = state.knowledgeSelection.has(documentRecord.id);
  selection.disabled = !documentRecord.enabled || documentRecord.status !== "ready" || workspaceLocked();
  selection.title = "Include this immutable generation in the next run";
  selection.addEventListener("change", () => {
    if (selection.checked) state.knowledgeSelection.add(documentRecord.id);
    else state.knowledgeSelection.delete(documentRecord.id);
    renderKnowledgeSelection();
  });
  const heading = document.createElement("div");
  const title = document.createElement("h3"); title.textContent = documentRecord.title;
  const meta = document.createElement("p");
  meta.textContent = `${displayStatus(documentRecord.source_type)} · generation ${documentRecord.generation} · ${documentRecord.chunk_count} passage${documentRecord.chunk_count === 1 ? "" : "s"} · ${documentRecord.acquisition_count || 0} verified run import${documentRecord.acquisition_count === 1 ? "" : "s"} · ${formatBytes(documentRecord.bytes)}`;
  heading.append(title, meta);
  const badge = document.createElement("span");
  badge.textContent = documentRecord.status === "ready"
    ? (documentRecord.enabled ? "CURRENT · READY" : "CURRENT · DISABLED")
    : effectiveStatus === "cancelled" ? "INDEX CANCELLED"
      : documentRecord.status === "indexing" ? "QUEUED / INDEXING" : documentRecord.status.toUpperCase();
  header.append(selection, heading, badge);

  const description = document.createElement("p");
  description.className = "knowledge-description";
  description.textContent = documentRecord.description || "No description.";
  const tags = document.createElement("small");
  tags.textContent = `${(documentRecord.tags || []).join(" · ") || "untagged"} · sha256 ${documentRecord.content_sha256.slice(0, 14)}…`;
  const routing = document.createElement("small");
  const routingMetadata = documentRecord.semantic_metadata?.routing || {};
  routing.textContent = effectiveStatus === "cancelled"
    ? "Semantic indexing was cancelled. Exact source bytes remain stored and the job can be retried; any published predecessor remains available."
    : documentRecord.status === "indexing"
    ? "Pending: Qwen will index exact text; Gemma will inspect only extracted images. The published predecessor remains available."
    : documentRecord.status === "index_failed"
      ? "Semantic indexing failed. Exact source bytes remain stored; use the failed job below to inspect events or retry."
      : `Retrieval routing: text ${routingMetadata.text || "exact lexical + Qwen descriptors when available"}; visuals ${routingMetadata.visual || "Gemma only when actual images exist"}.`;
  const actions = document.createElement("div"); actions.className = "knowledge-document-actions";
  const preview = document.createElement("button"); preview.type = "button"; preview.textContent = "Preview"; preview.addEventListener("click", () => openKnowledgePreview(documentRecord));
  const chunks = document.createElement("button"); chunks.type = "button"; chunks.textContent = "Indexed passages"; chunks.addEventListener("click", () => openKnowledgeChunks(documentRecord));
  const acquisitions = document.createElement("button"); acquisitions.type = "button"; acquisitions.textContent = "Import history"; acquisitions.addEventListener("click", () => openKnowledgeAcquisitions(documentRecord));
  const figures = document.createElement("button"); figures.type = "button"; figures.textContent = "Figures"; figures.addEventListener("click", () => loadKnowledgeVisuals(documentRecord));
  const download = document.createElement("a"); download.href = `/api/knowledge/${encodeURIComponent(documentRecord.id)}/download`; download.textContent = "Download";
  actions.append(preview, chunks, figures, acquisitions, download);
  if (documentRecord.status === "ready") {
    const edit = document.createElement("button"); edit.type = "button"; edit.textContent = "Edit metadata";
    const toggle = document.createElement("button"); toggle.type = "button"; toggle.textContent = documentRecord.enabled ? "Disable" : "Enable";
    const reindex = document.createElement("button"); reindex.type = "button"; reindex.textContent = "Reindex";
    const remove = document.createElement("button"); remove.type = "button"; remove.textContent = "Delete"; remove.className = "danger-link";
    actions.append(edit, toggle, reindex, remove);
    const editForm = buildKnowledgeEditForm(documentRecord, card);
    edit.addEventListener("click", () => editForm.classList.toggle("hidden"));
    toggle.addEventListener("click", () => mutateKnowledge(`/api/knowledge/${encodeURIComponent(documentRecord.id)}/enabled`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ enabled: !documentRecord.enabled, etag: documentRecord.etag }) }, "Knowledge availability updated."));
    reindex.addEventListener("click", () => mutateKnowledge(`/api/knowledge/${encodeURIComponent(documentRecord.id)}/reindex?etag=${documentRecord.etag}`, { method: "POST" }, "A new indexed generation was created."));
    remove.addEventListener("click", async () => {
      if (!window.confirm(`Delete “${documentRecord.title}” from future runs? Existing run snapshots remain reproducible.`)) return;
      await mutateKnowledge(`/api/knowledge/${encodeURIComponent(documentRecord.id)}?etag=${documentRecord.etag}`, { method: "DELETE" }, "Knowledge document deleted.");
    });
    card.append(header, description, tags, routing, actions, editForm);
  } else {
    card.append(header, description, tags, routing, actions);
  }
  return card;
}

function buildKnowledgeEditForm(documentRecord, card) {
  const form = document.createElement("form"); form.className = "knowledge-inline-edit hidden";
  const title = document.createElement("input"); title.value = documentRecord.title; title.maxLength = 300; title.required = true;
  const type = document.createElement("select");
  for (const value of ["primary_study", "review", "guideline", "documentation", "dataset", "web_page", "other"]) { const option = document.createElement("option"); option.value = value; option.textContent = displayStatus(value); option.selected = value === documentRecord.source_type; type.append(option); }
  const url = document.createElement("input"); url.type = "url"; url.maxLength = 2000; url.value = documentRecord.canonical_url || "";
  const tags = document.createElement("input"); tags.value = (documentRecord.tags || []).join(", "); tags.maxLength = 1000;
  const description = document.createElement("textarea"); description.rows = 3; description.maxLength = 4000; description.value = documentRecord.description || "";
  const save = document.createElement("button"); save.type = "submit"; save.textContent = "Save as new generation";
  form.append(knowledgeLabel("Title", title), knowledgeLabel("Source type", type), knowledgeLabel("Canonical URL", url), knowledgeLabel("Tags", tags), knowledgeLabel("Description", description), save);
  form.addEventListener("submit", async (event) => {
    event.preventDefault(); save.disabled = true;
    try {
      await mutateKnowledge(`/api/knowledge/${encodeURIComponent(documentRecord.id)}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: title.value, source_type: type.value, canonical_url: url.value, tags: tags.value.split(",").map((item) => item.trim()).filter(Boolean), description: description.value, etag: documentRecord.etag }) }, "Metadata saved as an immutable generation.");
    } finally { save.disabled = false; }
  });
  return form;
}

function renderKnowledgeCatalog() {
  const catalog = el("knowledge-catalog");
  if (!catalog) return;
  catalog.replaceChildren();
  const query = el("knowledge-filter").value.trim().toLowerCase();
  const records = state.knowledge.documents.filter((item) => !query || `${item.title} ${item.description} ${item.source_type} ${(item.tags || []).join(" ")}`.toLowerCase().includes(query));
  if (!records.length) { const empty = document.createElement("p"); empty.className = "file-empty"; empty.textContent = "No knowledge documents match this view."; catalog.append(empty); return; }
  for (const item of records) catalog.append(knowledgeDocumentCard(item));
}

function renderKnowledgeJobs() {
  const container = el("knowledge-jobs");
  container.replaceChildren();
  const allJobs = state.knowledge.jobs || [];
  const jobs = allJobs.slice(0, 20);
  if (!jobs.length) {
    const empty = document.createElement("p"); empty.className = "file-empty"; empty.textContent = "No semantic indexing jobs yet."; container.append(empty); return;
  }
  const documentById = new Map(state.knowledge.documents.map((item) => [item.id, item]));
  const queuedOrder = allJobs.filter((job) => job.status === "queued").sort((left, right) => left.created.localeCompare(right.created));
  for (const job of jobs) {
    const item = document.createElement("article"); item.className = `knowledge-job ${job.status}`;
    const header = document.createElement("header");
    const title = document.createElement("strong"); title.textContent = documentById.get(job.document_id)?.title || `Document ${job.document_id.slice(0, 8)}`;
    const status = document.createElement("span"); status.textContent = displayStatus(job.status).toUpperCase();
    header.append(title, status);
    const message = document.createElement("p"); message.textContent = job.message;
    const detail = document.createElement("small");
    const queuePosition = job.status === "queued" ? `queue position ${queuedOrder.findIndex((item) => item.id === job.id) + 1} · ` : "";
    detail.textContent = `${queuePosition}${displayStatus(job.operation)} · attempt ${job.attempt} · updated ${formatDate(job.updated)} · Qwen text → Gemma actual images only`;
    const controls = document.createElement("nav");
    if (["queued", "running", "cancel_requested"].includes(job.status)) {
      const cancel = document.createElement("button"); cancel.type = "button"; cancel.className = "danger-link"; cancel.textContent = job.status === "cancel_requested" ? "Cancelling…" : "Cancel"; cancel.disabled = job.status === "cancel_requested";
      cancel.addEventListener("click", async () => { cancel.disabled = true; try { await mutateKnowledge(`/api/knowledge/jobs/${encodeURIComponent(job.id)}/cancel`, { method: "POST" }, "Index cancellation requested."); } catch (error) { cancel.disabled = false; toast(error.message, "error"); } });
      controls.append(cancel);
    }
    if (["failed", "cancelled"].includes(job.status)) {
      const retry = document.createElement("button"); retry.type = "button"; retry.textContent = "Retry";
      retry.addEventListener("click", async () => { retry.disabled = true; try { await mutateKnowledge(`/api/knowledge/jobs/${encodeURIComponent(job.id)}/retry`, { method: "POST" }, "Indexing queued again."); } catch (error) { retry.disabled = false; toast(error.message, "error"); } });
      controls.append(retry);
    }
    const events = state.knowledgeJobEvents.get(job.id) || [];
    item.append(header, message, detail, controls);
    if (events.length) {
      const timeline = document.createElement("ol"); timeline.className = "knowledge-job-events";
      for (const event of events.slice(-6)) {
        const row = document.createElement("li"); row.textContent = `${event.actor}: ${event.message}`; timeline.append(row);
      }
      item.append(timeline);
    }
    container.append(item);
  }
}

function renderKnowledgeVisuals(visuals, title) {
  state.knowledgeVisuals = visuals;
  state.knowledgeVisualTitle = title;
  const gallery = el("knowledge-visual-gallery"); gallery.replaceChildren();
  el("knowledge-visual-help").textContent = visuals.length
    ? `${title} · ${visuals.length} hash-verified visual artifact${visuals.length === 1 ? "" : "s"}. Gemma descriptors aid retrieval but are not shown as evidence.`
    : `${title} has no registered visual artifacts.`;
  for (const visual of visuals) {
    const figure = document.createElement("figure"); figure.className = "knowledge-visual";
    const open = document.createElement("button"); open.type = "button"; open.title = "Open full visual preview";
    const image = document.createElement("img"); image.src = visual.preview_url; image.alt = visual.source_label || "Knowledge visual"; image.loading = "lazy";
    open.append(image); open.addEventListener("click", () => openKnowledgeVisualPreview(visual, title));
    const caption = document.createElement("figcaption"); caption.textContent = visual.source_label || "Extracted image";
    const hash = document.createElement("small"); hash.textContent = `sha256 ${visual.sha256.slice(0, 14)}…`; caption.append(hash);
    figure.append(open, caption); gallery.append(figure);
  }
}

async function loadKnowledgeVisuals(documentRecord) {
  el("knowledge-visual-help").textContent = `Loading visual artifacts for ${documentRecord.title}…`;
  const visuals = await api(`/api/knowledge/${encodeURIComponent(documentRecord.id)}/visuals`);
  renderKnowledgeVisuals(visuals, documentRecord.title);
}

function openKnowledgeVisualPreview(visual, title) {
  el("knowledge-visual-preview-title").textContent = `${title} — ${visual.source_label || "visual"}`;
  el("knowledge-visual-preview-meta").textContent = `Exact stored image · sha256 ${visual.sha256}`;
  el("knowledge-visual-preview-image").src = visual.preview_url;
  el("knowledge-visual-preview-image").alt = visual.source_label || title;
  el("knowledge-visual-preview-download").href = visual.preview_url;
  const dialog = el("knowledge-visual-preview-dialog"); if (!dialog.open) dialog.showModal();
}

async function mutateKnowledge(path, options, message) {
  await api(path, options);
  await loadKnowledge(el("knowledge-show-retired").checked, true);
  toast(message);
}

async function openKnowledgePreview(documentRecord) {
  const dialog = el("artifact-preview-dialog");
  state.previewArtifact = { kind: "knowledge", path: documentRecord.id };
  el("artifact-preview-title").textContent = documentRecord.title;
  el("artifact-preview-meta").textContent = "Loading extracted text…";
  el("artifact-preview-text").textContent = "";
  el("artifact-preview-download").href = `/api/knowledge/${encodeURIComponent(documentRecord.id)}/download`;
  if (!dialog.open) dialog.showModal();
  const requestId = ++state.previewRequest;
  try {
    const preview = await api(`/api/knowledge/${encodeURIComponent(documentRecord.id)}/preview`);
    if (requestId !== state.previewRequest || state.previewArtifact?.kind !== "knowledge" || state.previewArtifact?.path !== documentRecord.id) return;
    el("artifact-preview-text").textContent = preview.content;
    el("artifact-preview-meta").textContent = `${formatBytes(preview.bytes)} · ${preview.truncated ? "head + tail preview" : "complete extracted text"}`;
  } catch (error) { if (requestId === state.previewRequest) { el("artifact-preview-meta").textContent = "Preview unavailable"; el("artifact-preview-text").textContent = error.message; } }
}

async function openKnowledgeChunks(documentRecord) {
  const dialog = el("artifact-preview-dialog");
  state.previewArtifact = { kind: "knowledge-chunks", path: documentRecord.id };
  el("artifact-preview-title").textContent = `${documentRecord.title} — indexed passages`;
  el("artifact-preview-meta").textContent = "Loading immutable passage index…";
  el("artifact-preview-text").textContent = "";
  el("artifact-preview-download").href = `/api/knowledge/${encodeURIComponent(documentRecord.id)}/download`;
  if (!dialog.open) dialog.showModal();
  const requestId = ++state.previewRequest;
  try {
    const chunks = await api(`/api/knowledge/${encodeURIComponent(documentRecord.id)}/chunks?limit=200`);
    if (requestId !== state.previewRequest || state.previewArtifact?.kind !== "knowledge-chunks" || state.previewArtifact?.path !== documentRecord.id) return;
    el("artifact-preview-text").textContent = chunks.map((chunk) =>
      `Passage ${chunk.ordinal + 1}\ncharacters ${chunk.char_start}–${chunk.char_end}\nsha256 ${chunk.sha256}\nindexed characters ${chunk.chars}`
    ).join("\n\n");
    el("artifact-preview-meta").textContent = `${chunks.length} immutable indexed passage${chunks.length === 1 ? "" : "s"}${documentRecord.chunk_count > chunks.length ? " · first 200 shown" : ""}`;
  } catch (error) {
    if (requestId === state.previewRequest) {
      el("artifact-preview-meta").textContent = "Passage index unavailable";
      el("artifact-preview-text").textContent = error.message;
    }
  }
}

async function openKnowledgeAcquisitions(documentRecord) {
  const dialog = el("artifact-preview-dialog");
  state.previewArtifact = { kind: "knowledge-acquisitions", path: documentRecord.id };
  el("artifact-preview-title").textContent = `${documentRecord.title} — verified import history`;
  el("artifact-preview-meta").textContent = "Loading controller-recorded acquisitions…";
  el("artifact-preview-text").textContent = "";
  el("artifact-preview-download").href = `/api/knowledge/${encodeURIComponent(documentRecord.id)}/download`;
  if (!dialog.open) dialog.showModal();
  const requestId = ++state.previewRequest;
  try {
    const acquisitions = await api(`/api/knowledge/${encodeURIComponent(documentRecord.id)}/acquisitions`);
    if (requestId !== state.previewRequest || state.previewArtifact?.kind !== "knowledge-acquisitions" || state.previewArtifact?.path !== documentRecord.id) return;
    el("artifact-preview-text").textContent = acquisitions.length ? acquisitions.map((item) =>
      `Acquired ${item.acquired_at}\nworkspace ${item.workspace_id}\nrun ${item.run_id}\nsource ${item.source_id}\nPMID ${item.pmid || "—"}\nDOI ${item.doi || "—"}\noriginal sha256 ${item.original_sha256}\ntext sha256 ${item.content_sha256}`
    ).join("\n\n") : "No completed run has automatically imported this document. It was added or curated manually.";
    el("artifact-preview-meta").textContent = `${acquisitions.length} controller-verified run acquisition${acquisitions.length === 1 ? "" : "s"}`;
  } catch (error) {
    if (requestId === state.previewRequest) {
      el("artifact-preview-meta").textContent = "Import history unavailable";
      el("artifact-preview-text").textContent = error.message;
    }
  }
}

async function openKnowledgeLibrary() {
  await loadKnowledge(el("knowledge-show-retired").checked);
  const dialog = el("knowledge-library-dialog");
  if (!dialog.open) dialog.showModal();
  scheduleKnowledgePolling();
}

async function uploadKnowledge(event) {
  event.preventDefault();
  const files = [...el("knowledge-upload-file").files];
  if (!files.length) return;
  const button = el("knowledge-upload-form").querySelector("button[type=submit]");
  button.disabled = true;
  const failures = [];
  let completed = 0;
  let queued = 0;
  let reused = 0;
  try {
    for (const [index, file] of files.entries()) {
      button.textContent = `Uploading ${index + 1} of ${files.length}…`;
      const body = new FormData();
      body.append("upload", file);
      body.append("title", files.length === 1 ? el("knowledge-upload-title").value : "");
      body.append("description", el("knowledge-upload-description").value);
      body.append("tags", el("knowledge-upload-tags").value);
      body.append("source_type", el("knowledge-upload-source-type").value);
      body.append("canonical_url", files.length === 1 ? el("knowledge-upload-url").value : "");
      try {
        const result = await api("/api/knowledge", { method: "POST", body });
        completed += 1;
        if (result.job) queued += 1; else reused += 1;
      }
      catch (error) { failures.push(`${file.name}: ${error.message}`); }
    }
    el("knowledge-upload-form").reset();
    await loadKnowledge(el("knowledge-show-retired").checked, true);
    toast(`${completed} document${completed === 1 ? "" : "s"} accepted · ${queued} queued · ${reused} already ready${failures.length ? ` · ${failures.length} failed` : ""}.`, failures.length ? "error" : "");
    if (failures.length) {
      el("knowledge-index-summary").textContent = failures.join(" · ");
    }
  } finally { button.disabled = false; button.textContent = "Upload and queue indexing"; }
}

async function searchKnowledge(event) {
  event.preventDefault();
  const container = el("knowledge-search-results");
  container.replaceChildren();
  const request = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: el("knowledge-search-query").value, limit: 8 }),
  };
  const [response, visualResponse] = await Promise.all([
    api("/api/knowledge/search", request),
    api("/api/knowledge/search/visuals", request),
  ]);
  const textHeading = document.createElement("strong"); textHeading.textContent = `Passages (${response.passages.length})`; container.append(textHeading);
  if (!response.passages.length) {
    const empty = document.createElement("p"); empty.textContent = "No exact passage matched. Try a synonym; absence of a hit is not proof of absence."; container.append(empty);
  }
  for (const passage of response.passages) {
    const item = document.createElement("article");
    const title = document.createElement("strong"); title.textContent = passage.title;
    const meta = document.createElement("small"); meta.textContent = `${displayStatus(passage.retrieval_method)} retrieval · characters ${passage.char_start}–${passage.char_end} · sha256 ${passage.chunk_sha256.slice(0, 14)}…`;
    const text = document.createElement("p"); text.textContent = passage.untrusted_source_text;
    item.append(title, meta, text); container.append(item);
  }
  const visualHeading = document.createElement("strong"); visualHeading.textContent = `Figures (${visualResponse.visuals.length})`; container.append(visualHeading);
  if (visualResponse.visuals.length) {
    for (const visual of visualResponse.visuals) {
      const item = document.createElement("article");
      const title = document.createElement("strong"); title.textContent = visual.title;
      const meta = document.createElement("small"); meta.textContent = `${displayStatus(visual.retrieval_method)} retrieval · ${visual.source_label} · sha256 ${visual.sha256.slice(0, 14)}…`;
      const show = document.createElement("button"); show.type = "button"; show.textContent = "Show exact image"; show.addEventListener("click", () => renderKnowledgeVisuals([visual], visual.title));
      item.append(title, meta, show); container.append(item);
    }
  } else {
    const empty = document.createElement("p"); empty.textContent = "No visual descriptor matched. This does not imply that the documents contain no relevant figure."; container.append(empty);
  }
  const limitations = document.createElement("p"); limitations.className = "search-limitations";
  limitations.textContent = `Method limits: ${[...(response.limitations || []), ...(visualResponse.limitations || [])].join(" ")}`;
  container.append(limitations);
}

function configureIntegrationDownloads() {
  const byId = new Map((state.integrations?.downloads || []).map((item) => [item.id, item]));
  for (const [id, anchorId, checksumId] of [
    ["skill", "skill-integration-download", "skill-integration-checksum"],
    ["a2a", "a2a-integration-download", "a2a-integration-checksum"],
  ]) {
    const item = byId.get(id);
    if (!item) continue;
    const anchor = el(anchorId);
    anchor.href = item.url;
    anchor.download = item.filename;
    anchor.title = `${item.filename} · ${formatBytes(item.bytes)}`;
    el(checksumId).textContent = item.sha256;
  }
  el("integration-a2a-status").textContent = state.integrations?.a2a_enabled
    ? "A2A execution is enabled; its bearer token remains separate."
    : "This deployment does not currently accept A2A execution.";
}

function modelLoadText(model) {
  const parts = [`${model.active_requests} active`, `${model.queued_requests} queued`];
  if (model.slots_total !== null) parts.push(`${model.slots_busy}/${model.slots_total} slots busy`);
  if (model.cache_usage_percent !== null) parts.push(`${model.cache_usage_percent}% context cache`);
  return parts.join(" · ");
}

function renderModelStatus(payload = state.modelStatus, stale = false) {
  if (!payload) {
    el("model-queue-summary").textContent = "Model load is temporarily unavailable";
    el("model-queue-updated").textContent = "retrying";
    return;
  }
  el("model-queue-summary").textContent = `${payload.summary.message}${stale ? " · last known state" : ""}`;
  el("model-queue-updated").dateTime = payload.updated_at;
  el("model-queue-updated").textContent = `updated ${new Intl.DateTimeFormat(undefined, { timeStyle: "medium" }).format(new Date(payload.updated_at))}`;
  for (const model of payload.models) {
    const card = el(`model-status-${model.role}`);
    if (!card) continue;
    card.dataset.state = model.state;
    el(`model-status-${model.role}-name`).textContent = model.model.split("/").pop();
    el(`model-status-${model.role}-state`).textContent = displayStatus(model.state);
    const detail = model.reachable ? modelLoadText(model) : model.message;
    el(`model-status-${model.role}-load`).textContent = detail;
    card.title = model.message;
  }
}

function scheduleModelStatus(delay = document.hidden ? 15000 : 5000) {
  if (state.modelStatusTimer) window.clearTimeout(state.modelStatusTimer);
  state.modelStatusTimer = window.setTimeout(refreshModelStatus, delay);
}

async function refreshModelStatus() {
  if (state.modelStatusLoading) return;
  state.modelStatusLoading = true;
  try {
    state.modelStatus = await api("/api/model-status");
    renderModelStatus();
  } catch (_) {
    renderModelStatus(state.modelStatus, true);
  } finally {
    state.modelStatusLoading = false;
    scheduleModelStatus();
  }
}

function configuredBrowserUrl() {
  const browser = state.config?.browser;
  if (!browser?.enabled) return null;
  const url = browser.public_url ? new URL(browser.public_url, window.location.href) : new URL(window.location.href);
  if (!browser.public_url) {
    url.port = String(browser.novnc_port);
    url.pathname = "/vnc.html";
    url.search = "";
    url.hash = "";
  } else if (!url.pathname || url.pathname === "/") {
    url.pathname = "/vnc.html";
  }
  if (!url.searchParams.has("autoconnect")) url.searchParams.set("autoconnect", "1");
  if (!url.searchParams.has("resize")) url.searchParams.set("resize", "scale");
  if (!url.searchParams.has("show_dot")) url.searchParams.set("show_dot", "1");
  return url.toString();
}

function configureResearchBrowser() {
  const button = el("research-browser-button");
  try { state.browserUrl = configuredBrowserUrl(); }
  catch (_) { state.browserUrl = null; }
  button.disabled = !state.browserUrl;
  button.title = state.browserUrl ? "Open the shared service-owned Chromium session" : "Managed browser is not configured";
  el("research-browser-new-tab").href = state.browserUrl || "#";
}

function openResearchBrowser() {
  if (!state.browserUrl) { toast("Managed research browser is not configured", "error"); return; }
  const frame = el("research-browser-frame");
  if (frame.src !== state.browserUrl) frame.src = state.browserUrl;
  const dialog = el("research-browser-dialog");
  if (!dialog.open) dialog.showModal();
}

function reconnectResearchBrowser() {
  if (!state.browserUrl) return;
  const frame = el("research-browser-frame");
  frame.src = "about:blank";
  window.setTimeout(() => { frame.src = state.browserUrl; }, 50);
}

async function loadWorkspaces(selectId = null) {
  state.workspaces = await api("/api/workspaces");
  renderWorkspaces();
  const desired = selectId || state.workspace?.id;
  if (desired && state.workspaces.some((item) => item.id === desired)) {
    await selectWorkspace(desired, false);
  } else if (state.workspaces.length && !state.workspace) {
    await selectWorkspace(state.workspaces[0].id, false);
  } else if (!state.workspaces.length) {
    clearPoll();
    clearEventStream();
    state.workspace = null;
    state.selectedRun = null;
    state.activeRun = null;
    state.runEvents = [];
    renderWorkspace();
  }
}

function renderWorkspaces() {
  const list = el("workspace-list");
  list.replaceChildren();
  for (const workspace of state.workspaces) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `workspace-item ${state.workspace?.id === workspace.id ? "active" : ""}`;
    const glyph = document.createElement("span");
    glyph.className = "workspace-glyph";
    glyph.textContent = workspace.name.slice(0, 2).toUpperCase();
    const copy = document.createElement("span");
    copy.className = "workspace-copy";
    const name = document.createElement("strong");
    name.textContent = workspace.name;
    const meta = document.createElement("small");
    meta.textContent = workspace.active_runs ? "run active" : `${workspace.run_count} run${workspace.run_count === 1 ? "" : "s"}`;
    copy.append(name, meta);
    const count = document.createElement("span");
    count.className = "workspace-count";
    count.textContent = workspace.active_runs ? "LIVE" : "";
    button.append(glyph, copy, count);
    button.addEventListener("click", () => selectWorkspace(workspace.id));
    list.append(button);
  }
}

async function selectWorkspace(workspaceId, closeSidebar = true) {
  clearPoll();
  clearEventStream();
  const token = ++state.workspaceToken;
  const workspace = await api(`/api/workspaces/${workspaceId}`);
  if (token !== state.workspaceToken) return;
  state.workspace = workspace;
  const active = activeSummary(workspace);
  state.activeRun = active ? await api(`/api/runs/${active.id}`) : null;
  const latest = active || workspace.runs[0] || null;
  state.selectedRun = latest ? (active ? state.activeRun : await api(`/api/runs/${latest.id}`)) : null;
  await resetEvents(state.activeRun?.id || state.selectedRun?.id || null);
  if (state.activeRun) startEventStream(state.activeRun.id);
  state.tab = "article";
  state.discussion = [];
  state.discussionRunId = null;
  state.discussionLoading = false;
  renderWorkspaces();
  renderWorkspace();
  if (closeSidebar) toggleSidebar(false);
  if (state.activeRun) schedulePoll();
}

async function resetEvents(runId) {
  state.eventRunId = runId;
  state.eventCursor = 0;
  state.runEvents = [];
  if (!runId) return;
  await appendEvents(runId);
}

async function appendEvents(runId) {
  if (state.eventRunId !== runId) {
    await resetEvents(runId);
    return;
  }
  for (let page = 0; page < 40; page += 1) {
    const events = await api(`/api/runs/${runId}/events?after_id=${state.eventCursor}`);
    const fresh = events.filter((event) => event.id > state.eventCursor);
    if (fresh.length) {
      state.runEvents.push(...fresh);
      state.eventCursor = fresh.at(-1).id;
    }
    if (events.length < 500) break;
  }
}

function clearEventStream() {
  state.eventSource?.close();
  state.eventSource = null;
  state.eventSourceRunId = null;
  if (el("activity-stream-state")) el("activity-stream-state").textContent = "POLL";
}

function startEventStream(runId) {
  if (!window.EventSource || state.eventSourceRunId === runId) return;
  clearEventStream();
  const source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events/stream?after_id=${state.eventCursor}`);
  state.eventSource = source;
  state.eventSourceRunId = runId;
  el("activity-stream-state").textContent = "CONNECTING";
  source.onopen = () => { if (state.eventSourceRunId === runId) el("activity-stream-state").textContent = "LIVE"; };
  source.onerror = () => { if (state.eventSourceRunId === runId) el("activity-stream-state").textContent = "RECONNECTING"; };
  source.addEventListener("run_event", (message) => {
    if (state.eventSourceRunId !== runId) return;
    let event;
    try { event = JSON.parse(message.data); } catch (_) { return; }
    if (!Number.isInteger(event.id) || event.id <= state.eventCursor) return;
    state.runEvents.push(event);
    state.eventCursor = event.id;
    renderActiveRun();
    renderRail();
    renderActivityLog();
    schedulePoll(150);
  });
  source.addEventListener("stream_end", () => {
    if (state.eventSourceRunId !== runId) return;
    clearEventStream();
    schedulePoll(0);
  });
}

function renderWorkspace() {
  const present = Boolean(state.workspace);
  el("empty-state").classList.toggle("hidden", present);
  el("workspace-view").classList.toggle("hidden", !present);
  if (!present) return;
  el("workspace-title").textContent = state.workspace.name;
  el("workspace-meta").textContent = `Created ${formatDate(state.workspace.created_at)} · ${state.workspace.files.length} input${state.workspace.files.length === 1 ? "" : "s"} · ${state.workspace.runs.length} run${state.workspace.runs.length === 1 ? "" : "s"}`;
  reflectActiveProtocol();
  renderActiveRun();
  setWorkspaceLocked(workspaceLocked());
  renderFiles();
  renderHistory();
  renderRun();
  renderRail();
  renderActivityLog();
}

function reflectActiveProtocol() {
  const active = state.activeRun;
  if (!active) return;
  el("objective").value = active.objective;
  el("enable-code").checked = Boolean(active.enable_code);
  document.querySelectorAll("#mcp-options input").forEach((input) => {
    input.checked = active.mcp_servers.includes(input.value);
  });
  document.querySelectorAll("#requested-output-options input").forEach((input) => {
    input.checked = (active.requested_outputs || []).includes(input.value);
  });
  if (active.knowledge_snapshot?.documents) {
    state.knowledgeSelection = new Set(active.knowledge_snapshot.documents.map((item) => item.document_id));
    renderKnowledgeSelection();
  }
}

function renderActiveRun() {
  const active = state.activeRun;
  const banner = el("active-run-banner");
  banner.classList.toggle("hidden", !active);
  document.querySelector(".system-status span:nth-child(2)").textContent = active ? "run active" : "service ready";
  if (!active) return;
  const latest = state.runEvents.at(-1);
  el("active-run-heading").textContent = displayStatus(active.phase || active.status).replace(/\b\w/g, (value) => value.toUpperCase());
  el("active-run-message").textContent = active.message;
  el("active-run-actor").textContent = latest?.actor || phaseActors[active.phase] || "Controller";
  el("active-run-started").textContent = formatDate(active.started_at || active.created_at);
  const execution = active.enable_code ? "Python/R on" : "Python/R off";
  const research = active.mcp_servers.length ? active.mcp_servers.join(", ") : "research off";
  el("active-run-capabilities").textContent = `${execution} · ${research}`;
  el("active-run-id").textContent = active.id.slice(0, 8);
  const cancel = el("cancel-run-button");
  cancel.disabled = state.cancelPending || active.status === "cancel_requested";
  cancel.textContent = cancel.disabled ? "Cancelling…" : "Cancel run";
}

function setWorkspaceLocked(locked) {
  el("workspace-view").setAttribute("aria-busy", String(locked));
  el("workspace-lock-note").classList.toggle("hidden", !locked);
  document.querySelector(".task-panel")?.classList.toggle("is-locked", locked);
  document.querySelectorAll("[data-run-mutable]").forEach((control) => {
    if (control.matches("button,input,textarea,select")) {
      const unconfiguredMcp = control.matches("#mcp-options input") && state.config && !state.config.mcp[control.value];
      control.disabled = locked || Boolean(unconfiguredMcp);
      if (unconfiguredMcp) control.title = "Not configured by the service owner";
    }
  });
  const drop = el("drop-zone");
  drop.classList.toggle("locked", locked);
  drop.setAttribute("aria-disabled", String(locked));
  renderKnowledgeCatalog();
}

function renderFiles() {
  const list = el("file-list");
  list.replaceChildren();
  const inputHeading = document.createElement("h3");
  inputHeading.className = "workspace-files-heading";
  inputHeading.textContent = `Inputs (${state.workspace.files.length})`;
  list.append(inputHeading);
  if (!state.workspace.files.length) {
    const empty = document.createElement("div");
    empty.className = "file-empty";
    empty.textContent = "No inputs uploaded.";
    list.append(empty);
  }
  const locked = workspaceLocked();
  for (const file of state.workspace.files) {
    const row = document.createElement("div");
    row.className = "file-row";
    const info = document.createElement("div");
    const name = document.createElement(isTextArtifact(file.name) ? "button" : "strong");
    name.textContent = file.name;
    if (isTextArtifact(file.name)) {
      name.type = "button";
      name.className = "workspace-file-preview";
      name.title = "Preview in browser";
      name.addEventListener("click", () => openWorkspaceFilePreview(file.name, state.workspace.id));
    }
    const size = document.createElement("small");
    size.textContent = formatBytes(file.bytes);
    info.append(name, size);
    const download = document.createElement("a");
    download.href = `/api/workspaces/${state.workspace.id}/files/${encodeURIComponent(file.name)}`;
    download.textContent = "↓";
    download.title = "Download";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.title = locked ? "Locked during active run" : "Delete file";
    remove.disabled = locked;
    remove.addEventListener("click", () => deleteFile(file.name));
    row.append(info, download, remove);
    list.append(row);
  }
  const run = state.selectedRun;
  const artifacts = run?.artifacts || [];
  const outputHeading = document.createElement("h3");
  outputHeading.className = "workspace-files-heading produced";
  outputHeading.textContent = `Selected run files (${artifacts.length})`;
  list.append(outputHeading);
  if (!run) {
    const empty = document.createElement("div");
    empty.className = "file-empty";
    empty.textContent = "Select or start a run to browse its files.";
    list.append(empty);
  } else if (!artifacts.length) {
    const empty = document.createElement("div");
    empty.className = "file-empty";
    empty.textContent = "Run files will appear here as they are created.";
    list.append(empty);
  } else {
    const files = document.createElement("div");
    files.className = "workspace-run-files";
    for (const artifact of artifacts) files.append(artifactRow(artifact, run.id));
    list.append(files);
  }
}

function renderHistory() {
  const history = el("run-history");
  history.replaceChildren();
  if (!state.workspace.runs.length) {
    const empty = document.createElement("div");
    empty.className = "history-empty";
    empty.textContent = "No runs yet.";
    history.append(empty);
    return;
  }
  for (const run of state.workspace.runs) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `run-item ${state.selectedRun?.id === run.id ? "active" : ""}`;
    const dot = document.createElement("i");
    dot.className = statusClass(run.status);
    const copy = document.createElement("span");
    const objective = document.createElement("strong");
    objective.textContent = run.run_kind === "revision" ? `Revision: ${run.objective}` : run.objective;
    const date = document.createElement("small");
    date.textContent = `${displayStatus(run.status)} · ${formatDate(run.created_at)}`;
    copy.append(objective, date);
    const id = document.createElement("code");
    id.textContent = run.id.slice(0, 4);
    button.append(dot, copy, id);
    button.addEventListener("click", () => selectRun(run.id));
    history.append(button);
  }
}

async function selectRun(runId) {
  state.selectedRun = await api(`/api/runs/${runId}`);
  state.tab = "article";
  state.discussion = [];
  state.discussionRunId = null;
  state.discussionLoading = false;
  if (!state.activeRun) await resetEvents(runId);
  renderHistory();
  renderFiles();
  renderRun();
  renderRail();
  renderActivityLog();
}

function renderRun() {
  const run = state.selectedRun;
  if (!run || !run.report) {
    el("result-empty").classList.remove("hidden");
    el("result-content").classList.add("hidden");
    el("result-status").textContent = run ? displayStatus(run.status).toUpperCase() : "WAITING";
    el("result-status").className = `panel-state ${run ? statusClass(run.status) : ""}`;
    renderLiveArtifacts(run);
    return;
  }
  el("result-empty").classList.add("hidden");
  el("result-content").classList.remove("hidden");
  el("result-status").textContent = displayStatus(run.status).toUpperCase();
  el("result-status").className = `panel-state ${statusClass(run.status)}`;
  el("report-title").textContent = run.report.title;
  el("report-download").href = artifactUrl("report.md", run.id);
  el("bundle-download").href = `/api/runs/${run.id}/bundle`;
  renderRevisionLineage();
  renderTab();
  el("follow-up-panel").classList.toggle("hidden", !terminalStates.has(run.status));
}

function renderLiveArtifacts(run) {
  const target = el("result-empty");
  target.replaceChildren();
  const text = document.createElement("p");
  if (!run) text.textContent = "Run a task to populate the evidence record.";
  else if (activeStates.has(run.status)) text.textContent = "Work is in progress. Observable outputs and artifacts appear below as the controller records them.";
  else text.textContent = `This run ended as ${displayStatus(run.status)} without a complete article report.`;
  target.append(text);
  if (run?.artifacts?.length) {
    const list = document.createElement("div");
    list.className = "live-artifacts";
    for (const artifact of [...run.artifacts].reverse()) list.append(artifactRow(artifact, run.id));
    target.append(list);
  }
}

function monitoredRun() { return state.activeRun || state.selectedRun; }

function renderRail() {
  const run = monitoredRun();
  el("run-message").textContent = run ? run.message : "No run selected.";
  const current = run?.phase || "";
  let activeIndex = phaseOrder.indexOf(current);
  if (current === "plan-review") activeIndex = phaseOrder.indexOf("planning");
  if (current === "reporting") activeIndex = phaseOrder.indexOf("research");
  if (current === "complete") activeIndex = phaseOrder.length;
  if (["failed", "stopped", "cancelled", "canceling"].includes(current)) activeIndex = -1;
  document.querySelectorAll("#provenance-rail li").forEach((item, index) => {
    item.className = "";
    item.removeAttribute("aria-current");
    const marker = item.querySelector(":scope > span");
    if (!run) { marker.textContent = "—"; return; }
    if (index < activeIndex || activeIndex === phaseOrder.length) {
      item.classList.add("done"); marker.textContent = "OK";
    } else if (index === activeIndex && activeStates.has(run.status)) {
      item.classList.add("active"); item.setAttribute("aria-current", "step"); marker.textContent = "NOW";
    } else if (!activeStates.has(run.status) && index > Math.max(activeIndex, 0)) {
      item.classList.add("skipped"); marker.textContent = "—";
    } else { marker.textContent = "—"; }
  });
}

function renderActivityLog() {
  const log = el("activity-log");
  log.replaceChildren();
  el("activity-count").textContent = `${state.runEvents.length} event${state.runEvents.length === 1 ? "" : "s"}`;
  if (!state.runEvents.length) {
    const item = document.createElement("li");
    item.className = "activity-empty";
    item.textContent = "Activity will appear when a run starts.";
    log.append(item);
    refreshModelOutput();
    return;
  }
  const runId = state.eventRunId;
  for (const event of state.runEvents.slice(-100).reverse()) {
    const item = document.createElement("li");
    item.className = `activity-event ${statusClass(event.status)}`;
    const header = document.createElement("div");
    const actor = document.createElement("strong");
    actor.className = "event-actor";
    actor.textContent = event.actor;
    const time = document.createElement("time");
    time.dateTime = event.created_at;
    time.textContent = formatDate(event.created_at);
    header.append(actor, time);
    const message = document.createElement("p");
    message.textContent = event.message;
    const meta = document.createElement("small");
    meta.textContent = `${displayStatus(event.phase)} · ${displayStatus(event.event_type)}`;
    item.append(header, message, meta);
    if (event.artifact_path && runId) {
      if (isTextArtifact(event.artifact_path)) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "event-artifact-preview";
        button.textContent = `Preview ${event.artifact_path}`;
        button.addEventListener("click", () => openArtifactPreview(event.artifact_path, runId));
        item.append(button);
      } else {
        const link = document.createElement("a");
        link.href = artifactUrl(event.artifact_path, runId);
        link.textContent = `Open ${event.artifact_path}`;
        item.append(link);
      }
    }
    log.append(item);
  }
  refreshModelOutput();
}

async function refreshModelOutput() {
  const monitor = el("model-output-monitor");
  const runId = state.eventRunId;
  const event = [...state.runEvents].reverse().find((item) => item.event_type.startsWith("model_output") && item.artifact_path);
  if (!event || !runId) {
    monitor.classList.add("hidden");
    el("model-output-text").textContent = "";
    state.modelOutputArtifact = null;
    return;
  }
  monitor.classList.remove("hidden");
  el("model-output-actor").textContent = event.actor;
  state.modelOutputArtifact = { path: event.artifact_path, runId };
  const requestId = ++state.modelOutputRequest;
  try {
    const response = await fetch(artifactUrl(event.artifact_path, runId));
    if (!response.ok) return;
    const text = await response.text();
    if (requestId !== state.modelOutputRequest || state.eventRunId !== runId) return;
    el("model-output-text").textContent = text.length > 30000 ? `…${text.slice(-30000)}` : text;
    if (state.previewArtifact?.path === event.artifact_path && state.previewArtifact?.runId === runId) {
      refreshArtifactPreview(event.artifact_path, runId, false);
    }
  } catch (_) { /* polling will retry without a noisy toast */ }
}

function artifactUrl(path, runId = state.selectedRun?.id) {
  return `/api/runs/${runId}/artifacts?path=${encodeURIComponent(path)}`;
}

function artifactPreviewUrl(path, runId) {
  return `/api/runs/${runId}/artifact-preview?path=${encodeURIComponent(path)}`;
}

function isTextArtifact(path) {
  const name = String(path || "").split("/").pop();
  if (!name.includes(".")) return true;
  const extension = name.includes(".") ? name.split(".").pop().toLowerCase() : "";
  return textArtifactExtensions.has(extension);
}

async function openArtifactPreview(path, runId) {
  const dialog = el("artifact-preview-dialog");
  state.previewArtifact = { kind: "run", path, runId };
  el("artifact-preview-title").textContent = path;
  el("artifact-preview-meta").textContent = "Loading bounded UTF-8 preview…";
  el("artifact-preview-text").textContent = "";
  el("artifact-preview-download").href = artifactUrl(path, runId);
  if (!dialog.open) dialog.showModal();
  await refreshArtifactPreview(path, runId, true);
}

async function openWorkspaceFilePreview(filename, workspaceId) {
  const dialog = el("artifact-preview-dialog");
  state.previewArtifact = { kind: "workspace", path: filename, workspaceId };
  el("artifact-preview-title").textContent = filename;
  el("artifact-preview-meta").textContent = "Loading bounded UTF-8 preview…";
  el("artifact-preview-text").textContent = "";
  el("artifact-preview-download").href = `/api/workspaces/${encodeURIComponent(workspaceId)}/files/${encodeURIComponent(filename)}`;
  if (!dialog.open) dialog.showModal();
  const requestId = ++state.previewRequest;
  try {
    const preview = await api(`/api/workspaces/${encodeURIComponent(workspaceId)}/file-preview?filename=${encodeURIComponent(filename)}`);
    if (requestId !== state.previewRequest || !dialog.open || state.previewArtifact?.kind !== "workspace" || state.previewArtifact?.path !== filename || state.previewArtifact?.workspaceId !== workspaceId) return;
    el("artifact-preview-text").textContent = preview.content;
    el("artifact-preview-meta").textContent = `${formatBytes(preview.bytes)} · ${preview.truncated ? "head + tail preview; middle omitted" : "complete UTF-8 preview"}`;
    el("artifact-preview-text").focus();
  } catch (error) {
    if (requestId !== state.previewRequest) return;
    el("artifact-preview-meta").textContent = "Preview unavailable";
    el("artifact-preview-text").textContent = error.message;
  }
}

async function refreshArtifactPreview(path, runId, showErrors) {
  const dialog = el("artifact-preview-dialog");
  const requestId = ++state.previewRequest;
  try {
    const preview = await api(artifactPreviewUrl(path, runId));
    if (requestId !== state.previewRequest || !dialog.open || state.previewArtifact?.path !== path || state.previewArtifact?.runId !== runId) return;
    const previewText = el("artifact-preview-text");
    const wasAtEnd = previewText.scrollHeight - previewText.scrollTop - previewText.clientHeight < 48;
    previewText.textContent = preview.content;
    el("artifact-preview-meta").textContent = `${formatBytes(preview.bytes)} · ${preview.truncated ? "live head + tail preview; middle omitted" : "complete UTF-8 preview"}`;
    if (wasAtEnd) previewText.scrollTop = previewText.scrollHeight;
    if (showErrors) previewText.focus();
  } catch (error) {
    if (!showErrors || requestId !== state.previewRequest) return;
    el("artifact-preview-meta").textContent = "Preview unavailable";
    el("artifact-preview-text").textContent = error.message;
  }
}

function renderTab() {
  document.querySelectorAll(".report-tabs button").forEach((button) => {
    const active = button.dataset.tab === state.tab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  const content = el("tab-content");
  content.replaceChildren();
  const run = state.selectedRun;
  const report = run.report;
  if (state.tab === "article") {
    content.append(renderArticle(report, run));
  } else if (state.tab === "claims") {
    for (const claim of report.claims || []) content.append(claimCard(claim, run));
  } else if (state.tab === "sources") {
    for (const source of report.sources || []) content.append(sourceRow(source, run));
  } else if (state.tab === "artifacts") {
    for (const artifact of run.artifacts || []) content.append(artifactRow(artifact, run.id));
  } else if (state.tab === "review") {
    content.append(renderReview(run));
  } else {
    content.append(renderDiscussion(run));
  }
}

async function loadDiscussion(runId) {
  state.discussionLoading = true;
  if (state.tab === "discussion" && state.selectedRun?.id === runId) renderTab();
  try {
    const messages = await api(`/api/runs/${encodeURIComponent(runId)}/discussion`);
    if (state.selectedRun?.id !== runId) return;
    state.discussion = messages;
    state.discussionRunId = runId;
  } finally {
    state.discussionLoading = false;
    if (state.tab === "discussion" && state.selectedRun?.id === runId) renderTab();
  }
}

function useRevisionPrompt(prompt) {
  el("follow-up-query").value = prompt;
  el("follow-up-query").focus();
  el("follow-up-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  toast("Gemma's revision brief is ready for review. Starting it remains a separate audited action.");
}

function discussionMessage(message) {
  const article = document.createElement("article");
  article.className = `discussion-message ${message.role} ${message.status}`;
  const header = document.createElement("header");
  const actor = document.createElement("strong");
  actor.textContent = message.role === "assistant" ? (message.model || "Gemma") : "You";
  const time = document.createElement("time");
  time.textContent = formatDate(message.created_at);
  header.append(actor, time);
  const body = document.createElement("p");
  body.textContent = message.status === "generating" ? "Gemma is preparing an evidence-bounded answer…" : message.content;
  article.append(header, body);
  if (message.evidence_refs?.length) {
    const refs = document.createElement("small");
    refs.textContent = `Evidence: ${message.evidence_refs.join(", ")}`;
    article.append(refs);
  }
  if (message.unresolved_uncertainties?.length) {
    const uncertainties = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `${message.unresolved_uncertainties.length} unresolved uncertaint${message.unresolved_uncertainties.length === 1 ? "y" : "ies"}`;
    uncertainties.append(summary, makeList(message.unresolved_uncertainties));
    article.append(uncertainties);
  }
  if (message.suggested_revision_prompt) {
    const action = document.createElement("section");
    const label = document.createElement("strong");
    label.textContent = "Suggested audited revision";
    const prompt = document.createElement("p");
    prompt.textContent = message.suggested_revision_prompt;
    const use = document.createElement("button");
    use.type = "button";
    use.textContent = "Use as Qwen→Gemma revision prompt";
    use.disabled = workspaceLocked();
    use.addEventListener("click", () => useRevisionPrompt(message.suggested_revision_prompt));
    action.append(label, prompt, use);
    article.append(action);
  }
  return article;
}

function renderDiscussion(run) {
  const wrapper = document.createElement("section");
  wrapper.className = "gemma-discussion";
  const heading = document.createElement("div");
  heading.className = "discussion-heading";
  const copy = document.createElement("div");
  const eyebrow = document.createElement("span");
  eyebrow.className = "eyebrow";
  eyebrow.textContent = "Independent explanation lane";
  const title = document.createElement("h4");
  title.textContent = `Discuss this report with ${state.config?.models.critic || "Gemma"}`;
  const note = document.createElement("p");
  note.textContent = "Gemma can explain, challenge, and draft a revision brief. It cannot edit the immutable report or override deterministic validation.";
  copy.append(eyebrow, title, note);
  heading.append(copy);
  wrapper.append(heading);

  const messages = document.createElement("div");
  messages.className = "discussion-thread";
  if (state.discussionRunId !== run.id && !state.discussionLoading) {
    window.setTimeout(() => loadDiscussion(run.id).catch((error) => toast(error.message, "error")), 0);
  }
  if (state.discussionLoading && state.discussionRunId !== run.id) {
    const loading = document.createElement("p");
    loading.className = "discussion-empty";
    loading.textContent = "Loading discussion…";
    messages.append(loading);
  } else if (!state.discussion.length) {
    const empty = document.createElement("p");
    empty.className = "discussion-empty";
    empty.textContent = "Ask what a result means, why a limitation matters, whether a claim is adequately supported, or how to improve the report.";
    messages.append(empty);
  } else {
    for (const message of state.discussion) messages.append(discussionMessage(message));
  }
  wrapper.append(messages);

  const form = document.createElement("form");
  form.className = "discussion-form";
  const label = document.createElement("label");
  label.htmlFor = "discussion-query";
  label.textContent = "Question for Gemma";
  const textarea = document.createElement("textarea");
  textarea.id = "discussion-query";
  textarea.rows = 4;
  textarea.minLength = 3;
  textarea.maxLength = 20000;
  textarea.required = true;
  textarea.placeholder = "Explain the practical meaning of the primary result and identify any wording that exceeds the evidence.";
  const actions = document.createElement("div");
  const helper = document.createElement("small");
  helper.textContent = "Maximum thinking is enabled. Only final answers are stored; hidden reasoning is never shown.";
  const submit = document.createElement("button");
  submit.type = "submit";
  submit.className = "primary-button";
  submit.textContent = state.discussionSending ? "Gemma is answering…" : "Ask Gemma";
  submit.disabled = state.discussionSending || workspaceLocked();
  textarea.disabled = state.discussionSending || workspaceLocked();
  actions.append(helper, submit);
  form.append(label, textarea, actions);
  form.addEventListener("submit", (event) => {
    submitDiscussion(event, run.id, textarea.value).catch((error) => toast(error.message, "error"));
  });
  wrapper.append(form);
  return wrapper;
}

async function submitDiscussion(event, runId, message) {
  event.preventDefault();
  if (state.discussionSending || workspaceLocked()) return;
  state.discussionSending = true;
  renderTab();
  try {
    await api(`/api/runs/${encodeURIComponent(runId)}/discussion`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message.trim() }),
    });
    await loadDiscussion(runId);
  } finally {
    state.discussionSending = false;
    if (state.tab === "discussion" && state.selectedRun?.id === runId) renderTab();
  }
}

function externalSourceNumbers(report) {
  const numbers = new Map();
  let number = 0;
  for (const source of report.sources || []) {
    if (!source.url) continue;
    number += 1;
    numbers.set(source.source_id, number);
  }
  return numbers;
}

function appendCitationMarker(parent, citation, report, run) {
  const sources = new Map((report.sources || []).map((source) => [source.source_id, source]));
  const numbers = externalSourceNumbers(report);
  const marker = document.createElement("span");
  marker.className = "inline-citation";
  marker.append(document.createTextNode(" ["));
  citation.source_ids.forEach((sourceId, index) => {
    const source = sources.get(sourceId);
    const number = numbers.get(sourceId);
    if (!source || !number) return;
    if (index) marker.append(document.createTextNode(", "));
    const local = (run.reference_manifest?.references || []).find((item) => item.source_id === sourceId);
    const link = document.createElement(local?.markdown ? "button" : "a");
    link.textContent = String(number);
    link.title = source.title;
    if (local?.markdown) {
      link.type = "button";
      link.addEventListener("click", () => openArtifactPreview(local.markdown.path, run.id));
    } else {
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noreferrer";
    }
    marker.append(link);
  });
  marker.append(document.createTextNode("]"));
  parent.append(marker);
}

function appendCitedText(parent, text, report, run, sectionName) {
  const value = String(text || "Not reported.");
  const citations = (report.inline_citations || [])
    .filter((citation) => citation.section === sectionName)
    .map((citation) => ({ citation, start: value.indexOf(citation.anchor_text) }))
    .filter((item) => item.start >= 0)
    .sort((a, b) => a.start - b.start);
  let cursor = 0;
  for (const item of citations) {
    if (item.start < cursor) continue;
    const end = item.start + item.citation.anchor_text.length;
    parent.append(document.createTextNode(value.slice(cursor, end)));
    appendCitationMarker(parent, item.citation, report, run);
    cursor = end;
  }
  parent.append(document.createTextNode(value.slice(cursor)));
}

function appendParagraphs(section, text, report, run, sectionName) {
  const paragraphs = String(text || "Not reported.").split(/\n\s*\n/).filter(Boolean);
  for (const value of paragraphs) {
    const paragraph = document.createElement("p");
    appendCitedText(paragraph, value, report, run, sectionName);
    section.append(paragraph);
  }
}

function articleSection(title, text, report, run, sectionName) {
  const section = document.createElement("section");
  section.className = "article-section";
  const heading = document.createElement("h4");
  heading.textContent = title;
  section.append(heading);
  appendParagraphs(section, text, report, run, sectionName);
  return section;
}

function renderArticle(report, run) {
  const article = document.createElement("article");
  article.className = "report-article";
  const notice = document.createElement("p");
  notice.className = "report-boundary";
  if (supportedStates.has(run.status)) {
    notice.textContent = "Standards-derived exploratory report. Tests and evidence records outrank both models; publication use requires independent human and manuscript gates.";
  } else {
    notice.classList.add("unvalidated");
    notice.textContent = `NOT VALIDATED — run status: ${displayStatus(run.status)}. The article and claim labels are provisional model output and must not be treated as supported findings.`;
  }
  article.append(notice);
  article.append(articleSection("Abstract", report.executive_summary, report, run, "executive_summary"));
  article.append(articleSection("Introduction", report.introduction || "The legacy parent report did not contain a distinct Introduction.", report, run, "introduction"));
  const methods = articleSection("Methods", "", report, run, "methods");
  methods.querySelector("p")?.remove();
  const methodList = document.createElement("ul");
  methodList.className = "plain-list";
  for (const method of report.methods || ["Methods were not separately recorded."]) {
    const item = document.createElement("li");
    appendCitedText(item, method, report, run, "methods");
    methodList.append(item);
  }
  methods.append(methodList);
  appendDisplays(methods, run, "methods");
  article.append(methods);
  const results = articleSection("Results", report.results || report.narrative, report, run, "results");
  appendDisplays(results, run, "results");
  article.append(results);
  const discussion = articleSection("Discussion", report.discussion || "The legacy report did not contain a distinct Discussion.", report, run, "discussion");
  if (report.limitations?.length) {
    const heading = document.createElement("h5");
    heading.textContent = "Limitations";
    discussion.append(heading, makeList(report.limitations));
  }
  appendDisplays(discussion, run, "discussion");
  article.append(discussion);
  article.append(articleSection("Conclusions", report.conclusions || report.executive_summary, report, run, "conclusions"));
  return article;
}

function appendDisplays(section, run, placement) {
  const entries = run.display_manifest?.displays?.filter((item) => item.placement === placement) || [];
  for (const display of entries) {
    if (display.kind === "figure") section.append(renderFigure(display, run.id));
    else section.append(renderTable(display, run.id));
  }
}

function displayCaption(display) {
  const fragment = document.createDocumentFragment();
  const strong = document.createElement("strong");
  strong.textContent = `${display.kind === "figure" ? "Figure" : "Table"} ${display.number}. ${display.title}. `;
  fragment.append(strong, document.createTextNode(display.caption));
  return fragment;
}

function renderFigure(display, runId) {
  const figure = document.createElement("figure");
  figure.className = "report-figure";
  const link = document.createElement("a");
  link.href = `/api/runs/${runId}/displays/${encodeURIComponent(display.display_id)}/image`;
  link.target = "_blank";
  link.rel = "noreferrer";
  const image = document.createElement("img");
  image.src = link.href;
  image.alt = display.alt_text || "Scientific figure; alternative text was not supplied.";
  image.loading = "lazy";
  image.decoding = "async";
  link.append(image);
  const caption = document.createElement("figcaption");
  caption.append(displayCaption(display));
  figure.append(link, caption, evidenceLine(display));
  return figure;
}

function renderTable(display, runId) {
  const block = document.createElement("div");
  block.className = "report-table-block";
  const scroll = document.createElement("div");
  scroll.className = "report-table-scroll";
  const table = document.createElement("table");
  const caption = document.createElement("caption");
  caption.append(displayCaption(display));
  table.append(caption);
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const column of display.columns || []) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = column;
    headRow.append(cell);
  }
  head.append(headRow);
  const body = document.createElement("tbody");
  for (const row of display.rows || []) {
    const tr = document.createElement("tr");
    for (const value of row) { const cell = document.createElement("td"); cell.textContent = value; tr.append(cell); }
    body.append(tr);
  }
  table.append(head, body);
  scroll.append(table);
  const complete = document.createElement("a");
  complete.href = artifactUrl(display.path, runId);
  complete.textContent = display.truncated ? "Download complete table (preview truncated)" : "Download table";
  block.append(scroll, evidenceLine(display), complete);
  return block;
}

function evidenceLine(display) {
  const note = document.createElement("small");
  const refs = [...(display.claim_ids || []), ...(display.evidence_refs || [])];
  note.className = "display-evidence";
  note.textContent = `Evidence linkage: ${refs.join(", ") || "recorded artifact"}`;
  return note;
}

function claimCard(claim, run) {
  const card = document.createElement("article");
  const validated = supportedStates.has(run.status);
  card.className = `claim-card ${validated ? claim.status : "unvalidated"}`;
  const header = document.createElement("header");
  const id = document.createElement("code"); id.textContent = claim.claim_id;
  const status = document.createElement("span");
  status.textContent = validated
    ? displayStatus(claim.status)
    : `Model-labeled ${displayStatus(claim.status)} · run not validated`;
  header.append(id, status);
  const text = document.createElement("p"); text.textContent = claim.text;
  const refs = document.createElement("small"); refs.textContent = `Evidence: ${claim.evidence_refs.join(", ") || "none"}`;
  card.append(header, text, refs);
  return card;
}

function renderReview(run) {
  const wrapper = document.createElement("div");
  wrapper.className = "review-record";
  const validation = run.result?.deterministic_validation;
  const review = run.result?.scientific_review;
  const heading = document.createElement("h4");
  heading.textContent = "Standards and audit status";
  const status = document.createElement("p");
  status.textContent = `Deterministic validation: ${validation?.passed ? "passed" : "not passed or unavailable"}. Gemma review: ${displayStatus(review?.verdict || "unavailable")}. Model agreement is not proof.`;
  wrapper.append(heading, status);
  const findings = [...(validation?.findings || []).map((item) => `${item.code}: ${item.message}`), ...(review?.blocking_findings || []).map((item) => `Blocking: ${item.problem}`), ...(review?.nonblocking_findings || []).map((item) => `Comment: ${item.problem}`), ...(run.report.unresolved_issues || []).map((item) => `Unresolved: ${item}`), ...(run.report.limitations || []).map((item) => `Limitation: ${item}`)];
  wrapper.append(makeList(findings.length ? findings : ["No additional findings were recorded."]));
  return wrapper;
}

function makeList(items) {
  const list = document.createElement("ul");
  list.className = "plain-list";
  for (const value of items) { const item = document.createElement("li"); item.textContent = value; list.append(item); }
  return list;
}

function sourceRow(source, run) {
  const row = document.createElement("div"); row.className = "source-row";
  const id = document.createElement("code");
  const number = (run.report?.sources || []).filter((item) => item.url).findIndex((item) => item.source_id === source.source_id);
  id.textContent = number >= 0 ? `[${number + 1}] ${source.source_id}` : source.source_id;
  const copy = document.createElement("div");
  const local = (run.reference_manifest?.references || []).find((item) => item.source_id === source.source_id);
  const knowledge = (run.result?.retrieval_evidence?.knowledge_passages || []).find((item) => item.source_url === source.url);
  const knowledgeVisual = (run.result?.retrieval_evidence?.knowledge_visuals || []).find((item) => item.source_url === source.url);
  const link = document.createElement(local?.markdown ? "button" : source.url ? "a" : "span");
  link.textContent = source.title;
  if (local?.markdown) {
    link.type = "button";
    link.className = "source-preview-button";
    link.addEventListener("click", () => openArtifactPreview(local.markdown.path, run.id));
  } else if (source.url) {
    link.href = source.url; link.target = "_blank"; link.rel = "noreferrer";
  }
  const note = document.createElement("small"); note.textContent = source.supporting_passage;
  copy.append(link, note);
  const actions = document.createElement("div"); actions.className = "source-row-actions";
  if (local?.pdf) {
    const pdf = document.createElement("a");
    pdf.href = `/api/runs/${encodeURIComponent(run.id)}/references/${encodeURIComponent(source.source_id)}/pdf`;
    pdf.textContent = "PDF"; pdf.target = "_blank"; pdf.rel = "noreferrer";
    actions.append(pdf);
  }
  if (local?.markdown) {
    const markdown = document.createElement("button");
    markdown.type = "button"; markdown.textContent = source.full_text_status === "abstract_only" ? "Abstract" : "Text";
    markdown.addEventListener("click", () => openArtifactPreview(local.markdown.path, run.id));
    actions.append(markdown);
  }
  if (knowledge) {
    const text = document.createElement("a");
    text.href = `/api/runs/${encodeURIComponent(run.id)}/knowledge/documents/${encodeURIComponent(knowledge.document_id)}/text`;
    text.textContent = "Full text"; text.target = "_blank"; text.rel = "noreferrer";
    const original = document.createElement("a");
    original.href = `/api/runs/${encodeURIComponent(run.id)}/knowledge/documents/${encodeURIComponent(knowledge.document_id)}/original`;
    original.textContent = knowledge.document_filename?.toLowerCase().endsWith(".pdf") ? "PDF" : "Original";
    original.target = "_blank"; original.rel = "noreferrer";
    actions.append(text, original);
  }
  if (knowledgeVisual) {
    const preview = document.createElement("button");
    preview.type = "button"; preview.textContent = "Preview image";
    preview.addEventListener("click", () => openKnowledgeVisualPreview({
      preview_url: knowledgeVisual.source_url,
      source_label: knowledgeVisual.source_label,
      sha256: knowledgeVisual.artifact_sha256,
    }, knowledgeVisual.title));
    actions.append(preview);
  }
  if (source.url) {
    const canonical = document.createElement("a"); canonical.href = source.url;
    canonical.textContent = source.pmid ? `PMID ${source.pmid}` : "Source";
    canonical.target = "_blank"; canonical.rel = "noreferrer"; actions.append(canonical);
  }
  const type = document.createElement("span");
  type.textContent = displayStatus(local?.full_text_status || source.source_type);
  actions.append(type);
  row.append(id, copy, actions); return row;
}

function artifactRow(artifact, runId) {
  const row = document.createElement("div"); row.className = "artifact-row";
  const id = document.createElement("code"); id.textContent = artifact.path.split("/").pop().split(".").pop().toUpperCase();
  const copy = document.createElement("div");
  const link = document.createElement(isTextArtifact(artifact.path) ? "button" : "a");
  link.textContent = artifact.path;
  if (isTextArtifact(artifact.path)) {
    link.type = "button";
    link.className = "artifact-preview-button";
    link.addEventListener("click", () => openArtifactPreview(artifact.path, runId));
  } else {
    link.href = artifactUrl(artifact.path, runId);
  }
  const note = document.createElement("small");
  note.textContent = artifact.sha256 ? `sha256 ${artifact.sha256.slice(0, 14)}…` : "Live artifact · final hash pending";
  copy.append(link, note);
  const actions = document.createElement("div"); actions.className = "artifact-row-actions";
  const size = document.createElement("span"); size.textContent = formatBytes(artifact.bytes);
  const download = document.createElement("a"); download.href = artifactUrl(artifact.path, runId); download.textContent = "↓"; download.setAttribute("aria-label", `Download ${artifact.path}`);
  actions.append(size, download);
  row.append(id, copy, actions); return row;
}

function renderRevisionLineage() {
  const current = state.selectedRun;
  const runs = state.workspace.runs;
  const relatedIds = new Set([current.id]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const run of runs) {
      if ((run.parent_run_id && relatedIds.has(run.parent_run_id)) || (run.parent_run_id && relatedIds.has(run.id))) {
        if (!relatedIds.has(run.id)) { relatedIds.add(run.id); changed = true; }
        if (run.parent_run_id && !relatedIds.has(run.parent_run_id)) { relatedIds.add(run.parent_run_id); changed = true; }
      }
    }
  }
  const related = runs.filter((run) => relatedIds.has(run.id)).sort((a, b) => a.created_at.localeCompare(b.created_at));
  const currentIndex = related.findIndex((run) => run.id === current.id);
  el("revision-label").textContent = related.length > 1 ? `Revision ${currentIndex + 1} of ${related.length} · immutable lineage` : "Original audited record";
  const nav = el("revision-lineage");
  nav.replaceChildren();
  for (const [index, run] of related.entries()) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${index + 1}. ${run.run_kind === "revision" ? "Revision" : "Original"}`;
    button.classList.toggle("active", run.id === current.id);
    button.disabled = run.id === current.id;
    button.addEventListener("click", () => selectRun(run.id));
    nav.append(button);
  }
}

async function createWorkspace(name) {
  const workspace = await api("/api/workspaces", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
  el("new-workspace-form").reset();
  el("new-workspace-form").classList.add("hidden");
  state.workspace = null;
  await loadWorkspaces(workspace.id);
  toast("Workspace created.");
}

async function deleteWorkspace() {
  if (!state.workspace || workspaceLocked()) return;
  if (!window.confirm(`Delete “${state.workspace.name}” and all of its run artifacts?`)) return;
  await api(`/api/workspaces/${state.workspace.id}`, { method: "DELETE" });
  state.workspace = null; state.selectedRun = null; state.activeRun = null;
  await loadWorkspaces();
  toast("Workspace deleted.");
}

async function uploadFiles(files) {
  if (!state.workspace || !files.length) return;
  if (workspaceLocked()) { toast("Inputs are locked while a run is active.", "error"); return; }
  const guidance = el("upload-guidance");
  try {
    for (const [index, file] of files.entries()) {
      if (file.size > state.config.max_upload_bytes) throw new Error(`${file.name} exceeds the ${formatBytes(state.config.max_upload_bytes)} per-file limit.`);
      await new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      request.open("PUT", `/api/workspaces/${encodeURIComponent(state.workspace.id)}/files/${encodeURIComponent(file.name)}`);
      request.setRequestHeader("Content-Type", "application/octet-stream");
      request.upload.addEventListener("progress", (event) => {
        const progress = event.lengthComputable ? ` · ${Math.round((event.loaded / event.total) * 100)}%` : "";
        guidance.textContent = `Uploading ${index + 1}/${files.length}: ${file.name}${progress}`;
      });
      request.addEventListener("load", () => {
        if (request.status >= 200 && request.status < 300) resolve();
        else {
          let message = `${request.status} ${request.statusText}`;
          try { message = JSON.parse(request.responseText).detail || message; } catch (_) { /* not JSON */ }
          reject(new Error(message));
        }
      });
      request.addEventListener("error", () => reject(new Error(`Upload failed for ${file.name}`)));
      request.addEventListener("abort", () => reject(new Error(`Upload cancelled for ${file.name}`)));
      request.send(file);
      });
    }
  } finally {
    guidance.textContent = `Select many files, or ZIP a large directory · ${formatBytes(state.config.max_upload_bytes)} per file`;
  }
  await selectWorkspace(state.workspace.id, false);
  toast(`${files.length} file${files.length === 1 ? "" : "s"} added.`);
}

async function deleteFile(name) {
  if (workspaceLocked()) return;
  if (!window.confirm(`Delete input file “${name}”?`)) return;
  await api(`/api/workspaces/${state.workspace.id}/files/${encodeURIComponent(name)}`, { method: "DELETE" });
  await selectWorkspace(state.workspace.id, false);
}

async function startRun(event) {
  event.preventDefault();
  if (workspaceLocked()) return;
  const mcp = [...document.querySelectorAll("#mcp-options input:checked")].map((input) => input.value);
  const requestedOutputs = [...document.querySelectorAll("#requested-output-options input:checked")].map((input) => input.value);
  if (requestedOutputs.length && !el("enable-code").checked) throw new Error("Additional PPTX/notebook/data artifacts require Python + R execution.");
  const body = {
    objective: el("objective").value,
    enable_code: el("enable-code").checked,
    mcp_servers: mcp,
    knowledge_document_ids: [...state.knowledgeSelection],
    requested_outputs: requestedOutputs,
  };
  const run = await api(`/api/workspaces/${state.workspace.id}/runs`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  state.activeRun = await api(`/api/runs/${run.id}`);
  state.selectedRun = state.activeRun;
  state.workspace = await api(`/api/workspaces/${state.workspace.id}`);
  await resetEvents(run.id);
  startEventStream(run.id);
  renderWorkspace();
  schedulePoll();
  toast("Audited run queued.");
}

async function submitFollowUp(event) {
  event.preventDefault();
  if (!state.selectedRun?.report || workspaceLocked()) return;
  const request = el("follow-up-query").value.trim();
  const enable_code = el("follow-up-enable-code").checked;
  const run = await api(`/api/runs/${state.selectedRun.id}/follow-ups`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ request, enable_code }) });
  el("follow-up-form").reset();
  state.activeRun = await api(`/api/runs/${run.id}`);
  state.selectedRun = state.activeRun;
  state.workspace = await api(`/api/workspaces/${state.workspace.id}`);
  await resetEvents(run.id);
  startEventStream(run.id);
  renderWorkspace();
  schedulePoll();
  toast("Audited child revision queued.");
}

function openCancelDialog() {
  if (!state.activeRun || state.cancelPending) return;
  el("cancel-dialog").showModal();
}

async function cancelActiveRun() {
  if (!state.activeRun || state.cancelPending) return;
  state.cancelPending = true;
  renderActiveRun();
  try {
    const cancelled = await api(`/api/runs/${state.activeRun.id}/cancel`, { method: "POST" });
    state.activeRun = { ...state.activeRun, ...cancelled };
    if (state.selectedRun?.id === cancelled.id) state.selectedRun = { ...state.selectedRun, ...cancelled };
    await appendEvents(cancelled.id);
    renderWorkspace();
    schedulePoll(500);
    toast("Cancellation requested. Partial artifacts remain available.");
  } finally {
    state.cancelPending = false;
  }
}

function schedulePoll(delay = state.pollBackoff) {
  clearPoll();
  if (!state.activeRun) return;
  state.pollTimer = window.setTimeout(pollActiveRun, delay);
}

async function pollActiveRun() {
  if (!state.activeRun || !state.workspace) return;
  const runId = state.activeRun.id;
  const workspaceId = state.workspace.id;
  try {
    const [run, workspace] = await Promise.all([
      api(`/api/runs/${runId}`),
      api(`/api/workspaces/${workspaceId}`),
      appendEvents(runId),
    ]);
    if (state.workspace?.id !== workspaceId) return;
    state.workspace = workspace;
    state.activeRun = activeStates.has(run.status) ? run : null;
    if (state.activeRun) startEventStream(runId);
    else clearEventStream();
    if (state.selectedRun?.id === runId) state.selectedRun = run;
    state.pollBackoff = 1500;
    state.pollErrorShown = false;
    renderWorkspaces();
    renderWorkspace();
    if (state.activeRun) schedulePoll();
    else {
      try { await loadKnowledge(false, true); } catch (_) { /* run result remains authoritative */ }
      toast(`Run finished: ${displayStatus(run.status)}.`);
    }
  } catch (error) {
    if (!state.pollErrorShown) { toast(`Live monitor delayed: ${error.message}`, "error"); state.pollErrorShown = true; }
    state.pollBackoff = Math.min(15000, state.pollBackoff * 2);
    schedulePoll();
  }
}

function clearPoll() { if (state.pollTimer) window.clearTimeout(state.pollTimer); state.pollTimer = null; }
function toggleSidebar(open) { const sidebar = el("sidebar"); sidebar.classList.toggle("open", open); el("sidebar-toggle").setAttribute("aria-expanded", String(open)); }

function bindEvents() {
  el("new-workspace-button").addEventListener("click", () => { el("new-workspace-form").classList.toggle("hidden"); if (!el("new-workspace-form").classList.contains("hidden")) el("workspace-name").focus(); });
  el("empty-create-button").addEventListener("click", () => { toggleSidebar(true); el("new-workspace-form").classList.remove("hidden"); el("workspace-name").focus(); });
  el("new-workspace-form").addEventListener("submit", async (event) => { event.preventDefault(); try { await createWorkspace(el("workspace-name").value); } catch (error) { toast(error.message, "error"); } });
  el("delete-workspace-button").addEventListener("click", () => deleteWorkspace().catch((error) => toast(error.message, "error")));
  el("task-form").addEventListener("submit", (event) => startRun(event).catch((error) => toast(error.message, "error")));
  el("follow-up-form").addEventListener("submit", (event) => submitFollowUp(event).catch((error) => toast(error.message, "error")));
  el("file-input").addEventListener("change", (event) => uploadFiles([...event.target.files]).catch((error) => toast(error.message, "error")));
  el("sidebar-toggle").addEventListener("click", () => toggleSidebar(!el("sidebar").classList.contains("open")));
  el("cancel-run-button").addEventListener("click", openCancelDialog);
  el("model-output-expand").addEventListener("click", () => { if (state.modelOutputArtifact) openArtifactPreview(state.modelOutputArtifact.path, state.modelOutputArtifact.runId); });
  el("artifact-preview-close").addEventListener("click", () => el("artifact-preview-dialog").close());
  el("artifact-preview-dialog").addEventListener("close", () => { state.previewArtifact = null; state.previewRequest += 1; });
  el("research-browser-button").addEventListener("click", openResearchBrowser);
  el("research-browser-close").addEventListener("click", () => el("research-browser-dialog").close());
  el("research-browser-reconnect").addEventListener("click", reconnectResearchBrowser);
  el("integration-downloads-button").addEventListener("click", () => el("integration-downloads-dialog").showModal());
  el("integration-downloads-close").addEventListener("click", () => el("integration-downloads-dialog").close());
  el("knowledge-library-button").addEventListener("click", () => openKnowledgeLibrary().catch((error) => toast(error.message, "error")));
  el("knowledge-manage-inline").addEventListener("click", () => openKnowledgeLibrary().catch((error) => toast(error.message, "error")));
  el("knowledge-library-close").addEventListener("click", () => el("knowledge-library-dialog").close());
  el("knowledge-library-dialog").addEventListener("close", stopKnowledgePolling);
  el("knowledge-visual-preview-close").addEventListener("click", () => el("knowledge-visual-preview-dialog").close());
  el("knowledge-visual-preview-dialog").addEventListener("close", () => { el("knowledge-visual-preview-image").removeAttribute("src"); });
  el("knowledge-upload-form").addEventListener("submit", (event) => uploadKnowledge(event).catch((error) => toast(error.message, "error")));
  el("knowledge-search-form").addEventListener("submit", (event) => searchKnowledge(event).catch((error) => toast(error.message, "error")));
  el("knowledge-filter").addEventListener("input", renderKnowledgeCatalog);
  el("knowledge-show-retired").addEventListener("change", () => loadKnowledge(el("knowledge-show-retired").checked).catch((error) => toast(error.message, "error")));
  el("knowledge-upload-file").addEventListener("change", () => { const files = el("knowledge-upload-file").files; if (files.length !== 1) el("knowledge-upload-title").value = ""; });
  el("knowledge-select-all").addEventListener("click", () => { for (const item of state.knowledge.documents) if (item.status === "ready" && item.enabled) state.knowledgeSelection.add(item.id); renderKnowledgeSelection(); renderKnowledgeCatalog(); });
  el("knowledge-select-none").addEventListener("click", () => { state.knowledgeSelection.clear(); renderKnowledgeSelection(); renderKnowledgeCatalog(); });
  el("knowledge-reindex-all").addEventListener("click", async () => { if (!window.confirm("Create a new immutable indexed generation for every current document? Existing run snapshots will remain unchanged.")) return; try { await mutateKnowledge("/api/knowledge/reindex-all", { method: "POST" }, "Knowledge library reindexed."); } catch (error) { toast(error.message, "error"); } });
  el("cancel-dialog").addEventListener("close", () => { if (el("cancel-dialog").returnValue === "cancel") cancelActiveRun().catch((error) => toast(error.message, "error")); });
  const drop = el("drop-zone");
  ["dragenter", "dragover"].forEach((name) => drop.addEventListener(name, (event) => { event.preventDefault(); if (!workspaceLocked()) drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((name) => drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.remove("dragging"); }));
  drop.addEventListener("drop", (event) => uploadFiles([...event.dataTransfer.files]).catch((error) => toast(error.message, "error")));
  document.querySelectorAll(".report-tabs button").forEach((button) => button.addEventListener("click", () => { state.tab = button.dataset.tab; renderTab(); }));
  document.querySelectorAll(".workflow-starters button").forEach((button) => button.addEventListener("click", () => {
    if (workspaceLocked()) return;
    el("objective").value = workflowTemplates[button.dataset.template];
    if (button.dataset.template === "audit") document.querySelectorAll("#mcp-options input:not(:disabled)").forEach((input) => { input.checked = true; });
    el("objective").focus();
  }));
  document.querySelectorAll("[data-follow-up]").forEach((button) => button.addEventListener("click", () => { if (!workspaceLocked()) { el("follow-up-query").value = button.dataset.followUp; el("follow-up-query").focus(); } }));
  el("reuse-run").addEventListener("click", () => {
    if (!state.selectedRun || workspaceLocked()) return;
    el("objective").value = state.selectedRun.objective;
    el("enable-code").checked = state.selectedRun.enable_code;
    document.querySelectorAll("#mcp-options input").forEach((input) => { input.checked = state.selectedRun.mcp_servers.includes(input.value) && !input.disabled; });
    document.querySelectorAll("#requested-output-options input").forEach((input) => { input.checked = (state.selectedRun.requested_outputs || []).includes(input.value); });
    state.knowledgeSelection = new Set((state.selectedRun.knowledge_snapshot?.documents || []).map((item) => item.document_id));
    renderKnowledgeSelection();
    renderKnowledgeCatalog();
    el("objective").focus();
    el("task-form").scrollIntoView({ behavior: "smooth", block: "start" });
    toast("Protocol loaded for review. Starting it remains a separate action.");
  });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      if (state.modelStatusTimer) window.clearTimeout(state.modelStatusTimer);
      state.modelStatusTimer = null;
      refreshModelStatus();
    }
  });
}

async function init() {
  bindEvents();
  try { await loadConfig(); refreshModelStatus(); await loadWorkspaces(); }
  catch (error) { toast(error.message, "error"); }
}

document.addEventListener("DOMContentLoaded", init);
