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
};

const el = (id) => document.getElementById(id);
const activeStates = new Set(["queued", "running", "cancel_requested"]);
const supportedStates = new Set(["supported", "supported_with_comments"]);
const warningStates = new Set(["inconclusive", "requires_more_evidence", "requires_human_decision", "contradicted"]);
const terminalStates = new Set(["supported", "supported_with_comments", "contradicted", "inconclusive", "requires_more_evidence", "requires_human_decision", "failed", "interrupted", "cancelled"]);
const textArtifactExtensions = new Set(["bib", "c", "cfg", "conf", "cpp", "css", "csv", "env", "go", "h", "html", "ini", "ipynb", "java", "js", "json", "jsonl", "log", "md", "py", "qmd", "r", "rmd", "rs", "rst", "sh", "sql", "stan", "svg", "tex", "toml", "ts", "tsv", "txt", "xml", "yaml", "yml"]);
const phaseOrder = ["planning", "research", "validation", "scientific-review", "repair", "finalizing"];
const phaseActors = {
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
  state.config = await api("/api/config");
  el("app-version").textContent = `v${state.config.version}`;
  el("executor-model").textContent = state.config.models.executor.split("/").pop();
  el("critic-model").textContent = state.config.models.critic.split("/").pop();
  document.querySelectorAll("#mcp-options input").forEach((input) => {
    input.checked = state.config.default_mcp_servers.includes(input.value) && state.config.mcp[input.value];
  });
  configureResearchBrowser();
  setWorkspaceLocked(workspaceLocked());
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
  const token = ++state.workspaceToken;
  const workspace = await api(`/api/workspaces/${workspaceId}`);
  if (token !== state.workspaceToken) return;
  state.workspace = workspace;
  const active = activeSummary(workspace);
  state.activeRun = active ? await api(`/api/runs/${active.id}`) : null;
  const latest = active || workspace.runs[0] || null;
  state.selectedRun = latest ? (active ? state.activeRun : await api(`/api/runs/${latest.id}`)) : null;
  await resetEvents(state.activeRun?.id || state.selectedRun?.id || null);
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
  const events = await api(`/api/runs/${runId}/events?after_id=0`);
  state.runEvents = events;
  state.eventCursor = events.at(-1)?.id || 0;
}

async function appendEvents(runId) {
  if (state.eventRunId !== runId) {
    await resetEvents(runId);
    return;
  }
  const events = await api(`/api/runs/${runId}/events?after_id=${state.eventCursor}`);
  if (events.length) {
    state.runEvents.push(...events);
    state.eventCursor = events.at(-1).id;
  }
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
}

function renderFiles() {
  const list = el("file-list");
  list.replaceChildren();
  if (!state.workspace.files.length) {
    const empty = document.createElement("div");
    empty.className = "file-empty";
    empty.textContent = "No inputs uploaded.";
    list.append(empty);
    return;
  }
  const locked = workspaceLocked();
  for (const file of state.workspace.files) {
    const row = document.createElement("div");
    row.className = "file-row";
    const info = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = file.name;
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
  if (current === "plan-review" || current === "reporting") activeIndex = current === "plan-review" ? 0 : 1;
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
  state.previewArtifact = { path, runId };
  el("artifact-preview-title").textContent = path;
  el("artifact-preview-meta").textContent = "Loading bounded UTF-8 preview…";
  el("artifact-preview-text").textContent = "";
  el("artifact-preview-download").href = artifactUrl(path, runId);
  if (!dialog.open) dialog.showModal();
  await refreshArtifactPreview(path, runId, true);
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
    for (const claim of report.claims || []) content.append(claimCard(claim));
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

function appendParagraphs(section, text) {
  const paragraphs = String(text || "Not reported.").split(/\n\s*\n/).filter(Boolean);
  for (const value of paragraphs) {
    const paragraph = document.createElement("p");
    paragraph.textContent = value;
    section.append(paragraph);
  }
}

function articleSection(title, text) {
  const section = document.createElement("section");
  section.className = "article-section";
  const heading = document.createElement("h4");
  heading.textContent = title;
  section.append(heading);
  appendParagraphs(section, text);
  return section;
}

function renderArticle(report, run) {
  const article = document.createElement("article");
  article.className = "report-article";
  const notice = document.createElement("p");
  notice.className = "report-boundary";
  notice.textContent = "Standards-derived exploratory report. Tests and evidence records outrank both models; publication use requires independent human and manuscript gates.";
  article.append(notice);
  article.append(articleSection("Abstract", report.executive_summary));
  article.append(articleSection("Introduction", report.introduction || "The legacy parent report did not contain a distinct Introduction."));
  const methods = articleSection("Methods", "");
  methods.querySelector("p")?.remove();
  methods.append(makeList(report.methods || ["Methods were not separately recorded."]));
  appendDisplays(methods, run, "methods");
  article.append(methods);
  const results = articleSection("Results", report.results || report.narrative);
  appendDisplays(results, run, "results");
  article.append(results);
  const discussion = articleSection("Discussion", report.discussion || "The legacy report did not contain a distinct Discussion.");
  if (report.limitations?.length) {
    const heading = document.createElement("h5");
    heading.textContent = "Limitations";
    discussion.append(heading, makeList(report.limitations));
  }
  appendDisplays(discussion, run, "discussion");
  article.append(discussion);
  article.append(articleSection("Conclusions", report.conclusions || report.executive_summary));
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

function claimCard(claim) {
  const card = document.createElement("article");
  card.className = `claim-card ${claim.status}`;
  const header = document.createElement("header");
  const id = document.createElement("code"); id.textContent = claim.claim_id;
  const status = document.createElement("span"); status.textContent = displayStatus(claim.status);
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
  const id = document.createElement("code"); id.textContent = source.source_id;
  const copy = document.createElement("div");
  const local = (run.reference_manifest?.references || []).find((item) => item.source_id === source.source_id);
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
  for (const file of files) {
    const body = new FormData(); body.append("upload", file);
    await api(`/api/workspaces/${state.workspace.id}/files`, { method: "POST", body });
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
  const body = { objective: el("objective").value, enable_code: el("enable-code").checked, mcp_servers: mcp };
  const run = await api(`/api/workspaces/${state.workspace.id}/runs`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  state.activeRun = await api(`/api/runs/${run.id}`);
  state.selectedRun = state.activeRun;
  state.workspace = await api(`/api/workspaces/${state.workspace.id}`);
  await resetEvents(run.id);
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
    if (state.selectedRun?.id === runId) state.selectedRun = run;
    state.pollBackoff = 1500;
    state.pollErrorShown = false;
    renderWorkspaces();
    renderWorkspace();
    if (state.activeRun) schedulePoll();
    else toast(`Run finished: ${displayStatus(run.status)}.`);
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
    el("objective").focus();
    el("task-form").scrollIntoView({ behavior: "smooth", block: "start" });
    toast("Protocol loaded for review. Starting it remains a separate action.");
  });
}

async function init() {
  bindEvents();
  try { await loadConfig(); await loadWorkspaces(); }
  catch (error) { toast(error.message, "error"); }
}

document.addEventListener("DOMContentLoaded", init);
