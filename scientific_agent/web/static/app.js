"use strict";

const state = {
  config: null,
  workspaces: [],
  workspace: null,
  run: null,
  tab: "claims",
  pollTimer: null,
};

const el = (id) => document.getElementById(id);
const activeStates = new Set(["queued", "running"]);
const supportedStates = new Set(["supported", "supported_with_comments"]);
const warningStates = new Set(["inconclusive", "requires_more_evidence", "requires_human_decision", "contradicted"]);
const phaseOrder = ["planning", "research", "validation", "scientific-review", "repair", "finalizing"];
const workflowTemplates = {
  analyze: "Analyze the uploaded dataset end to end. First inspect schema, missingness, duplicates, units, and plausible outliers. Lock the primary analysis before inspecting its result; report effect sizes with uncertainty and sensitivity analyses; save machine-readable tables and a publication-ready figure. Do not claim causality. Every numerical claim must cite an exact computation artifact.",
  crosscheck: "Independently analyze the uploaded dataset in both Python and R using the same prespecified estimand and statistical method. Save machine-readable results from each language, reconcile numerical differences against an explicit tolerance, run assumption and sensitivity checks, and report effect sizes with uncertainty. Every numerical claim must cite an exact computation artifact.",
  audit: "Audit the scientific claim described below using retrieved primary or authoritative sources. Separate observed evidence, computation, inference, and unresolved uncertainty. Verify identifiers and supporting passages, look for corrections or contradictory evidence, and do not include any citation that was not retrieved by a configured tool. Claim to audit: ",
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

function formatBytes(bytes) {
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

function statusClass(status) {
  if (supportedStates.has(status)) return "supported";
  if (warningStates.has(status)) return "warning";
  if (["failed", "interrupted"].includes(status)) return "failed";
  if (activeStates.has(status)) return "running";
  return "";
}

async function loadConfig() {
  state.config = await api("/api/config");
  el("app-version").textContent = `v${state.config.version}`;
  el("executor-model").textContent = state.config.models.executor.split("/").pop();
  el("critic-model").textContent = state.config.models.critic.split("/").pop();
  document.querySelectorAll("#mcp-options input").forEach((input) => {
    input.disabled = !state.config.mcp[input.value];
    input.title = input.disabled ? "Not configured by the service owner" : "";
  });
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
    state.workspace = null;
    state.run = null;
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
  state.workspace = await api(`/api/workspaces/${workspaceId}`);
  state.run = null;
  const active = state.workspace.runs.find((run) => activeStates.has(run.status));
  const latest = active || state.workspace.runs[0];
  if (latest) state.run = await api(`/api/runs/${latest.id}`);
  renderWorkspaces();
  renderWorkspace();
  if (closeSidebar) toggleSidebar(false);
  if (state.run && activeStates.has(state.run.status)) schedulePoll();
}

function renderWorkspace() {
  const present = Boolean(state.workspace);
  el("empty-state").classList.toggle("hidden", present);
  el("workspace-view").classList.toggle("hidden", !present);
  if (!present) return;
  el("workspace-title").textContent = state.workspace.name;
  el("workspace-meta").textContent = `Created ${formatDate(state.workspace.created_at)} · ${state.workspace.files.length} input${state.workspace.files.length === 1 ? "" : "s"} · ${state.workspace.runs.length} run${state.workspace.runs.length === 1 ? "" : "s"}`;
  renderFiles();
  renderHistory();
  renderRun();
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
  const locked = state.workspace.runs.some((run) => activeStates.has(run.status));
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
    button.className = `run-item ${state.run?.id === run.id ? "active" : ""}`;
    const dot = document.createElement("i");
    dot.className = statusClass(run.status);
    const copy = document.createElement("span");
    const objective = document.createElement("strong");
    objective.textContent = run.objective;
    const date = document.createElement("small");
    date.textContent = `${run.status.replaceAll("_", " ")} · ${formatDate(run.created_at)}`;
    copy.append(objective, date);
    const id = document.createElement("code");
    id.textContent = run.id.slice(0, 4);
    button.append(dot, copy, id);
    button.addEventListener("click", () => selectRun(run.id));
    history.append(button);
  }
}

async function selectRun(runId) {
  clearPoll();
  state.run = await api(`/api/runs/${runId}`);
  renderHistory();
  renderRun();
  if (activeStates.has(state.run.status)) schedulePoll();
}

function renderRun() {
  renderRail();
  const running = state.run && activeStates.has(state.run.status);
  el("run-button").disabled = running;
  el("file-input").disabled = running;
  if (!state.run || !state.run.report) {
    el("result-empty").classList.remove("hidden");
    el("result-content").classList.add("hidden");
    el("result-status").textContent = state.run ? state.run.status.replaceAll("_", " ").toUpperCase() : "WAITING";
    el("result-status").className = `panel-state ${state.run ? statusClass(state.run.status) : ""}`;
    return;
  }
  el("result-empty").classList.add("hidden");
  el("result-content").classList.remove("hidden");
  el("result-status").textContent = state.run.status.replaceAll("_", " ").toUpperCase();
  el("result-status").className = `panel-state ${statusClass(state.run.status)}`;
  el("report-title").textContent = state.run.report.title;
  el("report-summary").textContent = state.run.report.executive_summary;
  el("report-download").href = artifactUrl("report.md");
  el("bundle-download").href = `/api/runs/${state.run.id}/bundle`;
  renderTab();
}

function renderRail() {
  const message = state.run ? state.run.message : "No run selected.";
  el("run-message").textContent = message;
  const current = state.run?.phase || "";
  let activeIndex = phaseOrder.indexOf(current);
  if (current === "plan-review") activeIndex = 0;
  if (current === "complete") activeIndex = phaseOrder.length;
  if (current === "failed" || current === "stopped") activeIndex = -1;
  document.querySelectorAll("#provenance-rail li").forEach((item, index) => {
    item.className = "";
    const marker = item.querySelector(":scope > span");
    if (!state.run) { marker.textContent = "—"; return; }
    if (index < activeIndex || activeIndex === phaseOrder.length) {
      item.classList.add("done"); marker.textContent = "OK";
    } else if (index === activeIndex && activeStates.has(state.run.status)) {
      item.classList.add("active"); marker.textContent = "NOW";
    } else if (!activeStates.has(state.run.status) && index > Math.max(activeIndex, 0)) {
      item.classList.add("skipped"); marker.textContent = "—";
    } else { marker.textContent = "—"; }
  });
}

function artifactUrl(path) {
  return `/api/runs/${state.run.id}/artifacts?path=${encodeURIComponent(path)}`;
}

function renderTab() {
  document.querySelectorAll(".report-tabs button").forEach((button) => button.classList.toggle("active", button.dataset.tab === state.tab));
  const content = el("tab-content");
  content.replaceChildren();
  const report = state.run.report;
  if (state.tab === "claims") {
    for (const claim of report.claims) {
      const card = document.createElement("article");
      card.className = `claim-card ${claim.status}`;
      const header = document.createElement("header");
      const id = document.createElement("code"); id.textContent = claim.claim_id;
      const status = document.createElement("span"); status.textContent = claim.status.replaceAll("_", " ");
      header.append(id, status);
      const text = document.createElement("p"); text.textContent = claim.text;
      const refs = document.createElement("small"); refs.textContent = `Evidence: ${claim.evidence_refs.join(", ") || "none"}`;
      card.append(header, text, refs);
      content.append(card);
    }
  } else if (state.tab === "methods") {
    content.append(makeList(report.methods));
  } else if (state.tab === "sources") {
    for (const source of report.sources) content.append(sourceRow(source));
  } else if (state.tab === "artifacts") {
    for (const artifact of state.run.artifacts) content.append(artifactRow(artifact));
  } else {
    const items = [...report.unresolved_issues.map((x) => `Unresolved: ${x}`), ...report.limitations.map((x) => `Limitation: ${x}`)];
    content.append(makeList(items.length ? items : ["No additional limitations were recorded."]));
  }
}

function makeList(items) {
  const list = document.createElement("ul"); list.className = "plain-list";
  for (const value of items) { const item = document.createElement("li"); item.textContent = value; list.append(item); }
  return list;
}

function sourceRow(source) {
  const row = document.createElement("div"); row.className = "source-row";
  const id = document.createElement("code"); id.textContent = source.source_id;
  const copy = document.createElement("div");
  const link = document.createElement(source.url ? "a" : "span"); link.textContent = source.title;
  if (source.url) { link.href = source.url; link.target = "_blank"; link.rel = "noreferrer"; }
  const note = document.createElement("small"); note.textContent = source.supporting_passage;
  copy.append(link, note);
  const type = document.createElement("span"); type.textContent = source.source_type.replaceAll("_", " ");
  row.append(id, copy, type); return row;
}

function artifactRow(artifact) {
  const row = document.createElement("div"); row.className = "artifact-row";
  const id = document.createElement("code"); id.textContent = artifact.path.split("/").pop().split(".").pop().toUpperCase();
  const copy = document.createElement("div");
  const link = document.createElement("a"); link.textContent = artifact.path; link.href = artifactUrl(artifact.path);
  const note = document.createElement("small"); note.textContent = `sha256 ${artifact.sha256.slice(0, 14)}…`;
  copy.append(link, note);
  const size = document.createElement("span"); size.textContent = formatBytes(artifact.bytes);
  row.append(id, copy, size); return row;
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
  if (!state.workspace || !window.confirm(`Delete “${state.workspace.name}” and all of its run artifacts?`)) return;
  await api(`/api/workspaces/${state.workspace.id}`, { method: "DELETE" });
  state.workspace = null; state.run = null;
  await loadWorkspaces();
  toast("Workspace deleted.");
}

async function uploadFiles(files) {
  if (!state.workspace) return;
  for (const file of files) {
    const body = new FormData(); body.append("upload", file);
    await api(`/api/workspaces/${state.workspace.id}/files`, { method: "POST", body });
  }
  await selectWorkspace(state.workspace.id, false);
  toast(`${files.length} file${files.length === 1 ? "" : "s"} added.`);
}

async function deleteFile(name) {
  if (!window.confirm(`Delete input file “${name}”?`)) return;
  await api(`/api/workspaces/${state.workspace.id}/files/${encodeURIComponent(name)}`, { method: "DELETE" });
  await selectWorkspace(state.workspace.id, false);
}

async function startRun(event) {
  event.preventDefault();
  const mcp = [...document.querySelectorAll("#mcp-options input:checked")].map((input) => input.value);
  const body = {
    objective: el("objective").value,
    enable_code: el("enable-code").checked,
    mcp_servers: mcp,
  };
  state.run = await api(`/api/workspaces/${state.workspace.id}/runs`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  state.workspace = await api(`/api/workspaces/${state.workspace.id}`);
  renderWorkspace(); schedulePoll(); toast("Audited run queued.");
}

function schedulePoll() {
  clearPoll();
  state.pollTimer = window.setTimeout(pollRun, 1500);
}

async function pollRun() {
  if (!state.run) return;
  try {
    state.run = await api(`/api/runs/${state.run.id}`);
    const currentWorkspace = state.workspace.id;
    state.workspace = await api(`/api/workspaces/${currentWorkspace}`);
    renderWorkspace();
    if (activeStates.has(state.run.status)) schedulePoll();
    else toast(`Run finished: ${state.run.status.replaceAll("_", " ")}.`);
  } catch (error) { toast(error.message, "error"); schedulePoll(); }
}

function clearPoll() { if (state.pollTimer) window.clearTimeout(state.pollTimer); state.pollTimer = null; }
function toggleSidebar(open) { const sidebar = el("sidebar"); sidebar.classList.toggle("open", open); el("sidebar-toggle").setAttribute("aria-expanded", String(open)); }

function bindEvents() {
  el("new-workspace-button").addEventListener("click", () => { el("new-workspace-form").classList.toggle("hidden"); if (!el("new-workspace-form").classList.contains("hidden")) el("workspace-name").focus(); });
  el("empty-create-button").addEventListener("click", () => { toggleSidebar(true); el("new-workspace-form").classList.remove("hidden"); el("workspace-name").focus(); });
  el("new-workspace-form").addEventListener("submit", async (event) => { event.preventDefault(); try { await createWorkspace(el("workspace-name").value); } catch (error) { toast(error.message, "error"); } });
  el("delete-workspace-button").addEventListener("click", () => deleteWorkspace().catch((error) => toast(error.message, "error")));
  el("task-form").addEventListener("submit", (event) => startRun(event).catch((error) => toast(error.message, "error")));
  el("file-input").addEventListener("change", (event) => uploadFiles([...event.target.files]).catch((error) => toast(error.message, "error")));
  el("sidebar-toggle").addEventListener("click", () => toggleSidebar(!el("sidebar").classList.contains("open")));
  const drop = el("drop-zone");
  ["dragenter", "dragover"].forEach((name) => drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((name) => drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.remove("dragging"); }));
  drop.addEventListener("drop", (event) => uploadFiles([...event.dataTransfer.files]).catch((error) => toast(error.message, "error")));
  document.querySelectorAll(".report-tabs button").forEach((button) => button.addEventListener("click", () => { state.tab = button.dataset.tab; renderTab(); }));
  document.querySelectorAll(".workflow-starters button").forEach((button) => button.addEventListener("click", () => {
    el("objective").value = workflowTemplates[button.dataset.template];
    if (button.dataset.template === "audit") {
      document.querySelectorAll("#mcp-options input:not(:disabled)").forEach((input) => { input.checked = true; });
    }
    el("objective").focus();
  }));
  el("reuse-run").addEventListener("click", () => {
    if (!state.run) return;
    el("objective").value = state.run.objective;
    el("enable-code").checked = state.run.enable_code;
    document.querySelectorAll("#mcp-options input").forEach((input) => {
      input.checked = state.run.mcp_servers.includes(input.value) && !input.disabled;
    });
    el("objective").focus();
    el("task-form").scrollIntoView({ behavior: "smooth", block: "start" });
    toast("Protocol loaded for review. Starting it remains a separate action.");
  });
}

async function init() {
  bindEvents();
  try { await Promise.all([loadConfig(), loadWorkspaces()]); }
  catch (error) { toast(error.message, "error"); }
}

document.addEventListener("DOMContentLoaded", init);
