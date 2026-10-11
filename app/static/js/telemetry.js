/*
 * Per-engine (Ollama/Matricxon) telemetry for the Stats "Telemetry" entry (a data-view slot in stats.html/stats.js, so
 * no nav switching here; stats.js's activateStatsView calls loadTelemetry on each activation). Separate from stats.js
 * so the two dashboards' fetch/render logic stay independent. Charts: telemetry_charts.js/telemetry_bar_chart.js.
 * 
 * One shared dashboard, not a panel per engine: both engines' summaries have the same shape (app/schemas/telemetry.py),
 * so switching #telemetry-source-select re-fetches into the same elements.
 */

// The engine whose history is shown; "ollama" only until the first load resolves the active engine and corrects it
// (loadTelemetry).
let currentTelemetrySource = "ollama";
// Set once, on the first activation (like stats.js's window.workflowDiagramInitialized): later refreshes must not
// override the engine an admin since picked; only the first "what should this page open on" decision.
let telemetryDefaultEngineApplied = false;

document.getElementById("telemetry-source-select").addEventListener("change", (e) => {
  currentTelemetrySource = e.target.value;
  loadTelemetry();
});

function telemetryStatTile(value, label) {
  const div = document.createElement("div");
  div.className = "rounded-lg bg-slate-900 border border-slate-800 px-4 py-3";
  div.innerHTML = `<p class="text-2xl font-semibold text-slate-100">${value}</p><p class="text-xs text-slate-500 mt-0.5">${escapeHtml(label)}</p>`;
  return div;
}

function telemetryRow(left, right) {
  const row = document.createElement("div");
  row.className = "flex items-center justify-between px-4 py-2 text-sm";
  row.innerHTML = `<span class="text-slate-300 truncate">${left}</span><span class="font-mono text-xs text-slate-400 shrink-0 ml-2">${right}</span>`;
  return row;
}

function renderSummaryTiles(summary) {
  const el = document.getElementById("telemetry-summary-tiles");
  el.innerHTML = "";
  const errorRate = summary.total_requests ? Math.round((summary.error_count / summary.total_requests) * 100) : 0;
  el.appendChild(telemetryStatTile(summary.total_requests, "Total requests"));
  el.appendChild(telemetryStatTile(summary.error_count, "Errors"));
  el.appendChild(telemetryStatTile(`${errorRate}%`, "Error rate"));
  el.appendChild(telemetryStatTile(summary.loaded_models.length, "Models loaded now"));
}

function renderByModel(summary) {
  const el = document.getElementById("telemetry-by-model");
  el.innerHTML = '<p class="px-4 py-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Requests by model</p>';
  if (!summary.by_model.length) {
    el.innerHTML += '<p class="px-4 py-2 text-xs text-slate-500">No requests yet.</p>';
  }
  for (const entry of summary.by_model) {
    el.appendChild(telemetryRow(escapeHtml(entry.key), entry.count));
  }
}

function renderLatencyByModel(summary) {
  const el = document.getElementById("telemetry-latency-by-model");
  el.innerHTML = '<p class="px-4 py-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Latency by model</p>';
  if (!summary.latency_by_model.length) {
    el.innerHTML += '<p class="px-4 py-2 text-xs text-slate-500">No completed requests yet.</p>';
  }
  for (const stat of summary.latency_by_model) {
    const label = `avg ${Math.round(stat.avg_duration_ms)}ms · p95 ${Math.round(stat.p95_duration_ms)}ms · ${stat.request_count}x`;
    el.appendChild(telemetryRow(escapeHtml(stat.model), label));
  }
}

/* "Currently loaded" is a recent /api/ps snapshot (app/services/ollama_ps_poller.py/matricxon_ps_poller.py for the selected engine), not a live push. expires_at is a plain formatted time, not a countdown, to avoid implying more precision than a ~30s poll. */
function renderLoadedModels(summary) {
  const el = document.getElementById("telemetry-loaded-models");
  el.innerHTML = '';
  if (!summary.loaded_models.length) {
    el.innerHTML = '<p class="px-4 py-2 text-xs text-slate-500">Nothing currently loaded.</p>';
    return;
  }
  let totalRam = 0;
  let totalVram = 0;
  for (const m of summary.loaded_models) {
    // size_bytes is the model's whole footprint, size_vram_bytes the GPU part; the rest is RAM. A CPU-only engine
    // (Matricxon) reports 0 VRAM, so RAM is what matters there. != null checks, not truthiness: 0 is a real value,
    // distinct from "never reported".
    const hasSize = m.size_bytes != null;
    const vramBytes = m.size_vram_bytes || 0;
    const ramBytes = hasSize ? Math.max(m.size_bytes - vramBytes, 0) : 0;
    totalRam += ramBytes;
    totalVram += vramBytes;
    const mem = hasSize ? `${formatGb(ramBytes)} RAM · ${formatGb(vramBytes)} VRAM` : "size unknown";
    const expires = m.expires_at ? `expires ${new Date(m.expires_at).toLocaleTimeString()}` : "no expiry set";
    const left = `${escapeHtml(m.model_name)} <span class="text-slate-500">on ${escapeHtml(new URL(m.host).host)}</span>`;
    el.appendChild(telemetryRow(left, `${mem} · ${expires}`));
  }
  el.appendChild(telemetryRow("Total loaded", `${formatGb(totalRam)} RAM · ${formatGb(totalVram)} VRAM`));
}

function formatGb(bytes) {
  return `${(bytes / 1e9).toFixed(1)} GB`;
}

function renderTelemetry(summary) {
  renderSummaryTiles(summary);
  renderByModel(summary);
  renderLatencyByModel(summary);
  renderLoadedModels(summary);
  renderLatencyLineChart(summary.latency_time_series);
  renderHostBarChart(summary.by_host);
}

/* Resolves the active engine (the /health endpoint settings.js's Model tab uses) and, the first time this view opens, points the dropdown at it, rather than "Ollama" regardless of what's running. Also fills the "you're currently on: X" note, which stays accurate when an admin switches the dropdown to the other engine's history: it always shows what serves live traffic. */
// Each engine's display name (EngineOption in app/schemas/common.py), filled once by populateTelemetryEngineOptionsOnce
// and read by applyDefaultTelemetryEngineOnce's "Currently on: X" note.
let telemetryEngineDisplayNames = {};

/* Fills #telemetry-source-select's <option>s from GET /api/settings/engine/options (every engine in app.services.engines.registry) instead of two hardcoded options, as settings.js's Active-engine picker. Runs once; the registry doesn't change at runtime. */
async function populateTelemetryEngineOptionsOnce() {
  const selectEl = document.getElementById("telemetry-source-select");
  if (selectEl.options.length > 0) return;
  const { engines } = await api("/api/settings/engine/options");
  for (const { name, display_name: displayName, capabilities } of engines) {
    telemetryEngineDisplayNames[name] = displayName;
    // Telemetry exists only for local engines (Ollama/Matricxon); a remote/serverless connector engine (e.g. RunPod)
    // has nothing to instrument, so it is never offered.
    if (!capabilities.local_process) continue;
    const option = document.createElement("option");
    option.value = name;
    option.textContent = displayName;
    selectEl.appendChild(option);
  }
}

async function applyDefaultTelemetryEngineOnce() {
  const selectEl = document.getElementById("telemetry-source-select");
  await populateTelemetryEngineOptionsOnce();
  let activeEngine;
  try {
    ({ active_engine: activeEngine } = await api("/health"));
  } catch (_err) {
    // Keep the current fallback selection: a /health hiccup shouldn't block the dashboard.
    selectEl.classList.remove("invisible");
    return;
  }
  const noteEl = document.getElementById("telemetry-active-engine-note");
  noteEl.textContent = `Currently on: ${telemetryEngineDisplayNames[activeEngine] || activeEngine}`;
  if (!telemetryDefaultEngineApplied) {
    telemetryDefaultEngineApplied = true;
    // A non-local active engine has no option here: keep the first local engine selected.
    if ([...selectEl.options].some((o) => o.value === activeEngine)) {
      currentTelemetrySource = activeEngine;
      selectEl.value = activeEngine;
    } else {
      currentTelemetrySource = selectEl.value;
    }
  }
  // Revealed only once it shows the real active engine (see stats.html's comment on its `invisible` class).
  selectEl.classList.remove("invisible");
}

async function loadTelemetry() {
  await applyDefaultTelemetryEngineOnce();
  const errorEl = document.getElementById("telemetry-error");
  try {
    const summary = await api(`/api/telemetry/${currentTelemetrySource}/summary`);
    renderTelemetry(summary);
    // Called on a repeating timer (stats.js STATS_VIEW_REFRESH): a transient failure must not leave a stale error
    // banner after a later refresh succeeds.
    errorEl.classList.add("hidden");
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.classList.remove("hidden");
  }
}
