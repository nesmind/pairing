/**
 * Per-engine (Ollama/Matricxon) telemetry rendering, for the Stats page's own "Telemetry" nav entry (see
 * stats.html/stats.js — this is a data-view slot inside that page, not a standalone one, so this file has no
 * top-level nav-switching logic of its own; stats.js's activateStatsView calls loadTelemetry() every time that
 * entry is activated). Kept as its own file (not folded into stats.js) so the two dashboards' fetch/render logic
 * don't depend on each other's internals. Chart rendering itself lives in telemetry_charts.js/
 * telemetry_bar_chart.js.
 *
 * One shared dashboard (not a separate #telemetry-source-view-<engine> panel per engine, unlike an early version
 * of this file) — both engines' summaries are the exact same shape (see app/schemas/telemetry.py), so switching
 * #telemetry-source-select just re-fetches and re-renders into the same elements rather than toggling between
 * two parallel copies of the same markup.
 */

// Which engine's history is currently shown — defaults to "ollama" only until the first real load below
// resolves the actually-active engine and corrects it (see loadTelemetry).
let currentTelemetrySource = "ollama";
// Set once, the first time this view is ever activated (mirrors stats.js's own window.workflowDiagramInitialized
// one-time-init idiom for the workflow diagram) — a *later* refresh/re-visit must never override whatever engine
// an admin has since picked from the dropdown, only the very first "what should this page open on" decision.
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

/** "Currently loaded" is a recent /api/ps snapshot (see app/services/ollama_ps_poller.py/matricxon_ps_poller.py,
 * whichever matches the currently selected engine), not a live push — expires_at is shown as a plain formatted
 * time, not a ticking countdown, to avoid implying more real-time precision than a ~30s poll interval has. */
function renderLoadedModels(summary) {
  const el = document.getElementById("telemetry-loaded-models");
  el.innerHTML = '';
  if (!summary.loaded_models.length) {
    el.innerHTML = '<p class="px-4 py-2 text-xs text-slate-500">Nothing currently loaded.</p>';
    return;
  }
  for (const m of summary.loaded_models) {
    // != null (not a plain truthy check) — 0 is a real, meaningful value here (no VRAM used, e.g. a CPU-only
    // model), not the same as Ollama never having reported a size at all.
    const vram = m.size_vram_bytes != null ? `${(m.size_vram_bytes / 1e9).toFixed(1)} GB VRAM` : "size unknown";
    const expires = m.expires_at ? `expires ${new Date(m.expires_at).toLocaleTimeString()}` : "no expiry set";
    const left = `${escapeHtml(m.model_name)} <span class="text-slate-500">on ${escapeHtml(new URL(m.host).host)}</span>`;
    el.appendChild(telemetryRow(left, `${vram} · ${expires}`));
  }
}

function renderTelemetry(summary) {
  renderSummaryTiles(summary);
  renderByModel(summary);
  renderLatencyByModel(summary);
  renderLoadedModels(summary);
  renderLatencyLineChart(summary.latency_time_series);
  renderHostBarChart(summary.by_host);
}

/** Resolves the currently-active engine (same /health endpoint settings.js's Model tab already reads its own
 * active-engine badge from) and, the very first time this view opens, points the dropdown at it — "Show the
 * right engine telemetry when entering this page," not just "Ollama" every time regardless of what's actually
 * running. Also fills in the small "you're currently on: X" note next to the selector, which stays accurate
 * even after an admin switches the dropdown to look at the *other* engine's history — that note always reflects
 * what's really serving live traffic, independent of which engine this dashboard happens to be showing. */
async function applyDefaultTelemetryEngineOnce() {
  const selectEl = document.getElementById("telemetry-source-select");
  let activeEngine;
  try {
    ({ active_engine: activeEngine } = await api("/health"));
  } catch (_err) {
    // Leave the "ollama" fallback in place — a /health hiccup here shouldn't block the dashboard itself.
    selectEl.classList.remove("invisible");
    return;
  }
  const noteEl = document.getElementById("telemetry-active-engine-note");
  noteEl.textContent = `Currently on: ${activeEngine === "matricxon" ? "Matricxon" : "Ollama"}`;
  if (!telemetryDefaultEngineApplied) {
    telemetryDefaultEngineApplied = true;
    currentTelemetrySource = activeEngine;
    selectEl.value = activeEngine;
  }
  // Only revealed once it shows the real active engine — see stats.html's own comment on its `invisible` class.
  selectEl.classList.remove("invisible");
}

async function loadTelemetry() {
  await applyDefaultTelemetryEngineOnce();
  const errorEl = document.getElementById("telemetry-error");
  try {
    const summary = await api(`/api/telemetry/${currentTelemetrySource}/summary`);
    renderTelemetry(summary);
    // Now called on a repeating timer (see stats.js's STATS_VIEW_REFRESH) — a transient failure must not leave
    // a stale error banner shown forever once a later refresh actually succeeds.
    errorEl.classList.add("hidden");
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.classList.remove("hidden");
  }
}
