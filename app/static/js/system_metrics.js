/**
 * System resource dashboard, for the Stats page's own "System" nav
 * entry (see stats.html/stats.js — a data-view slot inside that page,
 * same pattern as telemetry.js). Chart rendering lives in
 * system_charts.js; this file only fetches GET /api/system-metrics/summary
 * and wires the numbers/gauges into the DOM.
 */

function formatGB(bytes) {
  return (bytes / 1e9).toFixed(1);
}

function systemGaugeBlock(id, title) {
  const wrap = document.createElement("div");
  wrap.className = "flex flex-col items-center gap-2 rounded-lg bg-slate-900 border border-slate-800 p-4";
  wrap.innerHTML = `<div id="${id}"></div><p class="text-xs font-medium text-slate-300">${escapeHtml(title)}</p><p id="${id}-detail" class="text-xs text-slate-500"></p>`;
  return wrap;
}

function renderSystemGauges(summary) {
  const el = document.getElementById("system-gauges");
  el.innerHTML = "";
  el.appendChild(systemGaugeBlock("system-gauge-cpu", "CPU"));
  el.appendChild(systemGaugeBlock("system-gauge-mem", "Memory"));
  el.appendChild(systemGaugeBlock("system-gauge-disk", "Disk"));

  const memPercent = summary.mem_total_bytes ? (summary.mem_used_bytes / summary.mem_total_bytes) * 100 : 0;
  const diskPercent = summary.disk_total_bytes ? (summary.disk_used_bytes / summary.disk_total_bytes) * 100 : 0;

  renderSystemGauge("system-gauge-cpu", summary.cpu_percent, ThemeColors.chart(1));
  renderSystemGauge("system-gauge-mem", memPercent, ThemeColors.chart(2));
  renderSystemGauge("system-gauge-disk", diskPercent, ThemeColors.chart(3));

  document.getElementById("system-gauge-cpu-detail").textContent = `${summary.cpu_count} core(s)`;
  document.getElementById("system-gauge-mem-detail").textContent =
    `${formatGB(summary.mem_used_bytes)} / ${formatGB(summary.mem_total_bytes)} GB`;
  document.getElementById("system-gauge-disk-detail").textContent =
    `${formatGB(summary.disk_used_bytes)} / ${formatGB(summary.disk_total_bytes)} GB`;
}

/** Same color-per-gpu_index assignment renderGpuHistoryChart (system_charts.js) uses for its lines — kept in
 * sync by construction: both iterate gpu_index values in ascending order and cycle through GPU_LINE_COLORS by
 * position, and summary.gpus already comes back gpu_index-ordered from the API. */
function gpuColor(positionInSortedList) {
  return GPU_LINE_COLORS[positionInSortedList % GPU_LINE_COLORS.length];
}

function renderGpuGauges(gpus) {
  const gaugesEl = document.getElementById("gpu-gauges");
  const emptyEl = document.getElementById("gpu-empty");
  gaugesEl.innerHTML = "";
  if (!gpus.length) {
    emptyEl.classList.remove("hidden");
    gaugesEl.classList.add("hidden");
    return;
  }
  emptyEl.classList.add("hidden");
  gaugesEl.classList.remove("hidden");

  gpus.forEach((gpu, i) => {
    const id = `gpu-gauge-${gpu.gpu_index}`;
    gaugesEl.appendChild(systemGaugeBlock(id, `GPU ${gpu.gpu_index}: ${gpu.name}`));
    renderSystemGauge(id, gpu.utilization_percent, gpuColor(i));
    const mem = `${formatGB(gpu.mem_used_bytes)} / ${formatGB(gpu.mem_total_bytes)} GB`;
    const temp = gpu.temperature_c != null ? ` · ${Math.round(gpu.temperature_c)}°C` : "";
    document.getElementById(`${id}-detail`).textContent = mem + temp;
  });
}

/** The GPU history chart's color key — built dynamically (unlike the CPU/Memory chart's two hardcoded legend
 * entries in stats.html) since GPU count varies per machine. */
function renderGpuLegend(gpus) {
  const el = document.getElementById("gpu-history-legend");
  el.innerHTML = "";
  gpus.forEach((gpu, i) => {
    const span = document.createElement("span");
    span.className = "flex items-center gap-1.5";
    span.innerHTML = `<span class="h-2 w-2 rounded-full inline-block" style="background:${gpuColor(i)}"></span>${escapeHtml(`GPU ${gpu.gpu_index}`)}`;
    el.appendChild(span);
  });
}

function renderSystemStatTiles(summary) {
  const el = document.getElementById("system-stat-tiles");
  el.innerHTML = "";
  // statTile is defined in stats.js — reused here rather than a third near-identical copy (telemetry.js already
  // has its own, kept separate for that page's own reasons; this view lives on the same page as stats.js's
  // Overview, so reusing its generic tile helper directly is the simpler choice).
  el.appendChild(statTile(summary.cpu_count, "CPU cores"));
  el.appendChild(statTile(summary.load_avg_1m != null ? summary.load_avg_1m.toFixed(2) : "—", "Load avg (1m)"));
  el.appendChild(statTile(`${formatGB(summary.mem_total_bytes)} GB`, "Total memory"));
  el.appendChild(statTile(`${formatGB(summary.disk_total_bytes)} GB`, "Total disk"));
}

async function loadSystemMetrics() {
  const errorEl = document.getElementById("system-error");
  try {
    const summary = await api("/api/system-metrics/summary");
    renderSystemGauges(summary);
    renderSystemStatTiles(summary);
    renderSystemHistoryChart(summary.history);
    renderGpuGauges(summary.gpus);
    renderGpuLegend(summary.gpus);
    renderGpuHistoryChart(summary.gpu_history);
    // Now called on a repeating timer (see stats.js's STATS_VIEW_REFRESH) — a transient failure must not leave
    // a stale error banner shown forever once a later refresh actually succeeds.
    errorEl.classList.add("hidden");
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.classList.remove("hidden");
  }
}
