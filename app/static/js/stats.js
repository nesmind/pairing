/*
 * Stats page controller: fetches GET /api/stats/summary for the Overview dashboard and drives the sidebar's data-view
 * switch between "Overview", "Visual chat workflow", "Telemetry" and "System" (like settings.html's
 * data-tab/activateTab). The D3 workflow diagram (workflow_diagram.js) is fixed documentation: built once on first
 * open, never refetched.
 * 
 * The other three views fetch fresh data on every activation (not just the first, so a revisit doesn't show data from
 * before you left) and auto-refresh on a timer (STATS_VIEW_REFRESH) while active. Only one timer runs at a time
 * (switching clears it), and the visibilitychange listener pauses it in a background tab.
 */

const STATS_VIEWS = ["overview", "workflow", "telemetry", "system", "architectures"];

// Refresh interval per view, matching its data cadence: System mirrors its poller's 15s sample interval
// (app.services.system_metrics_poller); Telemetry/Overview use 30s, as new calls/messages/conversations don't need
// sub-15s freshness.
const STATS_VIEW_REFRESH = {
  overview: { fn: loadStatsOverviewSummary, intervalMs: 30000 },
  telemetry: { fn: loadTelemetry, intervalMs: 30000 },
  system: { fn: loadSystemMetrics, intervalMs: 15000 },
  // Changes only when Matricxon gains support for something; a slow poll is plenty.
  architectures: { fn: loadEngineSupport, intervalMs: 60000 },
};

let _activeStatsView = "overview";
let _statsRefreshTimer = null;

function _stopStatsAutoRefresh() {
  if (_statsRefreshTimer !== null) {
    clearInterval(_statsRefreshTimer);
    _statsRefreshTimer = null;
  }
}

function _startStatsAutoRefresh(name) {
  _stopStatsAutoRefresh();
  const config = STATS_VIEW_REFRESH[name];
  if (!config || document.hidden) return;
  _statsRefreshTimer = setInterval(config.fn, config.intervalMs);
}

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    _stopStatsAutoRefresh();
  } else {
    // A tab hidden for a while has stale data when looked at again: refresh at once rather than wait out the rest of
    // the interval.
    STATS_VIEW_REFRESH[_activeStatsView]?.fn();
    _startStatsAutoRefresh(_activeStatsView);
  }
});

function activateStatsView(name) {
  _activeStatsView = name;
  document.querySelectorAll(".stats-nav-btn").forEach((btn) => {
    const active = btn.dataset.view === name;
    btn.classList.toggle("bg-slate-800", active);
    btn.classList.toggle("text-slate-100", active);
    btn.classList.toggle("text-slate-300", !active);
    btn.classList.toggle("hover:bg-slate-800", !active);
  });
  for (const view of STATS_VIEWS) {
    document.getElementById(`stats-view-${view}`).classList.toggle("hidden", view !== name);
  }
  // The D3 diagram is fixed documentation, costly to build and never stale: built once, on first open, unlike the data
  // views below.
  if (name === "workflow" && !window.workflowDiagramInitialized) {
    window.workflowDiagramInitialized = true;
    initWorkflowDiagram();
  }
  // Every other view fetches fresh data on every activation, so a revisit doesn't show data from before you left.
  STATS_VIEW_REFRESH[name]?.fn();
  _startStatsAutoRefresh(name);
}

document.querySelectorAll(".stats-nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => activateStatsView(btn.dataset.view));
});

/* One small "N — label" tile, the building block of the stat-tile grids below. */
function statTile(value, label) {
  const div = document.createElement("div");
  div.className = "rounded-lg bg-slate-900 border border-slate-800 px-4 py-3";
  div.innerHTML = `<p class="text-2xl font-semibold text-slate-100">${value}</p><p class="text-xs text-slate-500 mt-0.5">${escapeHtml(label)}</p>`;
  return div;
}

function capitalizeFirst(value) {
  return value.length ? value.charAt(0).toUpperCase() + value.slice(1) : value;
}

/* Display-only relabeling for the Messages "By role" breakdown: the API's roles (user/assistant/system) mirror Ollama's chat roles (app/models/conversation.py), so "assistant" reads "AI (model)" and the rest are capitalized. The data and other breakdowns sharing the CountByKey shape (e.g. Users' "By role", which has no "assistant") are unchanged. */
const MESSAGE_ROLE_DISPLAY_LABELS = { assistant: "AI (model)" };

function relabelMessageRoleEntries(entries) {
  return entries.map((entry) => ({
    ...entry,
    key: MESSAGE_ROLE_DISPLAY_LABELS[entry.key] || capitalizeFirst(entry.key),
  }));
}

/* Capitalize-only treatment for the Messages "By status" breakdown (complete/deleted/error/streaming; Message.status in app/models/conversation.py). */
function capitalizeEntries(entries) {
  return entries.map((entry) => ({ ...entry, key: capitalizeFirst(entry.key) }));
}

/* A labeled "key — count" breakdown list (model usage, message role/status, user role); every summary by_X field uses the CountByKey shape (app/schemas/stats.py), so one renderer covers all. */
function breakdownList(title, entries) {
  const wrap = document.createElement("div");
  wrap.className = "rounded-lg bg-slate-900 border border-slate-800 divide-y divide-slate-800";
  const heading = document.createElement("p");
  heading.className = "px-4 py-2 text-xs font-semibold uppercase tracking-wide text-slate-400";
  heading.textContent = title;
  wrap.appendChild(heading);
  if (entries.length === 0) {
    const empty = document.createElement("p");
    empty.className = "px-4 py-2 text-xs text-slate-500";
    empty.textContent = "Nothing yet.";
    wrap.appendChild(empty);
  }
  for (const entry of entries) {
    const row = document.createElement("div");
    row.className = "flex items-center justify-between px-4 py-2 text-sm";
    row.innerHTML = `<span class="text-slate-300 truncate">${escapeHtml(entry.key)}</span><span class="font-mono text-slate-400">${entry.count}</span>`;
    wrap.appendChild(row);
  }
  return wrap;
}

function statSection(title, tiles, breakdowns) {
  const section = document.createElement("section");
  section.className = "space-y-3";
  const heading = document.createElement("h3");
  heading.className = "text-sm font-semibold text-slate-200";
  heading.textContent = title;
  section.appendChild(heading);

  if (tiles.length) {
    const grid = document.createElement("div");
    grid.className = "grid grid-cols-2 sm:grid-cols-4 gap-3";
    for (const [value, label] of tiles) grid.appendChild(statTile(value, label));
    section.appendChild(grid);
  }
  if (breakdowns.length) {
    const grid = document.createElement("div");
    grid.className = "grid grid-cols-1 sm:grid-cols-2 gap-3";
    for (const [breakdownTitle, entries] of breakdowns) grid.appendChild(breakdownList(breakdownTitle, entries));
    section.appendChild(grid);
  }
  return section;
}

function renderOverview(summary) {
  const contentEl = document.getElementById("stats-overview-content");
  contentEl.innerHTML = "";

  contentEl.appendChild(
    statSection(
      "Conversations",
      [
        [summary.conversations.total, "Total"],
        [summary.conversations.personal, "Personal"],
        [summary.conversations.channel, "Channel"],
      ],
      [["By model", summary.conversations.by_model]]
    )
  );
  contentEl.appendChild(
    // "Live" is what the server can observe: a reply mid-generation, or a message updated in the last 5 minutes
    // (covering a streaming reply's flushes). The app has no presence tracking, so per-viewer things like "scrolling"
    // aren't measurable.
    statSection(
      "Live conversations",
      [
        [summary.live_conversations.total, "Total"],
        [summary.live_conversations.streaming_now, "Waiting for reply"],
        [summary.live_conversations.recently_active, "Recently active"],
      ],
      []
    )
  );
  contentEl.appendChild(
    statSection(
      "Messages",
      [[summary.messages.total, "Total"]],
      [
        ["By role", relabelMessageRoleEntries(summary.messages.by_role)],
        ["By status", capitalizeEntries(summary.messages.by_status)],
      ]
    )
  );
  contentEl.appendChild(
    statSection(
      "Users",
      [
        [summary.users.total, "Total"],
        [summary.users.active, "Active"],
        [summary.users.disabled, "Disabled"],
      ],
      [["By role", summary.users.by_role]]
    )
  );
  contentEl.appendChild(
    statSection(
      "Knowledge base (RAG data)",
      [
        [summary.knowledge_base.total_documents, "Documents"],
        [summary.knowledge_base.total_chunks, "Chunks"],
        [summary.knowledge_base.global_documents, "Shared"],
        [summary.knowledge_base.private_documents, "Private"],
      ],
      []
    )
  );
}

async function loadStatsOverviewSummary() {
  const errorEl = document.getElementById("stats-overview-error");
  try {
    const summary = await api("/api/stats/summary");
    renderOverview(summary);
    // Called on a repeating timer (STATS_VIEW_REFRESH): a transient failure must not leave a stale error banner after a
    // later refresh succeeds.
    errorEl.classList.add("hidden");
  } catch (err) {
    // err.message is already readable: api runs formatErrorDetail on the server's `detail` before throwing.
    errorEl.textContent = err.message;
    errorEl.classList.remove("hidden");
  }
}

// Overview is the default view, so no click loads it or starts its timer; do both here, once, on page load.
activateStatsView("overview");
