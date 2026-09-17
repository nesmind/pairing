/**
 * Stats page controller: fetches GET /api/stats/summary and renders the
 * Overview dashboard, and drives the page-local sidebar's data-view
 * switch between "Overview", "Visual chat workflow", "Telemetry", and
 * "System" — mirrors settings.html's data-tab/activateTab() show-hide
 * convention, just four views instead of five tabs. "Visual chat
 * workflow" (the D3 diagram, workflow_diagram.js) is fixed
 * documentation, not live data — built once, the first time it's
 * actually opened, and never refetched.
 *
 * The other three are real data views (see STATS_VIEW_REFRESH below):
 * each fetches fresh data every time it's activated (not just the
 * first time — switching back to a tab you already visited shouldn't
 * show whatever was current as of the last refresh tick before you
 * left) and then keeps auto-refreshing on its own timer for as long as
 * it stays the active view. Only ever one timer running at a time
 * (switching views clears the previous one), and it's paused via the
 * visibilitychange listener whenever this tab isn't visible, so an
 * idle background tab doesn't keep polling the server for nothing.
 */

const STATS_VIEWS = ["overview", "workflow", "telemetry", "system"];

// Refresh interval per view, matching each one's own natural data cadence: System mirrors its own poller's 15s
// sample interval (see app.services.system_metrics_poller), Telemetry/Overview use a slower 30s — new Ollama
// calls/messages/conversations don't need sub-15s freshness the way live resource usage does.
const STATS_VIEW_REFRESH = {
  overview: { fn: loadStatsOverviewSummary, intervalMs: 30000 },
  telemetry: { fn: loadOllamaTelemetry, intervalMs: 30000 },
  system: { fn: loadSystemMetrics, intervalMs: 15000 },
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
    // A tab that was hidden for a while has stale data the moment it's looked at again — refresh immediately
    // rather than waiting out however much of the interval was left when it was backgrounded.
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
  // The D3 diagram is fixed documentation, expensive to (re)build, and never goes stale — built once, the first
  // time this view is actually opened, unlike the three data views below.
  if (name === "workflow" && !window.workflowDiagramInitialized) {
    window.workflowDiagramInitialized = true;
    initWorkflowDiagram();
  }
  // Every other view fetches fresh data on *every* activation (not just the first) — switching back to a tab
  // you already visited shouldn't show whatever was current as of the last refresh tick before you left.
  STATS_VIEW_REFRESH[name]?.fn();
  _startStatsAutoRefresh(name);
}

document.querySelectorAll(".stats-nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => activateStatsView(btn.dataset.view));
});

/** One small "N — label" tile — the building block every stat-tile grid
 * below is made of. */
function statTile(value, label) {
  const div = document.createElement("div");
  div.className = "rounded-lg bg-slate-900 border border-slate-800 px-4 py-3";
  div.innerHTML = `<p class="text-2xl font-semibold text-slate-100">${value}</p><p class="text-xs text-slate-500 mt-0.5">${escapeHtml(label)}</p>`;
  return div;
}

function capitalizeFirst(value) {
  return value.length ? value.charAt(0).toUpperCase() + value.slice(1) : value;
}

/** Display-only relabeling for the Messages section's "By role" breakdown — the API's role values (user/
 * assistant/system) mirror Ollama's own chat roles directly (see app/models/conversation.py), so "assistant"
 * reads as "AI (model)" and the rest ("user"/"system") just get their first letter capitalized, without
 * changing the underlying data or any other breakdown that reuses this same CountByKey shape (e.g. Users' own
 * "By role" list, which never has an "assistant" value anyway). */
const MESSAGE_ROLE_DISPLAY_LABELS = { assistant: "AI (model)" };

function relabelMessageRoleEntries(entries) {
  return entries.map((entry) => ({
    ...entry,
    key: MESSAGE_ROLE_DISPLAY_LABELS[entry.key] || capitalizeFirst(entry.key),
  }));
}

/** Same capitalize-only treatment for the Messages section's "By status" breakdown (complete/deleted/error/
 * streaming — see Message.status in app/models/conversation.py). */
function capitalizeEntries(entries) {
  return entries.map((entry) => ({ ...entry, key: capitalizeFirst(entry.key) }));
}

/** A labeled "key — count" breakdown list (model usage, message role/
 * status, user role) — every summary section's by_X field uses this
 * same CountByKey shape (see app/schemas/stats.py), so one renderer
 * covers all of them. */
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
    // "Live" is only what's actually observable server-side — a reply
    // mid-generation, or a message updated in the last 5 minutes (which
    // also covers a streaming reply's own in-progress flushes). This app
    // has no presence/heartbeat tracking, so per-viewer things like
    // "scrolling" aren't measurable and aren't represented here.
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
    // Now called on a repeating timer (see STATS_VIEW_REFRESH) — a transient failure must not leave a stale
    // error banner shown forever once a later refresh actually succeeds.
    errorEl.classList.add("hidden");
  } catch (err) {
    // err.message is already a readable string here — api() itself runs
    // formatErrorDetail on the server's raw `detail` before throwing.
    errorEl.textContent = err.message;
    errorEl.classList.remove("hidden");
  }
}

// Overview is the default view — no click ever fires to load it or start its refresh timer the way the other
// three views do, so this does both explicitly, once, on page load.
activateStatsView("overview");
