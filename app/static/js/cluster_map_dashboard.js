/**
 * The summary above the dot map for very large clusters (over 50 servers), above the dot map: totals, a status bar, the leader, and a short list of the servers that need attention (or match
 * the search box). A click on a row opens that server's panel.
 */

class ClusterDashboard {
  static LIST_LIMIT = 100;

  constructor(el, controller) {
    this.el = el;
    this.controller = controller;
    this.byId = new Map();
    el.addEventListener("click", (event) => {
      const row = event.target.closest("[data-node-id]");
      if (row && this.byId.has(row.dataset.nodeId)) this.controller.openNode(this.byId.get(row.dataset.nodeId));
    });
  }

  render(map) {
    const servers = map.nodes.filter((n) => n.kind === "matricxon" || n.kind === "voter");
    const count = (status) => servers.filter((n) => n.status === status).length;
    const [up, warn, down] = [count("up"), count("warn"), count("down")];
    const total = Math.max(1, servers.length);
    const voters = servers.filter((n) => n.badges.includes("voter")).length;
    const leader = servers.find((n) => n.badges.includes("leader"));
    const tile = (value, label, color) =>
      `<div class="rounded-lg border border-slate-800 bg-slate-900 px-4 py-2"><p class="text-xl font-semibold ${color}">${value}</p><p class="text-xs text-slate-500">${label}</p></div>`;
    const pct = (n) => `${((n / total) * 100).toFixed(2)}%`;
    const query = this.controller.hasFilter();
    const shown = servers
      .filter((n) => (query ? this.controller.matches(n) : n.status !== "up"))
      .sort((a, b) => (a.status === b.status ? a.subtitle.localeCompare(b.subtitle) : a.status === "down" ? -1 : 1));
    this.byId = new Map(shown.slice(0, ClusterDashboard.LIST_LIMIT).map((n) => [n.id, n]));
    const rows = [...this.byId.values()]
      .map((n) => `<li data-node-id="${escapeHtml(n.id)}" class="flex cursor-pointer items-center gap-2 px-3 py-1.5 text-xs hover:bg-slate-800">
        <span class="h-2 w-2 shrink-0 rounded-full ${n.badges.includes("not in cluster") ? "bg-slate-500" : n.status === "up" ? "bg-emerald-500" : n.status === "warn" ? "bg-amber-500" : "bg-red-500"}"></span>
        <span class="text-slate-200">${escapeHtml(n.name)}</span><span class="text-slate-500">${escapeHtml(n.subtitle)}</span>
        <span class="ml-auto text-slate-500">${escapeHtml(n.badges.join(" · "))}</span></li>`)
      .join("");
    const more = shown.length > ClusterDashboard.LIST_LIMIT ? `<p class="px-3 py-1.5 text-xs text-slate-500">…and ${shown.length - ClusterDashboard.LIST_LIMIT} more: refine the search.</p>` : "";
    this.el.innerHTML = `
      <div class="grid grid-cols-2 gap-3 sm:grid-cols-6">
        ${tile(servers.length, "servers", "text-slate-100")}${tile(up, "running", "text-emerald-400")}
        ${tile(warn, "need attention", "text-amber-400")}${tile(down, "not responding", "text-red-400")}
        ${tile(voters, "voters", "text-sky-400")}${tile(leader ? escapeHtml(leader.subtitle) : "none", "leader", "text-slate-100 text-base")}
      </div>
      <div class="mt-3 flex h-2 overflow-hidden rounded-full bg-slate-800">
        <div style="width:${pct(up)};background:${DOT_COLORS.up}"></div><div style="width:${pct(warn)};background:${DOT_COLORS.warn}"></div>
        <div style="width:${pct(down)};background:${DOT_COLORS.down}"></div></div>
      <p class="mt-3 text-xs font-semibold uppercase tracking-wide text-slate-500">${query ? "Matching servers" : "Servers that need attention"} (${shown.length})</p>
      <ul class="mt-1 max-h-40 divide-y divide-slate-800 overflow-y-auto rounded-lg border border-slate-800 bg-slate-950">
        ${rows || '<li class="px-3 py-2 text-xs text-slate-500">None.</li>'}</ul>${more}`;
  }
}
