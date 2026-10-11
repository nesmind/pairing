/**
 * The right-hand panel of the Cluster page's Nodes map: one node's state, its details, its checklist and
 * the actions that make sense for it (activate it in the cluster, make it a learner, remove it from the cluster). Opened by cluster_map.js; markup in app/templates/_cluster_node_panel.html. It re-renders
 * whenever the map refreshes, so what it shows is live.
 */

const CLUSTER_STATUS_STYLE = {
  up: { label: "Running", dot: "bg-emerald-500", text: "text-emerald-400" },
  warn: { label: "Needs attention", dot: "bg-amber-500", text: "text-amber-400" },
  down: { label: "Not responding", dot: "bg-red-500", text: "text-red-400" },
};

class ClusterNodePanel {
  constructor() {
    this.panel = document.getElementById("cluster-node-panel");
    this.backdrop = document.getElementById("cluster-node-backdrop");
    this.body = document.getElementById("cluster-node-body");
    this.hostList = null; // {nodeId, peers}: the open "Debug cluster host list", drawn again with each record
    this.message = null; // what the last action said: shown under the buttons, kept until the next action
    this.currentId = null;
    this.onChanged = () => {};
    document.getElementById("cluster-node-close").addEventListener("click", () => this.close());
    this.backdrop.addEventListener("click", () => this.close());
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && this.currentId) this.close();
    });
    this.body.addEventListener("click", (event) => this._onAction(event));
  }

  get isOpen() {
    return this.currentId !== null;
  }

  open(node) {
    this.currentId = node.id;
    this.hostList = null;
    this._say("");
    this.render(node);
    this.backdrop.classList.remove("hidden");
    this.panel.classList.remove("translate-x-full");
  }

  close() {
    this.currentId = null;
    this.backdrop.classList.add("hidden");
    this.panel.classList.add("translate-x-full");
  }

  /** Fresh full record of the open node (the map polls compact, so the panel asks for its own). */
  refresh(node) {
    if (this.isOpen && node.id === this.currentId) {
      this.render(node); // (the message under the buttons stays: it is drawn again with the record)
    }
  }

  /** The node's record could not be fetched (it may have left the cluster). */
  fail(message) {
    // A message that says what just happened ("Removed from the cluster.") is not replaced by "gone".
    if (this.isOpen && !(this.message && !this.message.bad)) this._say(message, true);
  }

  render(node) {
    this._lastNode = node;
    const style = CLUSTER_STATUS_STYLE[node.status] || CLUSTER_STATUS_STYLE.down;
    document.getElementById("cluster-node-title").textContent = node.name;
    document.getElementById("cluster-node-subtitle").textContent = node.subtitle;
    const badges = node.badges
      .map((b) => `<span class="rounded-md border border-slate-700 bg-slate-800/60 px-2 py-0.5 text-xs uppercase tracking-wide text-slate-300">${escapeHtml(b)}</span>`)
      .join(" ");
    const details = Object.entries(node.details)
      .map(([key, value]) => `<div class="flex justify-between gap-4 px-3 py-2"><dt class="text-xs text-slate-500">${escapeHtml(key)}</dt><dd class="min-w-0 whitespace-pre-line break-all text-right text-xs text-slate-200">${escapeHtml(String(value))}</dd></div>`)
      .join("");
    const checks = node.checks.length
      ? `<div><p class="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Checklist</p>
         <ul class="divide-y divide-slate-800 rounded-lg border border-slate-800 bg-slate-950">${node.checks.map(_clusterCheckRow).join("")}</ul></div>`
      : "";
    this.body.innerHTML = `
      <div class="flex items-center gap-2"><span class="h-2.5 w-2.5 rounded-full ${style.dot}"></span>
        <span class="text-sm font-medium ${style.text}">${style.label}</span></div>
      ${badges ? `<div class="flex flex-wrap gap-1.5">${badges}</div>` : ""}
      <dl class="divide-y divide-slate-800 rounded-lg border border-slate-800 bg-slate-950">${details}</dl>
      ${checks}
      <div class="flex flex-wrap gap-2 pt-1">${this._actions(node)}</div>
      <p data-field="message" class="hidden rounded-lg border px-3 py-2 text-xs"></p>
      ${this._debugButton(node)}`;
    this._paintMessage();
    this._paintHostList();
  }

  /** The same buttons on every Matricxon node: one for its role (Activate / Make it a voter / Make it a learner)
   * and Remove. One that does not apply is greyed out, and the reason is written under the buttons. */
  _actions(node) {
    if (node.kind === "voter") return this._ghostActions(node);
    if (node.kind !== "matricxon") return "";
    const notes = node.action_notes || {};
    const primary = "rounded-lg bg-brand-500 px-3 py-1.5 text-sm font-medium text-white hover:opacity-90";
    const plain = "rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-200 hover:bg-slate-800";
    const voter = node.badges.includes("voter");
    const learner = node.badges.includes("learner");
    const outside = node.badges.includes("not in cluster");  // listed by pAIring, but the cluster does not have it
    const role = voter
      ? { action: "demote", label: "Make it a learner", ok: node.demotable, why: notes.demote, css: plain }
      : { action: "activate", label: outside ? "Add to cluster" : learner ? "Make it a voter" : "Activate in cluster", ok: node.activatable, why: notes.activate, css: primary };
    const remove = { action: "remove", label: "Remove from cluster", ok: node.removable, why: notes.remove, css: `${plain} hover:text-red-400` };
    // A voter that moved (it advertises another address than the cluster has for it): Activate records it.
    const moved = node.address_mismatch
      ? [{ action: "activate", label: "Update its address", ok: true, why: "", css: primary }]
      : [];
    const buttons = [role, ...moved, remove].map((b) => {
      const off = !b.ok || clusterLock.lost;
      const why = clusterLock.lost ? QUORUM_LOCK_TIP : b.why || "";
      return { ...b, off, why, html: `<button type="button" data-action="${b.action}" data-value="${escapeHtml(node.address || "")}"${off ? ` disabled title="${escapeHtml(why)}"` : ""} class="${b.css}${off ? " opacity-50 cursor-not-allowed" : ""}">${b.label}</button>` };
    });
    const reasons = buttons.filter((b) => b.off && b.why).map((b) => `<li>${b.label}: ${escapeHtml(b.why)}</li>`).join("");
    return buttons.map((b) => b.html).join("") + (reasons ? `<ul class="basis-full space-y-0.5 pt-1 text-xs text-slate-500">${reasons}</ul>` : "");
  }

  /** A voter that is not one of pAIring's nodes (its node left the host list while it still voted): the only
   * thing to do is take it out of the voters, which needs a quorum. */
  _ghostActions(node) {
    const off = !node.removable || clusterLock.lost;
    const why = clusterLock.lost ? QUORUM_LOCK_TIP : node.removable ? "" : "it is the cluster's only voter";
    const id = escapeHtml((node.details || {})["Voter id"] || "");
    return `<button type="button" data-action="dropvoter" data-value="${id}"${off ? ` disabled title="${escapeHtml(why)}"` : ""} class="rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-200 hover:bg-slate-800 hover:text-red-400${off ? " opacity-50 cursor-not-allowed" : ""}">Remove from voters</button>${off && why ? `<p class="basis-full pt-1 text-xs text-slate-500">${escapeHtml(why)}</p>` : ""}`;
  }

  async _onAction(event) {
    const button = event.target.closest("button[data-action]");
    if (!button) return;
    const { action, value } = button.dataset;
    try {
      if (action === "hosts") {
        await this._toggleHostList(value);
      } else if (action === "activate") {
        button.disabled = true;
        if (this._lastNode.badges.includes("not in cluster")) {  // not a member yet: join it, as Add nodes does
          try {
            await api("/api/cluster/nodes", { method: "POST", body: JSON.stringify({ address: value }) });
          } catch (err) {
            const notFresh = err.status === 409 && !/quorum/i.test(err.message);
            throw new Error(notFresh ? `${err.message}. Use Reset to fresh on the Add nodes page.` : err.message);
          }
          this._say("Added. The node is restarting.");
          this.onChanged();
          return;
        }
        const state = await api("/api/cluster/activate", { method: "POST" });
        this._say(state.message || "Done.");  // the message names nodes that are still catching up
        this.onChanged();
      } else if (action === "demote") {
        if (!confirm(`Make ${value} a learner?\n\nIt stays in the cluster, keeps serving chats and following the log, but no longer votes. The remaining voters must be reachable: a change of the voters needs a majority.`)) return;
        await api("/api/cluster/nodes/demote", { method: "POST", body: JSON.stringify({ address: value }) });
        this._say("It no longer votes.");
        this.onChanged();
      } else if (action === "dropvoter") {
        if (!confirm("Take this voter out of the cluster's voters?\n\nIts node is not in pAIring's host list. The other voters keep running; this needs a majority of them to be reachable.")) return;
        await api("/api/cluster/voters", { method: "DELETE", body: JSON.stringify({ id: value }) });
        this._say("Taken out of the voters.");
        this.onChanged();
      } else if (action === "remove") {
        if (!confirm(`Remove ${value} from the cluster? It receives no chats and is reset to a fresh install (it restarts; the shared models and caches are not touched), so it can be added again.\n\nIf this is the first node (the one this app starts and stops), this app stops managing it and uses the remaining nodes as remote hosts.`)) return;
        await api("/api/cluster/nodes", { method: "DELETE", body: JSON.stringify({ address: value }) });
        this._say("Removed from the cluster and reset (restarting).");
        this.onChanged();
      }
    } catch (err) {
      this._say(err.message, true);
    }
  }

  _say(text, bad = false) {
    this.message = text ? { text, bad } : null;
    this._paintMessage();
  }

  /** The long address book is behind a button: it can be big (the Cluster host names page takes the really long ones). */
  _debugButton(node) {
    if (node.kind !== "matricxon" || !(node.badges.includes("voter") || node.badges.includes("learner"))) return "";
    const open = this.hostList && this.hostList.nodeId === node.id;
    return `<div><button type="button" data-action="hosts" data-value="${escapeHtml(node.address)}" class="text-xs text-slate-400 underline underline-offset-2 hover:text-slate-200">${open ? "Hide cluster host list" : "Debug cluster host list"}</button>
      <div data-field="hostlist" class="mt-2"></div></div>`;
  }

  async _toggleHostList(address) {
    if (this.hostList) {
      this.hostList = null;
      this.render(this._lastNode);
      return;
    }
    const list = await fetchClusterPeers(address, { limit: CLUSTER_HOSTS_INLINE_MAX });
    if (list.total > CLUSTER_HOSTS_INLINE_MAX) {  // too long for this panel: the Cluster host names page shows it in pages
      this.close();
      openClusterHostPage(address);
      return;
    }
    this.hostList = { nodeId: this.currentId, peers: list.peers };
    this.render(this._lastNode);
  }

  _paintHostList() {
    const el = this.body.querySelector('[data-field="hostlist"]');
    if (!el || !this.hostList || this.hostList.nodeId !== this.currentId) return;
    el.innerHTML = `<table class="w-full text-left text-xs"><tbody class="divide-y divide-slate-800 font-mono text-slate-300">${clusterPeerRows(this.hostList.peers)}</tbody></table>`;
  }

  /** The message sits right under the action buttons (the body is redrawn on every refresh, so this is
   * called from render too). */
  _paintMessage() {
    const el = this.body.querySelector('[data-field="message"]');
    if (!el) return;
    el.textContent = this.message ? this.message.text : "";
    el.classList.toggle("hidden", !this.message);
    const bad = Boolean(this.message && this.message.bad);
    el.classList.toggle("border-red-800/60", bad);
    el.classList.toggle("bg-red-900/20", bad);
    el.classList.toggle("text-red-200", bad);
    el.classList.toggle("border-slate-700", !bad);
    el.classList.toggle("bg-slate-950", !bad);
    el.classList.toggle("text-slate-300", !bad);
  }
}

const clusterNodePanel = new ClusterNodePanel();
