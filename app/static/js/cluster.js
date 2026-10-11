/**
 * Cluster page. Three views behind the page-local nav (same convention as stats.js: data-view buttons plus
 * #cluster-view-<name> panels): "overview" (cluster mode on/off, shared folders, and the seed's checklist, from
 * GET/PUT /api/cluster, see app/services/cluster_service.py), "nodes" (the live map, cluster_map.js) and "add"
 * (the form that adds a fresh Matricxon).
 */

const CLUSTER_VIEWS = ["overview", "nodes", "add", "hosts"];
const CLUSTER_REFRESH_MS = 15000;
let _clusterTimer = null;

function activateClusterView(name) {
  document.querySelectorAll(".cluster-nav-btn").forEach((btn) => {
    const active = btn.dataset.view === name;
    btn.classList.toggle("bg-slate-800", active);
    btn.classList.toggle("text-slate-100", active);
    btn.classList.toggle("text-slate-300", !active);
  });
  for (const view of CLUSTER_VIEWS) {
    document.getElementById(`cluster-view-${view}`).classList.toggle("hidden", view !== name);
  }
  clusterMap.setActive(name === "nodes");  // the map only polls while it is on screen
}

function _clusterShowError(message) {
  const el = document.getElementById("cluster-error");
  el.textContent = message || "";
  el.classList.toggle("hidden", !message);
}

function _clusterSetSwitch(enabled, busy) {
  const button = document.getElementById("cluster-switch");
  button.setAttribute("aria-checked", String(enabled));
  button.disabled = busy;
  button.classList.toggle("bg-brand-500", enabled);
  button.classList.toggle("bg-slate-700", !enabled);
  document.getElementById("cluster-switch-knob").style.transform = enabled ? "translateX(20px)" : "";
  document.getElementById("cluster-switch-note").textContent = enabled
    ? "On — the Matricxon nodes work together on shared storage."
    : "Off — Matricxon runs standalone.";
}

function _clusterCheckRow(check) {
  const ok = check.status === "ok";
  const hint = check.hint ? `<p class="mt-0.5 text-xs text-slate-400">${escapeHtml(check.hint)}</p>` : "";
  const setting = !ok && check.setting ? `<p class="text-xs text-slate-500">Setting: <code>${escapeHtml(check.setting)}</code></p>` : "";
  return `<li class="flex gap-3 px-4 py-2.5">
    <span class="mt-0.5 h-2.5 w-2.5 shrink-0 rounded-full ${ok ? "bg-emerald-500" : "bg-amber-500"}"></span>
    <div class="min-w-0"><p class="text-sm text-slate-200">${escapeHtml(check.what)}</p>
    <p class="text-xs text-slate-500">${escapeHtml(check.why)}</p>${hint}${setting}</div></li>`;
}

function _clusterHostCard(host, enabled) {
  // Same title style as the pAIring card: the section name, then grey details beside it.
  const title = (detail) => `<p class="text-sm font-medium text-slate-100">Matricxon
    <span class="ml-2 text-xs font-normal text-slate-500">${detail}</span></p>`;
  if (!host.reachable) {
    return `<div class="rounded-lg border border-slate-800 bg-slate-900 px-4 py-3 max-w-2xl">
      ${title(escapeHtml(host.host))}
      <p class="mt-1 text-xs text-red-400">Unreachable: ${escapeHtml(host.error || "")}</p></div>`;
  }
  const state = host.ready
    ? host.active ? "active in the cluster" : enabled ? "ready — the cluster is starting" : "ready — turn cluster mode on"
    : "action needed";
  return `<div class="rounded-lg border border-slate-800 bg-slate-900 max-w-2xl">
    <div class="px-4 py-3 border-b border-slate-800">
      ${title(`${escapeHtml(host.host)} · role: ${escapeHtml(host.role)} · ${state}`)}
      <p class="text-xs text-slate-500">${host.nodes.length} node(s) announced in the shared folder</p>
    </div>
    <ul class="divide-y divide-slate-800">${host.checks.map(_clusterCheckRow).join("")}</ul></div>`;
}

let _clusterPathsDirty = false;

function _clusterFillPaths(paths) {
  // Never overwrite what the admin is typing; the 15 s refresh only fills untouched fields.
  if (_clusterPathsDirty) return;
  document.getElementById("cluster-models-path").value = paths.models_path || "";
  document.getElementById("cluster-cache-path").value = paths.prompt_cache_path || "";
}

function renderCluster(state) {
  clusterLock.lost = state.quorum === "lost";
  const canAdd = state.supported && state.enabled;
  document.getElementById("cluster-node-form").classList.toggle("hidden", !canAdd);
  document.getElementById("cluster-add-off").classList.toggle("hidden", canAdd);
  _clusterFillPaths(state.paths);
  renderClusterInit(state);
  renderClusterFirstReady(state);
  renderClusterJoin(state);
  renderClusterReset(state);
  renderClusterTimers(state);
  renderClusterCache(state);
  _clusterLockPaths();
  document.getElementById("cluster-paths-form").classList.toggle("hidden", !state.supported);
  _clusterSetSwitch(state.enabled, !state.supported);
  document.getElementById("cluster-unsupported").classList.toggle("hidden", state.supported);
  document.getElementById("cluster-restart-note").classList.toggle("hidden", !state.restart_needed);
  document.getElementById("cluster-quorum").classList.toggle("hidden", state.quorum !== "lost");
  _clusterLockAdd();
  document.getElementById("cluster-secret-state").textContent = state.admin_auth ? "on: admin calls are signed" : "not set: nodes are unprotected";
  const message = document.getElementById("cluster-message");
  message.textContent = state.message || "";
  message.classList.toggle("hidden", !state.message);
  // This view is about the seed; the other nodes live on the Nodes map (and are added under Add nodes).
  const seeds = state.hosts.filter((host) => !(state.node_hosts || []).includes(host.host));
  // The card and its checklist are for first setup. Once the cluster runs consensus the nodes have agreed, and
  // each node's own panel on the Nodes map shows the same checklist.
  const settingUp = state.quorum === "off";
  document.getElementById("cluster-setup-hint").classList.toggle("hidden", !settingUp);
  // Setting up shows only the first node, once it was chosen and saved with the folders: no card before that.
  const first = state.paths.first_node;
  const shown = seeds.filter((host) => host.host === first);
  document.getElementById("cluster-hosts").innerHTML = settingUp ? shown.map((host) => _clusterHostCard(host, state.enabled)).join("") : "";
}

async function loadCluster() {
  try {
    renderCluster(await api("/api/cluster"));
    _clusterShowError("");
  } catch (err) {
    _clusterShowError(err.message);
  }
}

async function toggleCluster() {
  const button = document.getElementById("cluster-switch");
  const enabled = button.getAttribute("aria-checked") !== "true";
  const refused = document.getElementById("cluster-switch-error");
  refused.classList.add("hidden");
  _clusterSetSwitch(enabled, true);
  try {
    const body = { enabled };
    renderCluster(await api("/api/cluster", { method: "PUT", body: JSON.stringify(body) }));
    _clusterShowError("");
  } catch (err) {
    // Next to the switch, and kept: the reload below puts the switch back but must not wipe the reason.
    refused.textContent = err.message;
    refused.classList.remove("hidden");
    await loadCluster();
  }
}

async function saveClusterPaths(event) {
  event.preventDefault();
  const note = document.getElementById("cluster-paths-note");
  const body = {
    models_path: document.getElementById("cluster-models-path").value,
    prompt_cache_path: document.getElementById("cluster-cache-path").value,
    ...clusterInitChoice(),
  };
  if ("first_node" in body && !body.first_node) {
    note.textContent = "";
    _clusterShowError("Choose the first node.");
    return;
  }
  try {
    _clusterPathsDirty = false;
    for (const id of ["cluster-seed-host", "cluster-advertise"]) delete document.getElementById(id).dataset.touched;
    renderCluster(await api("/api/cluster/paths", { method: "PUT", body: JSON.stringify(body) }));
    note.textContent = "Saved.";
    _clusterShowError("");
    if (body.first_node) {  // all green: the switch at the top is next; else the card with the checklist
      const card = document.getElementById("cluster-hosts");
      const top = !document.getElementById("cluster-first-ready").classList.contains("hidden");
      if (top) requestAnimationFrame(() => document.getElementById("cluster-view-overview").scrollTo({ top: 0, behavior: "smooth" }));
      else if (card.children.length) requestAnimationFrame(() => card.scrollIntoView({ behavior: "smooth", block: "start" }));
    }
  } catch (err) {
    _clusterShowError(err.message);
  }
}

function _clusterNodeAddress() {
  return document.getElementById("cluster-node-address").value.trim();
}

function _clusterNodeNote(text, bad) {
  const el = document.getElementById("cluster-node-note");
  el.textContent = text;
  el.classList.toggle("text-red-400", !!bad);
  el.classList.toggle("text-slate-500", !bad);
}

async function checkClusterNode() {
  const address = _clusterNodeAddress();
  if (!address) return;
  _clusterNodeNote("Checking…");
  try {
    const result = await api("/api/cluster/nodes/check", { method: "POST", body: JSON.stringify({ address }) });
    // A reachable node can be reset whatever its state, but not while it is still a node of this cluster.
    const resettable = result.reachable && !result.in_cluster;  // a hard reset works in any state
    document.getElementById("cluster-node-reset").classList.toggle("hidden", !resettable);
    if (result.in_cluster) _clusterNodeNote(`${address} is already a node of this cluster: remove it first (Nodes map), then add it again.`, true);
    else if (!result.reachable) _clusterNodeNote(`Unreachable: ${result.error}`, true);
    else if (!result.fresh) _clusterNodeNote(`Not a fresh install: ${result.reasons.join("; ")}`, true);
    else _clusterNodeNote("Fresh Matricxon — it can join.");
  } catch (err) {
    _clusterNodeNote(err.message, true);
  }
}

async function addClusterNode(event) {
  event.preventDefault();
  const address = _clusterNodeAddress();
  if (!address) return;
  const button = document.getElementById("cluster-node-add");
  button.disabled = true;
  _clusterNodeNote("Configuring the node…");
  try {
    const advertise_url = document.getElementById("cluster-node-advertise").value.trim() || null;
    renderCluster(await api("/api/cluster/nodes", { method: "POST", body: JSON.stringify({ address, advertise_url, node_kind: document.getElementById("cluster-node-kind").value }) }));
    document.getElementById("cluster-node-address").value = "";
    document.getElementById("cluster-node-advertise").value = "";
    document.getElementById("cluster-node-kind").value = "compute";
    document.getElementById("cluster-node-reset").classList.add("hidden");
    _clusterNodeNote("Added. The node is restarting.");
  } catch (err) {
    _clusterNodeNote(err.message, true);
    // Refused as not fresh (409, not the quorum lock): it can be reset right here, as after Check.
    const notFresh = err.status === 409 && !/quorum/i.test(err.message);
    document.getElementById("cluster-node-reset").classList.toggle("hidden", !notFresh);
  } finally {
    button.disabled = false;
  }
}

async function saveClusterSecret(event) {
  event.preventDefault();
  const input = document.getElementById("cluster-secret");
  const note = document.getElementById("cluster-secret-note");
  note.textContent = "Updating every node…";
  try {
    renderCluster(await api("/api/cluster/admin-secret", { method: "PUT", body: JSON.stringify({ secret: input.value }) }));
    input.value = "";
    note.textContent = "Saved on every node.";
  } catch (err) {
    note.textContent = err.message;
  }
}

function _clusterStartRefresh() {
  clearInterval(_clusterTimer);
  _clusterTimer = document.hidden ? null : setInterval(loadCluster, CLUSTER_REFRESH_MS);
}

document.querySelectorAll(".cluster-nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => activateClusterView(btn.dataset.view));
});
document.getElementById("cluster-switch").addEventListener("click", toggleCluster);
document.getElementById("cluster-paths-form").addEventListener("submit", saveClusterPaths);
document.getElementById("cluster-paths-form").addEventListener("input", () => {
  _clusterPathsDirty = true;
  document.getElementById("cluster-paths-note").textContent = "";
});
document.getElementById("cluster-node-form").addEventListener("submit", addClusterNode);
document.getElementById("cluster-secret-form").addEventListener("submit", saveClusterSecret);
document.getElementById("cluster-node-check").addEventListener("click", checkClusterNode);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) loadCluster();
  _clusterStartRefresh();
});

activateClusterView("overview");
loadCluster();
_clusterStartRefresh();

// The switch's refusal stays until the user does something else on the page.
document.addEventListener("click", (event) => {
  if (event.target.closest("#cluster-switch")) return;
  document.getElementById("cluster-switch-error")?.classList.add("hidden");
});
document.addEventListener("input", () => {
  document.getElementById("cluster-switch-error")?.classList.add("hidden");
});
