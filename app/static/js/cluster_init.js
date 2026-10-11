/** Setting the cluster up (while it runs no consensus yet): which node starts it and the address the others reach
 * it by, saved with the shared folders (saveClusterPaths in cluster.js). pAIring may have no local Matricxon, so
 * the first node is a choice among its hosts, empty until the admin picks one. Hidden once the cluster runs. */

let _clusterInitHosts = "";

function renderClusterInit(state) {
  // Also with cluster mode off while the nodes still run a cluster (after "Leave cluster"): the first node is picked here.
  const settingUp = state.supported && (state.quorum === "off" || !state.enabled);
  document.getElementById("cluster-init").classList.toggle("hidden", !settingUp);
  document.getElementById("cluster-paths-title").textContent = settingUp ? "First node and shared folders" : "Shared folders";
  const select = document.getElementById("cluster-seed-host");
  const key = state.hosts.map((h) => `${h.host}:${h.reachable}:${h.fresh}:${!!h.consensus}`).join("|") + "#" + (state.paths.first_node || "");
  if (key !== _clusterInitHosts) {  // only when something changed (a host answering now too): keep what was picked
    const picked = select.dataset.touched ? select.value : state.paths.first_node || "";
    _clusterInitHosts = key;
    const option = (value, text) => `<option value="${escapeHtml(value)}" label="${escapeHtml(text)}"></option>`;
    document.getElementById("cluster-seed-hosts").innerHTML = state.hosts
      .map((h) => option(h.host, h.host + (h.reachable ? "" : " (not answering)") + (h.consensus ? " (runs a cluster)" : h.fresh === false ? " (has data)" : "")))
      .join("");
    select.value = picked;
  }
  const advertise = document.getElementById("cluster-advertise");
  if (!advertise.dataset.touched) advertise.value = state.paths.advertise_url || "";
}

/** Once the first node is chosen and its whole checklist is green, say that the next step is the switch. */
function renderClusterFirstReady(state) {
  const first = state.paths.first_node;
  const host = state.hosts.find((h) => h.host === first);
  const ready = !state.enabled && !!host && host.reachable && host.ready;
  document.getElementById("cluster-first-ready").classList.toggle("hidden", !ready);
  document.getElementById("cluster-first-ready-host").textContent = first || "";
  return ready;
}

/** What the folders form adds while the cluster is being set up: the chosen first node and its address. */
function clusterInitChoice() {
  if (document.getElementById("cluster-init").classList.contains("hidden")) return {};
  return {
    first_node: document.getElementById("cluster-seed-host").value.trim(),
    advertise_url: document.getElementById("cluster-advertise").value.trim(),
  };
}

document.getElementById("cluster-advertise").addEventListener("input", (e) => (e.target.dataset.touched = "1"));
document.getElementById("cluster-seed-host").addEventListener("input", (e) => (e.target.dataset.touched = "1"));
