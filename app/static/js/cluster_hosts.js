/** Cluster host names: one node's address book, from GET /api/cluster/peers. Two uses: the "Debug cluster host list" button
 * of the node panel shows a short list inline (cluster_node_panel.js), and this page shows any list, paged and
 * searchable, for the ones that are too long for the panel. */

const CLUSTER_HOSTS_INLINE_MAX = 100; // longer than this opens the Cluster host names page instead of the panel
const CLUSTER_HOSTS_PAGE = 200;
const _clusterHostsState = { node: "", query: "", offset: 0, total: 0 };

async function fetchClusterPeers(node, { query = "", offset = 0, limit = CLUSTER_HOSTS_PAGE } = {}) {
  const params = new URLSearchParams({ id: node, q: query, offset, limit });
  return api(`/api/cluster/peers?${params}`);
}

/** The role column: this node, voter / learner (a node of the cluster) or just a remembered address, and how many
 * older identities sit at the same address. */
function clusterPeerRole(peer) {
  const kind = peer.kind === "old" ? "remembered address" : peer.kind;
  const older = peer.old_ids ? `${peer.old_ids} older id${peer.old_ids > 1 ? "s" : ""} on that IP/host` : "";
  return [peer.this_node ? "this node" : "", kind, peer.seed ? "start-up seed" : "", older].filter(Boolean).join(" · ");
}

function clusterPeerRows(peers) {
  return peers
    .map((p) => `<tr><td class="px-3 py-1.5" title="${escapeHtml(p.id)}">${escapeHtml(p.id.slice(0, 8))}</td>` +
      `<td class="px-3 py-1.5 break-all">${escapeHtml(p.address)}</td><td class="px-3 py-1.5 text-slate-400">${escapeHtml(clusterPeerRole(p))}</td></tr>`)
    .join("");
}

/** Opens the Cluster host names page on `node` (used by the panel for lists that are too long to show there). */
function openClusterHostPage(node) {
  activateClusterView("hosts");
  document.getElementById("cluster-hosts-node").value = node;
  document.getElementById("cluster-hosts-search").value = "";
  loadClusterHostPage(node, "", 0);
}

async function loadClusterHostPage(node, query, offset) {
  const summary = document.getElementById("cluster-hosts-summary");
  if (!node) {
    summary.textContent = "Pick a node first.";
    return;
  }
  summary.textContent = "Loading…";
  try {
    const list = await fetchClusterPeers(node, { query, offset });
    Object.assign(_clusterHostsState, { node, query, offset: list.offset, total: list.total });
    document.getElementById("cluster-hosts-rows").innerHTML = clusterPeerRows(list.peers);
    const from = list.total ? list.offset + 1 : 0;
    const advertises = list.advertise_url ? ` · it advertises ${list.advertise_url}` : "";
    summary.textContent = list.total
      ? `${from}-${list.offset + list.peers.length} of ${list.total} hosts known to ${node}${advertises}`
      : `${node} knows no hosts (it may not run consensus)`;
    document.getElementById("cluster-hosts-prev").disabled = list.offset <= 0;
    document.getElementById("cluster-hosts-next").disabled = list.offset + list.peers.length >= list.total;
  } catch (err) {
    summary.textContent = err.message;
    document.getElementById("cluster-hosts-rows").innerHTML = "";
  }
}

function _clusterHostsFillNodes() {
  const nodes = (clusterMap.map ? clusterMap.map.nodes : []).filter((n) => n.kind === "matricxon");
  document.getElementById("cluster-hosts-nodes").innerHTML = nodes.map((n) => `<option value="${escapeHtml(n.id)}"></option>`).join("");
}

const _hostsInput = (id) => document.getElementById(id);
_hostsInput("cluster-hosts-load").addEventListener("click", () =>
  loadClusterHostPage(_hostsInput("cluster-hosts-node").value.trim(), _hostsInput("cluster-hosts-search").value, 0));
_hostsInput("cluster-hosts-search").addEventListener("keydown", (event) => {
  if (event.key === "Enter") _hostsInput("cluster-hosts-load").click();
});
_hostsInput("cluster-hosts-node").addEventListener("focus", _clusterHostsFillNodes);
_hostsInput("cluster-hosts-prev").addEventListener("click", () =>
  loadClusterHostPage(_clusterHostsState.node, _clusterHostsState.query, Math.max(0, _clusterHostsState.offset - CLUSTER_HOSTS_PAGE)));
_hostsInput("cluster-hosts-next").addEventListener("click", () =>
  loadClusterHostPage(_clusterHostsState.node, _clusterHostsState.query, _clusterHostsState.offset + CLUSTER_HOSTS_PAGE));
