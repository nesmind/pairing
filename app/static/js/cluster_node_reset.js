/** "Reset to fresh" on the Add nodes form: for a reachable node that is not fresh, typically one removed from
 * this cluster earlier (it still points at the shared folders and keeps its consensus journal). The node clears
 * only its own cluster setup and history; the shared folders are never touched. Uses the helpers of cluster.js. */

async function resetClusterNode() {
  const address = _clusterNodeAddress();
  if (!address) return;
  const ok = confirm(
    `Reset ${address} to a fresh install?\n\nIt forgets its cluster setup and history (its folders go back to their defaults) and restarts. ` +
      "Its own saved chat caches (only a speed-up, rebuilt on use) are deleted too. The shared models and caches are not touched."
  );
  if (!ok) return;
  _clusterNodeNote("Resetting…");
  try {
    await api("/api/cluster/nodes/reset", { method: "POST", body: JSON.stringify({ address }) });
    document.getElementById("cluster-node-reset").classList.add("hidden");
    _clusterNodeNote("Reset: the node is restarting. Press Check again in a few seconds, then Add node.");
  } catch (err) {
    _clusterNodeNote(err.message, true);
  }
}

document.getElementById("cluster-node-reset").addEventListener("click", resetClusterNode);
