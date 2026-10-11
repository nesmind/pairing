/** What is locked while the cluster has no quorum (clusterLock.lost, set from every poll): the forms that
 * change which nodes or settings the cluster has. The server refuses these requests too. */

/** Changing the shared folders goes through the cluster, so it is locked without a quorum too. */
function _clusterLockPaths() {
  const form = document.getElementById("cluster-paths-form");
  for (const el of form.querySelectorAll("input, button")) {
    el.disabled = clusterLock.lost;
    el.title = clusterLock.lost ? QUORUM_LOCK_TIP : "";
  }
}

/** The Add node form needs a quorum (the new node becomes a voter), so it is locked without one. */
function _clusterLockAdd() {
  document.getElementById("cluster-add-quorum").classList.toggle("hidden", !clusterLock.lost);
  for (const id of ["cluster-node-address", "cluster-node-check", "cluster-node-add"]) {
    const el = document.getElementById(id);
    el.disabled = clusterLock.lost;
    el.title = clusterLock.lost ? QUORUM_LOCK_TIP : "";
  }
}
