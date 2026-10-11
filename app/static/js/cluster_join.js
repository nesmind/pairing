/** The cluster id (made by the cluster itself, shown with a copy button) and the form that joins this pAIring to
 * an existing cluster by the address of one of its nodes and that id. Uses the helpers of cluster.js. */

function renderClusterJoin(state) {
  const box = document.getElementById("cluster-id-box");
  box.classList.toggle("hidden", !state.cluster_id);
  document.getElementById("cluster-id").textContent = state.cluster_id || "";
  document.getElementById("cluster-join").classList.toggle("hidden", !(state.supported && !state.enabled));
}

async function joinCluster() {
  const note = document.getElementById("cluster-join-note");
  const address = document.getElementById("cluster-join-address").value.trim();
  const clusterId = document.getElementById("cluster-join-id").value.trim();
  if (!address || !clusterId) {
    note.textContent = "Enter both the address and the cluster id.";
    return;
  }
  note.textContent = "Joining…";
  try {
    renderCluster(await api("/api/cluster/join", { method: "POST", body: JSON.stringify({ address, cluster_id: clusterId, admin_secret: document.getElementById("cluster-join-secret").value }) }));
    document.getElementById("cluster-join-secret").value = "";
    note.textContent = "";
  } catch (err) {
    note.textContent = err.message;
  }
}

document.getElementById("cluster-join-btn").addEventListener("click", joinCluster);
document.getElementById("cluster-id-copy").addEventListener("click", () => {
  navigator.clipboard?.writeText(document.getElementById("cluster-id").textContent);
});

async function leaveCluster() {
  const ok = confirm(
    "Leave the cluster from this pAIring?\n\nCluster mode goes off here only. No node is changed: none is reset and none leaves the cluster. " +
      "The nodes stay listed as plain remote hosts (a Matricxon this app managed becomes a remote host too). You can join again with the cluster id."
  );
  if (!ok) return;
  try {
    renderCluster(await api("/api/cluster/leave", { method: "POST" }));
    _clusterShowError("");
  } catch (err) {
    _clusterShowError(err.message);
  }
}

document.getElementById("cluster-leave-btn").addEventListener("click", leaveCluster);
