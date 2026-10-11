/** The "Cluster timers" form: election timeout and heartbeat, written through the cluster (see
 * _cluster_timers_form.html). Locked while there is no quorum or no consensus. */

let _clusterTimersDirty = false;

function renderClusterTimers(state) {
  const off = !state.timers;
  document.getElementById("cluster-timers-form").classList.toggle("hidden", off);  // not while setting up
  const lost = state.quorum === "lost";
  document.getElementById("cluster-timers-off").classList.toggle("hidden", !off);
  document.getElementById("cluster-timers-quorum").classList.toggle("hidden", !lost);
  document.getElementById("cluster-timers-fields").classList.toggle("hidden", off);
  for (const id of ["cluster-election", "cluster-heartbeat", "cluster-timers-save", "cluster-timers-fast", "cluster-timers-tolerant"]) {
    const el = document.getElementById(id);
    el.disabled = off || lost;
    el.title = lost ? QUORUM_LOCK_TIP : "";
  }
  if (state.timers && !_clusterTimersDirty) {
    document.getElementById("cluster-election").value = state.timers.election_seconds;
    document.getElementById("cluster-heartbeat").value = state.timers.heartbeat_seconds;
  }
}

function _clusterSetTimers(election, heartbeat) {
  document.getElementById("cluster-election").value = election;
  document.getElementById("cluster-heartbeat").value = heartbeat;
  _clusterTimersDirty = true;
}

async function saveClusterTimers(event) {
  event.preventDefault();
  const note = document.getElementById("cluster-timers-note");
  const body = {
    election_seconds: Number(document.getElementById("cluster-election").value),
    heartbeat_seconds: Number(document.getElementById("cluster-heartbeat").value),
  };
  try {
    _clusterTimersDirty = false;
    renderCluster(await api("/api/cluster/timers", { method: "PUT", body: JSON.stringify(body) }));
    note.textContent = "Saved: every voter uses these within a few seconds.";
    _clusterShowError("");
  } catch (err) {
    _clusterTimersDirty = true;
    note.textContent = "";
    _clusterShowError(err.message);
  }
}

document.getElementById("cluster-timers-form").addEventListener("submit", saveClusterTimers);
document.getElementById("cluster-timers-fast").addEventListener("click", () => _clusterSetTimers(1, 0.2));
document.getElementById("cluster-timers-tolerant").addEventListener("click", () => _clusterSetTimers(5, 1));
for (const id of ["cluster-election", "cluster-heartbeat"]) {
  document.getElementById(id).addEventListener("input", () => (_clusterTimersDirty = true));
}
