/** The cluster's "Saved chat caches" form (_cluster_cache_form.html): the limits are written through the cluster,
 * so it is locked without a quorum. Deleting the saved caches is not a cluster setting and always works. */

const CLUSTER_CACHE_URL = "/api/cluster/cache";
const CLUSTER_CACHE_STATS_URL = "/api/settings/matricxon/cache-persistence";
let _clusterCacheDirty = false;

function _clusterCacheFill(status) {
  if (!_clusterCacheDirty) {
    document.getElementById("cluster-cache-enabled").checked = status.enabled;
    document.getElementById("cluster-cache-budget").value = status.budget_mb;
    document.getElementById("cluster-cache-ttl").value = status.ttl_hours;
  }
  const mb = (status.used_bytes / 1048576).toFixed(1);
  const lost = status.unreachable.length ? ` — unreachable: ${status.unreachable.join(", ")}` : "";
  document.getElementById("cluster-cache-note").textContent = `${status.files} saved chat cache(s), ${mb} MB (one shared folder, read by ${status.hosts} node(s))${lost}`;
}

async function loadClusterCache() {
  try {
    _clusterCacheFill(await api(CLUSTER_CACHE_STATS_URL));
  } catch (err) {
    document.getElementById("cluster-cache-note").textContent = err.message;
  }
}

function renderClusterCache(state) {
  const form = document.getElementById("cluster-cache-form");
  form.classList.toggle("hidden", !(state.supported && state.enabled && state.quorum !== "off"));  // not while setting up
  document.getElementById("cluster-cache-quorum").classList.toggle("hidden", state.quorum !== "lost");
  for (const id of ["cluster-cache-enabled", "cluster-cache-budget", "cluster-cache-ttl", "cluster-cache-save"]) {
    const el = document.getElementById(id);
    el.disabled = state.quorum === "lost";
    el.title = state.quorum === "lost" ? QUORUM_LOCK_TIP : "";
  }
  if (!form.classList.contains("hidden") && !form.dataset.loaded) {
    form.dataset.loaded = "1";
    loadClusterCache();
  }
}

async function saveClusterCache(event) {
  event.preventDefault();
  const body = {
    enabled: document.getElementById("cluster-cache-enabled").checked,
    budget_mb: Number(document.getElementById("cluster-cache-budget").value),
    ttl_hours: Number(document.getElementById("cluster-cache-ttl").value),
  };
  try {
    _clusterCacheDirty = false;
    _clusterCacheFill(await api(CLUSTER_CACHE_URL, { method: "PUT", body: JSON.stringify(body) }));
    document.getElementById("cluster-cache-note").textContent = "Saved: every node applies it within a few seconds.";
    setTimeout(loadClusterCache, 5000);
    _clusterShowError("");
  } catch (err) {
    _clusterCacheDirty = true;
    _clusterShowError(err.message);
  }
}

async function clearClusterCache() {
  if (!confirm("Delete every saved chat cache on every node? Chats will re-read their history the next time they are used.")) return;
  try {
    _clusterCacheFill(await api(CLUSTER_CACHE_STATS_URL, { method: "DELETE" }));
  } catch (err) {
    _clusterShowError(err.message);
  }
}

document.getElementById("cluster-cache-form").addEventListener("submit", saveClusterCache);
document.getElementById("cluster-cache-clear").addEventListener("click", clearClusterCache);
for (const id of ["cluster-cache-enabled", "cluster-cache-budget", "cluster-cache-ttl"]) {
  document.getElementById(id).addEventListener("input", () => (_clusterCacheDirty = true));
}
