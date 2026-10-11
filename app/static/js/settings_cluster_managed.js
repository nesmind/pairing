/** While cluster mode is on, the folders and the list of nodes are cluster settings (Cluster page), so the
 * Matricxon form hides its own copies (the saved-chat-cache panel does the same in
 * settings_cache_persistence.js). The fields stay in the form with their loaded values: saving the rest of
 * the form changes nothing about them. The nodes are shown read-only instead of the host editor. */

document.addEventListener("DOMContentLoaded", async () => {
  try {
    const config = await api("/api/settings/matricxon/config");
    const managed = config.cluster_role !== "off";
    document.querySelectorAll("[data-cluster-managed]").forEach((el) => el.classList.toggle("hidden", managed));
    document.getElementById("matricxon-paths-managed").classList.toggle("hidden", !managed);
    if (managed) await showManagedNodes(config);
  } catch (_err) {
    /* not an admin: nothing to hide */
  }
});

/** The nodes, read-only, in place of the host editor of the Matricxon form (see collectRemoteHosts). The editor is
 * hidden at once, from the saved config; /health (which pings every host, so it can be slow) is asked only when
 * pAIring manages a local Matricxon, whose address is not in the config. */
async function showManagedNodes(config) {
  const section = document.querySelector('[data-server="matricxon"]');
  if (!section) return;
  section.querySelector('[data-field="remote-hosts-list"]').classList.add("hidden");
  document.getElementById("matricxon-add-host").classList.add("hidden");
  document.getElementById("matricxon-hosts-hint").classList.add("hidden");
  document.getElementById("matricxon-hosts-managed").classList.remove("hidden");
  let nodes = config.mode === "remote" ? [...config.remote_hosts, ...config.cluster_hosts] : [];
  if (!nodes.length) nodes = Object.keys((await api("/health")).engine_hosts || {});
  if (!nodes.length) return;
  section.dataset.clusterHosts = JSON.stringify(nodes);
  document.getElementById("matricxon-hosts-managed-list").innerHTML = nodes.map((n) => `<li>${escapeHtml(n)}</li>`).join("");
}
