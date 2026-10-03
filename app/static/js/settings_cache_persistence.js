/**
 * Settings > Matricxon > saved chat caches (app/templates/_matricxon_cache_persistence.html).
 * Reads and changes Matricxon's encrypted on-disk prompt cache through
 * /api/settings/matricxon/cache-persistence. Uses api() from app.js.
 */
class MatricxonCachePanel {
  static URL = "/api/settings/matricxon/cache-persistence";

  constructor(root) {
    this.root = root;
    this.enabled = root.querySelector("#matricxon-cache-enabled");
    this.budget = root.querySelector("#matricxon-cache-budget");
    this.ttl = root.querySelector("#matricxon-cache-ttl");
    this.saveButton = root.querySelector("#matricxon-cache-save");
    this.clearButton = root.querySelector("#matricxon-cache-clear");
    this.status = root.querySelector("#matricxon-cache-status");
    this.enabled.addEventListener("change", () => this.syncFields());
    this.saveButton.addEventListener("click", () => this.save());
    this.clearButton.addEventListener("click", () => this.clear());
  }

  static formatBytes(bytes) {
    if (bytes < 1048576) return `${Math.round(bytes / 1024)} KB`;
    if (bytes < 1073741824) return `${(bytes / 1048576).toFixed(1)} MB`;
    return `${(bytes / 1073741824).toFixed(2)} GB`;
  }

  /** The limits only matter while the switch is on. */
  syncFields() {
    this.budget.disabled = this.ttl.disabled = !this.enabled.checked;
  }

  show(state) {
    this.enabled.checked = state.enabled;
    this.syncFields();
    this.budget.value = state.budget_mb;
    this.ttl.value = state.ttl_hours;
    const hosts = state.hosts > 1 ? ` on ${state.hosts} hosts` : "";
    let text = `${state.files} saved chat${state.files === 1 ? "" : "s"}, ${MatricxonCachePanel.formatBytes(state.used_bytes)}${hosts}`;
    if (state.unreachable.length) text += ` — unreachable: ${state.unreachable.join(", ")}`;
    this.status.textContent = text;
  }

  async run(request, successText) {
    this.saveButton.disabled = this.clearButton.disabled = true;
    try {
      this.show(await request());
      if (successText) this.status.textContent = `${successText} — ${this.status.textContent}`;
    } catch (error) {
      this.status.textContent = error.message;
    } finally {
      this.saveButton.disabled = this.clearButton.disabled = false;
    }
  }

  load() {
    return this.run(() => api(MatricxonCachePanel.URL));
  }

  save() {
    const body = {
      enabled: this.enabled.checked,
      budget_mb: Number(this.budget.value),
      ttl_hours: Number(this.ttl.value),
    };
    return this.run(() => api(MatricxonCachePanel.URL, { method: "PUT", body: JSON.stringify(body) }), "Saved");
  }

  clear() {
    if (!confirm("Delete every saved chat cache? Chats will re-read their history the next time they are used.")) return;
    return this.run(() => api(MatricxonCachePanel.URL, { method: "DELETE" }), "Deleted");
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  const root = document.getElementById("matricxon-cache-panel");
  if (!root) return;
  // Matricxon-only feature: with Ollama (or anything else) as the active engine the panel stays hidden.
  const { active_engine: activeEngine } = await api("/health");
  if (activeEngine !== "matricxon") return;
  root.classList.remove("hidden");
  new MatricxonCachePanel(root).load();
});
