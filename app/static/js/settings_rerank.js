/* Admin settings > System: the Knowledge base reranker (app/templates/_rerank_model.html, /api/settings/rerank-model). Uses api from app.js. */
class RerankSetting {
  static URL = "/api/settings/rerank-model";

  constructor(root) {
    this.select = root.querySelector("#rerank-model");
    this.note = root.querySelector("#rerank-model-note");
    this.saveBtn = root.querySelector("#rerank-model-save-btn");
    this.status = root.querySelector("#rerank-model-status");
    this.saveBtn.addEventListener("click", () => this.save());
    document.addEventListener("pairing:rerankers-changed", () => this.load());
    this.load();
  }

  render(setting) {
    const options = [["", "Off"], ...setting.installed.map((tag) => [tag, tag])];
    this.select.replaceChildren(...options.map(([value, label]) => new Option(label, value)));
    this.select.value = setting.enabled && setting.model ? setting.model : "";
    const note = !setting.engine_supports
      ? "The active engine can't rerank. Reranking works with Matricxon."
      : setting.installed.length
        ? ""
        : "No reranker installed. Pull one under Models > Reranker models.";
    this.note.textContent = note;
    this.note.classList.toggle("hidden", !note);
    this.select.disabled = this.saveBtn.disabled = !setting.engine_supports || !setting.installed.length;
  }

  async load() {
    try {
      this.render(await api(RerankSetting.URL));
    } catch (err) {
      this.status.textContent = `Could not load: ${err.message}`;
    }
  }

  async save() {
    this.status.textContent = "Saving…";
    try {
      const model = this.select.value || null;
      this.render(await api(RerankSetting.URL, { method: "PUT", body: JSON.stringify({ model }) }));
      this.status.textContent = model ? "Saved." : "Saved — reranking is off.";
      document.dispatchEvent(new Event("pairing:reranker-selected"));
    } catch (err) {
      this.status.textContent = `Failed to save: ${err.message}`;
    }
  }
}

const rerankPanelEl = document.getElementById("rerank-panel");
if (rerankPanelEl) new RerankSetting(rerankPanelEl);
