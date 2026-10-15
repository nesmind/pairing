/* Settings > Models > Reranker models (#rerank-models-section): the curated rerankers (default_models.json "rerank_models") with Pull/Uninstall, via /api/settings/rerank-model-catalog. Reuses pullModel/uninstallModel/formatSize/escapeHtml from settings.js; tells the System tab's Reranker setting to reload after a change. */
class RerankModelCatalog {
  static URL = "/api/settings/rerank-model-catalog";

  constructor(section) {
    this.list = section.querySelector("#rerank-model-catalog");
    this.note = section.querySelector("#rerank-model-catalog-note");
    document.addEventListener("pairing:reranker-selected", () => this.load());
    this.load();
  }

  async load() {
    try {
      this.render(await api(RerankModelCatalog.URL));
    } catch (err) {
      this.showNote(`Could not load reranker models: ${err.message}`);
    }
  }

  showNote(text) {
    this.note.textContent = text;
    this.note.classList.toggle("hidden", !text);
  }

  async changed() {
    await this.load();
    document.dispatchEvent(new Event("pairing:rerankers-changed"));
  }

  render(catalog) {
    this.list.replaceChildren();
    this.showNote(catalog.engine_supports ? "" : "Reranking works with Matricxon; the active engine can't rerank.");
    if (!catalog.engine_supports) return;
    for (const entry of catalog.entries) this.list.appendChild(this.row(entry));
  }

  row(entry) {
    const row = document.createElement("div");
    row.className = "rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-sm flex flex-col justify-between gap-1.5";
    const inUse = entry.in_use
      ? ` <span class="inline-flex items-center rounded-full bg-[rgb(var(--color-brand-500)/0.15)] border border-[rgb(var(--color-brand-500)/0.4)] px-1.5 py-0.5 text-[10px] font-medium text-brand-500 align-middle">In use</span>`
      : "";
    const meta = [entry.vendor, entry.download_gb ? formatSize(entry.download_gb) : null].filter(Boolean).join(" · ");
    const label = document.createElement("div");
    label.className = "min-w-0";
    label.innerHTML =
      `<span class="block truncate font-medium text-slate-100">${escapeHtml(`${entry.family} ${entry.parameter_size}`)}${inUse}</span>` +
      `<span class="block truncate text-xs text-slate-500">${escapeHtml(meta)}</span>`;

    const action = document.createElement("div");
    action.className = "flex flex-wrap items-center gap-1.5";
    const progress = document.createElement("p");
    progress.className = "mt-1.5 text-xs text-slate-500 hidden";
    const done = () => this.changed();
    if (entry.installed) {
      const badge = document.createElement("span");
      badge.className = "text-xs font-medium text-emerald-400";
      badge.textContent = "Installed";
      const remove = this.button("Uninstall", "rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-400 hover:border-red-500 hover:text-red-400 transition-colors");
      remove.addEventListener("click", () => uninstallModel(entry, remove, progress, done));
      action.append(badge, remove);
    } else {
      const pull = this.button("Pull", "rounded-md border border-slate-700 px-2.5 py-1 text-xs text-slate-200 hover:bg-slate-800 transition-colors");
      pull.addEventListener("click", () => pullModel(entry, pull, progress, done));
      action.appendChild(pull);
    }
    row.append(label, action, progress);
    return row;
  }

  button(text, className) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = className;
    button.textContent = text;
    return button;
  }
}

const rerankModelsSectionEl = document.getElementById("rerank-models-section");
if (rerankModelsSectionEl) new RerankModelCatalog(rerankModelsSectionEl);
