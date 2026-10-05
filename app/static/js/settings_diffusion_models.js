/**
 * Settings > Model > "Diffusion models": the image models for the Images page — curated defaults
 * (default_models.json), image models an admin added from Hugging Face's "Browse more models" search, and anything
 * else already installed. Same list whichever chat engine (Ollama/Matricxon) is active; they install into the local
 * stable-diffusion.cpp engine's models folder (see app/services/image_model_service.py), via
 * /api/settings/image-models. Reuses pullModel/uninstallModel/formatSize/isAdmin from settings.js; reloads on the
 * "pairing:models-changed" event settings.js fires after any model action.
 */
class DiffusionModelsSection {
  static URL = "/api/settings/image-models/catalog";

  constructor(root) {
    this.grid = root.querySelector("#diffusion-model-catalog");
    this.note = root.querySelector("#diffusion-model-note");
    document.addEventListener("pairing:models-changed", () => this.load());
    this.load();
  }

  async load() {
    try {
      const { entries, local } = await api(DiffusionModelsSection.URL);
      this.render(entries, local);
    } catch (err) {
      this.grid.innerHTML = "";
      this.note.textContent = `Could not load: ${err.message}`;
      this.note.classList.remove("hidden");
    }
  }

  render(entries, local) {
    this.grid.innerHTML = "";
    this.note.textContent = local
      ? ""
      : "The image engine is in Remote mode — it manages its own models. Switch it to Local (Settings > Image) to download them here.";
    this.note.classList.toggle("hidden", local);
    for (const entry of entries) this.grid.appendChild(this.renderRow(entry, local));
  }

  renderRow(entry, local) {
    const row = document.createElement("div");
    row.className = "rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-sm flex flex-col justify-between gap-1.5";
    row.dataset.tag = entry.tag; // lets "Add" from the Hugging Face search scroll straight to it
    row.title = entry.tag;

    const label = document.createElement("div");
    label.className = "min-w-0";
    label.innerHTML =
      `<span class="block truncate font-medium text-slate-100">${escapeHtml(entry.family)}</span>` +
      `<span class="block truncate text-xs text-slate-500">${escapeHtml([entry.vendor, formatSize(entry.download_gb)].filter(Boolean).join(" · "))}</span>`;

    const action = document.createElement("div");
    action.className = "flex flex-wrap items-center gap-1.5";
    const progress = document.createElement("p");
    progress.className = "mt-1.5 text-xs text-slate-500 hidden";
    const reload = () => this.load();

    if (entry.installed) {
      action.appendChild(this.textSpan("Installed", "text-xs font-medium text-emerald-400"));
      if (isAdmin && local) action.appendChild(this.button("Uninstall", (btn) => uninstallModel(entry, btn, progress, reload), "Uninstall — deletes this model file for everyone", true));
    } else if (!local) {
      action.appendChild(this.textSpan("Needs the local engine", "text-xs text-slate-500"));
    } else if (isAdmin) {
      action.appendChild(this.button("Pull", (btn) => pullModel(entry, btn, progress, reload)));
      if (entry.removable) action.appendChild(this.button("Remove", (btn) => this.remove(entry, btn, progress), "Remove from this list — nothing is installed", true));
    } else {
      action.appendChild(this.textSpan("Not installed — ask an admin", "text-xs text-slate-500"));
    }

    const top = document.createElement("div");
    top.className = "flex flex-col gap-1.5";
    top.append(label, action);
    row.append(top, progress);
    return row;
  }

  textSpan(text, className) {
    const span = document.createElement("span");
    span.className = className;
    span.textContent = text;
    return span;
  }

  button(text, onClick, title = "", danger = false) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.title = title;
    btn.textContent = text;
    btn.className = danger
      ? "rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-400 hover:border-red-500 hover:text-red-400 transition-colors"
      : "rounded-md border border-slate-700 px-2.5 py-1 text-xs text-slate-200 hover:bg-slate-800 transition-colors";
    btn.addEventListener("click", () => onClick(btn));
    return btn;
  }

  async remove(entry, btn, progress) {
    btn.disabled = true;
    progress.classList.remove("hidden");
    progress.textContent = "Removing…";
    try {
      await api("/api/settings/model-catalog/extended", { method: "DELETE", body: JSON.stringify({ tag: entry.tag }) });
      await this.load();
    } catch (err) {
      progress.textContent = `Failed: ${err.message}`;
      btn.disabled = false;
    }
  }
}

const diffusionModelsRoot = document.getElementById("diffusion-models-section");
if (diffusionModelsRoot) new DiffusionModelsSection(diffusionModelsRoot);
