/**
 * Admin settings > Image: the "Active image engine" picker (stable-diffusion.cpp / ComfyUI), via
 * /api/settings/image-engine. The engines' own sections are driven by the generic controller in settings.js.
 * Uses api() from app.js.
 */
class ImageEnginePicker {
  static URL = "/api/settings/image-engine";

  constructor(root) {
    this.select = root.querySelector("#image-engine-select");
    this.status = root.querySelector("#image-engine-status");
    root.querySelector("#image-engine-save-btn").addEventListener("click", () => this.save());
    this.select.addEventListener("change", () => this.showSelectedEngineOnly());
    this.showSelectedEngineOnly();
    this.load();
  }

  /** Only the selected engine's section is shown (the select's values match the sections' data-server). */
  showSelectedEngineOnly() {
    for (const option of this.select.options) {
      document.querySelector(`[data-server="${option.value}"]`)?.classList.toggle("hidden", option.value !== this.select.value);
    }
  }

  async load() {
    try {
      this.select.value = (await api(ImageEnginePicker.URL)).active_image_engine;
      this.showSelectedEngineOnly();
    } catch (err) {
      this.status.textContent = `Could not load: ${err.message}`;
    }
  }

  async save() {
    this.status.textContent = "Saving…";
    try {
      await api(ImageEnginePicker.URL, {
        method: "PUT",
        body: JSON.stringify({ active_image_engine: this.select.value }),
      });
      this.status.textContent = "Saved.";
    } catch (err) {
      this.status.textContent = `Failed: ${err.message}`;
    }
  }
}

if (document.getElementById("image-engine-select")) new ImageEnginePicker(document);
