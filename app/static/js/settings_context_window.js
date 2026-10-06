/**
 * Settings > System > "Context window": the one num_ctx every chat uses (see
 * app/services/context_window_setting.py). Admin-only page section; reads/writes /api/settings/context-window.
 * Reuses api() from app.js.
 */
(() => {
  const section = document.getElementById("context-window-section");
  if (!section) return;
  const slider = document.getElementById("context-window-input");
  const valueEl = document.getElementById("context-window-value");
  const saveBtn = document.getElementById("context-window-save-btn");
  const status = document.getElementById("context-window-status");

  // The page ships with the built-in default (4096, same as app.config.DEFAULT_GENERATION_PARAMS) so a new install
  // shows a value even before anything is saved or if loading fails.
  slider.addEventListener("input", () => (valueEl.textContent = slider.value));

  api("/api/settings/context-window")
    .then(({ num_ctx }) => {
      slider.value = num_ctx;
      valueEl.textContent = slider.value;
    })
    .catch((err) => (status.textContent = `Could not load: ${err.message}`));

  saveBtn.addEventListener("click", async () => {
    saveBtn.disabled = true;
    status.textContent = "Saving…";
    try {
      await api("/api/settings/context-window", { method: "PUT", body: JSON.stringify({ num_ctx: parseInt(slider.value, 10) }) });
      status.textContent = "Saved — applies from the next message.";
    } catch (err) {
      status.textContent = `Failed to save: ${err.message}`;
    } finally {
      saveBtn.disabled = false;
    }
  });
})();
