/* Shows warnings for what the active engine can't do. Any element with data-engine-note="<capability>" (tools, vision,
embeddings; see EngineCapabilities in app/services/engines/base.py) is shown only when the active engine lacks it, and
its [data-engine-name] spans get the engine's name. */

class EngineCapabilityNotes {
  async apply() {
    const notes = document.querySelectorAll("[data-engine-note]");
    if (!notes.length) return;
    const [{ active_engine: activeName }, { engines }] = await Promise.all([
      api("/health"),
      api("/api/settings/engine/options"),
    ]);
    const engine = engines.find((e) => e.name === activeName);
    if (!engine) return;
    notes.forEach((note) => {
      note.classList.toggle("hidden", Boolean(engine.capabilities[note.dataset.engineNote]));
      note.querySelectorAll("[data-engine-name]").forEach((el) => (el.textContent = engine.display_name));
    });
  }
}

new EngineCapabilityNotes().apply().catch(() => {
  // Notes are a hint only; a failed check leaves them hidden.
});
