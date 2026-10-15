/* Settings > Models: why the model list couldn't load. Tells "the engine isn't running" apart from a slow or
   flaky engine, for Ollama and Matricxon. Used by settings.js initModelTab. */
class EngineDownReason {
  static LABELS = { ollama: "Ollama", matricxon: "Matricxon" };

  // GET /health answers 503 while the engine is down, but its body still names the engine and its hosts.
  static async health() {
    try {
      const response = await fetch("/health", { cache: "no-store" });
      return await response.json();
    } catch (_err) {
      return null;
    }
  }

  // Admins only: local mode plus a stopped process means "start it".
  static async localStopped(engine) {
    try {
      const config = await api(`/api/settings/${engine}/config`);
      if (config.mode !== "local") return false;
      return !(await api(`/api/settings/${engine}/status`)).running;
    } catch (_err) {
      return false;
    }
  }

  static async describe(err, isAdmin) {
    const health = await EngineDownReason.health();
    const engine = health?.active_engine;
    const label = EngineDownReason.LABELS[engine] || "the active engine";
    const hosts = Object.values(health?.engine_hosts || {});
    const allDown = hosts.length > 0 && hosts.every((up) => !up);
    if (allDown && engine in EngineDownReason.LABELS) {
      if (!isAdmin) return `${label} isn't running right now, so no models can be listed. Please ask your admin to start it.`;
      if (await EngineDownReason.localStopped(engine)) {
        return `${label} isn't running. Please start it in Admin settings > Inference engines, then refresh this page.`;
      }
      return `${label} isn't answering on any of its hosts. Please check that it's running, then refresh this page.`;
    }
    const detail = err?.message ? ` (${err.message})` : "";
    return (
      `Couldn't reach ${label} to load the model list${detail} — it may be still starting up ` +
      "or slow to respond right now. Try refreshing in a moment."
    );
  }
}
