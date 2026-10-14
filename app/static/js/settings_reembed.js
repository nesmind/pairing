/* Settings > System > RAG: shows how the knowledge base matches the current embedding model and embeds every
   document again on request (POST /api/documents/reembed), e.g. after switching embedding models. */

class ReembedPanel {
  constructor(root) {
    this.root = root;
    this.field = (name) => root.querySelector(`[data-field="${name}"]`);
    this.button = root.querySelector('[data-action="reembed"]');
    this.button.addEventListener("click", () => this.reembed());
    this.load();
  }

  async load() {
    try {
      this.render(await api("/api/documents/embedding-status"));
    } catch (err) {
      this.field("summary").textContent = `Could not read the knowledge base: ${err.message}`;
    }
  }

  render(status) {
    const model = status.current_model || "none set";
    this.field("summary").textContent =
      `${status.total} document${status.total === 1 ? "" : "s"} · current embedding model: ${model}`;
    const parts = [];
    if (status.outdated) parts.push(`${status.outdated} embedded with another model are left out of searches`);
    if (status.unknown) parts.push(`${status.unknown} from before models were recorded`);
    const note = this.field("note");
    note.textContent = parts.length ? `${parts.join("; ")}. Re-embed to use them with the current model.` : "";
    note.classList.toggle("hidden", !parts.length);
    this.button.disabled = !status.total || !status.current_model;
  }

  async reembed() {
    const confirmed = confirm(
      "Embed every document again with the current embedding model? On a local engine without a GPU this can " +
        "take a long time; with a cloud engine it uses some of your usage allowance.",
    );
    if (!confirmed) return;
    this.button.disabled = true;
    this.field("status").textContent = "Re-embedding… keep this page open.";
    try {
      const result = await api("/api/documents/reembed", { method: "POST" });
      this.render(result.status);
      const errors = result.errors.length ? ` ${result.errors.length} problem(s): ${result.errors.join("; ")}` : "";
      this.field("status").textContent = `Done: ${result.updated} document(s) embedded again.${errors}`;
    } catch (err) {
      this.field("status").textContent = `Re-embedding failed: ${err.message}`;
      this.button.disabled = false;
    }
  }
}

const reembedRoot = document.getElementById("rag-reembed-panel");
if (reembedRoot) new ReembedPanel(reembedRoot);
