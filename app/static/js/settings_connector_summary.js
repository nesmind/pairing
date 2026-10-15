/* Settings > Inference engines: when the Active engine picker shows a connector (Cloudflare, RunPod), this panel
   shows its state, account, model counts and usage (GET /api/connectors/{id}/summary). settings.js sends an
   "active-engine-shown" event with the picked engine each time it updates the page for the picker. */

class ConnectorSummaryPanel {
  constructor(root) {
    this.root = root;
    this.field = (name) => root.querySelector(`[data-field="${name}"]`);
    this.connectorByEngine = null; // engine name -> connector id, read once
    this.current = null;
    root.querySelector('[data-action="refresh"]').addEventListener("click", () => this.load(this.current));
    document.addEventListener("active-engine-shown", (event) => this.show(event.detail));
  }

  async connectorFor(engine) {
    if (this.connectorByEngine === null) {
      try {
        const { connectors } = await api("/api/connectors");
        this.connectorByEngine = Object.fromEntries(connectors.map((c) => [c.engine_name, c.id]));
      } catch (_err) {
        this.connectorByEngine = {};
      }
    }
    return this.connectorByEngine[engine] ?? null;
  }

  async show(engine) {
    const connectorId = await this.connectorFor(engine);
    this.root.classList.toggle("hidden", connectorId === null);
    if (connectorId !== null && connectorId !== this.current) await this.load(connectorId);
  }

  async load(connectorId) {
    if (!connectorId) return;
    this.current = connectorId;
    this.field("status").textContent = "Loading…";
    this.field("body").classList.add("opacity-50");
    try {
      this.render(await api(`/api/connectors/${connectorId}/summary`));
      this.field("status").textContent = "";
    } catch (err) {
      this.field("status").textContent = `Could not load the summary: ${err.message}`;
    } finally {
      this.field("body").classList.remove("opacity-50");
    }
  }

  static el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  static pill(text, ok) {
    const tone = ok
      ? "bg-emerald-500/15 border-emerald-500/40 text-emerald-400"
      : "bg-warning-500/15 border-warning-500/40 text-warning-400";
    return ConnectorSummaryPanel.el("span", `rounded-full border px-2 py-0.5 text-xs font-medium ${tone}`, text);
  }

  static tile(value, label) {
    const box = ConnectorSummaryPanel.el("div");
    box.append(
      ConnectorSummaryPanel.el("p", "text-lg font-semibold text-slate-100", value),
      ConnectorSummaryPanel.el("p", "text-xs text-slate-500", label),
    );
    return box;
  }

  render(summary) {
    const { el, pill, tile } = ConnectorSummaryPanel;
    const number = (n) => Number(n).toLocaleString();
    this.field("title").textContent = summary.display_name;

    const pills = this.field("pills");
    pills.replaceChildren(
      pill(summary.enabled ? "Enabled" : "Disabled", summary.enabled),
      pill(summary.last_test_passed ? "Connection test passed" : "Not tested", summary.last_test_passed),
      pill(summary.ready ? "Ready" : "Not ready", summary.ready),
    );

    const details = this.field("details");
    details.replaceChildren(
      ...summary.details.map(({ label, value }) => {
        const row = el("div", "flex justify-between gap-3");
        row.append(el("dt", "text-slate-500", label), el("dd", "text-slate-200 font-mono truncate", value));
        return row;
      }),
    );

    const models = this.field("models");
    if (summary.chat_models !== null) {
      models.replaceChildren(tile(number(summary.chat_models), "chat models"), tile(number(summary.embedding_models), "embedding models"));
    } else {
      models.replaceChildren(el("p", "text-xs text-slate-500", summary.models_error || "Shown once the connector is ready."));
    }

    const usage = summary.usage;
    this.field("usage").classList.toggle("hidden", !usage);
    if (usage) {
      this.field("usage-title").textContent = `Usage — ${usage.period} (as reported by the provider)`;
      this.field("usage-tiles").replaceChildren(
        tile(number(usage.requests), "requests"),
        tile(number(usage.units), usage.units_label),
        tile(number(usage.input_tokens + usage.output_tokens), "tokens in + out"),
        tile(`${number(usage.today_units)}`, `${usage.units_label} today (${number(usage.today_requests)} requests)`),
      );
      this.field("top-models").replaceChildren(
        el("p", "text-slate-400", usage.top_models.length ? "Most used models" : "No requests in this period."),
        ...usage.top_models.map(({ model, requests, units }) => {
          const row = el("div", "flex justify-between gap-3");
          row.append(
            el("span", "text-slate-200 font-mono truncate", model),
            el("span", "text-slate-500 shrink-0", `${number(requests)} requests · ${number(units)} ${usage.units_label}`),
          );
          return row;
        }),
      );
    }
    const note = this.field("usage-note");
    note.textContent = summary.usage_note || "";
    note.classList.toggle("hidden", !summary.usage_note);
  }
}

const connectorSummaryRoot = document.getElementById("connector-summary-panel");
if (connectorSummaryRoot) new ConnectorSummaryPanel(connectorSummaryRoot);
