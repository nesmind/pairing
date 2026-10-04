/**
 * Admin settings > MCP servers (app/templates/_mcp_servers.html): a list of the MCP tool servers,
 * with a form (opened by "Add server" or a row's Edit) to add or change one and test its connection, via
 * /api/settings/mcp-servers. Also the global on/off switch in Admin settings > System
 * (app/templates/_mcp_global_switch.html, /api/settings/mcp-enabled). Uses api() from app.js.
 */
class McpServersPanel {
  static URL = "/api/settings/mcp-servers";

  constructor(root) {
    this.root = root;
    this.list = root.querySelector("#mcp-list");
    this.empty = root.querySelector("#mcp-empty");
    this.form = root.querySelector("#mcp-form");
    this.title = root.querySelector("#mcp-form-title");
    this.name = root.querySelector("#mcp-name");
    this.url = root.querySelector("#mcp-url");
    this.headers = root.querySelector("#mcp-headers");
    this.headersHint = root.querySelector("#mcp-headers-hint");
    this.enabled = root.querySelector("#mcp-enabled");
    this.timeout = root.querySelector("#mcp-timeout");
    this.cancel = root.querySelector("#mcp-cancel");
    this.status = root.querySelector("#mcp-status");
    this.addButton = root.querySelector("#mcp-add");
    this.listHeader = root.querySelector("#mcp-list-header");
    this.testButton = root.querySelector("#mcp-test");
    this.testResult = root.querySelector("#mcp-test-result");
    this.editingId = null;
    this.form.addEventListener("submit", (event) => {
      event.preventDefault();
      this.save();
    });
    this.cancel.addEventListener("click", () => this.closeForm());
    this.addButton.addEventListener("click", () => this.openForm());
    this.testButton.addEventListener("click", () => this.testForm());
  }

  openForm(server = null) {
    this.resetForm();
    if (server) this.fill(server);
    this.form.classList.remove("hidden");
    this.setListShown(false);
    this.name.focus();
  }

  closeForm() {
    this.resetForm();
    this.form.classList.add("hidden");
    this.setListShown(true);
  }

  /** The list (and Add button) are hidden while the add/edit form is open. */
  setListShown(shown) {
    this.list.classList.toggle("hidden", !shown);
    this.addButton.classList.toggle("hidden", !shown);
    this.listHeader.classList.toggle("hidden", !shown);
    if (shown) this.empty.classList.toggle("hidden", this.list.children.length > 0);
    else this.empty.classList.add("hidden");
  }

  /** Tries the form's current values (before saving) and says what it found. */
  async testForm() {
    this.testResult.textContent = "Connecting…";
    this.testButton.disabled = true;
    try {
      const body = { name: this.name.value || "server", url: this.url.value, enabled: true };
      const headers = this.parseHeaders();
      // Editing with the headers box empty: the server uses the saved headers.
      if (!this.editingId || Object.keys(headers).length) body.headers = headers;
      const query = this.editingId ? `?server_id=${this.editingId}` : "";
      const result = await api(`${McpServersPanel.URL}/test${query}`, { method: "POST", body: JSON.stringify(body) });
      this.testResult.textContent = McpServersPanel.describe(result);
    } catch (error) {
      this.testResult.textContent = error.message;
    } finally {
      this.testButton.disabled = false;
    }
  }

  /** The test result as readable lines: what the server says about itself, how heavy its tools are for a
   * model (their definitions are re-read every round), and each tool with its parameters (* = required). */
  static describe(result) {
    if (!result.ok) return `Failed: ${result.error}`;
    const clip = (text, max) => (text.length > max ? `${text.slice(0, max).trimEnd()}…` : text);
    const server = result.server || {};
    const who = [server.name, server.version].filter(Boolean).join(" ");
    const lines = [`Connected in ${result.seconds}s${who ? ` — ${who}` : ""}`];
    const facts = [];
    if (server.protocol) facts.push(`protocol ${server.protocol}`);
    if (server.capabilities?.length) facts.push(`offers: ${server.capabilities.join(", ")}`);
    if (facts.length) lines.push(facts.join(" · "));
    if (server.instructions) lines.push(`Instructions: ${clip(server.instructions.replace(/\s+/g, " "), 300)}`);
    const n = result.tools.length;
    lines.push(
      `${n} tool${n === 1 ? "" : "s"}, about ${result.definition_tokens} tokens of definitions sent to the model each round`,
    );
    for (const tool of result.tools) {
      const params = tool.params.length ? ` (${tool.params.join(", ")})` : "";
      const description = tool.description ? ` — ${clip(tool.description.replace(/\s+/g, " "), 140)}` : "";
      lines.push(`• ${tool.name}${params}${description}`);
    }
    return lines.join("\n");
  }

  /** "Name: value" lines -> {Name: value}; throws on a malformed line. */
  parseHeaders() {
    const headers = {};
    for (const line of this.headers.value.split("\n")) {
      if (!line.trim()) continue;
      const at = line.indexOf(":");
      if (at < 1) throw new Error(`Header line needs "Name: value": ${line}`);
      headers[line.slice(0, at).trim()] = line.slice(at + 1).trim();
    }
    return headers;
  }

  resetForm() {
    this.editingId = null;
    this.form.reset();
    this.enabled.checked = true;
    this.timeout.value = "";
    this.title.textContent = "Add a server";
    this.headersHint.classList.add("hidden");
    this.testResult.textContent = "";
  }

  fill(server) {
    this.editingId = server.id;
    this.name.value = server.name;
    this.url.value = server.url;
    this.headers.value = "";
    this.enabled.checked = server.enabled;
    this.timeout.value = server.call_timeout_seconds;
    this.title.textContent = `Edit ${server.name}`;
    this.headersHint.classList.remove("hidden");
  }

  button(label, onClick) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.className = "rounded-md border border-slate-700 px-2.5 py-1 text-xs text-slate-200 hover:bg-slate-800";
    button.addEventListener("click", onClick);
    return button;
  }

  row(server) {
    const row = document.createElement("div");
    row.className = "rounded-lg bg-slate-900 border border-slate-800 px-4 py-2.5 space-y-1";
    const top = document.createElement("div");
    top.className = "flex items-center gap-2";
    const label = document.createElement("div");
    label.className = "min-w-0 flex-1";
    const name = document.createElement("p");
    name.className = "text-sm text-slate-200";
    name.textContent = server.name + (server.enabled ? "" : " (disabled)");
    const url = document.createElement("p");
    url.className = "text-xs text-slate-500 truncate";
    const headerNote = server.header_names.length ? ` — headers: ${server.header_names.join(", ")}` : "";
    url.textContent = `${server.url}${headerNote} — tool timeout ${server.call_timeout_seconds}s`;
    label.append(name, url);
    const result = document.createElement("p");
    result.className = "text-xs text-slate-400 whitespace-pre-wrap break-words";
    top.append(
      label,
      this.button(server.enabled ? "Disable" : "Enable", () => this.setEnabled(server)),
      this.button("Test", () => this.test(server, result)),
      this.button("Edit", () => this.openForm(server)),
      this.button("Delete", () => this.remove(server)),
    );
    row.append(top, result);
    return row;
  }

  /** Flips a server on or off without opening the form (its saved headers are kept: none are sent). */
  async setEnabled(server) {
    try {
      await api(`${McpServersPanel.URL}/${server.id}`, {
        method: "PUT",
        body: JSON.stringify({
          name: server.name,
          url: server.url,
          enabled: !server.enabled,
          call_timeout_seconds: server.call_timeout_seconds,
        }),
      });
      await this.load();
    } catch (error) {
      this.status.textContent = error.message;
    }
  }

  async test(server, output) {
    output.textContent = "Connecting…";
    try {
      const result = await api(`${McpServersPanel.URL}/${server.id}/test`, { method: "POST" });
      output.textContent = McpServersPanel.describe(result);
    } catch (error) {
      output.textContent = error.message;
    }
  }

  async remove(server) {
    if (!confirm(`Remove the MCP server "${server.name}"?`)) return;
    try {
      await api(`${McpServersPanel.URL}/${server.id}`, { method: "DELETE" });
      await this.load();
    } catch (error) {
      this.status.textContent = error.message;
    }
  }

  async save() {
    try {
      const body = {
        name: this.name.value,
        url: this.url.value,
        enabled: this.enabled.checked,
        call_timeout_seconds: Number(this.timeout.value) || 20, // empty = the 20 s default shown as the placeholder
      };
      const headers = this.parseHeaders();
      // On an edit, an empty box keeps the saved headers (the server never sends their values back).
      if (!this.editingId || Object.keys(headers).length) body.headers = headers;
      const path = this.editingId ? `${McpServersPanel.URL}/${this.editingId}` : McpServersPanel.URL;
      await api(path, { method: this.editingId ? "PUT" : "POST", body: JSON.stringify(body) });
      this.closeForm();
      this.status.textContent = "Saved";
      await this.load();
    } catch (error) {
      this.status.textContent = error.message;
    }
  }

  async load() {
    const servers = await api(McpServersPanel.URL);
    this.list.replaceChildren(...servers.map((server) => this.row(server)));
    this.empty.classList.toggle("hidden", servers.length > 0);
  }
}

/** Admin settings > System: one switch that turns MCP off (or on) for every user. */
class McpGlobalSwitch {
  static URL = "/api/settings/mcp-enabled";

  constructor(root) {
    this.checkbox = root.querySelector("#mcp-global-enabled");
    this.status = root.querySelector("#mcp-global-status");
    this.checkbox.addEventListener("change", () => this.save());
  }

  async load() {
    this.checkbox.checked = (await api(McpGlobalSwitch.URL)).enabled;
  }

  async save() {
    const wanted = this.checkbox.checked;
    try {
      await api(McpGlobalSwitch.URL, { method: "PUT", body: JSON.stringify({ enabled: wanted }) });
      this.status.textContent = wanted ? "MCP tools are on for everyone." : "MCP tools are off for everyone.";
    } catch (error) {
      this.checkbox.checked = !wanted;
      this.status.textContent = error.message;
    }
  }
}

/** Admin settings > System: the limits every MCP server shares (tool rounds per reply, characters of a result). */
class McpLimits {
  static URL = "/api/settings/mcp-limits";

  constructor(root) {
    this.rounds = root.querySelector("#mcp-max-rounds");
    this.chars = root.querySelector("#mcp-max-result-chars");
    this.status = root.querySelector("#mcp-limits-status");
    root.querySelector("#mcp-limits-save").addEventListener("click", () => this.save());
  }

  show(limits) {
    this.rounds.value = limits.rounds;
    this.chars.value = limits.result_chars;
  }

  async load() {
    this.show(await api(McpLimits.URL)); // on failure the form keeps its defaults (5 and 4000)
  }

  async save() {
    try {
      const body = JSON.stringify({ rounds: Number(this.rounds.value), result_chars: Number(this.chars.value) });
      this.show(await api(McpLimits.URL, { method: "PUT", body }));
      this.status.textContent = "Saved";
    } catch (error) {
      this.status.textContent = error.message;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const limitsRoot = document.getElementById("mcp-limits-panel");
  if (limitsRoot) {
    new McpLimits(limitsRoot).load().catch((error) => {
      limitsRoot.querySelector("#mcp-limits-status").textContent = `Couldn't load the saved values (${error.message}); restart the app if it was just updated.`;
    });
  }
  const root = document.getElementById("mcp-panel");
  if (root) new McpServersPanel(root).load().catch((error) => (root.querySelector("#mcp-status").textContent = error.message));
  const globalRoot = document.getElementById("mcp-global-panel");
  if (globalRoot) new McpGlobalSwitch(globalRoot).load().catch((error) => (globalRoot.querySelector("#mcp-global-status").textContent = error.message));
});
