/**
 * Chat page: MCP tool support. ToolBlocks shows the tool calls a reply made (a collapsible block at the
 * top of its bubble); ToolToggle is the composer's "Tools" switch that turns tools on for a conversation
 * (conversation.params.use_tools, switched from a popover that also lists the tools — see app/services/tool_loop_service.py). Uses api() from app.js.
 */
class ToolBlocks {
  static RESULT_PREVIEW_CHARS = 2000;

  /** Draws `bubble._toolEvents` as a block at the top of the bubble (replacing any earlier one). Called
   * after every renderReplyBody, which rewrites the bubble's contents. */
  static render(bubble) {
    bubble.querySelector(":scope > .tool-blocks")?.remove();
    const events = bubble._toolEvents;
    if (!events || !events.length) return;
    const block = document.createElement("details");
    block.className = "tool-blocks mb-2 rounded-lg border border-white/10 text-xs";
    const summary = document.createElement("summary");
    summary.className = "cursor-pointer px-2 py-1 text-slate-300";
    summary.textContent = `🔧 Used ${events.length} tool call${events.length === 1 ? "" : "s"}`;
    block.append(summary, ...events.map((event) => ToolBlocks.entry(event)));
    bubble.prepend(block);
  }

  static entry(event) {
    const entry = document.createElement("div");
    entry.className = "border-t border-white/10 px-2 py-1 space-y-1";
    const title = document.createElement("div");
    title.className = event.is_error ? "font-medium text-red-400" : "font-medium text-slate-200";
    title.textContent = `${event.server ? event.server + " · " : ""}${event.tool}${event.is_error ? " (failed)" : ""}`;
    const args = document.createElement("pre");
    args.className = "whitespace-pre-wrap break-words text-slate-400";
    args.textContent = JSON.stringify(event.arguments);
    const result = document.createElement("pre");
    result.className = "whitespace-pre-wrap break-words text-slate-300";
    const text = event.result || "";
    const long = text.length > ToolBlocks.RESULT_PREVIEW_CHARS;
    result.textContent = long ? text.slice(0, ToolBlocks.RESULT_PREVIEW_CHARS) + "\n…" : text;
    entry.append(title, args, result);
    return entry;
  }

  /** A live `tool` event for a bubble that is still streaming. */
  static add(bubble, event) {
    bubble._toolEvents = [...(bubble._toolEvents || []), event];
    ToolBlocks.render(bubble);
  }
}

class ToolToggle {
  constructor(button) {
    this.button = button;
    this.conversationId = null;
    this.params = null;
    this.tools = [];
    this.canManage = false;
    this.popover = document.createElement("div");
    this.popover.className =
      "hidden absolute bottom-full left-0 mb-2 z-20 w-80 max-h-72 overflow-y-auto rounded-xl border border-slate-700 bg-slate-900 p-3 text-xs shadow-lg space-y-2";
    button.parentElement.classList.add("relative");
    button.parentElement.append(this.popover);
    this.button.addEventListener("click", (event) => {
      event.stopPropagation();
      this.popover.classList.toggle("hidden");
      this.drawPopover();
    });
    document.addEventListener("click", (event) => {
      if (!this.popover.contains(event.target)) this.popover.classList.add("hidden");
    });
  }

  /** Called when a conversation is opened: show the switch only if some MCP tool exists. */
  async sync(conversationId, conversation, canManage) {
    this.conversationId = conversationId;
    this.params = conversation.params || {};
    this.canManage = canManage;
    this.popover.classList.add("hidden");
    let tools = [];
    try {
      tools = await api("/api/mcp/tools");
    } catch (_) {
      /* no tools -> no switch */
    }
    if (this.conversationId !== conversationId) return; // the user moved on while this loaded
    this.tools = tools;
    this.button.classList.toggle("hidden", tools.length === 0);
    this.paint();
  }

  get on() {
    return Boolean(this.params && this.params.use_tools);
  }

  paint() {
    this.button.setAttribute("aria-pressed", String(this.on));
    this.button.classList.toggle("text-brand-500", this.on);
    this.button.classList.toggle("text-slate-400", !this.on);
    this.button.title = this.on ? "Tools are on for this chat" : "Tools are off for this chat";
  }

  /** The popover: the on/off switch plus every tool the model would be offered. */
  drawPopover(message = "") {
    const head = document.createElement("div");
    head.className = "flex items-center justify-between gap-2 text-slate-200 text-sm";
    const text = document.createElement("span");
    text.textContent = this.on ? "Tools are on" : "Tools are off";
    const switchButton = document.createElement("button");
    switchButton.type = "button";
    switchButton.setAttribute("role", "switch");
    switchButton.setAttribute("aria-checked", String(this.on));
    switchButton.disabled = !this.canManage;
    switchButton.className = `relative h-6 w-11 shrink-0 rounded-full transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${this.on ? "bg-brand-600" : "bg-slate-700"}`;
    const knob = document.createElement("span");
    knob.className = `absolute left-1 top-1 h-4 w-4 rounded-full bg-white transition-transform ${this.on ? "translate-x-5" : ""}`;
    switchButton.append(knob);
    switchButton.addEventListener("click", () => this.flip());
    head.append(text, switchButton);
    const note = document.createElement("p");
    note.className = "text-slate-500";
    note.textContent =
      message || (this.canManage ? "The model may call these during its reply." : "Only a channel manager can change this.");
    const items = this.tools.map((tool) => {
      const item = document.createElement("div");
      const name = document.createElement("p");
      name.className = "font-medium text-slate-300";
      name.textContent = tool.name;
      const description = document.createElement("p");
      description.className = "text-slate-500 line-clamp-2";
      description.textContent = tool.description;
      item.append(name, description);
      return item;
    });
    this.popover.replaceChildren(head, note, ...items);
  }

  async flip() {
    const params = { ...this.params, use_tools: !this.on };
    const id = this.conversationId;
    let message = "";
    try {
      const conversation = await api(`/api/conversations/${id}`, { method: "PATCH", body: JSON.stringify({ params }) });
      if (this.conversationId === id) this.params = conversation.params;
    } catch (error) {
      message = error.message;
    }
    this.paint();
    this.drawPopover(message);
  }
}

const toolToggle = new ToolToggle(document.getElementById("tools-btn"));
