/* The Active engine picker drawn as a list in two groups, Internal (engines pAIring runs or reaches on your servers)
and External (cloud connectors, each with a "Cloud" badge); a native <select> can't show either. The <select id="active-engine-select"> stays the source of truth, hidden: settings.js fills it (each option's
data-cloud) and reads it; picking here sets its value and fires its "change", so everything bound to it still runs. */
class EnginePicker {
  constructor(select) {
    this.select = select;
    this.activeIndex = -1;
    this.wrap = document.createElement("div");
    this.wrap.className = "relative inline-block";
    this.button = document.createElement("button");
    this.button.type = "button";
    this.button.title = select.title;
    this.button.setAttribute("aria-haspopup", "listbox");
    this.button.setAttribute("aria-expanded", "false");
    this.button.className =
      "flex min-w-[14rem] items-center justify-between gap-3 rounded-md border border-slate-700 bg-slate-950 " +
      "px-2.5 py-1.5 text-left text-sm focus:outline-none focus:border-brand-500";
    this.list = document.createElement("ul");
    this.list.setAttribute("role", "listbox");
    this.list.tabIndex = -1;
    this.list.hidden = true;
    this.list.className =
      "absolute left-0 z-30 mt-1 max-h-72 min-w-full overflow-y-auto rounded-md border border-slate-700 " +
      "bg-slate-800 p-1 text-sm text-slate-100 shadow-xl focus:outline-none";
    select.after(this.wrap);
    this.wrap.append(select, this.button, this.list);
    select.hidden = true;

    this.button.addEventListener("click", () => (this.list.hidden ? this.open() : this.close()));
    this.button.addEventListener("keydown", (e) => {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(e.key)) {
        e.preventDefault();
        this.open();
      }
    });
    this.list.addEventListener("keydown", (e) => this.onListKey(e));
    document.addEventListener("click", (e) => {
      if (!this.wrap.contains(e.target)) this.close();
    });
    // settings.js refills the options and sets the value in one go: redraw after it.
    new MutationObserver(() => this.render()).observe(select, { childList: true });
    select.addEventListener("change", () => this.render());
    this.render();
  }

  /* A native select with the same classes, measured off screen (the tab may be hidden at load): the button takes its
  height, so it lines up with the other select boxes in every browser. */
  static nativeHeight(select) {
    const probe = document.createElement("select");
    probe.className = select.className;
    probe.innerHTML = "<option>Engine</option>";
    probe.style.cssText = "position:absolute;left:-9999px;top:0;visibility:hidden";
    document.body.appendChild(probe);
    const height = probe.getBoundingClientRect().height;
    probe.remove();
    return height;
  }

  static badge() {
    const badge = document.createElement("span");
    badge.className =
      "engine-badge shrink-0 rounded-full border border-brand-500/40 bg-brand-500/10 px-1.5 py-0.5 text-[9px] " +
      "font-medium uppercase leading-none tracking-wide text-brand-400";
    badge.textContent = "Cloud";
    return badge;
  }

  static label(option) {
    const row = document.createElement("span");
    row.className = "flex min-w-0 items-center gap-2";
    const name = document.createElement("span");
    name.className = "truncate";
    name.textContent = option.textContent;
    row.appendChild(name);
    if (option.dataset.cloud === "true") row.appendChild(EnginePicker.badge());
    return row;
  }

  render() {
    // Measured on every draw, not once: the page's styles are built just after load (Tailwind in the browser).
    this.button.style.height = `${EnginePicker.nativeHeight(this.select)}px`;
    const options = [...this.select.options];
    const chosen = options.find((o) => o.value === this.select.value);
    const chevron = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    chevron.setAttribute("viewBox", "0 0 24 24");
    chevron.setAttribute("class", "h-4 w-4 shrink-0 text-slate-500");
    chevron.innerHTML = '<path d="m6 9 6 6 6-6" fill="none" stroke="currentColor" stroke-width="2"/>';
    this.button.replaceChildren(chosen ? EnginePicker.label(chosen) : document.createTextNode("—"), chevron);
    // External shows even when empty: connectors appear here once switched on (and tested) on the Connectors page.
    const groups = [
      ["Internal", options.filter((o) => o.dataset.cloud !== "true"), ""],
      ["External", options.filter((o) => o.dataset.cloud === "true"), "No connectors"],
    ];
    const rows = [];
    groups.forEach(([title, members, emptyText], g) => {
      if (!members.length && !emptyText) return;
      const head = document.createElement("li");
      head.setAttribute("role", "presentation");
      head.className =
        "px-2.5 pt-2 text-[10px] font-semibold uppercase tracking-wider text-warning-400" +
        (g > 0 ? " mt-1 border-t border-slate-700" : "");
      head.textContent = title;
      rows.push(head);
      for (const option of members) rows.push(this.item(option, options.indexOf(option)));
      if (!members.length) {
        const empty = document.createElement("li");
        empty.setAttribute("role", "presentation");
        empty.className = "px-2.5 py-1.5 text-xs text-slate-500";
        empty.textContent = emptyText;
        rows.push(empty);
      }
    });
    this.list.replaceChildren(...rows);
  }

  item(option, index) {
    const item = document.createElement("li");
    item.setAttribute("role", "option");
    item.setAttribute("aria-selected", String(option.value === this.select.value));
    item.dataset.index = String(index);
    item.className = "flex cursor-pointer items-center justify-between gap-3 rounded-md px-2.5 py-1.5";
    item.appendChild(EnginePicker.label(option));
    item.addEventListener("click", () => this.choose(index));
    item.addEventListener("mouseenter", () => this.highlight(this.items().indexOf(item)));
    return item;
  }

  /* The option rows in the order shown (groups can reorder the <select>'s options). */
  items() {
    return [...this.list.querySelectorAll("[role='option']")];
  }

  open() {
    this.list.hidden = false;
    this.button.setAttribute("aria-expanded", "true");
    const shown = this.items().findIndex((item) => Number(item.dataset.index) === this.select.selectedIndex);
    this.highlight(Math.max(0, shown));
    this.list.focus();
  }

  close() {
    if (this.list.hidden) return;
    this.list.hidden = true;
    this.button.setAttribute("aria-expanded", "false");
  }

  highlight(index) {
    const items = this.items();
    this.activeIndex = Math.max(0, Math.min(index, items.length - 1));
    // The highlighted row takes the selection colours, as in a native select; its Cloud badge turns white to stay
    // readable on them.
    items.forEach((item, i) => {
      const on = i === this.activeIndex;
      item.classList.toggle("bg-brand-600", on);
      item.classList.toggle("text-white", on);
      item.querySelector(".engine-badge")?.classList.toggle("engine-badge-on", on);
    });
    items[this.activeIndex]?.scrollIntoView({ block: "nearest" });
  }

  choose(index) {
    const option = this.select.options[index];
    this.close();
    this.button.focus();
    if (!option || option.value === this.select.value) return;
    this.select.value = option.value;
    this.select.dispatchEvent(new Event("change", { bubbles: true }));
  }

  onListKey(e) {
    const moves = { ArrowDown: 1, ArrowUp: -1 };
    if (e.key in moves) {
      e.preventDefault();
      this.highlight(this.activeIndex + moves[e.key]);
    } else if (e.key === "Home" || e.key === "End") {
      e.preventDefault();
      this.highlight(e.key === "Home" ? 0 : this.items().length - 1);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      const item = this.items()[this.activeIndex];
      if (item) this.choose(Number(item.dataset.index));
    } else if (e.key === "Escape" || e.key === "Tab") {
      this.close();
      if (e.key === "Escape") this.button.focus();
    }
  }
}

const activeEngineSelect = document.getElementById("active-engine-select");
if (activeEngineSelect) new EnginePicker(activeEngineSelect);
