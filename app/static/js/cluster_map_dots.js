/**
 * The dot map of the Nodes map, for clusters of ~50 to thousands of servers: one canvas, one dot per server
 * in a grid (leader and voters first, then problems, then the healthy ones), so what needs attention is
 * always in the same corner. Wheel to zoom, drag to pan, hover for the name, click for the panel. Drawn only
 * when something changes (data, filter, hover, zoom): there is no animation loop, so it costs nothing at rest.
 */

const DOT_COLORS = { up: "#22c55e", warn: "#f59e0b", down: "#ef4444" };
const DOT_RING = { leader: MAP_LEADER_COLOR, voter: "#38bdf8" };
const DOT_SQUARE_KINDS = new Set(["app", "storage", "image"]);
const DOT_STATUS_ORDER = { down: 0, warn: 1, up: 2 };

class DotRenderer {
  constructor(canvas, tooltip, controller) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.tooltip = tooltip;
    this.controller = controller;
    this.items = [];
    this.layout = { cols: 1, rows: 1, cell: 20, ox: 0, oy: 0, radius: 6 };
    this.view = { scale: 1, tx: 0, ty: 0 };
    this.hover = null;
    this.scheduled = false;
    this.width = 0;
    this.height = 0;
    this._bind();
    new ResizeObserver(() => this._resize()).observe(canvas.parentElement);
  }

  render(map) {
    const first = this.items.length === 0;
    const order = (n) => (n.badges.includes("leader") ? -2 : n.badges.includes("voter") ? -1 : DOT_STATUS_ORDER[n.status] ?? 3);
    this.items = [...map.nodes]
      .sort((a, b) => order(a) - order(b) || a.subtitle.localeCompare(b.subtitle))
      .map((data) => ({ data, x: 0, y: 0 }));
    this._layout();
    if (first) this.view = { scale: 1, tx: 0, ty: 0 };
    this.draw();
  }

  applyFilter() {
    this.draw();
  }

  clear() {
    this.items = [];
    this.tooltip.classList.add("hidden");
    this.draw();
  }

  // ---- layout and drawing ------------------------------------------------------------------

  _layout() {
    const n = Math.max(1, this.items.length);
    const w = Math.max(100, this.width);
    const h = Math.max(100, this.height);
    const cols = Math.max(1, Math.ceil(Math.sqrt((n * w) / h)));
    const rows = Math.ceil(n / cols);
    const cell = Math.max(4, Math.min(w / cols, h / rows));
    const ox = (w - cols * cell) / 2;
    const oy = Math.max(4, (h - rows * cell) / 2);
    this.layout = { cols, rows, cell, ox, oy, radius: Math.max(1.6, Math.min(11, cell * 0.36)) };
    this.items.forEach((item, i) => {
      item.x = ox + ((i % cols) + 0.5) * cell;
      item.y = oy + (Math.floor(i / cols) + 0.5) * cell;
    });
  }

  _resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const dpr = window.devicePixelRatio || 1;
    this.width = rect.width;
    this.height = rect.height;
    this.canvas.width = Math.round(rect.width * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height}px`;
    this._layout();
    this.draw();
  }

  draw() {
    if (this.scheduled) return;
    this.scheduled = true;
    requestAnimationFrame(() => {
      this.scheduled = false;
      this._paint();
    });
  }

  _paint() {
    const { ctx, view, layout } = this;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.translate(view.tx, view.ty);
    ctx.scale(view.scale, view.scale);
    const r = layout.radius;
    const left = -view.tx / view.scale - layout.cell;
    const top = -view.ty / view.scale - layout.cell;
    const right = left + this.width / view.scale + 2 * layout.cell;
    const bottom = top + this.height / view.scale + 2 * layout.cell;
    const labels = layout.cell * view.scale >= 44;
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillStyle = getComputedStyle(this.canvas).color;
    for (const item of this.items) {
      if (item.x < left || item.x > right || item.y < top || item.y > bottom) continue;
      const { data } = item;
      ctx.globalAlpha = this.controller.matches(data) ? 1 : 0.12;
      const color = mapNodeColor(data);
      const size = item === this.hover ? r * 1.35 : r;
      ctx.fillStyle = color;
      ctx.beginPath();
      mapShapePath(ctx, data.kind, item.x, item.y, size);
      ctx.fill();
      const ring = data.badges.includes("leader") ? DOT_RING.leader : data.badges.includes("voter") ? DOT_RING.voter : null;
      if (ring && size >= 3) {
        ctx.strokeStyle = ring;
        ctx.lineWidth = Math.max(1.2, size * 0.28);
        ctx.beginPath();
        ctx.arc(item.x, item.y, size + Math.max(2, size * 0.3), 0, 2 * Math.PI);
        ctx.stroke();
      }
      if (labels && ctx.globalAlpha === 1) {
        ctx.fillStyle = getComputedStyle(this.canvas).color;
        ctx.fillText(this._label(data, Math.max(6, Math.floor(layout.cell / 5.6))), item.x, item.y + size + 12);
      }
    }
    ctx.globalAlpha = 1;
  }

  /** What is written under a dot: the address without the usual port (or the name, for pAIring/storage), cut
   * with an ellipsis rather than in the middle of a number. */
  _label(data, fit) {
    const text = DOT_SQUARE_KINDS.has(data.kind) ? data.name : data.subtitle.replace(/:8420$/, "");
    return text.length > fit ? `${text.slice(0, fit - 1)}…` : text;
  }

  // ---- pointer: hover, pan, zoom, click ----------------------------------------------------

  _bind() {
    const canvas = this.canvas;
    let drag = null;
    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      drag = { x: event.clientX, y: event.clientY, tx: this.view.tx, ty: this.view.ty, moved: false };
    });
    canvas.addEventListener("pointermove", (event) => {
      if (drag) {
        const dx = event.clientX - drag.x;
        const dy = event.clientY - drag.y;
        if (drag.moved || Math.hypot(dx, dy) > 5) {
          drag.moved = true;
          this.view.tx = drag.tx + dx;
          this.view.ty = drag.ty + dy;
          this.draw();
          return;
        }
      }
      this._hover(event);
    });
    canvas.addEventListener("pointerup", (event) => {
      if (drag && !drag.moved) {
        const item = this._itemAt(event);
        if (item) this.controller.openNode(item.data);
      }
      drag = null;
    });
    canvas.addEventListener("pointerleave", () => this._setHover(null, null));
    canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const mx = event.clientX - rect.left;
        const my = event.clientY - rect.top;
        const factor = event.deltaY < 0 ? 1.2 : 1 / 1.2;
        const scale = Math.min(40, Math.max(0.6, this.view.scale * factor));
        const real = scale / this.view.scale;
        this.view.tx = mx - (mx - this.view.tx) * real;
        this.view.ty = my - (my - this.view.ty) * real;
        this.view.scale = scale;
        this.draw();
      },
      { passive: false },
    );
    canvas.addEventListener("dblclick", () => {
      this.view = { scale: 1, tx: 0, ty: 0 };
      this.draw();
    });
  }

  _itemAt(event) {
    const rect = this.canvas.getBoundingClientRect();
    const wx = (event.clientX - rect.left - this.view.tx) / this.view.scale;
    const wy = (event.clientY - rect.top - this.view.ty) / this.view.scale;
    const { cols, cell, ox, oy, radius } = this.layout;
    const col = Math.floor((wx - ox) / cell);
    const row = Math.floor((wy - oy) / cell);
    if (col < 0 || col >= cols || row < 0) return null;
    const item = this.items[row * cols + col];
    if (!item) return null;
    return Math.hypot(item.x - wx, item.y - wy) <= Math.max(radius * 1.5, cell * 0.5) ? item : null;
  }

  _hover(event) {
    const item = this._itemAt(event);
    this._setHover(item, event);
  }

  _setHover(item, event) {
    if (item !== this.hover) {
      this.hover = item;
      this.draw();
    }
    if (!item || !event) {
      this.tooltip.classList.add("hidden");
      return;
    }
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const { data } = item;
    this.tooltip.textContent = `${data.name} · ${data.subtitle} · ${data.status}${data.badges.length ? " · " + data.badges.join(", ") : ""}`;
    this.tooltip.style.left = `${Math.min(rect.width - 260, event.clientX - rect.left + 14)}px`;
    this.tooltip.style.top = `${Math.max(4, event.clientY - rect.top + 14)}px`;
    this.tooltip.classList.remove("hidden");
  }
}
