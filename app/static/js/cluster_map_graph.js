/**
 * The SVG renderer of the Nodes map, for up to ~50 servers: the detailed diagram (boxes with names, status
 * lights, badges, arcs between voters) and, from 11 servers, the compact ring layout (small discs, no labels
 * until you hover). Drag to rearrange (remembered per browser). No animation unless "Motion" is on, and then
 * only at a low frame rate: continuous SVG animation is what used to take a laptop to 100% CPU.
 */

class GraphRenderer {
  constructor(svg, controller) {
    this.svg = svg;
    this.controller = controller;
    this.compact = false;
    this.nodes = new Map(); // id -> { data, x, y, el }
    this.links = new Map(); // key -> { data, el }
    this.positions = this._loadPositions();
    svg.setAttribute("viewBox", `0 0 ${MAP_W} ${MAP_H}`);
    this.linkLayer = svgEl("g", {}, svg);
    this.nodeLayer = svgEl("g", {}, svg);
    this.hoverLabel = svgEl("text", { class: "mx-hover fill-slate-100 stroke-slate-950", "font-size": 13, "font-weight": 600, "paint-order": "stroke", "stroke-width": 5, "text-anchor": "middle", "pointer-events": "none" }, svg);
  }

  /** compact = the ring layout for 11-50 servers. Switching clears the drawing (the layouts differ). */
  render(map, compact) {
    if (compact !== this.compact) {
      this.clear();
      this.compact = compact;
    }
    const layout = compact ? MapLayouts.ring(map.nodes) : MapLayouts.row(map.nodes);
    const seen = new Set(map.nodes.map((n) => n.id));
    for (const [id, node] of this.nodes) {
      if (!seen.has(id)) {
        node.el.remove();
        this.nodes.delete(id);
      }
    }
    for (const data of map.nodes) this._upsertNode(data, layout[data.id]);
    this._upsertLinks(compact ? this._ringLinks(map) : map.links);
    this.applyFilter();
  }

  /** In the ring, pAIring and the storage get one dashed line each to the leader (a line per server would
   * bury the ring): they belong to the cluster, they are not part of it. */
  _ringLinks(map) {
    const servers = map.nodes.filter((n) => n.kind === "matricxon" || n.kind === "voter");
    const hub = servers.find((n) => n.badges.includes("leader")) || servers[0];
    const own = map.links.filter((l) => l.kind === "consensus");
    if (!hub) return own;
    return [...own, ...["app", "storage"].map((id) => ({ source: id, target: hub.id, kind: "affinity", active: true, leader: false }))];
  }

  clear() {
    this.nodeLayer.replaceChildren();
    this.linkLayer.replaceChildren();
    this.nodes.clear();
    this.links.clear();
  }

  _loadPositions() {
    try {
      return JSON.parse(localStorage.getItem(MAP_POSITIONS_KEY) || "{}");
    } catch (_err) {
      return {}; // private mode / blocked storage: the default layout is used
    }
  }

  _savePositions() {
    try {
      localStorage.setItem(MAP_POSITIONS_KEY, JSON.stringify(this.positions));
    } catch (_err) {
      /* nothing to do: positions just are not remembered */
    }
  }

  // ---- nodes -------------------------------------------------------------------------------

  _upsertNode(data, fallback) {
    let node = this.nodes.get(data.id);
    if (!node) {
      const saved = this.compact ? null : this.positions[data.id];
      const where = saved || fallback || { x: 500, y: 300 };
      node = { data, x: where.x, y: where.y, el: this._createNodeEl(data.kind) };
      this.nodes.set(data.id, node);
      this._wireNode(node);
    }
    node.data = data;
    this._paintNode(node);
  }

  _createNodeEl(kind) {
    const radius = this.compact ? 15 : 32;
    const g = svgEl("g", { class: "mx-node" }, this.nodeLayer);
    const points = this.compact ? mapShapePoints(kind, radius) : null;
    const shape = (attrs) => (points ? svgEl("polygon", { points, "stroke-linejoin": "round", ...attrs }, g) : svgEl("circle", { r: radius, ...attrs }, g));
    shape({ class: "mx-body fill-slate-900" });
    shape({ class: "mx-ring", fill: "none", "stroke-width": this.compact ? 2.5 : 3 });
    if (this.compact) {
      svgEl("circle", { class: "mx-lead", r: radius + 5, fill: "none", stroke: MAP_LEADER_COLOR, "stroke-width": 2, display: "none" }, g);
      svgEl("circle", { class: "mx-vote", cx: 11, cy: -11, r: 4, fill: MAP_BADGE_COLOR.voter, display: "none" }, g);
      svgEl("title", {}, g);
      return g;
    }
    const icon = svgEl("g", { class: "mx-icon stroke-slate-300", fill: "none", "stroke-width": 1.8, "stroke-linecap": "round", "stroke-linejoin": "round" }, g);
    for (const [tag, attrs] of MAP_ICONS[kind] || MAP_ICONS.matricxon) svgEl(tag, attrs, icon);
    svgEl("circle", { class: "mx-led stroke-slate-950", cx: 23, cy: -23, r: 6, "stroke-width": 2 }, g);
    // paint-order puts a background-coloured outline behind the letters, so a link passing behind a label
    // does not cut through it.
    const halo = { "stroke-width": 5, "stroke-linejoin": "round", "paint-order": "stroke", "text-anchor": "middle" };
    svgEl("text", { class: "mx-name fill-slate-200 stroke-slate-950", y: 55, "font-size": 13, "font-weight": 600, ...halo }, g);
    svgEl("text", { class: "mx-sub fill-slate-500 stroke-slate-950", y: 71, "font-size": 11, ...halo }, g);
    svgEl("g", { class: "mx-badges", transform: "translate(0 85)" }, g);
    svgEl("title", {}, g);
    return g;
  }

  _paintNode(node) {
    const { data, el } = node;
    el.setAttribute("transform", `translate(${node.x} ${node.y})`);
    el.setAttribute("class", `mx-node mx-${data.status}`);
    const color = mapNodeColor(data);
    el.querySelector(".mx-ring").setAttribute("stroke", color);
    // A voter-only node: dashed ring, and (full size) a hexagon-with-tick icon instead of the server rack.
    const voterOnly = data.badges.includes("voter only");
    const ring = el.querySelector(".mx-ring");
    if (voterOnly) ring.setAttribute("stroke-dasharray", "6 4");
    else ring.removeAttribute("stroke-dasharray");
    const iconEl = el.querySelector(".mx-icon");
    if (iconEl && iconEl.dataset.kind !== (voterOnly ? "voteronly" : data.kind)) {
      iconEl.dataset.kind = voterOnly ? "voteronly" : data.kind;
      iconEl.replaceChildren();
      for (const [tag, attrs] of MAP_ICONS[iconEl.dataset.kind] || MAP_ICONS.matricxon) svgEl(tag, attrs, iconEl);
    }
    el.querySelector("title").textContent = `${data.name} ${data.subtitle} - ${data.status}`;
    if (this.compact) {
      el.querySelector(".mx-lead").setAttribute("display", data.badges.includes("leader") ? "" : "none");
      el.querySelector(".mx-vote").setAttribute("display", data.badges.includes("voter") ? "" : "none");
      el.querySelector(".mx-body").setAttribute("fill-opacity", data.status === "down" ? 0.5 : 1);
      return;
    }
    el.querySelector(".mx-led").setAttribute("fill", color);
    el.querySelector(".mx-name").textContent = data.name;
    el.querySelector(".mx-sub").textContent = data.subtitle;
    this._paintBadges(el.querySelector(".mx-badges"), data.badges);
  }

  _paintBadges(group, badges) {
    const key = badges.join("|");
    if (group.dataset.key === key) return;
    group.dataset.key = key;
    group.replaceChildren();
    // Rows of at most three, each centred, so a node with many badges stays inside the map.
    for (let first = 0, row = 0; first < badges.length; first += 3, row += 1) {
      const rowBadges = badges.slice(first, first + 3);
      const widths = rowBadges.map((b) => b.length * 6.4 + 12);
      let x = -(widths.reduce((a, b) => a + b + 4, -4) / 2);
      const y = row * 21;
      rowBadges.forEach((badge, i) => {
        const color = MAP_BADGE_COLOR[badge] || "#94a3b8";
        svgEl("rect", { x, y: y - 9, width: widths[i], height: 17, rx: 8, fill: color, "fill-opacity": 0.18, stroke: color, "stroke-opacity": 0.6 }, group);
        const label = svgEl("text", { x: x + widths[i] / 2, y: y + 3, "text-anchor": "middle", "font-size": 9.5, "font-weight": 600, fill: color }, group);
        label.textContent = badge.toUpperCase();
        x += widths[i] + 4;
      });
    }
  }

  _wireNode(node) {
    let start = null;
    node.el.addEventListener("pointerdown", (event) => {
      node.el.setPointerCapture(event.pointerId);
      start = { x: event.clientX, y: event.clientY, moved: false };
    });
    node.el.addEventListener("pointermove", (event) => {
      if (!start) return;
      if (!start.moved && Math.hypot(event.clientX - start.x, event.clientY - start.y) < 5) return;
      start.moved = true;
      const point = this._toMap(event);
      node.x = Math.min(MAP_W - 30, Math.max(30, point.x));
      node.y = Math.min(MAP_H - (this.compact ? 30 : 100), Math.max(30, point.y));
      this._paintNode(node);
      this._moveLinks();
    });
    node.el.addEventListener("pointerup", () => {
      if (start && start.moved) {
        if (!this.compact) {
          this.positions[node.data.id] = { x: node.x, y: node.y };
          this._savePositions();
        }
      } else if (start) {
        this.controller.openNode(node.data);
      }
      start = null;
    });
    node.el.addEventListener("pointerenter", () => this._focus(node));
    node.el.addEventListener("pointerleave", () => this._focus(null));
  }

  _toMap(event) {
    const point = this.svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    return point.matrixTransform(this.svg.getScreenCTM().inverse());
  }

  /** Hovering a node fades everything not connected to it (and names it, in the compact layout). */
  _focus(hovered) {
    const id = hovered ? hovered.data.id : null;
    const linked = new Set([id]);
    for (const { data } of this.links.values()) {
      if (data.source === id) linked.add(data.target);
      if (data.target === id) linked.add(data.source);
    }
    for (const [nodeId, node] of this.nodes) node.el.classList.toggle("mx-dim", id !== null && !linked.has(nodeId));
    for (const { data, el } of this.links.values()) {
      el.classList.toggle("mx-dim", id !== null && data.source !== id && data.target !== id);
    }
    this.hoverLabel.textContent = hovered && this.compact ? `${hovered.data.name} ${hovered.data.subtitle}` : "";
    if (hovered && this.compact) {
      this.hoverLabel.setAttribute("x", Math.min(MAP_W - 110, Math.max(110, hovered.x)));
      this.hoverLabel.setAttribute("y", hovered.y - 26);
    }
  }

  /** The search box and status filter: nodes that do not match are faded. */
  applyFilter() {
    for (const node of this.nodes.values()) {
      node.el.classList.toggle("mx-nomatch", !this.controller.matches(node.data));
    }
  }

  // ---- links -------------------------------------------------------------------------------

  _upsertLinks(links) {
    const keys = new Set();
    for (const data of links) {
      const key = `${data.kind}|${data.source}|${data.target}`;
      keys.add(key);
      let link = this.links.get(key);
      if (!link) {
        link = { data, el: svgEl("path", { fill: "none" }, this.linkLayer) };
        this.links.set(key, link);
      }
      link.data = data;
      const color = data.leader ? MAP_LEADER_COLOR : data.kind === "consensus" ? "#38bdf8" : "#64748b";
      link.el.setAttribute("stroke", color);
      link.el.setAttribute("stroke-width", data.leader ? (this.compact ? 1.5 : 2.5) : data.kind === "consensus" ? 2 : 1.5);
      link.el.setAttribute("class", `mx-link mx-link-${data.kind} ${data.active ? "mx-active" : "mx-off"}`);
    }
    for (const [key, link] of this.links) {
      if (!keys.has(key)) {
        link.el.remove();
        this.links.delete(key);
      }
    }
    this._moveLinks();
  }

  _moveLinks() {
    for (const { data, el } of this.links.values()) {
      const a = this.nodes.get(data.source);
      const b = this.nodes.get(data.target);
      el.style.display = a && b ? "" : "none";
      if (a && b) el.setAttribute("d", this._path(a, b, data.kind === "consensus" && !this.compact));
    }
  }

  /** A straight line, or - for voters in the detailed map - an arc bowed upward (longer links higher), so
   * the links of a leader to nodes standing in a row do not run behind the nodes between them. */
  _path(a, b, bowed) {
    if (!bowed) return `M${a.x} ${a.y}L${b.x} ${b.y}`;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const length = Math.hypot(dx, dy) || 1;
    const sign = dx >= 0 ? -1 : 1; // the normal that points up
    const bow = Math.min(140, length * 0.28);
    const cx = (a.x + b.x) / 2 + (sign * dy * bow) / length;
    const cy = (a.y + b.y) / 2 - Math.abs((dx * bow) / length);
    return `M${a.x} ${a.y}Q${cx} ${cy} ${b.x} ${b.y}`;
  }
}
