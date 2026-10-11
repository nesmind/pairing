/**
 * The Nodes map controller. Which picture is drawn depends on how many servers there are:
 *   up to 6    - the detailed diagram            (cluster_map_graph.js, boxes with names and badges)
 *   7 to 50    - the compact ring                (cluster_map_graph.js, small discs, hover for the name)
 *   51 to 500  - the dot map under a summary     (cluster_map_dots.js, one canvas, zoom and pan)
 *   over 500   - the same, the summary listing only what needs attention (cluster_map_dashboard.js)
 * The data is polled compact (no per-node details: GET /api/cluster/map?compact=true) and a node's full
 * record is fetched when it is clicked, so a big cluster costs one small response, not one per server.
 * Add ?demo=N to the page address to see any size without the servers (made-up, slowly changing data).
 */

const MAP_REFRESH_MS = 4000;
const MAP_MOTION_KEY = "pairingClusterMapMotion";
const MAP_TIERS = [
  { name: "detail", max: 6, label: "Detailed map" },
  { name: "ring", max: 50, label: "Compact map" },
  { name: "dots", max: 500, label: "Dot map" },
  { name: "summary", max: Infinity, label: "Summary and dot map" },
];

function mapTierOf(servers) {
  return MAP_TIERS.find((tier) => servers <= tier.max);
}

/** Made-up cluster of `count` servers for ?demo=N: mostly healthy, a few warning, a few down, changing a little each call. */
function demoMap(count, step) {
  const rng = (i) => {
    let t = (i * 2654435761 + Math.floor(step / 3) * 40503) >>> 0;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const infra = (id, kind, name, subtitle) => ({ id, kind, name, subtitle, status: "up", badges: [], address: null, activatable: false, removable: false, details: {}, checks: [] });
  const nodes = [infra("app", "app", "pAIring", "control"), infra("storage", "storage", "Shared storage", "models + chat caches")];
  const servers = [];
  for (let i = 0; i < count; i += 1) {
    const roll = rng(i);
    const status = i < 3 ? "up" : roll < 0.03 ? "down" : roll < 0.1 ? "warn" : "up";
    const badges = i === 0 ? ["seed", "active", "voter", "leader", "pulls", "cleans"] : i < 3 ? ["joiner", "active", "voter"] : status === "down" ? ["joiner"] : ["joiner", "active"];
    const address = `http://10.${Math.floor(i / 62500) % 250}.${Math.floor(i / 250) % 250}.${(i % 250) + 1}:8420`;
    const check = { id: "models", status: status === "up" ? "ok" : "action_needed", what: "models: demo", why: "demo data" };
    servers.push({ id: address, kind: "matricxon", name: "Matricxon", subtitle: address.replace("http://", ""), status, badges, address, activatable: false, removable: i >= 3, details: { Address: address, "Cluster role": i === 0 ? "seed" : "joiner", Demo: "made-up data" }, checks: [check] });
  }
  const links = [];
  const voters = servers.slice(0, Math.min(3, servers.length));
  voters.slice(1).forEach((v) => links.push({ source: voters[0].id, target: v.id, kind: "consensus", active: true, leader: true }));
  if (count <= 6) {
    for (const s of servers) {
      links.push({ source: "app", target: s.id, kind: "api", active: s.status !== "down", leader: false });
      links.push({ source: s.id, target: "storage", kind: "storage", active: s.status === "up", leader: false });
    }
  }
  return { enabled: true, supported: true, nodes: [...nodes, ...servers], links };
}

class ClusterMap {
  constructor(panel) {
    this.panel = panel;
    this.demo = Number(new URLSearchParams(window.location.search).get("demo")) || 0;
    this.demoStep = 0;
    this.svg = document.getElementById("cluster-map");
    this.canvas = document.getElementById("cluster-map-canvas");
    this.emptyEl = document.getElementById("cluster-map-empty");
    this.tierEl = document.getElementById("cluster-map-tier");
    this.dashEl = document.getElementById("cluster-map-dashboard");
    this.graph = new GraphRenderer(this.svg, this);
    this.dots = new DotRenderer(this.canvas, document.getElementById("cluster-map-tooltip"), this);
    this.dashboard = new ClusterDashboard(this.dashEl, this);
    this.tier = null;
    this.map = null;
    this.timer = null;
    this.active = false;
    this.filter = { text: "", status: "" };
    this._wireToolbar();
    this._renderLegend();
  }

  // ---- live data ---------------------------------------------------------------------------

  setActive(active) {
    this.active = active;
    clearInterval(this.timer);
    this.timer = null;
    if (!active) return;
    this.load();
    if (!document.hidden) this.timer = setInterval(() => this.load(), MAP_REFRESH_MS);
  }

  async load() {
    try {
      const map = this.demo ? demoMap(this.demo, this.demoStep++) : await api("/api/cluster/map?compact=true");
      this.update(map);
      await this._refreshPanel();
    } catch (err) {
      this.emptyEl.textContent = err.message;
      this.emptyEl.classList.remove("hidden");
    }
  }

  update(map) {
    this.map = map;
    document.getElementById("cluster-map-quorum").classList.toggle("hidden", map.quorum !== "lost");
    const outside = document.getElementById("cluster-map-outside");  // listed hosts that are not in the cluster
    outside.textContent = (map.outside || []).length
      ? (map.enabled ? "Not in the cluster (not drawn): " : "Cluster mode is off. Hosts, used as plain hosts: ") +
        `${map.outside.join(", ")}. ` + (map.enabled ? "Add them under Add nodes." : "")
      : "";
    outside.classList.toggle("hidden", !(map.outside || []).length);
    clusterLock.lost = map.quorum === "lost";
    _clusterLockAdd();
    const servers = map.nodes.filter((n) => n.kind === "matricxon" || n.kind === "voter").length;
    const tier = mapTierOf(servers);
    if (!map.supported || map.nodes.length === 0) {
      this.emptyEl.textContent = "Nothing to show: cluster mode is for the Matricxon engine.";
      this.emptyEl.classList.remove("hidden");
    } else {
      this.emptyEl.classList.add("hidden");
    }
    if (!this.tier || tier.name !== this.tier.name) this._switchTier(tier);
    this.tierEl.textContent = `${tier.label} · ${servers} server${servers === 1 ? "" : "s"}${this.demo ? " · demo data" : ""}`;
    if (tier.name === "detail" || tier.name === "ring") this.graph.render(map, tier.name === "ring");
    else this.dots.render(map);
    if (tier.name === "dots" || tier.name === "summary") this.dashboard.render(map);
  }

  _switchTier(tier) {
    const vector = tier.name === "detail" || tier.name === "ring";
    this.svg.classList.toggle("hidden", !vector);
    this.canvas.classList.toggle("hidden", vector);
    this.dashEl.classList.toggle("hidden", vector);
    if (vector) this.dots.clear();
    else this.graph.clear();
    this.tier = tier;
    this._syncMotion();
  }

  // ---- search, filter, motion --------------------------------------------------------------

  hasFilter() {
    return Boolean(this.filter.text || this.filter.status);
  }

  matches(node) {
    if (this.filter.status && node.status !== this.filter.status) return false;
    const text = this.filter.text;
    if (!text) return true;
    return [node.name, node.subtitle, node.address || "", node.id, node.badges.join(" ")].join(" ").toLowerCase().includes(text);
  }

  _applyFilter() {
    if (!this.map) return;
    this.graph.applyFilter();
    this.dots.applyFilter();
    if (this.tier && !this.tier.name.match(/^(detail|ring)$/)) this.dashboard.render(this.map);
  }

  _wireToolbar() {
    const search = document.getElementById("cluster-map-search");
    const status = document.getElementById("cluster-map-status");
    let pending = null;
    search.addEventListener("input", () => {
      clearTimeout(pending);
      pending = setTimeout(() => {
        this.filter.text = search.value.trim().toLowerCase();
        this._applyFilter();
      }, 150);
    });
    status.addEventListener("change", () => {
      this.filter.status = status.value;
      this._applyFilter();
    });
    const motion = document.getElementById("cluster-map-motion");
    const reduced = Boolean(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    let saved = false;
    try {
      saved = localStorage.getItem(MAP_MOTION_KEY) === "1";
    } catch (_err) {
      /* storage blocked: motion stays off */
    }
    motion.checked = saved && !reduced;
    motion.disabled = reduced;
    motion.addEventListener("change", () => {
      try {
        localStorage.setItem(MAP_MOTION_KEY, motion.checked ? "1" : "0");
      } catch (_err) {
        /* not remembered */
      }
      this._syncMotion();
    });
    this._syncMotion();
  }

  /** Animation is off unless asked for, only in the detailed map, and slow (see the page's CSS): a moving
   * SVG repaints every frame, which is what used to take a laptop to 100% CPU. */
  _syncMotion() {
    const on = document.getElementById("cluster-map-motion").checked && this.tier && this.tier.name === "detail";
    this.svg.classList.toggle("mx-motion", Boolean(on));
  }

  // ---- one node's panel --------------------------------------------------------------------

  async openNode(node) {
    this.panel.open(node);
    await this._refreshPanel(node);
  }

  async _refreshPanel(node = null) {
    const id = node ? node.id : this.panel.currentId;
    if (!id || !this.panel.isOpen) return;
    try {
      const full = this.demo ? this._demoRecord(id) : await api(`/api/cluster/map/node?id=${encodeURIComponent(id)}`);
      this.panel.refresh(full);
    } catch (err) {
      this.panel.fail(err.message);
    }
  }

  _demoRecord(id) {
    const node = (this.map ? this.map.nodes : []).find((n) => n.id === id);
    if (!node) throw new Error("This node is gone.");
    return { ...node, details: { ...node.details, Status: node.status } };
  }

  _renderLegend() {
    const dot = (color, text) => `<span class="flex items-center gap-1.5"><span class="h-2.5 w-2.5 rounded-full" style="background:${color}"></span>${text}</span>`;
    const line = (color, text) => `<span class="flex items-center gap-1.5"><span class="inline-block h-0 w-5 border-t-2 border-dashed" style="border-color:${color}"></span>${text}</span>`;
    document.getElementById("cluster-map-legend").innerHTML =
      dot(MAP_STATUS_COLOR.up, "running") + dot(MAP_STATUS_COLOR.warn, "needs attention") + dot(MAP_STATUS_COLOR.down, "not responding") + dot(MAP_OUTSIDE_COLOR, "not in cluster") +
      line("#38bdf8", "voters") + line(MAP_LEADER_COLOR, "leader");
  }
}

const clusterMap = new ClusterMap(clusterNodePanel);
document.addEventListener("visibilitychange", () => {
  if (clusterMap.active) clusterMap.setActive(true);
});
