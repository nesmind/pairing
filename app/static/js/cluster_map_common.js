/**
 * Shared pieces of the Nodes map renderers: sizes, colours, icons, a tiny SVG helper. Loaded first.
 */

const QUORUM_LOCK_TIP = "No quorum: a majority of the voters is not reachable. Restore it first.";
const clusterLock = { lost: false }; // set from every poll; the forms and buttons that change membership obey it

const MAP_W = 1000;
const MAP_H = 620;
const MAP_POSITIONS_KEY = "pairingClusterMapPositions";
const MAP_STATUS_COLOR = { up: "#22c55e", warn: "#f59e0b", down: "#ef4444" };
const MAP_OUTSIDE_COLOR = "#64748b";  // a host that is not in the cluster (yet): neither running nor failing

/** Ring/dot colour of a node: grey while it is not in the cluster, else by status. */
function mapNodeColor(node) {
  if (node.badges.includes("not in cluster")) return MAP_OUTSIDE_COLOR;
  return MAP_STATUS_COLOR[node.status] || MAP_STATUS_COLOR.down;
}
const MAP_LEADER_COLOR = "#f8fafc"; // not amber: amber already means "needs attention"
const MAP_BADGE_COLOR = { leader: MAP_LEADER_COLOR, voter: "#38bdf8", seed: "#22c55e", joiner: "#2dd4bf", active: "#94a3b8", learner: "#64748b", pulls: "#a78bfa", cleans: "#f472b6", "no quorum": "#ef4444", "voter only": "#c084fc", "not in cluster": "#f59e0b" };
const SVG_NS = "http://www.w3.org/2000/svg";

/** Icons drawn around (0,0), about 30 units wide: [tag, attributes] pairs. */
const MAP_ICONS = {
  app: [["rect", { x: -14, y: -12, width: 28, height: 18, rx: 2 }], ["path", { d: "M-8 14h16M0 6v8" }]],
  matricxon: [
    ["rect", { x: -13, y: -14, width: 26, height: 11, rx: 2 }],
    ["rect", { x: -13, y: 2, width: 26, height: 11, rx: 2 }],
    ["path", { d: "M-8 -8.5h.01M-8 7.5h.01" }],
  ],
  storage: [
    ["ellipse", { cx: 0, cy: -10, rx: 13, ry: 5 }],
    ["path", { d: "M-13 -10v18c0 3 6 5 13 5s13-2 13-5v-18M-13 -1c0 3 6 5 13 5s13-2 13-5" }],
  ],
  image: [
    ["rect", { x: -14, y: -12, width: 28, height: 24, rx: 3 }],
    ["circle", { cx: -6, cy: -4, r: 3 }],
    ["path", { d: "M-14 8l8-8 6 6 5-4 9 9" }],
  ],
  voter: [["path", { d: "M0 -14l12 7v14l-12 7-12-7v-14z" }]],
  // A voter-only node (badge "voter only"): the voter hexagon with a tick, no server rack.
  voteronly: [["path", { d: "M0 -14l12 7v14l-12 7-12-7v-14z" }], ["path", { d: "M-5 0l4 4 7-8" }]],
};

function svgEl(name, attrs = {}, parent = null) {
  const el = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  if (parent) parent.appendChild(el);
  return el;
}

const STATUS_ORDER = { down: 0, warn: 1, up: 2 };

/** pAIring is a diamond and the shared storage a hexagon (servers are circles), in small maps and dots. */
const MAP_SHAPES = {
  app: [[0, -1.25], [1.25, 0], [0, 1.25], [-1.25, 0]],
  storage: [[0, -1.2], [1.04, -0.6], [1.04, 0.6], [0, 1.2], [-1.04, 0.6], [-1.04, -0.6]],
};

function mapShapePoints(kind, r) {
  return MAP_SHAPES[kind] ? MAP_SHAPES[kind].map(([x, y]) => `${x * r},${y * r}`).join(" ") : null;
}

/** Adds the outline of a node to a canvas path: a circle, or the shape of its kind (images: a square). */
function mapShapePath(ctx, kind, x, y, r) {
  const shape = MAP_SHAPES[kind];
  if (shape) {
    shape.forEach(([sx, sy], i) => (i ? ctx.lineTo(x + sx * r, y + sy * r) : ctx.moveTo(x + sx * r, y + sy * r)));
    ctx.closePath();
  } else if (kind === "image") {
    ctx.rect(x - r, y - r, 2 * r, 2 * r);
  } else {
    ctx.arc(x, y, r, 0, 2 * Math.PI);
  }
}
