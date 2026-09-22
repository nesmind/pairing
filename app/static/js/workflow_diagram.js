/**
 * D3-based rendering for the Stats page's "Visual chat workflow" diagram —
 * mechanics only, no descriptive content (see workflow_diagram_data.js
 * for WORKFLOW_NODES/WORKFLOW_EDGES). Lazily invoked via
 * initWorkflowDiagram(), called once by stats.js the first time "Visual
 * workflow" is actually clicked.
 *
 * Deliberately NOT a D3 force simulation — every node's x/y is fixed,
 * hand-laid-out data, since this is a real, fixed pipeline meant to
 * read as documentation, not an arbitrary graph that should jitter into
 * some equilibrium. Laid out top-to-bottom (vertical) and taller than
 * any viewport, so the wrapping container (#workflow-canvas-wrap, see
 * stats.html) scrolls it natively — no drag-to-pan/wheel-to-zoom, which
 * had no visible affordance telling a first-time viewer either was
 * needed.
 */

const WORKFLOW_ARROW_MARKER_ID = "wf-arrow";

/** A smooth cubic-bezier path between two nodes' centers, exiting/
 * entering whichever side faces the other node (left/right for a
 * mostly-horizontal edge, top/bottom for a mostly-vertical one) — every
 * edge in WORKFLOW_EDGES is one of these two shapes, so one function
 * covers the whole diagram instead of hand-picking d3.linkHorizontal()
 * only for the strictly-horizontal main chain. */
function workflowEdgePath(from, to) {
  const x1 = from.x + from.w / 2;
  const y1 = from.y + from.h / 2;
  const x2 = to.x + to.w / 2;
  const y2 = to.y + to.h / 2;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const horizontal = Math.abs(dx) >= Math.abs(dy);

  let sx, sy, ex, ey;
  if (horizontal) {
    sx = dx >= 0 ? from.x + from.w : from.x;
    sy = y1;
    ex = dx >= 0 ? to.x : to.x + to.w;
    ey = y2;
  } else {
    sx = x1;
    sy = dy >= 0 ? from.y + from.h : from.y;
    ex = x2;
    ey = dy >= 0 ? to.y : to.y + to.h;
  }

  const path = d3.path();
  path.moveTo(sx, sy);
  if (horizontal) {
    const cdx = (ex - sx) / 2;
    path.bezierCurveTo(sx + cdx, sy, ex - cdx, ey, ex, ey);
  } else {
    const cdy = (ey - sy) / 2;
    path.bezierCurveTo(sx, sy + cdy, ex, ey - cdy, ex, ey);
  }
  return path.toString();
}

/** Builds the "what does each border color mean" key shown pinned in
 * the corner of the diagram — one swatch + label per entry in
 * WORKFLOW_CATEGORY_COLORS/LABELS (see workflow_diagram_data.js), so
 * adding a new category there picks it up here automatically. */
function renderWorkflowLegend() {
  const legendEl = document.getElementById("workflow-legend");
  legendEl.innerHTML =
    `<p class="text-xs font-semibold uppercase tracking-wide text-slate-400 mb-0.5">Legend</p>` +
    Object.keys(WORKFLOW_CATEGORY_COLORS)
      .map((category) => {
        const color = WORKFLOW_CATEGORY_COLORS[category];
        const label = escapeHtml(WORKFLOW_CATEGORY_LABELS[category] || category);
        return `<div class="flex items-center gap-2">
          <span class="h-3 w-3 shrink-0 rounded-sm border" style="border-color:${color};background:${color}22"></span>
          <span class="text-slate-300">${label}</span>
        </div>`;
      })
      .join("");
}

/** The category label is rendered with CSS `uppercase` — a blind visual
 * transform that would also cap every letter of this app's own
 * stylized name (APP_NAME — see workflow_diagram_data.js, e.g. lowercase
 * p, capital AI, lowercase ring for the "pAIring" default) into all
 * caps. Escapes the label first, then wraps any APP_NAME substring in
 * text-transform:none so it renders with its real casing even under
 * the ancestor's uppercase class. */
function formatCategoryLabel(label) {
  return escapeHtml(label).replace(
    new RegExp(APP_NAME, "g"),
    `<span style="text-transform:none">${APP_NAME}</span>`
  );
}

function showWorkflowDetail(node) {
  document.getElementById("workflow-detail-content").innerHTML = `
    <div class="flex items-center justify-between gap-2 pr-7">
      <p class="text-xs font-semibold uppercase tracking-wide" style="color:${WORKFLOW_CATEGORY_COLORS[node.category]}">
        ${formatCategoryLabel(WORKFLOW_CATEGORY_LABELS[node.category] || node.category)}
      </p>
      <span class="shrink-0 rounded-full border border-slate-700 px-2 py-0.5 text-xs font-mono text-slate-300">
        ${escapeHtml(node.detail.server)}
      </span>
    </div>
    <h3 class="text-base font-semibold text-slate-100 mt-1">${escapeHtml(node.detail.title)}</h3>
    <p class="text-xs font-mono text-slate-500 mt-1">${escapeHtml(node.detail.location)}</p>
    <p class="text-sm text-slate-300 mt-3 leading-relaxed">${escapeHtml(node.detail.body)}</p>
  `;
  document.getElementById("workflow-detail-panel").classList.remove("hidden");

  d3.selectAll(".wf-node rect").attr("stroke-width", 1.5);
  d3.select(`.wf-node[data-id="${node.id}"] rect`).attr("stroke-width", 3);
}

/** The diagram's own content bounds, from the hand-laid-out node
 * positions — the <svg> is sized to exactly this (plus a margin), so
 * the wrapping container's native scrollbars appear precisely where
 * the content actually overflows, rather than the SVG being stretched
 * to fill whatever space happens to be available. */
function workflowCanvasBounds() {
  const margin = 40;
  const minX = Math.min(...WORKFLOW_NODES.map((n) => n.x));
  const minY = Math.min(...WORKFLOW_NODES.map((n) => n.y));
  const maxX = Math.max(...WORKFLOW_NODES.map((n) => n.x + n.w));
  const maxY = Math.max(...WORKFLOW_NODES.map((n) => n.y + n.h));
  return {
    width: maxX - minX + margin * 2,
    height: maxY - minY + margin * 2,
    offsetX: margin - minX,
    offsetY: margin - minY,
  };
}

function initWorkflowDiagram() {
  renderWorkflowLegend();

  const svg = d3.select("#workflow-svg");
  const { width, height, offsetX, offsetY } = workflowCanvasBounds();
  svg.attr("width", width).attr("height", height);

  svg
    .append("defs")
    .append("marker")
    .attr("id", WORKFLOW_ARROW_MARKER_ID)
    .attr("viewBox", "0 0 10 10")
    .attr("refX", 9)
    .attr("refY", 5)
    .attr("markerWidth", 7)
    .attr("markerHeight", 7)
    .attr("orient", "auto-start-reverse")
    .append("path")
    .attr("d", "M0,0 L10,5 L0,10 Z")
    .attr("fill", ThemeColors.get("--color-slate-500"));

  const root = svg.append("g").attr("class", "wf-root").attr("transform", `translate(${offsetX},${offsetY})`);

  const nodesById = new Map(WORKFLOW_NODES.map((n) => [n.id, n]));

  root
    .append("g")
    .attr("class", "wf-edges")
    .selectAll("path")
    .data(WORKFLOW_EDGES)
    .join("path")
    .attr("class", (d) => "wf-edge" + (d.branch ? " wf-edge-branch" : ""))
    .attr("d", (d) => workflowEdgePath(nodesById.get(d.from), nodesById.get(d.to)))
    .attr("fill", "none")
    .attr("stroke", (d) => ThemeColors.get(d.branch ? "--color-slate-600" : "--color-slate-500"))
    .attr("stroke-width", (d) => (d.branch ? 1.5 : 2))
    .attr("stroke-dasharray", (d) => (d.branch ? "4 3" : null))
    .attr("marker-end", `url(#${WORKFLOW_ARROW_MARKER_ID})`);

  const nodeGroups = root
    .append("g")
    .attr("class", "wf-nodes")
    .selectAll("g")
    .data(WORKFLOW_NODES)
    .join("g")
    .attr("class", "wf-node")
    .attr("data-id", (d) => d.id)
    .attr("transform", (d) => `translate(${d.x},${d.y})`)
    .style("cursor", "pointer")
    .on("click", (_event, d) => showWorkflowDetail(d));

  nodeGroups
    .append("rect")
    .attr("width", (d) => d.w)
    .attr("height", (d) => d.h)
    .attr("rx", 8)
    .attr("fill", ThemeColors.get("--color-slate-900"))
    .attr("stroke", (d) => WORKFLOW_CATEGORY_COLORS[d.category] || ThemeColors.get("--color-slate-500"))
    .attr("stroke-width", 1.5);

  nodeGroups.each(function (d) {
    const lines = d.label.split("\n");
    const text = d3
      .select(this)
      .append("text")
      .attr("x", d.w / 2)
      .attr("y", d.h / 2 - ((lines.length - 1) * 14) / 2)
      .attr("text-anchor", "middle")
      .attr("fill", ThemeColors.get("--color-slate-200"))
      .attr("font-size", "12px")
      .attr("font-weight", "500");
    lines.forEach((line, i) => {
      text
        .append("tspan")
        .attr("x", d.w / 2)
        .attr("dy", i === 0 ? 0 : 14)
        .text(line);
    });
  });

  document.getElementById("workflow-detail-close").addEventListener("click", () => {
    document.getElementById("workflow-detail-panel").classList.add("hidden");
    d3.selectAll(".wf-node rect").attr("stroke-width", 1.5);
  });
}
