/**
 * D3 rendering for the Stats page's "System" view: a donut/arc gauge
 * (CPU/RAM/disk/GPU usage right now), a dual-line history chart (CPU %
 * and memory % over the last hour), and a multi-line GPU utilization
 * history chart (one line per GPU — see renderGpuHistoryChart, empty
 * and hidden on any machine with no GPU this app's own poller can read). All the history
 * charts reuse telemetry_charts.js's shared telemetryChartSetup/
 * telemetryDrawAxes helpers rather than duplicating axis/scale
 * boilerplate a third time in this app — those helpers were already
 * written generically, not Ollama-specific, despite the file name.
 */

const SYSTEM_GAUGE_SIZE = 120;
const SYSTEM_GAUGE_STROKE = 12;
// Cycled through for each GPU's own line/legend swatch — GPU count is dynamic (0 to however many are installed),
// unlike CPU/Memory's fixed two colors above, so this can't just be two hardcoded constants. Themed via
// ThemeColors (see app/static/js/theme.js) rather than fixed hex.
const GPU_LINE_COLORS = [
  ThemeColors.chart(1),
  ThemeColors.chart(3),
  ThemeColors.chart(4),
  ThemeColors.chart(5),
  ThemeColors.chart(6),
  ThemeColors.chart(2),
];

/** Renders one full-circle usage gauge into the <div id="containerId">
 * — a background ring plus a colored foreground arc proportional to
 * `percent`, with the rounded percentage centered inside. */
function renderSystemGauge(containerId, percent, color) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  const size = SYSTEM_GAUGE_SIZE;
  const radius = (size - SYSTEM_GAUGE_STROKE) / 2;
  const clamped = Math.max(0, Math.min(100, percent));

  const svg = d3.select(container).append("svg").attr("width", size).attr("height", size);
  const g = svg.append("g").attr("transform", `translate(${size / 2},${size / 2})`);

  const ring = d3
    .arc()
    .innerRadius(radius - SYSTEM_GAUGE_STROKE / 2)
    .outerRadius(radius + SYSTEM_GAUGE_STROKE / 2)
    .startAngle(-Math.PI)
    .endAngle(Math.PI);
  g.append("path").attr("d", ring).attr("fill", ThemeColors.get("--color-slate-800"));

  const fill = d3
    .arc()
    .innerRadius(radius - SYSTEM_GAUGE_STROKE / 2)
    .outerRadius(radius + SYSTEM_GAUGE_STROKE / 2)
    .startAngle(-Math.PI)
    .endAngle(-Math.PI + (clamped / 100) * 2 * Math.PI);
  g.append("path").attr("d", fill).attr("fill", color);

  g.append("text")
    .attr("text-anchor", "middle")
    .attr("dy", "0.35em")
    .attr("fill", ThemeColors.get("--color-slate-200"))
    .attr("font-size", "20px")
    .attr("font-weight", "600")
    .text(`${Math.round(clamped)}%`);
}

/** Renders the CPU%/Memory% dual-line history chart from
 * SystemMetricsSummary.history (already time-ordered, one point per
 * poll — see system_metrics_service.py; no bucketing needed here since
 * the poll interval is short and the window is bounded). */
function renderSystemHistoryChart(history) {
  const emptyEl = document.getElementById("system-history-empty");
  const svgEl = document.getElementById("system-history-chart");
  if (history.length < 2) {
    emptyEl.classList.remove("hidden");
    svgEl.classList.add("hidden");
    return;
  }
  emptyEl.classList.add("hidden");
  svgEl.classList.remove("hidden");

  const { g, innerWidth, innerHeight } = telemetryChartSetup("system-history-chart");
  const points = history.map((p) => ({
    time: new Date(p.polled_at),
    cpu: p.cpu_percent,
    mem: p.mem_percent,
  }));

  const xScale = d3
    .scaleTime()
    .domain(d3.extent(points, (d) => d.time))
    .range([0, innerWidth]);
  const yScale = d3.scaleLinear().domain([0, 100]).range([innerHeight, 0]);

  telemetryDrawAxes(g, xScale, yScale, innerHeight, d3.timeFormat("%H:%M"));

  const drawLine = (accessor, color) => {
    const line = d3
      .line()
      .x((d) => xScale(d.time))
      .y((d) => yScale(accessor(d)));
    g.append("path").datum(points).attr("fill", "none").attr("stroke", color).attr("stroke-width", 2).attr("d", line);
  };
  drawLine((d) => d.cpu, ThemeColors.chart(1));
  drawLine((d) => d.mem, ThemeColors.chart(2));

  g.append("text")
    .attr("x", -innerHeight / 2)
    .attr("y", -36)
    .attr("transform", "rotate(-90)")
    .attr("text-anchor", "middle")
    .attr("fill", ThemeColors.get("--color-slate-500"))
    .attr("font-size", "10px")
    .text("%");
}

/** Renders GPU utilization % over time — one line per gpu_index, cycling through GPU_LINE_COLORS (see
 * system_metrics.js's renderGpuLegend for the matching color-key it builds alongside this). Flat input (see
 * GpuHistoryPoint's own docstring) grouped by gpu_index here rather than server-side, since drawing is the only
 * place that actually needs it split up. */
function renderGpuHistoryChart(gpuHistory) {
  const emptyEl = document.getElementById("gpu-history-empty");
  const svgEl = document.getElementById("gpu-history-chart");
  if (gpuHistory.length < 2) {
    emptyEl.classList.remove("hidden");
    svgEl.classList.add("hidden");
    return;
  }
  emptyEl.classList.add("hidden");
  svgEl.classList.remove("hidden");

  const { g, innerWidth, innerHeight } = telemetryChartSetup("gpu-history-chart");

  const byGpu = new Map();
  for (const point of gpuHistory) {
    const series = byGpu.get(point.gpu_index) ?? [];
    series.push({ time: new Date(point.polled_at), value: point.utilization_percent });
    byGpu.set(point.gpu_index, series);
  }

  const xScale = d3
    .scaleTime()
    .domain(d3.extent(gpuHistory, (p) => new Date(p.polled_at)))
    .range([0, innerWidth]);
  const yScale = d3.scaleLinear().domain([0, 100]).range([innerHeight, 0]);
  telemetryDrawAxes(g, xScale, yScale, innerHeight, d3.timeFormat("%H:%M"));

  const line = d3
    .line()
    .x((d) => xScale(d.time))
    .y((d) => yScale(d.value));
  const gpuIndexes = [...byGpu.keys()].sort((a, b) => a - b);
  gpuIndexes.forEach((gpuIndex, i) => {
    const color = GPU_LINE_COLORS[i % GPU_LINE_COLORS.length];
    g.append("path")
      .datum(byGpu.get(gpuIndex))
      .attr("fill", "none")
      .attr("stroke", color)
      .attr("stroke-width", 2)
      .attr("d", line);
  });

  g.append("text")
    .attr("x", -innerHeight / 2)
    .attr("y", -36)
    .attr("transform", "rotate(-90)")
    .attr("text-anchor", "middle")
    .attr("fill", ThemeColors.get("--color-slate-500"))
    .attr("font-size", "10px")
    .text("%");
}
