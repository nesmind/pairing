/**
 * Shared D3 scale/axis helpers + the latency-over-time line chart for
 * the Stats page's "Telemetry" view — this app's first D3 chart with
 * real axes/scales (workflow_diagram.js's DAG has none, everything
 * there is hand-laid-out fixed positions). app/static/js/telemetry_bar_chart.js
 * is the sibling file for the by-host bar chart, split out to stay
 * under CLAUDE.md's 250-line cap rather than one big chart-drawing file.
 */

const TELEMETRY_CHART_MARGIN = { top: 12, right: 20, bottom: 28, left: 48 };
const TELEMETRY_CHART_HEIGHT = 220;

/** Sizes an <svg> to its own container's current width (recomputed on
 * every render rather than cached, since the container can resize) and
 * returns the {svg, innerWidth, innerHeight, g} an axis-based chart
 * needs — one helper shared by every chart type on this page. */
function telemetryChartSetup(svgId) {
  const svgEl = document.getElementById(svgId);
  const width = Math.max(svgEl.parentElement.clientWidth, 320);
  const { top, right, bottom, left } = TELEMETRY_CHART_MARGIN;
  const innerWidth = width - left - right;
  const innerHeight = TELEMETRY_CHART_HEIGHT - top - bottom;

  const svg = d3.select(svgEl).attr("width", width).attr("height", TELEMETRY_CHART_HEIGHT);
  svg.selectAll("*").remove();
  const g = svg.append("g").attr("transform", `translate(${left},${top})`);
  return { svg, g, innerWidth, innerHeight };
}

function telemetryDrawAxes(g, xScale, yScale, innerHeight, xTickFormat) {
  g.append("g")
    .attr("transform", `translate(0,${innerHeight})`)
    .call(d3.axisBottom(xScale).ticks(6).tickFormat(xTickFormat))
    .attr("color", ThemeColors.get("--color-slate-500"))
    .attr("font-size", "10px");
  g.append("g").call(d3.axisLeft(yScale).ticks(5)).attr("color", ThemeColors.get("--color-slate-500")).attr("font-size", "10px");
}

/** Renders the "avg latency per hour bucket" line chart from
 * OllamaTelemetrySummary.latency_time_series — bucket_start is a plain
 * "YYYY-MM-DDTHH:00:00" string (see telemetry_service.py's own
 * docstring on why this is computed in Python, not SQL). */
function renderLatencyLineChart(buckets) {
  const emptyEl = document.getElementById("telemetry-latency-empty");
  const svgEl = document.getElementById("telemetry-latency-chart");
  if (!buckets.length) {
    emptyEl.classList.remove("hidden");
    svgEl.classList.add("hidden");
    return;
  }
  emptyEl.classList.add("hidden");
  svgEl.classList.remove("hidden");

  const { g, innerWidth, innerHeight } = telemetryChartSetup("telemetry-latency-chart");
  const points = buckets.map((b) => ({ time: new Date(b.bucket_start), avg: b.avg_duration_ms }));

  const xScale = d3
    .scaleTime()
    .domain(d3.extent(points, (d) => d.time))
    .range([0, innerWidth]);
  const yScale = d3
    .scaleLinear()
    .domain([0, d3.max(points, (d) => d.avg) * 1.1 || 1])
    .nice()
    .range([innerHeight, 0]);

  telemetryDrawAxes(g, xScale, yScale, innerHeight, d3.timeFormat("%H:%M"));

  const line = d3
    .line()
    .x((d) => xScale(d.time))
    .y((d) => yScale(d.avg));

  g.append("path").datum(points).attr("fill", "none").attr("stroke", ThemeColors.chart(1)).attr("stroke-width", 2).attr("d", line);

  g.selectAll("circle")
    .data(points)
    .join("circle")
    .attr("cx", (d) => xScale(d.time))
    .attr("cy", (d) => yScale(d.avg))
    .attr("r", 3)
    .attr("fill", ThemeColors.chart(1))
    .append("title")
    .text((d) => `${d.time.toLocaleString()}: ${Math.round(d.avg)}ms avg`);

  g.append("text")
    .attr("x", -innerHeight / 2)
    .attr("y", -36)
    .attr("transform", "rotate(-90)")
    .attr("text-anchor", "middle")
    .attr("fill", ThemeColors.get("--color-slate-500"))
    .attr("font-size", "10px")
    .text("avg ms");
}
