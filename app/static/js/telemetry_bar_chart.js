/**
 * Requests-by-host bar chart for the Stats page's "Telemetry" view —
 * sibling of telemetry_charts.js (shared setup/axis helpers live
 * there), split out to stay under CLAUDE.md's 250-line cap.
 */

/** Renders OllamaTelemetrySummary.by_host as grouped bars (total request
 * count, with the error slice highlighted) — correct-looking for both a
 * single-host deployment (one bar) and a genuinely multi-host one. */
function renderHostBarChart(hostStats) {
  const emptyEl = document.getElementById("telemetry-host-empty");
  const svgEl = document.getElementById("telemetry-host-chart");
  if (!hostStats.length) {
    emptyEl.classList.remove("hidden");
    svgEl.classList.add("hidden");
    return;
  }
  emptyEl.classList.add("hidden");
  svgEl.classList.remove("hidden");

  const { g, innerWidth, innerHeight } = telemetryChartSetup("telemetry-host-chart");

  const xScale = d3
    .scaleBand()
    .domain(hostStats.map((h) => h.host))
    .range([0, innerWidth])
    .padding(0.3);
  const yScale = d3
    .scaleLinear()
    .domain([0, d3.max(hostStats, (h) => h.request_count) * 1.1 || 1])
    .nice()
    .range([innerHeight, 0]);

  telemetryDrawAxes(g, xScale, yScale, innerHeight, (host) => {
    // Hosts are full URLs (http://host:11434) — too wide for a tick label, so just the host:port part is shown.
    try {
      const u = new URL(host);
      return u.host;
    } catch {
      return host;
    }
  });

  const bars = g
    .selectAll("g.telemetry-host-bar")
    .data(hostStats)
    .join("g")
    .attr("class", "telemetry-host-bar");

  bars
    .append("rect")
    .attr("x", (h) => xScale(h.host))
    .attr("width", xScale.bandwidth())
    .attr("y", (h) => yScale(h.request_count))
    .attr("height", (h) => innerHeight - yScale(h.request_count))
    .attr("fill", ThemeColors.chart(1))
    .append("title")
    .text((h) => `${h.host}: ${h.request_count} request(s), ${h.error_count} error(s)`);

  // The error slice, stacked at the bottom of the same bar, in red — 0-height and invisible when there are none.
  bars
    .append("rect")
    .attr("x", (h) => xScale(h.host))
    .attr("width", xScale.bandwidth())
    .attr("y", (h) => yScale(h.error_count))
    .attr("height", (h) => innerHeight - yScale(h.error_count))
    .attr("fill", ThemeColors.chart(9));
}
