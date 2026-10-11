/**
 * Where the SVG renderer puts each server (cluster_map_graph.js): a row under pAIring for the detailed map,
 * rings around the leader and voters for the compact one. Pure functions of the node list.
 */

class MapLayouts {
  /** Servers in one row, pAIring above, the storage below. */
  static row(nodes) {
    const place = {};
    const row = nodes.filter((n) => n.kind === "matricxon" || n.kind === "voter");
    row.forEach((n, i) => {
      place[n.id] = { x: row.length === 1 ? 500 : 140 + (i * 720) / (row.length - 1), y: 300 };
    });
    place.app = { x: 500, y: 85 };
    place.storage = { x: 500, y: 520 };
    return place;
  }

  /** Leader in the middle, voters next, then everything else on rings around them, worst status first. */
  static ring(nodes) {
    const place = {};
    const servers = nodes
      .filter((n) => n.kind === "matricxon" || n.kind === "voter")
      .sort((a, b) => MapLayouts.rank(a) - MapLayouts.rank(b));
    const cx = 500;
    const cy = 320;
    let ring = 0;
    let placed = 0;
    while (placed < servers.length) {
      const radius = ring === 0 ? 0 : 70 + ring * 62;
      const capacity = ring === 0 ? 1 : Math.max(6, Math.floor((2 * Math.PI * radius) / 50));
      const members = servers.slice(placed, placed + capacity);
      members.forEach((n, i) => {
        const angle = (2 * Math.PI * i) / members.length - Math.PI / 2;
        place[n.id] = { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
      });
      placed += members.length;
      ring += 1;
    }
    // pAIring and the storage sit just outside the outer ring, left and right of it, linked to the leader.
    const reach = (ring <= 1 ? 0 : 70 + (ring - 1) * 62) + 85;
    place.app = { x: Math.max(60, cx - reach), y: cy - 70 };
    place.storage = { x: Math.min(MAP_W - 150, cx + reach), y: cy + 70 };
    return place;
  }

  static rank(node) {
    if (node.badges.includes("leader")) return -2;
    if (node.badges.includes("voter")) return -1;
    return STATUS_ORDER[node.status] ?? 3;
  }
}
