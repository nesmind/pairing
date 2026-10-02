/**
 * Message timestamps in the user's own timezone (Settings > Account > Timezone, stored on User.timezone and
 * echoed to every page as <body data-timezone>). Server timestamps are naive UTC (see app/models/_base.py's
 * utcnow), so they're tagged "Z" before parsing.
 */

const userTimezone = document.body.dataset.timezone || "UTC";

/** Shortest date+time form, e.g. "10/2, 14:05" — locale decides order/separators. */
function formatShortDateTime(isoUtc) {
  const date = isoUtc ? new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(isoUtc) ? isoUtc : isoUtc + "Z") : new Date();
  if (Number.isNaN(date.getTime())) return "";
  try {
    return date.toLocaleString(undefined, {
      timeZone: userTimezone,
      month: "numeric",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch (_err) {
    return ""; // unknown zone name — show no time rather than a wrong one
  }
}
