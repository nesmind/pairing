/* Message timestamps in the user's timezone (Settings > Account > Timezone, User.timezone, echoed as <body data-timezone>). Server timestamps are naive UTC (app/models/_base.py utcnow), so they are tagged "Z" before parsing. */

const userTimezone = document.body.dataset.timezone || "UTC";

/* Shortest date+time form, e.g. "10/2, 14:05"; the locale sets order/separators. */
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
    return "";
  }
}
