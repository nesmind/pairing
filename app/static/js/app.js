/**
 * Tiny shared helpers used by both chat.js and settings.js. Kept as
 * plain global functions (no bundler/framework) since the whole
 * frontend is just a handful of static files served directly by
 * FastAPI — see app/main.py's StaticFiles mount.
 */

/**
 * Turns a FastAPI error response's `detail` into a plain, readable
 * string. Most of this app's own HTTPException(detail=...) calls already
 * send a plain string, which passes through unchanged — this exists for
 * FastAPI/Pydantic's own 422 validation errors, where `detail` is
 * instead a *list* of {loc, msg, ...} objects. Without this,
 * `new Error(detail)` on that list coerces it through
 * Object.prototype.toString and shows the user a literal "[object
 * Object]" instead of the actual validation message.
 */
function formatErrorDetail(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((entry) => {
        if (entry && typeof entry === "object" && entry.msg) {
          const field = Array.isArray(entry.loc) ? entry.loc[entry.loc.length - 1] : null;
          return field ? `${field}: ${entry.msg}` : entry.msg;
        }
        return typeof entry === "string" ? entry : JSON.stringify(entry);
      })
      .join("; ");
  }
  if (detail && typeof detail === "object") return JSON.stringify(detail);
  return String(detail);
}

/**
 * Wraps `fetch` for our JSON API: adds the right headers, parses the
 * response body as JSON, and throws a readable Error if the server
 * responded with a non-2xx status (FastAPI puts the message in
 * `detail`). Every API call in the frontend goes through this so error
 * handling only needs to be written once.
 */
async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (response.status === 401) {
    // Session expired/missing — bounce to the login page rather than
    // surfacing a confusing "Not authenticated" error inline.
    window.location.href = "/login";
    throw new Error("Not authenticated");
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (body.detail !== undefined && body.detail !== null) detail = formatErrorDetail(body.detail);
    } catch (_) {
      /* response wasn't JSON — fall back to statusText above */
    }
    throw new Error(detail);
  }
  if (response.status === 204) return null;
  return response.json();
}

/** Escapes text that will be inserted as HTML outside of markdown
 * rendering (e.g. titles, filenames) to prevent it being interpreted as
 * markup. Message content itself goes through a different path —
 * `marked.parse()` followed by `DOMPurify.sanitize()`, see chat.js's
 * renderMessageContent — since marked performs no sanitization on its own. */
function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value;
  return div.innerHTML;
}

/** Fills `container` with a small round avatar: the real picture (an <img>) when `avatarUrl` is set, otherwise a
 * plain-color circle showing `initials` — the same fallback both chat.js (a message's sender, beside every
 * "user"-role bubble) and settings.js (Settings > Account's own profile preview) need, so the rendering rule lives
 * once here rather than twice. `container` is expected to already carry sizing/shape classes (e.g. "h-8 w-8
 * rounded-full overflow-hidden") — this only ever sets its content, never its own size, so both call sites can pick
 * whatever size fits their own layout. */
function renderAvatar(container, avatarUrl, initials) {
  container.innerHTML = "";
  if (avatarUrl) {
    const img = document.createElement("img");
    img.src = avatarUrl;
    img.alt = "";
    img.className = "h-full w-full object-cover";
    container.appendChild(img);
    return;
  }
  const span = document.createElement("span");
  span.textContent = initials || "?";
  span.className = "flex h-full w-full items-center justify-center bg-slate-700 text-slate-200 font-medium";
  container.appendChild(span);
}

/** Formats a byte-ish/relative timestamp as a short, human time. */
function formatTime(isoString) {
  return new Date(isoString).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// Key used to hand a short confirmation message from one page load to
// the next (see queueSystemMessage/showQueuedSystemMessage below).
const SYSTEM_MESSAGE_KEY = "pairing:systemMessage";

/**
 * Queues a short message to show once, on whichever page loads next —
 * for an action that redirects before the user could otherwise see its
 * own result (Settings saving something and immediately sending the
 * user back to chat, say). Showing a status message and then racing a
 * redirect against it on the *same* page means either cutting the
 * message off early or making the user wait through an artificial
 * delay just so they have time to read it; showing it fresh on the
 * destination page, at a fixed spot near the top, needs neither.
 */
function queueSystemMessage(text) {
  try {
    sessionStorage.setItem(SYSTEM_MESSAGE_KEY, text);
  } catch (_) {
    /* sessionStorage unavailable (private browsing, etc.) — the
       redirect itself still works, the destination page just won't
       have a message queued to show. */
  }
}

/**
 * Displays (and consumes) whatever message queueSystemMessage() left
 * behind, if this page has a #system-banner element — safe to call
 * unconditionally on every page load, including ones with no banner
 * element (settings.html) or nothing queued (the common case).
 */
function showQueuedSystemMessage() {
  const bannerEl = document.getElementById("system-banner");
  if (!bannerEl) return;
  let text;
  try {
    text = sessionStorage.getItem(SYSTEM_MESSAGE_KEY);
  } catch (_) {
    return;
  }
  if (!text) return;
  try {
    sessionStorage.removeItem(SYSTEM_MESSAGE_KEY);
  } catch (_) {}
  bannerEl.textContent = text;
  bannerEl.classList.remove("hidden");
  setTimeout(() => bannerEl.classList.add("hidden"), 4000);
}

// Matches any Hebrew or Arabic-script letter — the two RTL scripts a
// local chat model is realistically going to produce. Written with
// explicit \uXXXX escapes (Hebrew U+0591-U+05F4, Arabic U+0600-U+06FF,
// plus the Arabic Presentation Forms blocks) rather than pasting the
// literal characters, since bidi source text has a way of getting
// visually reordered by editors/terminals and silently corrupted.
const RTL_LETTER_RE = /[\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\uFB1D-\uFDFF\uFE70-\uFEFF]/;
const STRONG_LETTER_RE = /[A-Za-z\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\uFB1D-\uFDFF\uFE70-\uFEFF]/;

/**
 * Picks "rtl" or "ltr" for a whole block of text by finding the *first
 * actual letter* (Latin or Hebrew/Arabic) and using its script, rather
 * than trusting HTML's built-in `dir="auto"` or a whole-string majority
 * vote — both get mixed Hebrew/English sentences wrong, in opposite ways:
 *
 * - `dir="auto"` looks at the very first strong-directional *character*,
 *   full stop — a message that opens with a digit, punctuation, or
 *   markdown syntax before any real letter throws it off, guessing "ltr"
 *   for a mostly-Hebrew message and mirroring neutral characters like
 *   "?" or "!" to the wrong side.
 * - Counting letters across the *whole* string (this function's previous
 *   approach) has the opposite failure: "שלום world" is one short Hebrew
 *   word followed by one longer English word, so English letters win the
 *   count and the whole line renders under an "ltr" base direction —
 *   which, per the Unicode bidi algorithm, plants the *first* logical
 *   word (Hebrew) on the left and the *second* (English) to its right:
 *   backwards from the Hebrew-first reading order a Hebrew speaker
 *   expects, and backwards from what `dir="rtl"` would have produced.
 *
 * Skipping straight to the first actual *letter* — ignoring digits,
 * punctuation, and markdown syntax along the way — reproduces what the
 * Unicode bidi algorithm's own paragraph-direction rule is supposed to
 * do (decide direction from the first strong character) without the
 * digit/punctuation/markdown-prefix trap that makes native `dir="auto"`
 * unreliable here.
 */
function detectTextDirection(text) {
  const match = text.match(STRONG_LETTER_RE);
  if (!match) return "auto"; // no letters yet (digits/emoji/empty) — nothing to decide
  return RTL_LETTER_RE.test(match[0]) ? "rtl" : "ltr";
}

/**
 * Cross-tab UI theme sync: when ThemeManager.save() (see app/static/js/theme.js) writes the new theme to
 * localStorage on the tab where it was saved, every *other* open tab gets a native `storage` event (fired only
 * in tabs other than the one that wrote it) and reloads itself onto it. Upgrades the code-block theme picker's
 * existing "other tabs need a manual refresh" precedent to self-refreshing, since a UI theme is worth reflecting
 * everywhere immediately rather than leaving other tabs visibly mismatched until their next navigation.
 */
window.addEventListener("storage", (event) => {
  if (event.key === "pairing:uiTheme" && event.newValue) {
    window.location.reload();
  }
});
