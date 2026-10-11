/* Small helpers shared by chat.js and settings.js: plain globals (no bundler), served as static files by FastAPI (see app/main.py's StaticFiles mount). */

/* Turns a FastAPI error `detail` into a readable string. HTTPException details are already strings and pass through; this handles Pydantic's 422 `detail`, a list of {loc, msg,...} objects that `new Error(detail)` would show as "[object Object]". */
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

/* Wraps `fetch` for the JSON API: sets headers, parses JSON, and throws a readable Error on a non-2xx status (FastAPI puts the message in `detail`). All frontend API calls go through it. */
async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (response.status === 401) {
    // Session expired or missing: go to the login page instead of showing "Not authenticated".
    window.location.href = "/login";
    throw new Error("Not authenticated");
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (body.detail !== undefined && body.detail !== null) detail = formatErrorDetail(body.detail);
    } catch (_) {
      /* not JSON: fall back to statusText above */
    }
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  if (response.status === 204) return null;
  return response.json();
}

/* Escapes text inserted as HTML outside markdown rendering (titles, filenames). Message content goes through `marked.parse` then `DOMPurify.sanitize` (chat.js renderMessageContent), as marked does not sanitize. */
function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value;
  return div.innerHTML;
}

/* Fills `container` with a small round avatar: the picture (<img>) when `avatarUrl` is set, else a colored circle with `initials`. Shared by chat.js (sender beside "user" bubbles) and settings.js (Account preview). `container` should already carry size/shape classes; this sets only its content. */
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

/* Formats a timestamp as a short, human time. */
function formatTime(isoString) {
  return new Date(isoString).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// Key for handing a short confirmation from one page load to the next (queueSystemMessage/showQueuedSystemMessage).
const SYSTEM_MESSAGE_KEY = "pairing:systemMessage";

/* Queues a short message to show once on the next page load, for actions that redirect before the user could read the result (e.g. Settings saving, then returning to chat). Showing it on the destination page avoids cutting it off or adding a delay. */
function queueSystemMessage(text) {
  try {
    sessionStorage.setItem(SYSTEM_MESSAGE_KEY, text);
  } catch (_) {
    /* sessionStorage unavailable: the redirect still works, with no message queued. */
  }
}

/* Shows (and consumes) the message queueSystemMessage left, if the page has #system-banner. Safe to call on every page load. */
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

// Matches Hebrew or Arabic-script letters (the RTL scripts a local chat model is likely to produce). Uses \uXXXX
// escapes (Hebrew U+0591-U+05F4, Arabic U+0600-U+06FF, plus Arabic Presentation Forms) because bidi source text gets
// visually reordered by editors.
const RTL_LETTER_RE = /[\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\uFB1D-\uFDFF\uFE70-\uFEFF]/;
const STRONG_LETTER_RE = /[A-Za-z\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\uFB1D-\uFDFF\uFE70-\uFEFF]/;

/*
 * Picks "rtl" or "ltr" for a block of text from its first actual letter (Latin or Hebrew/Arabic), not HTML's
 * `dir="auto"` or a whole-string majority vote; both misjudge mixed Hebrew/English text:
 * 
 * - `dir="auto"` uses the first strong character, so a message opening with a digit, punctuation or markdown guesses
 * "ltr" for mostly-Hebrew text and mirrors "?" or "!".
 * - Counting letters over the whole string lets a longer English word outvote a short Hebrew one ("שלום world"), giving
 * an "ltr" base that puts the Hebrew word on the left, the reverse of Hebrew reading order.
 * 
 * Skipping digits, punctuation and markdown to the first letter matches the Unicode bidi paragraph rule without those
 * traps.
 */
function detectTextDirection(text) {
  const match = text.match(STRONG_LETTER_RE);
  if (!match) return "auto";
  return RTL_LETTER_RE.test(match[0]) ? "rtl" : "ltr";
}

/* Cross-tab UI theme sync: when ThemeManager.save (theme.js) writes the theme to localStorage, every other open tab gets a `storage` event and reloads onto it, so tabs aren't left mismatched. */
window.addEventListener("storage", (event) => {
  if (event.key === "pairing:uiTheme" && event.newValue) {
    window.location.reload();
  }
});
