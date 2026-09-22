/**
 * Chat page controller: loads the conversation list into the sidebar,
 * switches between conversations, and sends new messages — streaming the
 * model's reply into the page as it arrives.
 *
 * State is intentionally kept in a few plain variables rather than a
 * framework store; the page is small enough that this stays readable.
 */

// id of the conversation currently shown in the main pane, or null
// before any conversation has been created/selected.
let activeConversationId = null;

const messagesEl = document.getElementById("messages");
const emptyStateEl = document.getElementById("empty-state");
const conversationListEl = document.getElementById("conversation-list");
const channelListEl = document.getElementById("channel-list");
const conversationTitleEl = document.getElementById("conversation-title");
const pinnedBadgesEl = document.getElementById("pinned-badges");
const modelBadgeEl = document.getElementById("model-badge");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const askAiBtn = document.getElementById("ask-ai-btn");
const attachBtn = document.getElementById("attach-btn");
const attachmentInput = document.getElementById("attachment-input");
const attachmentPreviewEl = document.getElementById("attachment-preview");

// Emoji picker (see chat.html: #emoji-btn/#emoji-picker) — 50 common, simple, kid-friendly emoji (faces, hand
// gestures, hearts) picked for being widely recognized and inoffensive, not an exhaustive/exact "most used"
// ranking. Grid order below is the display order, 8 per row.
const EMOJI_CHOICES = [
  "😀", "😃", "😄", "😁", "😆", "😊", "🙂", "😉", "😍", "🥰",
  "😘", "🤗", "🤔", "😴", "😋", "😜", "🤩", "😎", "🥳", "😇",
  "🙃", "😢", "😭", "😡", "😱", "😨", "😅", "🤣", "😬", "🥺",
  "👍", "👎", "👏", "🙌", "🙏", "👋", "💪", "✌️", "🤝", "👌",
  "❤️", "🧡", "💛", "💚", "💙", "💜", "💕", "⭐", "✨", "🎉",
];
const emojiBtn = document.getElementById("emoji-btn");
const emojiPickerEl = document.getElementById("emoji-picker");
(function initEmojiPicker() {
  if (emojiPickerEl.childElementCount === 0) {
    for (const emoji of EMOJI_CHOICES) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = emoji;
      btn.className = "rounded-md p-1 text-lg leading-none hover:bg-slate-800 transition-colors";
      btn.addEventListener("click", () => insertEmoji(emoji));
      emojiPickerEl.appendChild(btn);
    }
  }

  function insertEmoji(emoji) {
    // Inserted at the cursor (not just appended) so picking one mid-sentence doesn't jumble the message —
    // selectionStart/End collapse to the same point when nothing's selected, so this also just works as "insert
    // at cursor" for the common case.
    const start = chatInput.selectionStart ?? chatInput.value.length;
    const end = chatInput.selectionEnd ?? chatInput.value.length;
    chatInput.value = chatInput.value.slice(0, start) + emoji + chatInput.value.slice(end);
    const cursor = start + emoji.length;
    chatInput.focus();
    chatInput.setSelectionRange(cursor, cursor);
    // Picker stays open — same "pick several in a row" convention as every mainstream chat app's emoji picker;
    // it only closes on outside click, Escape, or clicking the emoji button again (below).
  }

  emojiBtn.addEventListener("click", (event) => {
    event.stopPropagation(); // otherwise the document-level listener below would immediately close this same click
    emojiPickerEl.classList.toggle("hidden");
  });
  document.addEventListener("click", (event) => {
    if (!emojiPickerEl.classList.contains("hidden") && !emojiPickerEl.contains(event.target) && event.target !== emojiBtn) {
      emojiPickerEl.classList.add("hidden");
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !emojiPickerEl.classList.contains("hidden")) {
      emojiPickerEl.classList.add("hidden");
    }
  });
})();

// One-time "🎨 New: UI themes" sidebar announcement (see chat.html) — shown until dismissed, then never again on
// this browser. localStorage rather than a server-side per-user flag: this is a one-off UI announcement, not
// state worth a DB round trip or syncing across devices — same reasoning as theme.js's own cross-tab sync key,
// just for a simpler "seen it" flag instead of a value that matters.
const WHATS_NEW_THEMES_KEY = "pairing:whatsNewThemesDismissed";
const whatsNewThemesEl = document.getElementById("whats-new-themes");
(function initWhatsNewThemes() {
  let dismissed = false;
  try {
    dismissed = localStorage.getItem(WHATS_NEW_THEMES_KEY) === "1";
  } catch (_) {
    /* private-browsing/storage-disabled — falls back to always showing it, not fatal */
  }
  whatsNewThemesEl.classList.toggle("hidden", dismissed);
})();
document.getElementById("whats-new-themes-dismiss").addEventListener("click", () => {
  whatsNewThemesEl.classList.add("hidden");
  try {
    localStorage.setItem(WHATS_NEW_THEMES_KEY, "1");
  } catch (_) {
    /* private-browsing/storage-disabled — reappears next load, not fatal */
  }
});
// Same sessionStorage tab-handoff account_panel.js's own "Go to Settings" link uses (settings.js's
// SETTINGS_TAB_STORAGE_KEY reads this back and clears it on load) — without this, the link just landed on
// Settings' own default tab (Model), not Account, since a plain href has no way to say which tab to open.
document.getElementById("whats-new-themes-settings-link").addEventListener("click", () => {
  try {
    sessionStorage.setItem("pairingSettingsActiveTab", "account");
  } catch (_err) {
    // Private-browsing/storage-blocked — settings.js falls back to its own default tab, same as if this had
    // never run.
  }
});

// Per-message caps — kept in sync by hand with app.config.
// MAX_ATTACHMENT_DOCUMENTS/MAX_ATTACHMENT_IMAGES. Purely a same-values
// client-side nicety (an early, friendlier rejection); the server is the
// real authority and re-checks this regardless (see
// chat_attachment_service.save_attachments).
const MAX_ATTACHMENT_DOCUMENTS = 5;
const MAX_ATTACHMENT_IMAGES = 1;

// The File objects picked via #attachment-input — sent as FormData's
// repeated "attachments" field on the next send, then cleared (see
// sendMessage/clearPendingAttachment below). Module-level rather than
// read fresh from the <input> at send time so clearPendingAttachment can
// reset both the input and this in one place.
let pendingAttachments = [];

function clearPendingAttachment() {
  pendingAttachments = [];
  attachmentInput.value = "";
  attachmentPreviewEl.replaceChildren();
  attachmentPreviewEl.classList.add("hidden");
}

function removePendingAttachment(file) {
  pendingAttachments = pendingAttachments.filter((f) => f !== file);
  renderAttachmentPreview();
}

/** Rebuilds the pre-send preview chip row from `pendingAttachments` —
 * one small chip per file, each with its own "×" to drop just that one. */
function renderAttachmentPreview() {
  attachmentPreviewEl.replaceChildren();
  attachmentPreviewEl.classList.toggle("hidden", pendingAttachments.length === 0);
  for (const file of pendingAttachments) {
    const chip = document.createElement("div");
    chip.className =
      "flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-300";
    const name = document.createElement("span");
    name.className = "truncate max-w-[16rem]";
    name.textContent = file.name;
    const clear = document.createElement("button");
    clear.type = "button";
    clear.className = "shrink-0 text-slate-500 hover:text-red-400";
    clear.textContent = "×";
    clear.addEventListener("click", () => removePendingAttachment(file));
    chip.append(name, clear);
    attachmentPreviewEl.appendChild(chip);
  }
}

attachBtn.addEventListener("click", () => attachmentInput.click());
attachmentInput.addEventListener("change", () => {
  // Merged onto whatever was already picked in an earlier dialog open —
  // <input multiple> only ever reports *this* dialog's own selection via
  // .files, so replacing pendingAttachments wholesale here (rather than
  // combining) would silently drop every earlier pick the moment the
  // user attaches files one at a time across several clicks of
  // #attach-btn instead of selecting them all in one go.
  const picked = [...pendingAttachments, ...Array.from(attachmentInput.files || [])];
  // Cleared so the same underlying <input> can be reused for the next
  // pick — pendingAttachments (not attachmentInput.files) is the single
  // source of truth for what's actually queued to send.
  attachmentInput.value = "";
  const images = picked.filter((f) => f.type.startsWith("image/"));
  const documents = picked.filter((f) => !f.type.startsWith("image/"));
  if (images.length > MAX_ATTACHMENT_IMAGES || documents.length > MAX_ATTACHMENT_DOCUMENTS) {
    alert(`Up to ${MAX_ATTACHMENT_IMAGES} image and ${MAX_ATTACHMENT_DOCUMENTS} document files per message.`);
  }
  pendingAttachments = [...images.slice(0, MAX_ATTACHMENT_IMAGES), ...documents.slice(0, MAX_ATTACHMENT_DOCUMENTS)];
  renderAttachmentPreview();
});

// Whether the logged-in user is an admin — set server-side as a data
// attribute on <body> (see app/templates/base.html and settings.js'
// identical use of it). Combined with a channel's own is_manager flag
// (see channelsByConversationId below) to decide, client-side, whether
// the active chat's model/persona/rules/skill controls should act as
// editable or read-only — the server enforces the real rule regardless
// (see app.services.channel_service.can_manage_channel_conversation),
// this only avoids showing a control that would just 403 if used.
const isAdmin = document.body.dataset.isAdmin === "true";
// The logged-in user's own id — lets the optimistic "user" bubble
// sendMessage renders (in a channel) be clickable to open its own
// profile modal immediately, the same way currentAvatarUrl below lets
// it show the right picture without a round trip.
const currentUserId = document.body.dataset.userId || null;
// The logged-in user's own username — same data attribute settings.js
// already reads (see app/templates/base.html).
const currentUsername = document.body.dataset.username || "";
// First/last name (whichever are set) with username in parentheses,
// matching app/models/conversation.py's Message.sender_display_name
// exactly — used to label a just-sent message with the sender's own
// name in a channel chat, right away, without waiting on a round trip
// to learn what the server would compute for "you".
const currentFullName = [document.body.dataset.firstName, document.body.dataset.lastName]
  .filter(Boolean)
  .join(" ");
const currentDisplayName = currentFullName ? `${currentFullName} (${currentUsername})` : currentUsername;
// Same immediate-availability reasoning as currentFullName above, for
// the small round avatar beside the optimistic "user" bubble sendMessage
// renders before any round trip — mirrors app.models.user.User's own
// avatar_url/initials pair exactly (see app/templates/base.html).
const currentAvatarUrl = document.body.dataset.avatarUrl || null;
const currentInitials = (() => {
  const letters = [document.body.dataset.firstName, document.body.dataset.lastName].filter(Boolean).map((n) => n[0]);
  return (letters.length ? letters.join("") : currentUsername[0] || "?").toUpperCase();
})();

// Maps a channel's shared conversation id -> the ChannelSummary it came
// from (see GET /api/channels), so selectConversation() can tell a
// channel conversation apart from a personal one — and, for a channel
// one, whether *this* viewer can manage it — without a second request
// per conversation. Populated by loadChannelList().
let channelsByConversationId = {};

// Whether the active conversation's model/persona/rules/skill can be
// changed by this viewer. Always true for a personal chat (its owner
// always has full rights); for a channel chat, only an admin or that
// channel's manager. Read fresh (not captured in a closure) by the
// model-badge and pinned-badge click handlers below, since it changes
// every time selectConversation() runs.
let canManageActive = true;

// How other channel members find out about an in-progress/finished
// reply while sitting in that chat — "cheap" (poll) or "real" (pushed
// live) — see app.services.chat_settings_service.get_channel_delivery_mode.
// Fetched once in init() below; "cheap" is the safe default until that
// resolves, matching the server's own default.
let channelDeliveryMode = "cheap";

// message_ids this tab is itself driving via streamAssistantReply (the
// sender's own live typing effect) — the reply-watch/poll paths below
// must never touch these, or they'd fight the sender's own smoother,
// immediate rendering with a slower, catch-up-only one. Reset on every
// conversation switch (see selectConversation).
let liveMessageIds = new Set();
// True for exactly the duration of this tab's own chatForm submit
// handler below — a channel conversation's own reply-watch connection
// (see startReplyWatch) stays open while sitting in the chat, *before*
// activeConversationId's owner ever sends anything, so when they do
// send, that pre-existing watch and their new POST /stream request both
// end up subscribed to the exact same broadcast topic and can each
// receive its very first event before the other has had a chance to tag
// liveMessageIds — a genuine race, not just a timing nicety, and
// unlike later chunks (where the loser only sees an already-tagged id
// and skips cleanly), the *first* event racing this way creates a
// second, orphaned bubble that never updates again. Most visible on an
// instant failure, which has no earlier chunk to have already settled
// the race — see handleReplyLiveEvent's own check of this flag. Cheap
// mode's poll doesn't need the same guard: its ~3s tick interval is
// always slower than this tab's own stream tagging liveMessageIds.
let ownSendInFlight = false;
// Accumulated text for a reply this tab is watching live but did NOT
// send itself (a channel's "real" mode, or a personal chat's own owner
// reconnecting mid-reply), keyed by message_id — the source of truth
// for what's rendered in that bubble, independent of the DOM (see
// handleReplyLiveEvent), since a backstop reconcile can replace the DOM
// node from under an in-flight chunk sequence.
let replyLiveContent = new Map();
// The open live-updates connection/timer for the conversation currently
// on screen, if any — exactly one of these two is ever active at a
// time, closed and cleared on every conversation switch by
// stopReplyWatch() below.
// This tab's own in-flight POST /stream request, if any — aborted
// whenever the user leaves the conversation it belongs to (see
// selectConversation/clearMainPane), so that leaving actually closes
// the connection instead of leaving it running unseen in the
// background. What that disconnect then means server-side depends on
// conversation type (see app.services.reply_generation_service.stream_reply's
// cancel_on_disconnect): a channel's shared conversation keeps
// generating regardless; a personal chat's reply is cancelled, since
// only its owner could ever have been watching.
let activeStreamAbortController = null;

let replySubscribeSource = null;
// Whether the open replySubscribeSource should close itself the moment
// the one reply it's watching settles (a personal chat's reconnect
// watch — see selectConversation/startReplyWatch) or stay open
// indefinitely (a channel's "real" delivery mode).
let replyWatchIsOneShot = false;
let channelPollTimeoutId = null;
// The `server_time` echoed back by the last poll tick — the next
// tick's `since`, so each request only asks for what's actually new.
let channelPollSince = null;

// Best-effort, proactive-only mirror of the backend's real guard (see
// conversation_service.has_active_reply, enforced in
// app/routers/chat.py's stream_reply) — disables #ask-ai-btn while this
// tab believes some member's reply is already in progress in the active
// channel. Never itself the source of truth: the backend still rejects
// with 409 if this has drifted stale (see sendMessage's catch handler),
// and every checkpoint below (renderMessageList, handleReplyLiveEvent,
// runChannelPoll) re-syncs it from the latest known message state, so
// any drift self-heals on the next render/poll/live-tick.
let channelReplyInFlight = false;

function updateAskAiButtonState() {
  askAiBtn.disabled = channelReplyInFlight;
  askAiBtn.title = channelReplyInFlight
    ? "The AI is already replying to another message in this channel"
    : "";
}

// Show any message a redirecting page (e.g. Settings, after saving)
// queued for us — as early as possible, not gated behind the
// conversation-list fetch below, since it's independent of that data.
showQueuedSystemMessage();

// marked.js is configured once here so every place that renders model
// output (renderMessageContent below) gets consistent behavior: GitHub-
// flavored line breaks, and syntax highlighting via highlight.js for
// fenced code blocks. Syntax highlighting needs a custom `code` renderer
// rather than the old `highlight` setOptions key — marked dropped that
// key in v5 (this app is on v15), and it's silently ignored rather than
// erroring, so code blocks rendered as plain unstyled text with no
// warning until this was wired up properly.
marked.use({
  renderer: {
    code({ text, lang }) {
      // Only trust `lang` as an actual language name (guards against it
      // holding something unexpected before it ever reaches an HTML
      // class attribute) and only when highlight.js actually knows it —
      // hljs.getLanguage() is undefined for a lang hint it doesn't
      // recognize, e.g. a model inventing "pseudocode".
      const language = lang && /^[\w+-]+$/.test(lang) && hljs.getLanguage(lang) ? lang : null;
      const highlighted = language
        ? hljs.highlight(text, { language }).value
        : hljs.highlightAuto(text).value;
      const langClass = language ? ` language-${language}` : "";
      // The "hljs" class is what every vendored theme's CSS actually
      // targets (see app/static/vendor/highlightjs/styles/*.css) — the
      // color scheme silently has no effect without it.
      return `<pre><code class="hljs${langClass}">${highlighted}</code></pre>`;
    },
  },
});
marked.setOptions({ breaks: true });

function renderMessageContent(container, text) {
  container.innerHTML = DOMPurify.sanitize(marked.parse(text || ""));
  // Re-derive direction from the actual (possibly still-growing, mid-
  // stream) text every time — see app.js:detectTextDirection for why
  // this is more reliable than the browser's own dir="auto" heuristic
  // for Hebrew/Arabic text mixed with punctuation, numbers, or markdown.
  container.dir = detectTextDirection(text || "");
}

/** Appends a small "Sources: a.txt, b.pdf" line under a reply that used
 * RAG (see app/rag.py / app/routers/chat.py) — lets the user see which
 * uploaded documents contributed to that specific answer. Only appears
 * when RAG actually contributed (sources is empty otherwise), so its
 * presence itself is useful signal. Safe to call only once a bubble's
 * final text is set, since it doesn't survive a later
 * renderMessageContent() call (which replaces the whole innerHTML). */
function appendSourcesFooter(bubble, sources) {
  if (!sources || sources.length === 0) return;
  const footer = document.createElement("div");
  footer.className = "mt-2 pt-2 border-t border-white/10 text-xs text-slate-400";
  footer.textContent = "Sources: " + sources.join(", ");
  bubble.appendChild(footer);
}

/** `senderLabel`, when given, renders a small "who sent this" line above
 * the bubble — only ever passed for a channel's shared conversation
 * (see selectConversation/the chat-send handler below), never for a
 * personal chat: with only one human participant, a label there would
 * be pure noise. Channel messages need it precisely because more than
 * one member can post in the same thread.
 *
 * `avatarUrl`/`initials` are used for every "user"-role message, personal
 * chat included — a message's own sender always has a picture-or-initials
 * to show (see app.models.conversation.Message.sender_avatar_url/
 * sender_initials). Never rendered for "assistant"/"system" — no picture
 * concept for the AI in this app.
 *
 * `senderId`, unlike avatarUrl/initials, follows senderLabel's own
 * channel-only gating (always undefined/null for a personal chat's call
 * site) — it's what makes the avatar clickable to open the "enlarged
 * picture + basic details" modal (see openUserProfileModal), which only
 * makes sense where there's more than one possible sender to look up. */
function bubbleFor(role, senderLabel, avatarUrl, initials, senderId) {
  const wrapper = document.createElement("div");
  wrapper.className = "flex flex-col " + (role === "user" ? "items-end" : "items-start");

  if (senderLabel) {
    // A dedicated row (not just a bare <span>) so attachDeleteButtonIfEligible
    // below has somewhere to put a delete affordance next to the label —
    // "message-label-row" is what it looks for, and its presence at all
    // is also how it recognizes "this is a channel message" (a personal
    // chat's bubble never gets a label, or this row, in the first place).
    const labelRow = document.createElement("div");
    labelRow.className = "message-label-row mb-1 flex items-center gap-1.5 px-1";
    const label = document.createElement("span");
    label.className = "text-xs font-medium text-slate-500";
    label.textContent = senderLabel;
    labelRow.appendChild(label);
    wrapper.appendChild(labelRow);
  }

  const bubble = document.createElement("div");
  bubble.className =
    "markdown-body max-w-[75ch] rounded-2xl px-4 py-2.5 text-sm leading-relaxed " +
    (role === "user"
      ? "bg-brand-600 text-white rounded-br-sm"
      : "bg-slate-800 text-slate-100 rounded-bl-sm");
  // Text direction (rtl for Hebrew/Arabic, ltr otherwise) is set by
  // renderMessageContent() below, per render — not here, since it
  // depends on the actual message text, which doesn't exist yet at
  // bubble-creation time. This only affects text flow *inside* the
  // bubble — which side of the chat it sits on (user vs. assistant) is
  // controlled separately by the wrapper's items-end/items-start.

  if (role === "user") {
    // Bubble + avatar side by side, avatar last so it lands on the same
    // side as the text (right, matching items-end above).
    const row = document.createElement("div");
    row.className = "flex items-end gap-2";
    const avatar = document.createElement("div");
    avatar.className = "message-avatar h-6 w-6 shrink-0 rounded-full overflow-hidden text-[10px] leading-none";
    renderAvatar(avatar, avatarUrl, initials);
    if (senderId) {
      avatar.classList.add("cursor-pointer", "hover:opacity-80", "transition-opacity");
      avatar.addEventListener("click", () => openUserProfileModal(senderId));
    }
    row.append(bubble, avatar);
    wrapper.appendChild(row);
  } else {
    wrapper.appendChild(bubble);
  }
  return { wrapper, bubble };
}

/** Adds a small delete affordance to `wrapper`'s label row for a viewer
 * who can manage this channel (see canManageActive) — lets a channel
 * admin/manager remove any message, live if it's still generating (see
 * deleteChannelMessage below). No-op if there's nowhere to put it (a
 * personal chat's bubble never gets a label row at all — see bubbleFor)
 * or the viewer can't manage this channel. Safe to call more than once
 * for the same wrapper: a poll/live-watch bubble is often created before
 * its real id is known, so this runs again once it is (see
 * findOrCreateBubbleForMessage/readAssistantReplyStream below), and
 * button-already-present is checked for rather than assumed. */
function attachDeleteButtonIfEligible(wrapper, messageId) {
  if (!canManageActive) return;
  const labelRow = wrapper.querySelector(".message-label-row");
  if (!labelRow || labelRow.querySelector(".delete-message-btn")) return;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "delete-message-btn ms-auto shrink-0 text-slate-600 hover:text-red-400 transition-colors";
  btn.title = "Delete this message";
  // Same trash-can icon as the sidebar's own "delete chat" button (see
  // renderConversationListItem below) rather than an emoji, for visual
  // consistency across the app's delete affordances.
  btn.innerHTML =
    '<svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" ' +
    'stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0-1 14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2L4 6h16Z"/></svg>';
  btn.addEventListener("click", () => deleteChannelMessage(messageId, wrapper));
  labelRow.appendChild(btn);
}

/** Deletes a channel message for good — channel admins/managers only
 * (see app/routers/conversations.py's delete_message, the only enforcer
 * that actually matters; attachDeleteButtonIfEligible above is just the
 * UI-side reflection of the same rule). Removes `wrapper` from this tab
 * immediately on success rather than waiting for the next poll/live-
 * watch tick to notice the status="deleted" transition — other members
 * still learn about it that way (see runChannelPoll/handleReplyLiveEvent),
 * just not this tab, which already knows. */
async function deleteChannelMessage(messageId, wrapper) {
  if (!confirm("Delete this message? This can't be undone.")) return;
  let deletedMessageIds;
  try {
    ({ deleted_message_ids: deletedMessageIds } = await api(
      `/api/conversations/${activeConversationId}/messages/${messageId}`,
      { method: "DELETE" }
    ));
  } catch (err) {
    alert(err.message || "Failed to delete message.");
    return;
  }
  // Deleting a "user" message whose reply is still generating (see
  // conversation_service.delete_message) takes that reply down too —
  // the response names every id actually removed (one or two), not just
  // `messageId`, so its bubble disappears from this tab right away
  // instead of waiting for the next poll/live-watch tick to notice.
  for (const id of deletedMessageIds) {
    const el = id === messageId ? wrapper : messagesEl.querySelector(`[data-message-id="${id}"]`);
    if (el) el.remove();
    // *Adds* to liveMessageIds — not a typo for .delete(): a "cheap"
    // mode poll request already in flight when this DELETE was clicked
    // can still resolve afterward, reflecting whatever status the row
    // had *before* this delete committed (e.g. still "streaming") —
    // without this, runChannelPoll's own liveMessageIds check (see
    // below) wouldn't recognize that stale response as one to ignore,
    // and would recreate the exact bubble just removed here. A real,
    // shipped bug: the message reappeared until the whole conversation
    // was reloaded, which is the only path that re-reads from
    // conversation_service.visible_messages and so never shows it again.
    liveMessageIds.add(id);
  }
  if (messagesEl.children.length === 0) showEmptyState(true);
}

// Shown in place of a failed/interrupted reply's content — see
// renderReplyBody below. A reply that got partway before failing still
// shows that partial content above this, rather than losing it.
const REPLY_FAILED_NOTICE = "⚠️ This reply didn't finish — generation failed or was interrupted.";

/** Renders an assistant reply's body against its current known
 * content/status/sources (see Message.status in
 * app/models/conversation.py) — the one place that decides between
 * "still thinking" (typing dots — status is "streaming" and nothing has
 * come through yet), the reply text itself (streaming with partial
 * content, or complete), and a failure notice (status "error", any
 * partial content kept above it). Used for every render of an
 * assistant bubble: the initial history load, "cheap" mode's poll, and
 * "real" mode / personal-chat reconnect's live events — so a reply's
 * in-progress or failed state reads the same whether you watched it
 * happen or you're only just opening the conversation now. `status`
 * undefined/"complete" (a plain user message, or any pre-existing
 * caller that never passes one) just renders `content` as-is.
 * `errorText`, only meaningful with status "error", overrides the generic REPLY_FAILED_NOTICE with the real
 * failure reason whenever one is known — Message.error_message/MessageOut.error_message for the history-load/
 * poll paths (null for a row written before that field existed, or one that genuinely never started at all —
 * see sendMessage's own catch handler), or payload.error directly for a live SSE event, the exact same text
 * either way. Falls back to the generic notice only when nothing more specific is available. */
function renderReplyBody(bubble, content, status, sources, errorText) {
  if (status === "streaming" && !content) {
    bubble.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
    return;
  }
  renderMessageContent(bubble, content);
  if (status === "error") {
    const notice = document.createElement("div");
    notice.className = "mt-2 pt-2 border-t border-white/10 text-xs text-red-400";
    notice.textContent = errorText || REPLY_FAILED_NOTICE;
    bubble.appendChild(notice);
  } else {
    appendSourcesFooter(bubble, sources);
  }
}

/** Appends every one of a message's attachments (an image, or a small
 * download chip for anything else) to `bubble` — see MessageOut.attachments/
 * AttachmentOut in app/schemas/chat.py. No-op when `message` carries
 * none. Safe to call right after renderReplyBody on the same bubble
 * every time, even for an already-rendered one being updated again (e.g.
 * runChannelPoll) — that call replaces bubble.innerHTML wholesale, so
 * this needs to run again right after it every time, not just once at
 * creation. */
function renderAttachment(bubble, message) {
  if (!message || !message.attachments) return;
  for (const attachment of message.attachments) {
    if (attachment.type === "image") {
      const img = document.createElement("img");
      img.src = attachment.url;
      img.alt = attachment.filename || "attachment";
      img.className = "mt-2 max-w-full rounded-lg";
      // An <img> has no layout height until it actually loads, so the
      // plain scrollTop=scrollHeight every caller below does right after
      // this returns undershoots — the message list scrolls to what
      // looks like the bottom *before* the image has any height, then
      // the image loads and pushes everything down, leaving the view no
      // longer actually at the bottom. Re-scrolling once the real size
      // is known fixes it regardless of which caller (a fresh send, a
      // channel poll, a full reload) triggered this render.
      img.addEventListener("load", () => scrollMessagesToBottom());
      bubble.appendChild(img);
    } else {
      const chip = document.createElement("a");
      chip.href = attachment.url;
      chip.target = "_blank";
      chip.rel = "noopener";
      chip.className = "mt-2 flex items-center gap-1.5 text-xs underline decoration-dotted";
      chip.textContent = "📎 " + (attachment.filename || "attachment");
      bubble.appendChild(chip);
    }
  }
}

function addMessageBubble(
  role,
  content,
  sources,
  senderLabel,
  messageId,
  status,
  attachmentSource,
  avatarUrl,
  initials,
  senderId,
  errorText
) {
  const { wrapper, bubble } = bubbleFor(role, senderLabel, avatarUrl, initials, senderId);
  if (messageId) {
    wrapper.dataset.messageId = messageId;
    attachDeleteButtonIfEligible(wrapper, messageId);
  }
  renderReplyBody(bubble, content, status, sources, errorText);
  renderAttachment(bubble, attachmentSource);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return { wrapper, bubble };
}

/** Finds the bubble already tracking `messageId` (see addMessageBubble/
 * streamAssistantReply, which both tag a bubble's wrapper with
 * data-message-id once its id is known) or creates a fresh one — used by
 * the poll/live-watch paths below to update the same DOM node across
 * repeated events instead of appending a duplicate each time. `created`
 * tells the caller whether this is a message it hasn't shown any part of
 * yet (see handleReplyLiveEvent's reconcile-on-first-sight below). */
function findOrCreateBubbleForMessage(messageId, role, senderLabel, avatarUrl, initials, senderId) {
  const existing = messagesEl.querySelector(`[data-message-id="${messageId}"]`);
  if (existing) return { bubble: existing.querySelector(".markdown-body"), created: false };
  showEmptyState(false);
  const { wrapper, bubble } = bubbleFor(role, senderLabel, avatarUrl, initials, senderId);
  wrapper.dataset.messageId = messageId;
  attachDeleteButtonIfEligible(wrapper, messageId);
  messagesEl.appendChild(wrapper);
  return { bubble, created: true };
}

/** Renders a conversation's full message list from scratch — shared by
 * selectConversation (initial load) and reconcileConversationMessages
 * (the channel live-updates backstop) below. */
function renderMessageList(messages, channelInfo) {
  messagesEl.innerHTML = "";
  showEmptyState(messages.length === 0);
  for (const message of messages) {
    // Sender labels only ever show in a channel's shared conversation —
    // see bubbleFor's docstring. "AI" for every assistant reply;
    // message.sender_display_name (see MessageOut) for who actually
    // typed a user-role message, since more than one member can post
    // here.
    const senderLabel = channelInfo ? (message.role === "assistant" ? "AI" : message.sender_display_name) : null;
    // Clickable-to-enlarge only in a channel — see bubbleFor's own
    // docstring on why senderId follows senderLabel's exact gating.
    const senderId = channelInfo && message.role === "user" ? message.sender_id : null;
    addMessageBubble(
      message.role,
      message.content,
      message.sources,
      senderLabel,
      message.id,
      message.status,
      message,
      message.sender_avatar_url,
      message.sender_initials,
      senderId,
      message.error_message
    );
  }
  if (messages.length > 0) scrollMessagesToBottom();

  if (channelInfo) {
    // Covers both the initial load and every reconcileConversationMessages
    // backstop call, including right after a reply finishes — the single
    // most reliable checkpoint for this, since it reflects the database's
    // own authoritative state rather than a live event that might have
    // been missed.
    const last = messages[messages.length - 1];
    channelReplyInFlight = !!(last && last.role === "assistant" && last.status === "streaming");
    updateAskAiButtonState();
  }
}

/** Refetches and re-renders this conversation's full message history —
 * the correctness backstop for both channel delivery modes and a
 * personal chat's reconnect watch alike: the broadcast hub's bounded
 * queue can drop an event under load (see
 * app.services.reply_broadcast_service), and "real" mode has no other
 * way to learn about a *different* channel member's newly-sent user
 * message (the broadcast hub only ever carries reply-generation
 * events). Called once a watched reply finishes, or the first time this
 * tab sees a message it has no bubble for yet — cheap enough for how
 * infrequently either of those happens. */
async function reconcileConversationMessages(conversationId) {
  if (conversationId !== activeConversationId) return;
  let messages;
  try {
    messages = await api(`/api/conversations/${conversationId}/messages`);
  } catch (_) {
    return;
  }
  if (conversationId !== activeConversationId) return; // switched away while this was in flight
  renderMessageList(messages, channelsByConversationId[conversationId] || null);
}

/** Stops whichever live-updates mechanism (see startReplyWatch/
 * startChannelPolling below) is active for the conversation being left —
 * called on every conversation switch, so exactly one is ever running at
 * a time, for the conversation actually on screen. */
function stopReplyWatch() {
  if (replySubscribeSource) {
    replySubscribeSource.close();
    replySubscribeSource = null;
  }
  replyWatchIsOneShot = false;
  if (channelPollTimeoutId) {
    clearTimeout(channelPollTimeoutId);
    channelPollTimeoutId = null;
  }
  channelPollSince = null;
  replyLiveContent.clear();
}

/** A read-only SSE connection to app/routers/chat.py's `/subscribe`
 * endpoint, which relays whatever app.services.reply_broadcast_service
 * publishes for this conversation — live, token-by-token, without this
 * tab having to be the one that sent the message. Two callers (see
 * selectConversation below): a channel conversation in "real" delivery
 * mode, watching for any member's activity for as long as the chat stays
 * open; and a personal chat whose own last message is still "streaming"
 * when it's opened — a reconnect after whatever connection originally
 * sent it dropped — which only needs to watch until that one reply
 * settles. Plain EventSource (not the manual fetch+reader
 * streamAssistantReply below uses) is fine here: this is a GET with no
 * body, and the browser's automatic reconnect-on-drop is exactly the
 * behavior wanted for a long-lived "watch this chat" connection. */
function startReplyWatch(conversationId, oneShot) {
  const source = new EventSource(`/api/chat/${conversationId}/subscribe`);
  replySubscribeSource = source;
  replyWatchIsOneShot = !!oneShot;
  source.onmessage = (event) => {
    handleReplyLiveEvent(conversationId, JSON.parse(event.data));
  };
}

function handleReplyLiveEvent(conversationId, payload) {
  if (conversationId !== activeConversationId) return;
  // This tab's own streamAssistantReply is already rendering whatever
  // reply it just sent (or is about to start rendering, within the same
  // tick) — see ownSendInFlight's own declaration for the race this
  // avoids. Skips this connection's events for the whole send, not just
  // once liveMessageIds catches up, since that catch-up is exactly what
  // isn't guaranteed to win.
  if (ownSendInFlight) return;
  const messageId = payload.message_id;
  // Same idea, after the fact: once tagged, this message is known to be
  // one this tab is already rendering live via its own connection.
  if (!messageId || liveMessageIds.has(messageId)) return;

  if (payload.sync) {
    replyLiveContent.set(messageId, payload.content || "");
    if (payload.status !== "streaming") return; // already finished by the time this tab connected
    channelReplyInFlight = true;
    updateAskAiButtonState();
    const { bubble, created } = findOrCreateBubbleForMessage(messageId, "assistant", "AI");
    renderReplyBody(bubble, replyLiveContent.get(messageId), payload.status);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    if (created) reconcileConversationMessages(conversationId);
    return;
  }

  if (payload.chunk) {
    channelReplyInFlight = true;
    updateAskAiButtonState();
    const text = (replyLiveContent.get(messageId) || "") + payload.chunk;
    replyLiveContent.set(messageId, text);
    const { bubble, created } = findOrCreateBubbleForMessage(messageId, "assistant", "AI");
    renderReplyBody(bubble, text, "streaming");
    messagesEl.scrollTop = messagesEl.scrollHeight;
    // A brand-new message this tab hasn't shown any part of yet — pick
    // up the user message that prompted it (and anything else missed)
    // in one shot, rather than teaching the broadcast hub about
    // messages it was never meant to carry.
    if (created) reconcileConversationMessages(conversationId);
  }
  if (payload.deleted) {
    channelReplyInFlight = false;
    updateAskAiButtonState();
    replyLiveContent.delete(messageId);
    const existing = messagesEl.querySelector(`[data-message-id="${messageId}"]`);
    if (existing) existing.remove();
    // Deleting the *question* this reply was answering (see
    // conversation_service.delete_message) takes the reply down too —
    // this event only ever names the reply's own id (the broadcast hub
    // only ever tracks a conversation's *generation* state), so a full
    // reconcile is what actually removes the paired question's bubble
    // for this tab, the same way it's the authoritative source for
    // done/error below.
    reconcileConversationMessages(conversationId);
    if (replyWatchIsOneShot) stopReplyWatch();
    return;
  }
  if (payload.done || payload.error) {
    channelReplyInFlight = false;
    updateAskAiButtonState();
    replyLiveContent.delete(messageId);
    reconcileConversationMessages(conversationId); // authoritative final content/status/sources
    // A personal chat's reconnect watch only ever cares about the one
    // reply it caught up on — nothing else will ever arrive on it, so
    // close it rather than leaving an idle connection open. A channel's
    // "real"-mode watch is the opposite: it stays open for as long as
    // the chat does, since any member might send another message later.
    if (replyWatchIsOneShot) stopReplyWatch();
  }
}

// How often "cheap" channel delivery mode checks back for new/changed
// messages while its conversation stays open — frequent enough to feel
// responsive, cheap enough to leave running for as long as the chat
// stays open (unlike pollForGeneratedTitle below, this has no give-up-
// after-N-attempts cap: a channel chat is open-ended, not a one-shot
// check-back).
const CHANNEL_POLL_INTERVAL_MS = 3000;

function startChannelPolling(conversationId) {
  channelPollSince = null;
  scheduleChannelPoll(conversationId);
}

function scheduleChannelPoll(conversationId) {
  channelPollTimeoutId = setTimeout(() => runChannelPoll(conversationId), CHANNEL_POLL_INTERVAL_MS);
}

async function runChannelPoll(conversationId) {
  if (conversationId !== activeConversationId) return; // switched away while this tick was scheduled
  let result;
  try {
    const query = channelPollSince ? `?since=${encodeURIComponent(channelPollSince)}` : "";
    result = await api(`/api/conversations/${conversationId}/messages/latest${query}`);
  } catch (_) {
    scheduleChannelPoll(conversationId); // transient failure — just try again next tick
    return;
  }
  if (conversationId !== activeConversationId) return;

  // While this tab's own send is in flight, skip this tick entirely —
  // see ownSendInFlight's own declaration for why liveMessageIds can't
  // be relied on yet to filter just the sender's own message. A channel
  // reply's placeholder Message exists in the database (and so in a
  // poll response) from the moment generation starts, well before this
  // tab's own POST /stream has received a single live event to tag it
  // with — a slow-to-start model (loading into memory, a long queue,
  // anything past one ~3s poll tick) makes this a routine race, not a
  // rare one. Deliberately *not* advancing channelPollSince here: this
  // response (including any other member's message that arrived in the
  // same window) gets reprocessed once the send finishes, rather than
  // silently skipped forever.
  if (ownSendInFlight) {
    scheduleChannelPoll(conversationId);
    return;
  }
  channelPollSince = result.server_time;
  const channelInfo = channelsByConversationId[conversationId] || null;
  for (const message of result.messages) {
    if (message.role === "assistant") channelReplyInFlight = message.status === "streaming";
    if (message.status === "deleted") {
      // Checked *before* the liveMessageIds skip below, deliberately: a
      // message this very tab sent (and so already tagged) can still get
      // deleted later by someone else — that dedup guard exists to avoid
      // re-rendering a message this tab is actively driving live, which
      // deletion is the opposite of.
      const existing = messagesEl.querySelector(`[data-message-id="${message.id}"]`);
      if (existing) existing.remove();
      // .add(), not .delete() — see deleteChannelMessage's own comment
      // on the exact same choice: once a message is known deleted, this
      // tab must never recreate a bubble for it again, no matter how
      // stale a later response claims it still is.
      liveMessageIds.add(message.id);
      continue;
    }
    if (liveMessageIds.has(message.id)) continue; // this tab is already rendering it via its own send
    const senderLabel = channelInfo ? (message.role === "assistant" ? "AI" : message.sender_display_name) : null;
    const senderId = channelInfo && message.role === "user" ? message.sender_id : null;
    const { bubble } = findOrCreateBubbleForMessage(
      message.id,
      message.role,
      senderLabel,
      message.sender_avatar_url,
      message.sender_initials,
      senderId
    );
    renderReplyBody(bubble, message.content, message.status, message.sources, message.error_message);
    renderAttachment(bubble, message);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
  updateAskAiButtonState();
  if (messagesEl.children.length === 0) showEmptyState(true);
  scheduleChannelPoll(conversationId);
}

/**
 * Scrolls #messages to the bottom, re-checking across a few animation
 * frames until scrollHeight stops growing rather than trusting a single
 * synchronous measurement. Needed because this page's Tailwind build
 * (see base.html: the CDN runtime script, not a precompiled stylesheet)
 * restyles newly-added elements *asynchronously* — right after a big
 * batch of message bubbles is appended (e.g. opening a long chat),
 * scrollHeight still reflects their pre-styled, too-short size for a
 * beat, so scrolling immediately lands short of the real bottom. Capped
 * so a pathological case just gives up instead of polling forever.
 */
function scrollMessagesToBottom() {
  let lastHeight = -1;
  let stableFrames = 0;
  const maxFrames = 30; // ~0.5s at 60fps — comfortably past Tailwind's usual restyle time
  function step(frame) {
    const height = messagesEl.scrollHeight;
    stableFrames = height === lastHeight ? stableFrames + 1 : 0;
    lastHeight = height;
    messagesEl.scrollTop = height;
    if (stableFrames < 2 && frame < maxFrames) {
      requestAnimationFrame(() => step(frame + 1));
    }
  }
  step(0);
}

function showEmptyState(show) {
  emptyStateEl.classList.toggle("hidden", !show);
  messagesEl.classList.toggle("hidden", show);
}

function clearMainPane() {
  activeConversationId = null;
  canManageActive = true;
  activeStreamAbortController?.abort();
  activeStreamAbortController = null;
  stopReplyWatch();
  liveMessageIds = new Set();
  closeSlotEditor();
  closeModelPicker();
  conversationTitleEl.textContent = "";
  pinnedBadgesEl.innerHTML = "";
  modelBadgeEl.textContent = "";
  modelBadgeEl.classList.add("hidden"); // no conversation open — see its own template comment
  setModelBadgeManageable(true);
  messagesEl.innerHTML = "";
  showEmptyState(true);
  askAiBtn.classList.add("hidden");
  channelReplyInFlight = false;
  updateAskAiButtonState();
}

/** Toggles the model badge's look between "click to switch" (personal
 * chats, and channel chats for an admin/manager) and a plain read-only
 * display (a channel chat for anyone else) — purely cosmetic, the real
 * enforcement is server-side (see openModelPicker's own guard below). */
function setModelBadgeManageable(manageable) {
  modelBadgeEl.classList.toggle("cursor-default", !manageable);
  modelBadgeEl.classList.toggle("opacity-60", !manageable);
  modelBadgeEl.title = manageable
    ? "Switch model for this chat"
    : "Only admins and this channel's managers can change its model";
}

// Same icons used on the Notes page's pin chips (see notes.js) — kept
// as a small separate copy rather than a shared import since these are
// two independent static files with no bundler/module system tying
// them together.
const PIN_TYPE_ICONS = { persona: "🎭", rules: "📏", skill: "🛠️" };
const PIN_TYPE_LABELS = { persona: "Persona", rules: "Rules", skill: "Skill" };

const slotEditorEl = document.getElementById("slot-editor");
const slotEditorTitleEl = document.getElementById("slot-editor-title");
const slotEditorContentEl = document.getElementById("slot-editor-content");
const slotEditorOnlyThisChatEl = document.getElementById("slot-editor-only-this-chat");
const slotEditorStatusEl = document.getElementById("slot-editor-status");
const slotEditorUseDefaultBtn = document.getElementById("slot-editor-use-default");
const slotEditorTurnOffBtn = document.getElementById("slot-editor-turn-off");

// The slot currently open in the editor, so its buttons know which
// conversation/pin_type to act on — null while the panel is closed.
let editingSlot = null; // { conversationId, pinType }

function closeSlotEditor() {
  editingSlot = null;
  slotEditorEl.classList.add("hidden");
}

function openSlotEditor(conversationId, slot) {
  editingSlot = { conversationId, pinType: slot.pin_type };
  const state = !slot.active ? " (off)" : slot.is_override ? " (customized)" : " (default)";
  slotEditorTitleEl.textContent = `${PIN_TYPE_LABELS[slot.pin_type] || slot.pin_type} for this chat${state}`;
  slotEditorContentEl.value = slot.content || "";
  slotEditorContentEl.dir = detectTextDirection(slotEditorContentEl.value);
  // Reflects this slot's actual current state: checked when it's already
  // a per-chat override, unchecked when it's on the shared default —
  // otherwise re-opening an already-customized slot showed an unchecked
  // box right next to content that's clearly not the default, which read
  // as if the override had silently been lost.
  slotEditorOnlyThisChatEl.checked = slot.is_override;
  // "Use default"/"Enable" is the same action either way (POST
  // use-default) — only its label changes: an active override can
  // "revert" to the shared default, an off slot needs "enabling" back
  // on. Hidden only when there's nothing to do — already showing the
  // plain, active default.
  slotEditorUseDefaultBtn.classList.toggle("hidden", slot.active && !slot.is_override);
  slotEditorUseDefaultBtn.textContent = slot.active ? "Use default" : "Enable";
  // "Turn off" only makes sense for a slot that's currently on.
  slotEditorTurnOffBtn.classList.toggle("hidden", !slot.active);
  slotEditorStatusEl.textContent = "";
  slotEditorEl.classList.remove("hidden");
  slotEditorContentEl.focus();
}

document.getElementById("slot-editor-cancel").addEventListener("click", closeSlotEditor);

document.getElementById("slot-editor-save").addEventListener("click", async () => {
  if (!editingSlot) return;
  const { conversationId, pinType } = editingSlot;
  slotEditorStatusEl.textContent = "Saving…";
  try {
    await api(`/api/notes/slots/${conversationId}/${pinType}`, {
      method: "PUT",
      body: JSON.stringify({
        content: slotEditorContentEl.value,
        only_this_chat: slotEditorOnlyThisChatEl.checked,
      }),
    });
    closeSlotEditor();
    loadPinnedBadges(conversationId);
  } catch (err) {
    slotEditorStatusEl.textContent = err.message;
  }
});

slotEditorUseDefaultBtn.addEventListener("click", async () => {
  if (!editingSlot) return;
  const { conversationId, pinType } = editingSlot;
  try {
    await api(`/api/notes/slots/${conversationId}/${pinType}/use-default`, { method: "POST" });
    closeSlotEditor();
    loadPinnedBadges(conversationId);
  } catch (err) {
    slotEditorStatusEl.textContent = err.message;
  }
});

slotEditorTurnOffBtn.addEventListener("click", async () => {
  if (!editingSlot) return;
  const { conversationId, pinType } = editingSlot;
  try {
    await api(`/api/notes/slots/${conversationId}/${pinType}`, { method: "DELETE" });
    closeSlotEditor();
    loadPinnedBadges(conversationId);
  } catch (err) {
    slotEditorStatusEl.textContent = err.message;
  }
});

const modelPickerEl = document.getElementById("model-picker");
const modelPickerListEl = document.getElementById("model-picker-list");
const modelPickerStatusEl = document.getElementById("model-picker-status");

function closeModelPicker() {
  modelPickerEl.classList.add("hidden");
}

/** Fetches the installed-model list and renders it into the picker
 * opened by clicking the model badge — a plain name list (unlike
 * Settings' full catalog picker), since there's nothing to install
 * from here, only an existing conversation's model to change. */
async function openModelPicker() {
  if (!activeConversationId || !canManageActive) return;
  closeSlotEditor(); // only one floating panel open at a time
  modelPickerListEl.innerHTML = "";
  modelPickerStatusEl.textContent = "Loading…";
  modelPickerEl.classList.remove("hidden");

  let models;
  try {
    ({ models } = await api("/api/settings/installed-models"));
  } catch (err) {
    modelPickerStatusEl.textContent = err.message;
    return;
  }
  modelPickerStatusEl.textContent = "";
  if (models.length === 0) {
    modelPickerStatusEl.textContent = "No models installed.";
    return;
  }
  const currentModel = modelBadgeEl.textContent;
  for (const model of models) {
    const btn = document.createElement("button");
    btn.type = "button";
    // Full tag, wrapped rather than truncated — a tag like "hf.co/mistralai/Ministral-3-3B-Instruct-2512-
    // GGUF:Ministral-3-3B-Instruct-2512-Q4_K_M" is the real pull path an admin needs to actually distinguish
    // between similarly-named models, and single-line `truncate` inside this panel's old, narrower width cut
    // it down to something unreadably short ("hf.co/mistralai/Ministral-3-…" for every Ministral variant
    // alike). `break-all` (not just `break-words`) since a long unbroken segment like the GGUF filename itself
    // has no natural wrap point a browser would otherwise find. `title` stays as a plain hover fallback, same
    // convention as Settings' own model catalog rows (see settings.js: renderModelRow's pathLine).
    btn.className = "w-full break-all rounded-md px-2 py-1.5 text-start font-mono text-xs leading-snug transition-colors " + (
      model === currentModel ? "bg-slate-800 text-slate-100" : "text-slate-300 hover:bg-slate-800"
    );
    btn.textContent = model;
    btn.title = model;
    btn.addEventListener("click", () => switchModel(model));
    modelPickerListEl.appendChild(btn);
  }
}

/** Switches the active conversation to a different model going
 * forward — past messages stay attributed to whatever model actually
 * generated them, only new replies use the new one. */
async function switchModel(model) {
  if (model === modelBadgeEl.textContent) {
    closeModelPicker();
    return;
  }
  modelPickerStatusEl.textContent = "Switching…";
  try {
    const conversation = await api(`/api/conversations/${activeConversationId}`, {
      method: "PATCH",
      body: JSON.stringify({ model }),
    });
    modelBadgeEl.textContent = conversation.model || "";
    closeModelPicker();
  } catch (err) {
    modelPickerStatusEl.textContent = err.message;
  }
}

modelBadgeEl.addEventListener("click", () => {
  if (modelPickerEl.classList.contains("hidden")) openModelPicker();
  else closeModelPicker();
});

// ---- User profile modal (channels only) --------------------------------
// Opened by clicking a message's own round avatar — see bubbleFor's own
// docstring on why this is only ever wired up for a channel's shared
// conversation, never a personal chat. Enlarges the picture and shows
// whatever basic details GET /api/account/{id} returns; a natural place
// to grow more fields into later without touching every message's own
// payload for it.

const userProfileModalEl = document.getElementById("user-profile-modal");
const userProfileModalAvatarEl = document.getElementById("user-profile-modal-avatar");
const userProfileModalNameEl = document.getElementById("user-profile-modal-name");
const userProfileModalUsernameEl = document.getElementById("user-profile-modal-username");

function closeUserProfileModal() {
  userProfileModalEl.classList.add("hidden");
}

async function openUserProfileModal(userId) {
  userProfileModalNameEl.textContent = "";
  userProfileModalUsernameEl.textContent = "Loading…";
  renderAvatar(userProfileModalAvatarEl, null, "");
  userProfileModalEl.classList.remove("hidden");

  let profile;
  try {
    profile = await api(`/api/account/${userId}`);
  } catch (err) {
    userProfileModalUsernameEl.textContent = err.message;
    return;
  }

  const fullName = [profile.first_name, profile.last_name].filter(Boolean).join(" ");
  renderAvatar(userProfileModalAvatarEl, profile.avatar_url, profile.initials);
  // With no name set, the username alone is already the modal's whole
  // point — shown as the main line rather than leaving it empty above a
  // redundant second line repeating the same thing.
  userProfileModalNameEl.textContent = fullName || profile.username;
  userProfileModalUsernameEl.textContent = fullName ? `@${profile.username}` : "";
}

document.getElementById("user-profile-modal-close").addEventListener("click", closeUserProfileModal);
// Closes on a genuine backdrop click only (the check is against
// event.target, not currentTarget — a click that started inside the
// panel and bubbled up would otherwise close it too).
userProfileModalEl.addEventListener("click", (event) => {
  if (event.target === userProfileModalEl) closeUserProfileModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !userProfileModalEl.classList.contains("hidden")) closeUserProfileModal();
});

/** Shows one clickable icon per persona/rules/skill slot for this
 * conversation (see GET /api/notes/slots/{id}) — always all 3, in the
 * same order, so the row never jumps around; a turned-off slot just
 * fades instead of disappearing, and stays clickable so turning it back
 * on is exactly as easy as turning it off (see openSlotEditor's
 * "Enable" button). Best-effort: a failed lookup just means no icons
 * show, not a broken chat. */
async function loadPinnedBadges(conversationId) {
  let slots;
  try {
    slots = await api(`/api/notes/slots/${conversationId}`);
  } catch (_) {
    return;
  }
  // Not awaited by its caller (see selectConversation) — if the user
  // switched to a different conversation while this was in flight, its
  // result is now stale and shouldn't overwrite that conversation's icons.
  if (conversationId !== activeConversationId) return;

  pinnedBadgesEl.innerHTML = "";
  for (const slot of slots) {
    const badge = document.createElement("button");
    badge.type = "button";
    badge.className = "rounded-full px-1.5 py-0.5 transition-colors " + (
      slot.active
        ? "bg-slate-800" + (canManageActive ? " hover:bg-slate-700" : "")
        : "bg-slate-800/40 opacity-40" + (canManageActive ? " hover:opacity-70" : "")
    );
    const state = !slot.active ? " (off)" : slot.is_override ? " (customized)" : "";
    const readOnlyNote = canManageActive ? "" : " — read-only (admins/managers only)";
    badge.title = `${PIN_TYPE_LABELS[slot.pin_type] || slot.pin_type}${state}: ${slot.title || ""}${readOnlyNote}`;
    badge.textContent = PIN_TYPE_ICONS[slot.pin_type] || "📌";
    // Reads canManageActive fresh at click time (not the value captured
    // when this badge was rendered) — see its own declaration above.
    badge.addEventListener("click", () => {
      if (canManageActive) openSlotEditor(conversationId, slot);
    });
    pinnedBadgesEl.appendChild(badge);
  }
}

// ---- Sidebar / conversation list ------------------------------------

/** A plain "No chats"/"No channels" row for an empty sidebar section —
 * shown instead of just leaving the section blank under its heading, so
 * an empty Chats/Channels list still reads as "empty" rather than
 * "still loading" or "broken". */
function renderEmptyListPlaceholder(container, text) {
  const placeholder = document.createElement("p");
  placeholder.className = "px-3 py-2 text-xs text-slate-500";
  placeholder.textContent = text;
  container.appendChild(placeholder);
}

async function loadConversationList() {
  const conversations = await api("/api/conversations");
  conversationListEl.innerHTML = "";
  for (const conversation of conversations) {
    conversationListEl.appendChild(renderConversationListItem(conversation));
  }
  if (conversations.length === 0) renderEmptyListPlaceholder(conversationListEl, "No chats");
  return conversations;
}

function renderConversationListItem(conversation) {
  const item = document.createElement("div");
  item.dataset.id = conversation.id;
  item.className =
    "group flex items-center rounded-lg pr-1 text-sm transition-colors " +
    (conversation.id === activeConversationId
      ? "bg-slate-800 text-slate-100"
      : "text-slate-300 hover:bg-slate-800/60");

  const titleBtn = document.createElement("button");
  titleBtn.type = "button";
  // text-left (not text-start): the row itself always stays left-
  // aligned, in the sidebar's own reading direction, regardless of the
  // title's language — only the *text itself* (via dir below) follows
  // Hebrew/Arabic's own reading direction, so a mixed-language title
  // still renders its characters/word order correctly without flipping
  // which side of the row it sits on.
  titleBtn.className = "flex-1 min-w-0 truncate px-3 py-2 text-left";
  const titleText = conversation.title || "New chat";
  titleBtn.dir = detectTextDirection(titleText);
  titleBtn.textContent = titleText;
  titleBtn.addEventListener("click", () => selectConversation(conversation.id));

  const deleteBtn = document.createElement("button");
  deleteBtn.type = "button";
  deleteBtn.title = "Delete chat";
  deleteBtn.className =
    "shrink-0 rounded p-1.5 text-slate-500 opacity-0 group-hover:opacity-100 hover:bg-slate-700 hover:text-red-400 transition-opacity";
  deleteBtn.innerHTML =
    '<svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" ' +
    'stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0-1 14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2L4 6h16Z"/></svg>';
  deleteBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    deleteConversation(conversation.id);
  });

  item.appendChild(titleBtn);
  item.appendChild(deleteBtn);
  return item;
}

function highlightActiveInSidebar() {
  for (const item of [...conversationListEl.children, ...channelListEl.children]) {
    item.classList.toggle("bg-slate-800", item.dataset.id === activeConversationId);
    item.classList.toggle("text-slate-100", item.dataset.id === activeConversationId);
  }
}

// ---- Sidebar / channel list -------------------------------------------

async function loadChannelList() {
  const channels = await api("/api/channels");
  channelsByConversationId = {};
  channelListEl.innerHTML = "";
  for (const channel of channels) {
    channelsByConversationId[channel.conversation_id] = channel;
    channelListEl.appendChild(renderChannelListItem(channel));
  }
  if (channels.length === 0) renderEmptyListPlaceholder(channelListEl, "No channels");
}

function renderChannelListItem(channel) {
  const item = document.createElement("div");
  item.dataset.id = channel.conversation_id;
  item.className =
    "group flex items-center rounded-lg pr-1 text-sm transition-colors " +
    (channel.conversation_id === activeConversationId
      ? "bg-slate-800 text-slate-100"
      : "text-slate-300 hover:bg-slate-800/60");

  // No delete button here — a channel's lifecycle (create/rename/
  // membership/delete) is admin-only, managed from Settings > Channels,
  // not from the sidebar (see app/routers/channels.py).
  const titleBtn = document.createElement("button");
  titleBtn.type = "button";
  titleBtn.className = "flex-1 min-w-0 flex items-center gap-1.5 truncate px-3 py-2 text-left";
  titleBtn.innerHTML =
    `<span class="text-slate-500">#</span><span class="min-w-0 truncate"></span>`;
  const label = titleBtn.querySelector("span:last-child");
  label.dir = detectTextDirection(channel.name);
  label.textContent = channel.name;
  titleBtn.addEventListener("click", () => selectConversation(channel.conversation_id));

  item.appendChild(titleBtn);
  return item;
}

async function selectConversation(id) {
  activeConversationId = id;
  closeSlotEditor(); // an open editor belonged to whichever chat we're leaving
  closeModelPicker();
  highlightActiveInSidebar();
  // Actually closes the connection for whichever conversation we're
  // leaving, if this tab has one of its own sends still in flight there
  // — see activeStreamAbortController's own declaration for why this
  // matters (a channel keeps generating regardless; a personal chat's
  // reply is cancelled).
  activeStreamAbortController?.abort();
  activeStreamAbortController = null;
  stopReplyWatch(); // belonged to whichever conversation we're leaving, if any
  liveMessageIds = new Set();

  const channelInfo = channelsByConversationId[id] || null;
  canManageActive = !channelInfo || isAdmin || channelInfo.is_manager;
  setModelBadgeManageable(canManageActive);
  askAiBtn.classList.toggle("hidden", !channelInfo);

  const conversation = await api(`/api/conversations/${id}`);
  // A channel conversation's display name is the channel's own `name`
  // (set/edited via Settings > Channels), never conversation.title —
  // see chat_service.build_reply_stream's title-generation guard, which
  // deliberately leaves that field alone for channel conversations.
  const titleText = channelInfo ? channelInfo.name : (conversation.title || "New chat");
  conversationTitleEl.textContent = titleText;
  conversationTitleEl.dir = detectTextDirection(titleText);
  modelBadgeEl.textContent = conversation.model || "";
  modelBadgeEl.classList.remove("hidden");
  loadPinnedBadges(id); // not awaited — a badge lookup shouldn't delay showing messages

  const messages = await api(`/api/conversations/${id}/messages`);
  renderMessageList(messages, channelInfo);

  if (channelInfo) {
    // Watch for other members' activity in this channel — which
    // mechanism depends on the admin-configured channelDeliveryMode (see
    // its own declaration above). Runs for as long as this channel stays
    // open: any member might send another message at any time.
    if (channelDeliveryMode === "real") startReplyWatch(id);
    else startChannelPolling(id);
  } else {
    // A personal chat has only one possible sender (its owner) and no
    // delivery-mode choice — but if the last message here is still
    // "streaming" (see Message.status), that means whatever connection
    // originally sent it is gone (a disconnect, a reload, a different
    // tab/device) while generation itself, detached from that
    // connection, kept going in the background (see
    // app.services.reply_generation_service). Reconnect and watch just
    // that one reply through to done/error, then stop (see
    // handleReplyLiveEvent's replyWatchIsOneShot handling) — nothing
    // else will ever arrive on this watch until the owner sends a new
    // message themselves, which streamAssistantReply drives directly.
    const lastMessage = messages[messages.length - 1];
    if (lastMessage && lastMessage.role === "assistant" && lastMessage.status === "streaming") {
      startReplyWatch(id, /* oneShot */ true);
    }
  }
}

async function createNewConversation() {
  const conversation = await api("/api/conversations", { method: "POST", body: JSON.stringify({}) });
  // Select first, render the sidebar row second: selectConversation()
  // sets activeConversationId, which is what actually lets a message get
  // sent. If the sidebar row's rendering ever has a bug, the chat itself
  // still works — worst case the sidebar looks stale until the next
  // reload, instead of every Send silently doing nothing.
  await selectConversation(conversation.id);
  // The only <p> ever appended directly here is the "No chats"
  // placeholder (list rows are <div data-id> — see
  // renderConversationListItem) — clear it before adding this chat's
  // first-ever row, otherwise it'd keep showing alongside a non-empty list.
  const placeholder = conversationListEl.querySelector("p");
  if (placeholder) placeholder.remove();
  conversationListEl.prepend(renderConversationListItem(conversation));
}

/** Removes a conversation for good (confirmed first, since this can't be
 * undone). If the deleted chat was the one on screen, falls back to
 * whatever's now first in the list, or an empty "start a new chat" state
 * if that was the last one. */
async function deleteConversation(id) {
  if (!confirm("Delete this chat? This can't be undone.")) return;

  await api(`/api/conversations/${id}`, { method: "DELETE" });
  const item = conversationListEl.querySelector(`[data-id="${id}"]`);
  if (item) item.remove();
  if (!conversationListEl.querySelector("[data-id]")) {
    renderEmptyListPlaceholder(conversationListEl, "No chats");
  }

  if (id === activeConversationId) {
    const next = conversationListEl.querySelector("[data-id]");
    if (next) {
      await selectConversation(next.dataset.id);
    } else {
      clearMainPane();
    }
  }
}

document.getElementById("new-chat-btn").addEventListener("click", createNewConversation);

// ---- Sending a message + streaming the reply -------------------------

/** Grows the textarea with its content, up to a sensible max height,
 * instead of scrolling inside a fixed-size box for long messages, and
 * flips its typing direction live as Hebrew/Arabic text is entered. */
chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + "px";
  chatInput.dir = detectTextDirection(chatInput.value);
});

chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

/** Shows a small red note right under the user's own bubble for a send
 * that never produced any assistant-side record at all — a 409 "AI
 * busy" rejection (see conversation_service.has_active_reply), or a
 * plain (ask_ai=false) message whose request itself failed outright.
 * Deliberately distinct from renderReplyBody's REPLY_FAILED_NOTICE path,
 * which is reserved for a reply that actually started generating and
 * then failed — this matches exactly what reloading this conversation
 * would show either way: nothing at all, vs. a failed-reply record. */
function showInlineSendError(userWrapper, message) {
  const notice = document.createElement("div");
  notice.className = "mt-1 px-1 text-xs text-red-400";
  notice.textContent = message;
  userWrapper.appendChild(notice);
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  // Plain Send/Enter always posts a normal message in a channel — no AI
  // reply (that's what #ask-ai-btn below is for). A personal chat has no
  // such choice: it always asks the AI, exactly as before this feature
  // existed (see build_reply_stream's own ask_ai coercion for personal
  // chats, which enforces this server-side too, not just here).
  const activeChannelInfo = activeConversationId ? channelsByConversationId[activeConversationId] : null;
  await sendMessage(!activeChannelInfo);
});

askAiBtn.addEventListener("click", () => sendMessage(true));

/** Sends whatever's in #chat-input as either a normal channel message
 * (askAi=false — the AI is never invoked, see build_reply_stream's
 * early-return for it) or an explicit "ask AI" request (askAi=true —
 * this conversation's original, and for a personal chat only, behavior).
 * Shared by #chat-form's submit and #ask-ai-btn's click above. */
async function sendMessage(askAi) {
  const content = chatInput.value.trim();
  if (!content) return;

  if (!activeConversationId) {
    await createNewConversation();
  }
  // Captured once, used for the rest of this handler (including the
  // reconcile in `finally` below) — activeConversationId itself can
  // change out from under a long-running send if the user switches
  // conversations mid-stream (see activeStreamAbortController), and
  // every reconcile/poll helper already no-ops once its own
  // conversationId argument stops matching the (possibly different by
  // then) active one.
  const conversationId = activeConversationId;

  // Captured and cleared from the UI immediately — a new file picked
  // while this send is still in flight must never end up attached to it.
  const attachmentFiles = pendingAttachments;
  clearPendingAttachment();

  showEmptyState(false);
  chatInput.value = "";
  chatInput.style.height = "auto";
  sendBtn.disabled = true;
  askAiBtn.disabled = true;

  // Only ever shown for a channel's shared conversation (see
  // bubbleFor's docstring) — channelsByConversationId is populated by
  // loadChannelList.
  const activeChannelInfo = channelsByConversationId[conversationId] || null;
  // Shown immediately via local object URLs/file names, before the
  // server round trip even starts — findOrCreateBubbleForMessage's own
  // "already exists" path leaves this alone once the real attachment
  // urls are known, but a full renderMessageList rebuild (e.g.
  // reconcileConversationMessages) always re-renders from the server's
  // own MessageOut fields, so these local object URLs are never relied
  // on beyond this one optimistic render.
  const optimisticMessage = {
    attachments: attachmentFiles.map((file) => ({
      url: URL.createObjectURL(file),
      filename: file.name,
      type: file.type.startsWith("image/") ? "image" : "text",
    })),
  };
  const { wrapper: userWrapper } = addMessageBubble(
    "user",
    content,
    undefined,
    activeChannelInfo ? currentDisplayName : null,
    undefined,
    undefined,
    optimisticMessage,
    currentAvatarUrl,
    currentInitials,
    activeChannelInfo ? currentUserId : null
  );

  // A normal (askAi=false) message never gets a reply at all — no
  // "thinking" bubble for it, since build_reply_stream's ask_ai=False
  // early return never creates an assistant Message for one.
  let wrapper = null;
  let bubble = null;
  if (askAi) {
    ({ wrapper, bubble } = bubbleFor("assistant", activeChannelInfo ? "AI" : null));
    renderReplyBody(bubble, "", "streaming");
    messagesEl.appendChild(wrapper);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    if (activeChannelInfo) {
      channelReplyInFlight = true;
      updateAskAiButtonState();
    }
  }

  ownSendInFlight = true;
  try {
    await streamAssistantReply(conversationId, content, askAi, userWrapper, wrapper, bubble, attachmentFiles);
  } catch (err) {
    // AbortError means the user themselves navigated away (see
    // activeStreamAbortController) — that bubble is already gone from
    // the DOM (selectConversation/clearMainPane already rebuilt
    // #messages for wherever they went instead), so there's nothing
    // useful to show.
    if (err.name === "AbortError") {
      // Pane isn't on screen anymore either way — nothing to re-sync.
    } else if (err.status === 409) {
      // Nothing was ever created server-side (see app/routers/chat.py's
      // stream_reply) — someone else's reply really is still active, so
      // channelReplyInFlight is intentionally left as-is here, not
      // reset: this rejection confirms it rather than contradicting it.
      showInlineSendError(userWrapper, err.message);
    } else if (bubble && userWrapper.dataset.messageId) {
      // The user message really was persisted (see readAssistantReplyStream's
      // own "user_message_id" tagging) before generation itself failed
      // outright — matches exactly what reopening this conversation
      // later would show (see renderReplyBody): a real user message
      // followed by a real assistant message with status="error".
      renderReplyBody(bubble, "", "error");
      if (activeChannelInfo) channelReplyInFlight = false;
    } else if (bubble) {
      // Failed before the server ever saved anything at all (e.g. an
      // unsupported/oversized attachment, rejected by
      // chat_attachment_service.save_attachments before build_reply_stream
      // ever runs — see app/routers/chat.py's stream_reply) — no
      // user_message_id event ever arrived. Leaving these bubbles in
      // place would show a "sent" message and a failed reply that a
      // reload would never reproduce (neither was ever written to the
      // database), stuck until the next reload quietly removed them —
      // so both are removed here instead, right away, and the real
      // reason is surfaced via alert since there's no bubble left to
      // attach an inline note to.
      userWrapper.remove();
      wrapper.remove();
      if (activeChannelInfo) channelReplyInFlight = false;
      alert(err.message);
    } else {
      // askAi was false and the request itself failed outright (no
      // assistant bubble ever existed to show it against).
      showInlineSendError(userWrapper, err.message);
    }
  } finally {
    ownSendInFlight = false;
    sendBtn.disabled = false;
    updateAskAiButtonState();
    chatInput.focus();
  }
}

/**
 * Thin XMLHttpRequest wrapper standing in for fetch() + ReadableStream —
 * needed only because fetch has no upload-progress event at all, and a
 * real percentage for an attached file (see updateUploadProgress below)
 * requires XMLHttpRequest's `xhr.upload.onprogress`. Starts the request
 * synchronously (before returning) so `xhr` is available immediately for
 * abort-registration and upload-progress wiring, before any network
 * activity has to resolve. Returns `{ xhr, chunks }` — `chunks` is an
 * async generator of raw text deltas, playing the same role
 * `response.body.getReader()` used to for readAssistantReplyStream's own
 * SSE-frame buffering below (unchanged) — `xhr.responseText` is already
 * a decoded string, so there's no TextDecoder step needed here the way
 * raw stream bytes required.
 */
function startXhrRequest(url, formData) {
  const xhr = new XMLHttpRequest();
  xhr.open("POST", url);
  let seenLength = 0;
  let pending = null; // {resolve, reject} for whoever's awaiting the next chunk
  const queued = [];
  let finished = false;
  let failure = null;

  function deliver(value) {
    if (pending) {
      pending.resolve(value);
      pending = null;
    } else {
      queued.push(value);
    }
  }

  xhr.addEventListener("progress", () => {
    const newText = xhr.responseText.slice(seenLength);
    seenLength = xhr.responseText.length;
    if (newText) deliver({ done: false, value: newText });
  });
  xhr.addEventListener("load", () => deliver({ done: true }));
  xhr.addEventListener("error", () => {
    failure = new Error("Network error");
    finished = true;
    if (pending) pending.reject(failure);
  });
  xhr.addEventListener("abort", () => {
    failure = new DOMException("Aborted", "AbortError");
    finished = true;
    if (pending) pending.reject(failure);
  });

  async function* chunks() {
    while (true) {
      if (queued.length) {
        const next = queued.shift();
        if (next.done) return;
        yield next.value;
        continue;
      }
      if (finished) {
        if (failure) throw failure;
        return;
      }
      const next = await new Promise((resolve, reject) => {
        pending = { resolve, reject };
      });
      if (next.done) return;
      yield next.value;
    }
  }

  xhr.send(formData);
  return { xhr, chunks: chunks() };
}

/** Shows/updates a real visual progress bar under the just-sent user
 * message's own attachment(s) (see the optimisticMessage object built in
 * sendMessage) — created on first call, then just has its value updated
 * on every subsequent xhr.upload progress event. A native <progress>
 * element rather than anything hand-rolled — the simplest possible real
 * bar, no extra library. Since the upload is one multipart body
 * regardless of how many files are in it, the fraction xhr.upload
 * reports is already naturally the *combined* progress across every
 * attached file. */
// Floor on how long the bar stays on screen once shown, in ms — over
// loopback/localhost a small attachment can finish uploading (and headers
// come back) in a handful of milliseconds, faster than the browser's next
// paint, so without this the bar gets created and removeUploadProgress'd
// again before it was ever actually rendered — invisible, not just brief.
const MIN_UPLOAD_PROGRESS_VISIBLE_MS = 400;

function updateUploadProgress(userWrapper, fraction) {
  let bar = userWrapper.querySelector(".attachment-upload-progress");
  if (!bar) {
    bar = document.createElement("progress");
    bar.className = "attachment-upload-progress mt-1 block w-full h-1.5 accent-brand-500";
    bar.max = 1;
    bar.dataset.shownAt = String(performance.now());
    userWrapper.querySelector(".markdown-body").appendChild(bar);
  }
  bar.value = fraction;
}

/** Removes updateUploadProgress's element once the upload phase is over
 * — called right after headers come back, whether or not there ever was
 * an attachment (a harmless no-op query if not). Delays the actual
 * removal if needed so the bar was on screen for at least
 * MIN_UPLOAD_PROGRESS_VISIBLE_MS — see that constant's own comment. */
function removeUploadProgress(userWrapper) {
  const bar = userWrapper.querySelector(".attachment-upload-progress");
  if (!bar) return;
  const elapsed = performance.now() - Number(bar.dataset.shownAt || 0);
  setTimeout(() => bar.remove(), Math.max(0, MIN_UPLOAD_PROGRESS_VISIBLE_MS - elapsed));
}

/**
 * Wraps readAssistantReplyStream below, registering whatever it hands
 * back as activeStreamAbortController for the duration of the call so
 * leaving this conversation (see selectConversation/clearMainPane) can
 * actually abort the underlying request instead of leaving it running
 * unseen — see that variable's own declaration for why this matters,
 * and app.services.reply_generation_service.stream_reply's
 * cancel_on_disconnect for what the server does with the resulting
 * disconnect. An XMLHttpRequest's own `.abort()` closes the connection
 * the exact same way an AbortController's would.
 */
async function streamAssistantReply(conversationId, content, askAi, userWrapper, wrapper, bubble, attachmentFiles) {
  try {
    await readAssistantReplyStream(
      conversationId,
      content,
      askAi,
      userWrapper,
      wrapper,
      bubble,
      (xhr) => {
        activeStreamAbortController = xhr;
      },
      attachmentFiles
    );
  } finally {
    activeStreamAbortController = null;
  }
}

/**
 * Sends the user's message and reads the server's Server-Sent Events
 * response as it arrives, updating `bubble`'s contents on every chunk
 * so the reply appears to "type" in real time.
 *
 * XMLHttpRequest (via startXhrRequest above) rather than fetch — the
 * only way to get a real upload-progress percentage for an attached
 * file; browser EventSource can't be used either way since it only
 * supports GET, and we need to POST the message body.
 *
 * `wrapper`/`bubble` are the reply bubble's own elements (see
 * bubbleFor) — `null` when `askAi` is false, since a normal message
 * never gets a reply bubble at all (see build_reply_stream's ask_ai=False
 * early return, which never creates an assistant Message for one), so
 * every branch below that touches either one is guarded accordingly.
 * When present, `wrapper` is tagged with the reply's real message_id as
 * soon as the first event reveals it, and that id added to
 * liveMessageIds, so the poll/live-watch paths above know this
 * particular reply is already being driven live by this very function
 * and leave it alone. `userWrapper` is the *user* message's own wrapper
 * (already showing `content`, added optimistically before this call) —
 * tagged the same way, off a dedicated "user_message_id" event
 * app.services.chat_service.build_reply_stream yields right after saving
 * it: without this, the optimistic bubble never carries a
 * data-message-id at all, so once this tab's own ownSendInFlight-driven
 * skip (see runChannelPoll/handleReplyLiveEvent) ends and polling/live-
 * watch resumes, it would see this same user message in the catch-up
 * fetch as unrecognized and create a genuine duplicate for it — a real,
 * shipped bug, not just a hypothetical one.
 */
async function readAssistantReplyStream(
  conversationId,
  content,
  askAi,
  userWrapper,
  wrapper,
  bubble,
  registerAbortable,
  attachmentFiles
) {
  // FormData rather than JSON — the browser sets its own multipart
  // boundary in Content-Type, so no header is set explicitly here (see
  // app/routers/chat.py's stream_reply, which now takes Form()/File()
  // fields instead of a JSON ChatRequest body). "attachments" is
  // repeated once per file, matching the router's list[UploadFile].
  const formData = new FormData();
  formData.append("content", content);
  formData.append("ask_ai", askAi);
  for (const file of attachmentFiles) formData.append("attachments", file);

  const { xhr, chunks } = startXhrRequest(`/api/chat/${conversationId}/stream`, formData);
  registerAbortable(xhr);
  if (attachmentFiles.length > 0) {
    // Shown at 0% immediately, rather than waiting for the first real
    // progress event — over loopback/localhost a small file can finish
    // uploading before the browser ever gets around to firing one, which
    // would otherwise mean the bar never appears at all rather than just
    // briefly (see MIN_UPLOAD_PROGRESS_VISIBLE_MS for the other half of
    // that fix).
    updateUploadProgress(userWrapper, 0);
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) updateUploadProgress(userWrapper, e.loaded / e.total);
    });
  }

  // Status/headers aren't known until the response actually starts — an
  // error response (e.g. 409) still arrives through the same `chunks`
  // stream as a normal one, so the not-ok check happens once, on the
  // first readystatechange past HEADERS_RECEIVED — the same thing
  // fetch's own `!response.ok` checked, just observed differently.
  await new Promise((resolve, reject) => {
    xhr.addEventListener("readystatechange", function onReady() {
      if (xhr.readyState < XMLHttpRequest.HEADERS_RECEIVED) return;
      xhr.removeEventListener("readystatechange", onReady);
      if (xhr.status === 200) {
        resolve();
        return;
      }
      // A 409 ("AI already replying" — see conversation_service.has_active_reply)
      // carries a specific {"detail": "..."} body worth showing verbatim
      // near the user's own message (see sendMessage's catch handler);
      // anything else falls back to the generic status-based message.
      xhr.addEventListener("loadend", () => {
        let message = `Request failed (${xhr.status})`;
        try {
          const errorBody = JSON.parse(xhr.responseText);
          if (errorBody.detail) message = errorBody.detail;
        } catch (_) {
          // Not JSON (or no body at all) — keep the generic message.
        }
        const error = new Error(message);
        error.status = xhr.status;
        reject(error);
      });
    });
  });
  removeUploadProgress(userWrapper);

  let buffer = "";
  let fullText = "";

  for await (const newText of chunks) {
    // SSE frames are separated by a blank line; a single progress event
    // can contain zero, one, or several complete frames, so we buffer
    // and split rather than assuming one delta == one frame.
    buffer += newText;
    const frames = buffer.split("\n\n");
    buffer = frames.pop(); // last piece may be an incomplete frame

    for (const frame of frames) {
      const line = frame.trim();
      if (!line.startsWith("data:")) continue;
      const payload = JSON.parse(line.slice("data:".length).trim());

      // The very first event on every send, for every conversation type
      // (see app.services.chat_service.build_reply_stream) — tags the
      // *user* message's own bubble, which otherwise never carries a
      // data-message-id at all (see readAssistantReplyStream's own
      // docstring for why that's a real bug, not just tidiness).
      // attachDeleteButtonIfEligible here too: this is the sender's own
      // tab, which the poll/live-watch paths deliberately skip re-
      // processing for a message already in liveMessageIds (see
      // runChannelPoll/handleReplyLiveEvent) — without this, a channel
      // admin/manager would never see a delete icon on their *own*
      // freshly-sent message until something unrelated forced a full
      // reconcile (switching conversations and back, a reload).
      if (payload.user_message_id && !userWrapper.dataset.messageId) {
        userWrapper.dataset.messageId = payload.user_message_id;
        liveMessageIds.add(payload.user_message_id);
        attachDeleteButtonIfEligible(userWrapper, payload.user_message_id);
      }
      // Present on every event now (see app.services.reply_generation_service)
      // — tag the bubble with it the first time it's seen (before the
      // error check below: an instant failure's very first event can
      // *be* the error, and this tag needs to land regardless, so a
      // later reconcile finds this exact DOM node instead of creating a
      // duplicate for the same message_id). Only present at all when
      // askAi is true (wrapper/bubble are null otherwise).
      if (bubble && payload.message_id && !wrapper.dataset.messageId) {
        wrapper.dataset.messageId = payload.message_id;
        liveMessageIds.add(payload.message_id);
        attachDeleteButtonIfEligible(wrapper, payload.message_id);
      }
      // A channel conversation never gets an auto-generated title (see
      // chat_service.build_reply_stream) — its header/sidebar entry
      // always show the channel's own name instead. Declared up here
      // (rather than only where the title-specific check below needs it)
      // since the "deleted" branch just below needs it too.
      const isChannel = !!channelsByConversationId[conversationId];
      if (payload.deleted) {
        // A channel admin/manager deleted this exact reply while it was
        // still generating (see conversation_service.delete_message) —
        // this tab is the sender's own connection, so it's the one place
        // that can't just wait for the next poll/live-watch tick to
        // notice: remove the bubble right here, same as
        // deleteChannelMessage does for the tab that actually clicked
        // delete.
        if (bubble) wrapper.remove();
        if (isChannel) {
          channelReplyInFlight = false;
          updateAskAiButtonState();
        }
        return;
      }
      if (payload.error) {
        // Rendered the same way a reload of this conversation would show it (see renderReplyBody) — whatever
        // text had already streamed stays visible, with the real failure reason below it (see
        // Message.error_message/MessageOut.error_message — payload.error here is that exact same text, just
        // arriving live instead of from a reload) rather than a one-off plain-text error message that reload
        // could never reproduce. Shouldn't happen server-side when askAi is false (no reply is ever attempted —
        // see build_reply_stream), but fail safe rather than silently swallow it if it somehow did.
        if (bubble) {
          renderReplyBody(bubble, fullText, "error", null, payload.error);
          return;
        }
        throw new Error(payload.error);
      }
      // (isChannel is declared above, before the "deleted" branch.) A
      // payload.title here would just be the stale "New chat" placeholder
      // for a channel and must not overwrite what's already showing.
      // Two different events can carry a title, handled identically
      // either way: an early one sent right after the message is saved
      // (see build_reply_stream) — "simple" mode already knows the real
      // title by then, so the sidebar doesn't have to wait for the
      // *entire reply* to finish just to learn something already
      // decided before a token of it was generated — and "done" below,
      // whose title is only actually new for "smart" mode (still
      // "New chat" the first time around, until the check-back below
      // picks up whatever it eventually lands on).
      if (payload.title && !isChannel) {
        updateConversationTitleDisplay(conversationId, payload.title);
      }
      if (bubble && payload.chunk) {
        fullText += payload.chunk;
        renderReplyBody(bubble, fullText, "streaming");
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
      if (payload.done) {
        if (bubble) renderReplyBody(bubble, fullText, "complete", payload.sources);
        if (isChannel && askAi) {
          channelReplyInFlight = false;
          updateAskAiButtonState();
        }
        // Deliberately returns here instead of continuing the loop to
        // let the stream close naturally: the server yields this event
        // *before* "smart" title mode's model call, if that's the
        // admin-configured mode (Settings > System) — so this response
        // doesn't have to sit through that too.
        if (!isChannel && payload.title === "New chat") {
          pollForGeneratedTitle(conversationId);
        }
        return;
      }
    }
  }
}

/** Updates both the header and this conversation's sidebar row with a
 * new title — shared by the immediate update above (when a title is
 * already known) and pollForGeneratedTitle below (once a "smart" mode
 * title finishes generating). No-ops harmlessly if `conversationId` has
 * since scrolled out of the sidebar (e.g. paginated away, if that's
 * ever added) — the header only updates when `conversationId` is still
 * the active one. */
function updateConversationTitleDisplay(conversationId, title) {
  if (conversationId === activeConversationId) {
    conversationTitleEl.textContent = title;
    conversationTitleEl.dir = detectTextDirection(title);
  }
  const item = conversationListEl.querySelector(`[data-id="${conversationId}"] button`);
  if (item) {
    item.textContent = title;
    item.dir = detectTextDirection(title);
  }
}

// How many times, and how far apart, to check back for a title that was
// still generating when its first reply finished streaming ("smart"
// mode only — see chat_service.build_reply_stream/title_service.py) —
// bounded rather than indefinite, so an unusually slow model just means
// the sidebar keeps showing "New chat" until the next reload instead of
// a timer polling forever in the background.
const TITLE_POLL_ATTEMPTS = 6;
const TITLE_POLL_INTERVAL_MS = 5000;

async function pollForGeneratedTitle(conversationId, attempt = 1) {
  let conversation;
  try {
    conversation = await api(`/api/conversations/${conversationId}`);
  } catch (_) {
    return; // e.g. the conversation was deleted in the meantime
  }
  if (conversation.title && conversation.title !== "New chat") {
    updateConversationTitleDisplay(conversationId, conversation.title);
    return;
  }
  if (attempt < TITLE_POLL_ATTEMPTS) {
    setTimeout(() => pollForGeneratedTitle(conversationId, attempt + 1), TITLE_POLL_INTERVAL_MS);
  }
}

// ---- Boot -------------------------------------------------------------

/** Loads the admin-configured channel delivery mode (see
 * app.services.chat_settings_service.get_channel_delivery_mode) — every
 * logged-in user can read this, not just admins, since this page needs
 * it to decide how to behave in a channel chat. Best-effort: a failed
 * fetch just leaves channelDeliveryMode at its already-assigned "cheap"
 * default rather than blocking the rest of the page. */
async function loadChannelDeliveryMode() {
  try {
    ({ mode: channelDeliveryMode } = await api("/api/settings/channel-delivery-mode"));
  } catch (_) {
    // Keep the default already assigned above.
  }
}

(async function init() {
  // Always lands on the empty-state banner, never auto-opening whatever
  // chat happens to be newest — every route into this page (typing the
  // URL, the sidebar logo, a redirect back from Settings/Notes) should
  // land on the same neutral "nothing open yet" screen; picking a
  // specific chat is always an explicit click (sidebar row or "New
  // chat"), never a guess made on the user's behalf.
  await Promise.all([loadChannelList(), loadConversationList(), loadChannelDeliveryMode()]);
  showEmptyState(true);
})();
