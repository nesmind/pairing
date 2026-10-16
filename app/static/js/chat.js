/* Chat page controller: loads the conversation list into the sidebar, switches conversations, and sends messages, streaming the reply into the page as it arrives. State lives in a few plain variables, not a framework store. */

// id of the conversation in the main pane, or null before one is created/selected.
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

// Emoji picker (chat.html: #emoji-btn/#emoji-picker): 50 common, simple, kid-friendly emoji (faces, gestures, hearts),
// 8 per row in display order.
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
    // Inserted at the cursor, not appended (selectionStart/End coincide when nothing is selected).
    const start = chatInput.selectionStart ?? chatInput.value.length;
    const end = chatInput.selectionEnd ?? chatInput.value.length;
    chatInput.value = chatInput.value.slice(0, start) + emoji + chatInput.value.slice(end);
    const cursor = start + emoji.length;
    chatInput.focus();
    chatInput.setSelectionRange(cursor, cursor);
    // Picker stays open to pick several in a row; it closes on outside click, Escape, or the emoji button again.
  }

  emojiBtn.addEventListener("click", (event) => {
    event.stopPropagation();
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

// One-time "🎨 New: UI themes" sidebar announcement (chat.html), shown until dismissed. In localStorage rather than a
// per-user server flag: a one-off announcement isn't worth a DB round trip or cross-device sync.
const WHATS_NEW_THEMES_KEY = "pairing:whatsNewThemesDismissed";
const whatsNewThemesEl = document.getElementById("whats-new-themes");
(function initWhatsNewThemes() {
  let dismissed = false;
  try {
    dismissed = localStorage.getItem(WHATS_NEW_THEMES_KEY) === "1";
  } catch (_) {
    /* storage disabled: always show it */
  }
  whatsNewThemesEl.classList.toggle("hidden", dismissed);
})();
document.getElementById("whats-new-themes-dismiss").addEventListener("click", () => {
  whatsNewThemesEl.classList.add("hidden");
  try {
    localStorage.setItem(WHATS_NEW_THEMES_KEY, "1");
  } catch (_) {
    /* storage disabled: reappears next load */
  }
});
// Same sessionStorage tab handoff as account_panel.js's "Go to Settings" link (settings.js's SETTINGS_TAB_STORAGE_KEY
// reads and clears it); a plain href can't pick a tab and would land on Model.
document.getElementById("whats-new-themes-settings-link").addEventListener("click", () => {
  try {
    sessionStorage.setItem("pairingSettingsActiveTab", "account");
  } catch (_err) {
    // Storage blocked: settings.js falls back to its default tab.
  }
});

// Per-message caps, kept in sync by hand with app.config MAX_ATTACHMENT_DOCUMENTS/MAX_ATTACHMENT_IMAGES. A client-side
// early rejection; the server re-checks (chat_attachment_service.save_attachments).
const MAX_ATTACHMENT_DOCUMENTS = 5;
const MAX_ATTACHMENT_IMAGES = 1;

// Files picked via #attachment-input, sent as FormData's repeated "attachments" field on the next send, then cleared
// (sendMessage/clearPendingAttachment). Module-level so clearPendingAttachment resets the input and this together.
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

/* Rebuilds the pre-send preview chips from `pendingAttachments`, each with an "×" to drop that file. */
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
  // Merged with earlier picks: <input multiple> reports only this dialog's selection, so replacing the list would drop
  // files attached one at a time.
  const picked = [...pendingAttachments, ...Array.from(attachmentInput.files || [])];
  // Cleared so the input can be reused; pendingAttachments is the source of truth for what is queued.
  attachmentInput.value = "";
  const images = picked.filter((f) => f.type.startsWith("image/"));
  const documents = picked.filter((f) => !f.type.startsWith("image/"));
  if (images.length > MAX_ATTACHMENT_IMAGES || documents.length > MAX_ATTACHMENT_DOCUMENTS) {
    alert(`Up to ${MAX_ATTACHMENT_IMAGES} image and ${MAX_ATTACHMENT_DOCUMENTS} document files per message.`);
  }
  pendingAttachments = [...images.slice(0, MAX_ATTACHMENT_IMAGES), ...documents.slice(0, MAX_ATTACHMENT_DOCUMENTS)];
  renderAttachmentPreview();
});

// Whether the user is an admin (data attribute on <body>, see base.html and settings.js). With a channel's is_manager
// flag (channelsByConversationId) it decides whether the active chat's model/persona/rules/skill controls are editable
// or read-only. The server enforces the rule (channel_service.can_manage_channel_conversation); this only avoids
// showing a control that would 403.
const isAdmin = document.body.dataset.isAdmin === "true";
// The user's own id, so the optimistic "user" bubble in a channel can open their profile modal at once.
const currentUserId = document.body.dataset.userId || null;
// The user's own username (same data attribute as settings.js; see base.html).
const currentUsername = document.body.dataset.username || "";
// First/last name (whichever are set) with the username in parentheses, matching Message.sender_display_name
// (app/models/conversation.py); labels a just-sent channel message without a round trip.
const currentFullName = [document.body.dataset.firstName, document.body.dataset.lastName]
  .filter(Boolean)
  .join(" ");
const currentDisplayName = currentFullName ? `${currentFullName} (${currentUsername})` : currentUsername;
// Avatar URL and initials for the optimistic "user" bubble, mirroring User's avatar_url/initials (see base.html).
const currentAvatarUrl = document.body.dataset.avatarUrl || null;
const currentInitials = (() => {
  const letters = [document.body.dataset.firstName, document.body.dataset.lastName].filter(Boolean).map((n) => n[0]);
  return (letters.length ? letters.join("") : currentUsername[0] || "?").toUpperCase();
})();

// Maps a channel's shared conversation id to its ChannelSummary (GET /api/channels), so selectConversation can tell
// channel from personal chats and whether this viewer can manage it. Filled by loadChannelList.
let channelsByConversationId = {};

// Whether this viewer can change the active conversation's model/persona/rules/skill: always for a personal chat, only
// admins or the channel's manager in a channel. Read fresh by the badge click handlers, as it changes on each
// selectConversation.
let canManageActive = true;

// How other channel members see an in-progress or finished reply: "cheap" (poll) or "real" (pushed live); see
// chat_settings_service.get_channel_delivery_mode. Fetched in init; "cheap" is the default until then.
let channelDeliveryMode = "cheap";

// message_ids this tab drives itself via streamAssistantReply. The reply-watch/poll paths must skip them, or they'd
// fight the sender's smoother rendering. Reset on each conversation switch (selectConversation).
let liveMessageIds = new Set();
// True for the duration of this tab's chatForm submit handler. A channel's reply-watch connection (startReplyWatch) is
// already open before the owner sends, so the watch and the new POST /stream request both receive the first event on
// the same topic before either tags liveMessageIds, and the first event can create a second, orphaned bubble. Most
// visible on an instant failure (see handleReplyLiveEvent). Cheap-mode polling doesn't need this: its ~3s tick is
// slower than the stream tagging liveMessageIds.
let ownSendInFlight = false;
// Accumulated text per message_id for a reply watched live but not sent by this tab (channel "real" mode, or a personal
// chat reconnecting mid-reply). It is the source of truth for the bubble, not the DOM (handleReplyLiveEvent), since a
// reconcile can replace the node mid-sequence.
let replyLiveContent = new Map();
// The open live-updates connection/timer for the on-screen conversation; only one of the two is active, cleared on each
// switch by stopReplyWatch.
// This tab's in-flight POST /stream request, aborted when the user leaves its conversation
// (selectConversation/clearMainPane) so the connection actually closes. Server side
// (reply_generation_service.stream_reply's cancel_on_disconnect): a channel conversation keeps generating; a personal
// chat's reply is cancelled, since only its owner was watching.
let activeStreamAbortController = null;

let replySubscribeSource = null;
// Whether the open replySubscribeSource closes once its one reply settles (a personal chat's reconnect watch) or stays
// open (a channel's "real" mode).
let replyWatchIsOneShot = false;
let channelPollTimeoutId = null;
// The `server_time` from the last poll tick, used as the next tick's `since` so requests fetch only what's new.
let channelPollSince = null;

// Client-side mirror of the backend guard (conversation_service.has_active_reply, enforced in routers/chat.py
// stream_reply): disables #ask-ai-btn while some member's reply is in progress in the channel. Not authoritative: the
// backend answers 409 if this drifts (sendMessage's catch), and renderMessageList, handleReplyLiveEvent and
// runChannelPoll re-sync it.
let channelReplyInFlight = false;

function updateAskAiButtonState() {
  askAiBtn.disabled = channelReplyInFlight;
  askAiBtn.title = channelReplyInFlight
    ? "The AI is already replying to another message in this channel"
    : "";
}

// Show any message a redirecting page (e.g. Settings after saving) queued, early and independent of the
// conversation-list fetch below.
showQueuedSystemMessage();

// marked.js is configured once so all model output (renderMessageContent) renders alike: GitHub-flavored line breaks
// and highlight.js for fenced code. Highlighting needs a custom `code` renderer: marked v5+ (this app uses v15)
// silently ignores the old `highlight` option.
marked.use({
  renderer: {
    code({ text, lang }) {
      // Trust `lang` only if it is a real language name (it reaches an HTML class attribute) and highlight.js knows it;
      // hljs.getLanguage is undefined for hints like a model-invented "pseudocode".
      const language = lang && /^[\w+-]+$/.test(lang) && hljs.getLanguage(lang) ? lang : null;
      const highlighted = language
        ? hljs.highlight(text, { language }).value
        : hljs.highlightAuto(text).value;
      const langClass = language ? ` language-${language}` : "";
      // Every vendored theme's CSS targets the "hljs" class (app/static/vendor/highlightjs/styles/*.css); without it
      // the color scheme has no effect.
      return `<pre><code class="hljs${langClass}">${highlighted}</code></pre>`;
    },
  },
});
marked.setOptions({ breaks: true });

// Matches $$...$$ and \[...\] (display math) or \(...\) (inline math), deliberately not bare $...$: a single dollar
// sign is common in prose ("$5 and $10"). Extraction runs on the raw model text before marked.parse, whose CommonMark
// backslash-escape rule would drop the backslash from the delimiters before KaTeX saw them.
const MATH_SPAN_RE = /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)/g;

function mathPlaceholder(index) {
  // Plain letters/digits/"@", which marked and DOMPurify leave unchanged, to be swapped back for the trusted
  // KaTeX-generated HTML below.
  return `@@KATEX_MATH_${index}@@`;
}

/* Pulls math spans out of `text` before markdown parsing (see MATH_SPAN_RE), renders each with KaTeX, and leaves a placeholder. Returns the substituted text plus the rendered HTML per placeholder, for renderMessageContent to splice back after sanitization. A span still streaming in (opening \[ with no closing \]) stays plain text until it completes. */
function extractMathSpans(text) {
  const mathHtml = [];
  const placeholderText = text.replace(MATH_SPAN_RE, (match, dollarBlock, bracketBlock, parenInline) => {
    const displayMode = dollarBlock !== undefined || bracketBlock !== undefined;
    const tex = dollarBlock ?? bracketBlock ?? parenInline;
    let html;
    try {
      // throwOnError: false shows a parse error as inline red text, so a malformed \[...\] doesn't break the reply.
      html = katex.renderToString(tex, { throwOnError: false, displayMode });
    } catch (_err) {
      return match;
    }
    const index = mathHtml.length;
    mathHtml.push(html);
    return mathPlaceholder(index);
  });
  return { placeholderText, mathHtml };
}

function renderMessageContent(container, text) {
  const { placeholderText, mathHtml } = extractMathSpans(text || "");
  let html = DOMPurify.sanitize(marked.parse(placeholderText));
  // Substituted after sanitization: this HTML comes from KaTeX (trusted, not raw model text) and includes MathML tags
  // (<math>, <mrow>, <annotation>,...) that DOMPurify's HTML allowlist would strip.
  mathHtml.forEach((rendered, index) => {
    html = html.split(mathPlaceholder(index)).join(rendered);
  });
  container.innerHTML = html;
  // Direction is re-derived from the current (possibly still-streaming) text on each render; see app.js
  // detectTextDirection for why this beats `dir="auto"` on mixed Hebrew/Arabic text.
  container.dir = detectTextDirection(text || "");
}

/* Appends a "Sources: a.txt, b.pdf" line under a reply that used RAG (app/rag.py, app/routers/chat.py), showing which documents contributed. Call it after the bubble's final text is set: a later renderMessageContent replaces the innerHTML. */
function appendSourcesFooter(bubble, sources) {
  if (!sources || sources.length === 0) return;
  const footer = document.createElement("div");
  footer.className = "mt-2 pt-2 border-t border-white/10 text-xs text-slate-400";
  footer.textContent = "Sources: " + sources.join(", ");
  bubble.appendChild(footer);
}

/*
 * `senderLabel`, when given, renders a "who sent this" line above the bubble; passed only for a channel's shared
 * conversation (selectConversation, chat-send handler), where several members can post.
 * 
 * `avatarUrl`/`initials` apply to every "user"-role message, personal chats included
 * (Message.sender_avatar_url/sender_initials); never to "assistant"/"system".
 * 
 * `senderId` follows senderLabel's channel-only gating and makes the avatar clickable to open the profile modal
 * (openUserProfileModal).
 */
function bubbleFor(role, senderLabel, avatarUrl, initials, senderId) {
  const wrapper = document.createElement("div");
  wrapper.className = "flex flex-col " + (role === "user" ? "items-end" : "items-start");

  if (senderLabel) {
    // A dedicated row so attachDeleteButtonIfEligible can put a delete control beside the label; its presence also
    // marks "this is a channel message".
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
  // Text direction is set per render by renderMessageContent (it depends on the text). This affects text flow inside
  // the bubble; the side of the chat is set by the wrapper's items-end/items-start.

  if (role === "user") {
    // Bubble and avatar side by side, avatar last so it sits on the text's side (right, matching items-end).
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
  // Starts at "now" (right for a just-sent bubble); setBubbleTime overwrites it with the stored created_at for
  // persisted messages.
  const time = document.createElement("span");
  time.className = "message-time mt-0.5 px-1 text-[10px] text-slate-500";
  time.textContent = formatShortDateTime();
  wrapper.appendChild(time);
  return { wrapper, bubble };
}

/* A message's display time: when a reply finished (`completed_at`), else when it was created. */
function messageTime(message) {
  return message && (message.completed_at || message.created_at);
}

/* Sets a bubble's timestamp from a server time (no-op if unknown). */
function setBubbleTime(wrapper, createdAt) {
  const time = wrapper && wrapper.querySelector(".message-time");
  if (time && createdAt) time.textContent = formatShortDateTime(createdAt);
}

/* Adds a delete control to `wrapper`'s label row for viewers who can manage the channel (canManageActive), so an admin/manager can remove any message, even a generating one (deleteChannelMessage). No-op without a label row (personal chats) or without rights. Safe to call repeatedly: poll/live-watch bubbles are created before their id is known and rerun this (findOrCreateBubbleForMessage/readAssistantReplyStream), and an existing button is detected. */
function attachDeleteButtonIfEligible(wrapper, messageId) {
  if (!canManageActive) return;
  const labelRow = wrapper.querySelector(".message-label-row");
  if (!labelRow || labelRow.querySelector(".delete-message-btn")) return;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "delete-message-btn ms-auto shrink-0 text-slate-600 hover:text-red-400 transition-colors";
  btn.title = "Delete this message";
  // Same trash-can icon as the sidebar's "delete chat" button (renderConversationListItem), for consistency.
  btn.innerHTML =
    '<svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" ' +
    'stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0-1 14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2L4 6h16Z"/></svg>';
  btn.addEventListener("click", () => deleteChannelMessage(messageId, wrapper));
  labelRow.appendChild(btn);
}

/* Deletes a channel message for good (channel admins/managers only; enforced in routers/conversations.py delete_message, which this UI mirrors). Removes `wrapper` at once; other members see the status="deleted" change via runChannelPoll/handleReplyLiveEvent. */
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
  // Deleting a "user" message whose reply is still generating also removes that reply
  // (conversation_service.delete_message); the response lists every removed id so both bubbles go now.
  for (const id of deletedMessageIds) {
    const el = id === messageId ? wrapper : messagesEl.querySelector(`[data-message-id="${id}"]`);
    if (el) el.remove();
    //.add, not.delete: a "cheap" poll in flight at click time can resolve with the pre-delete status (e.g.
    // "streaming"), and without this runChannelPoll's liveMessageIds check would recreate the bubble, reappearing until
    // a full reload (a shipped bug).
    liveMessageIds.add(id);
  }
  if (messagesEl.children.length === 0) showEmptyState(true);
}

// Shown in place of a failed or interrupted reply's content (renderReplyBody); partial content stays above it.
const REPLY_FAILED_NOTICE = "⚠️ This reply didn't finish — generation failed or was interrupted.";

/*
 * Renders an assistant reply's body from its known content/status/sources (Message.status in
 * app/models/conversation.py): typing dots (streaming, nothing yet), the reply text (streaming or complete), or a
 * failure notice (status "error", partial content kept above). Used for every assistant bubble render (history load,
 * "cheap" poll, "real" live events and reconnect), so in-progress and failed states read the same however you arrive.
 * Undefined/"complete" status renders `content` as-is.
 * `errorText` (status "error" only) replaces the generic REPLY_FAILED_NOTICE with the real reason:
 * Message.error_message/MessageOut.error_message for history and poll (null for rows predating the field or replies
 * that never started; see sendMessage's catch), or payload.error for a live SSE event. Falls back to the generic notice
 * otherwise.
 */
function renderReplyBody(bubble, content, status, sources, errorText) {
  if (status === "streaming" && !content) {
    bubble.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
    ToolBlocks.render(bubble);
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
  ToolBlocks.render(bubble);
}

/* Appends a message's attachments (image, or a download chip for other files) to `bubble` (MessageOut.attachments/AttachmentOut in app/schemas/chat.py); no-op if none. Call it after each renderReplyBody on the same bubble, since that replaces bubble.innerHTML. */
function renderAttachment(bubble, message) {
  if (!message || !message.attachments) return;
  for (const attachment of message.attachments) {
    if (attachment.type === "image") {
      const img = document.createElement("img");
      img.src = attachment.url;
      img.alt = attachment.filename || "attachment";
      img.className = "mt-2 max-w-full rounded-lg";
      // An <img> has no height until it loads, so callers' scrollTop=scrollHeight undershoots; re-scroll once the real
      // size is known, whichever caller (send, poll, reload) rendered it.
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
  const sharedFrom = attachmentSource && attachmentSource.shared_from;
  const { wrapper, bubble } = bubbleFor(
    ChannelShare.visualRole(role, sharedFrom),
    senderLabel,
    avatarUrl,
    initials,
    senderId
  );
  if (messageId) {
    wrapper.dataset.messageId = messageId;
    attachDeleteButtonIfEligible(wrapper, messageId);
  }
  setBubbleTime(wrapper, messageTime(attachmentSource));
  // A finished private-chat message can be shared to a channel; a message shared into this channel carries a "Shared
  // from a private chat" remark (chat_share.js).
  if (messageId && (!status || status === "complete")) ChannelShare.decorate(wrapper, messageId, role);
  ChannelShare.remark(wrapper, sharedFrom, attachmentSource && attachmentSource.sender_display_name);
  bubble._toolEvents = attachmentSource && attachmentSource.tool_events;
  renderReplyBody(bubble, content, status, sources, errorText);
  renderAttachment(bubble, attachmentSource);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return { wrapper, bubble };
}

/* Finds the bubble already tracking `messageId` (tagged data-message-id by addMessageBubble/streamAssistantReply) or creates one, so poll/live-watch paths update the same node instead of appending duplicates. `created` tells the caller the message is new to this tab (handleReplyLiveEvent's reconcile-on-first-sight). */
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

/* Renders a conversation's full message list from scratch, for selectConversation (initial load) and reconcileConversationMessages (channel live-updates backstop). */
function renderMessageList(messages, channelInfo) {
  messagesEl.innerHTML = "";
  showEmptyState(messages.length === 0);
  for (const message of messages) {
    // Sender labels show only in a channel's shared conversation (see bubbleFor): "AI" for assistant replies,
    // message.sender_display_name (MessageOut) for user messages, since several members post.
    const senderLabel = channelInfo ? (message.role === "assistant" ? "AI" : message.sender_display_name) : null;
    // Clickable-to-enlarge only in a channel (see bubbleFor on senderId).
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
    // Covers the initial load and every reconcileConversationMessages call, including right after a reply finishes: the
    // most reliable checkpoint, since it reflects the database's state rather than a possibly missed live event.
    const last = messages[messages.length - 1];
    channelReplyInFlight = !!(last && last.role === "assistant" && last.status === "streaming");
    updateAskAiButtonState();
  }
}

/* Refetches and re-renders the conversation's full history: the backstop for both channel delivery modes and a personal chat's reconnect watch. The broadcast hub's bounded queue can drop events under load (reply_broadcast_service), and "real" mode has no other way to learn of another member's new user message (the hub carries only reply-generation events). Called when a watched reply finishes or when a message has no bubble yet; both are infrequent. */
async function reconcileConversationMessages(conversationId) {
  if (conversationId !== activeConversationId) return;
  let messages;
  try {
    messages = await api(`/api/conversations/${conversationId}/messages`);
  } catch (_) {
    return;
  }
  if (conversationId !== activeConversationId) return;
  renderMessageList(messages, channelsByConversationId[conversationId] || null);
}

/* Stops the live-updates mechanism (startReplyWatch/startChannelPolling) for the conversation being left; called on every switch, so one runs at a time. */
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

/* A read-only SSE connection to routers/chat.py's `/subscribe`, relaying reply_broadcast_service events for this conversation live, without this tab having sent the message. Two callers (selectConversation): a channel in "real" delivery mode, watching all activity while open; and a personal chat whose last message is still "streaming" when opened (the original connection dropped), watching until that reply settles. Plain EventSource suits it: a GET with no body, and its automatic reconnect is wanted for a long-lived watch. */
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
  // This tab's streamAssistantReply is already rendering (or about to render) the reply it sent; see ownSendInFlight
  // for the race. Skip events for the whole send, not just until liveMessageIds catches up, since that catch-up isn't
  // guaranteed to win.
  if (ownSendInFlight) return;
  const messageId = payload.message_id;
  // Once tagged, this message is known to be rendered live by this tab's own connection.
  if (!messageId || liveMessageIds.has(messageId)) return;

  if (payload.sync) {
    replyLiveContent.set(messageId, payload.content || "");
    if (payload.status !== "streaming") return;
    channelReplyInFlight = true;
    updateAskAiButtonState();
    const { bubble, created } = findOrCreateBubbleForMessage(messageId, "assistant", "AI");
    bubble._toolEvents = payload.tool_events;
    renderReplyBody(bubble, replyLiveContent.get(messageId), payload.status);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    if (created) reconcileConversationMessages(conversationId);
    return;
  }

  if (payload.tool) {
    const { bubble } = findOrCreateBubbleForMessage(messageId, "assistant", "AI");
    ToolBlocks.add(bubble, payload.tool);
  }
  if (payload.chunk) {
    channelReplyInFlight = true;
    updateAskAiButtonState();
    const text = (replyLiveContent.get(messageId) || "") + payload.chunk;
    replyLiveContent.set(messageId, text);
    const { bubble, created } = findOrCreateBubbleForMessage(messageId, "assistant", "AI");
    renderReplyBody(bubble, text, "streaming");
    messagesEl.scrollTop = messagesEl.scrollHeight;
    // A brand-new message this tab hasn't shown: fetch the prompting user message and anything missed in one reconcile,
    // rather than teaching the broadcast hub about messages it doesn't carry.
    if (created) reconcileConversationMessages(conversationId);
  }
  if (payload.deleted) {
    channelReplyInFlight = false;
    updateAskAiButtonState();
    replyLiveContent.delete(messageId);
    const existing = messagesEl.querySelector(`[data-message-id="${messageId}"]`);
    if (existing) existing.remove();
    // Deleting the question takes its reply down too (conversation_service.delete_message), but this event names only
    // the reply's id (the hub tracks generation state only), so a full reconcile removes the paired question's bubble,
    // as it is the authority for done/error below.
    reconcileConversationMessages(conversationId);
    if (replyWatchIsOneShot) stopReplyWatch();
    return;
  }
  if (payload.done || payload.error) {
    channelReplyInFlight = false;
    updateAskAiButtonState();
    replyLiveContent.delete(messageId);
    reconcileConversationMessages(conversationId);
    // A personal chat's reconnect watch cares only about the one reply it caught up on: close it rather than leave an
    // idle connection. A channel's "real"-mode watch stays open as long as the chat does, as any member may send again.
    if (replyWatchIsOneShot) stopReplyWatch();
  }
}

// How often "cheap" delivery mode checks for new or changed messages while the conversation is open: frequent enough to
// feel responsive, cheap enough to leave running. Unlike pollForGeneratedTitle, it has no attempt cap, as a channel
// chat is open-ended.
const CHANNEL_POLL_INTERVAL_MS = 3000;

function startChannelPolling(conversationId) {
  channelPollSince = null;
  scheduleChannelPoll(conversationId);
}

function scheduleChannelPoll(conversationId) {
  channelPollTimeoutId = setTimeout(() => runChannelPoll(conversationId), CHANNEL_POLL_INTERVAL_MS);
}

async function runChannelPoll(conversationId) {
  if (conversationId !== activeConversationId) return;
  let result;
  try {
    const query = channelPollSince ? `?since=${encodeURIComponent(channelPollSince)}` : "";
    result = await api(`/api/conversations/${conversationId}/messages/latest${query}`);
  } catch (_) {
    scheduleChannelPoll(conversationId);
    return;
  }
  if (conversationId !== activeConversationId) return;

  // Skip this tick while this tab's own send is in flight (see ownSendInFlight): liveMessageIds can't yet filter the
  // sender's own message. A channel reply's placeholder Message is in the database (and poll responses) as soon as
  // generation starts, before this tab's POST /stream tags it, so a slow-starting model makes this race routine.
  // channelPollSince is deliberately not advanced, so this response (including other members' messages in the window)
  // is reprocessed after the send.
  if (ownSendInFlight) {
    scheduleChannelPoll(conversationId);
    return;
  }
  channelPollSince = result.server_time;
  const channelInfo = channelsByConversationId[conversationId] || null;
  for (const message of result.messages) {
    if (message.role === "assistant") channelReplyInFlight = message.status === "streaming";
    if (message.status === "deleted") {
      // Checked before the liveMessageIds skip, deliberately: a message this tab sent (already tagged) can still be
      // deleted by someone else, and that dedup guard exists for messages being rendered live, which deletion is not.
      const existing = messagesEl.querySelector(`[data-message-id="${message.id}"]`);
      if (existing) existing.remove();
      //.add, not.delete (see deleteChannelMessage): once a message is known deleted, never recreate its bubble,
      // however stale a later response is.
      liveMessageIds.add(message.id);
      continue;
    }
    if (liveMessageIds.has(message.id)) continue;
    const senderLabel = channelInfo ? (message.role === "assistant" ? "AI" : message.sender_display_name) : null;
    const senderId = channelInfo && message.role === "user" ? message.sender_id : null;
    const { bubble } = findOrCreateBubbleForMessage(
      message.id,
      ChannelShare.visualRole(message.role, message.shared_from),
      senderLabel,
      message.sender_avatar_url,
      message.sender_initials,
      senderId
    );
    setBubbleTime(bubble.closest("[data-message-id]"), messageTime(message));
    // A message another member shared in from a private chat also arrives here while this tab is open.
    ChannelShare.remark(bubble.closest("[data-message-id]"), message.shared_from, message.sender_display_name);
    bubble._toolEvents = message.tool_events;
    renderReplyBody(bubble, message.content, message.status, message.sources, message.error_message);
    renderAttachment(bubble, message);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
  updateAskAiButtonState();
  if (messagesEl.children.length === 0) showEmptyState(true);
  scheduleChannelPoll(conversationId);
}

/* Scrolls #messages to the bottom, re-checking over a few animation frames until scrollHeight stops growing. The Tailwind CDN runtime restyles new elements asynchronously (see base.html), so right after a big batch of bubbles scrollHeight is briefly too small. Capped, so a pathological case gives up. */
function scrollMessagesToBottom() {
  let lastHeight = -1;
  let stableFrames = 0;
  const maxFrames = 30;
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
  modelBadgeEl.classList.add("hidden");
  setModelBadgeManageable(true);
  messagesEl.innerHTML = "";
  showEmptyState(true);
  askAiBtn.classList.add("hidden");
  channelReplyInFlight = false;
  updateAskAiButtonState();
}

/* Toggles the model badge between "click to switch" (personal chats, and channel chats for an admin/manager) and plain read-only display (other channel viewers). Cosmetic: enforcement is server-side (openModelPicker's guard below). */
function setModelBadgeManageable(manageable) {
  modelBadgeEl.classList.toggle("cursor-default", !manageable);
  modelBadgeEl.classList.toggle("opacity-60", !manageable);
  modelBadgeEl.title = manageable
    ? "Switch model for this chat"
    : "Only admins and this channel's managers can change its model";
}

// Same icons as the Notes page pin chips (notes.js), copied since these static files share no module system.
const PIN_TYPE_ICONS = { persona: "🎭", rules: "📏", skill: "🛠️" };
const PIN_TYPE_LABELS = { persona: "Persona", rules: "Rules", skill: "Skill" };

const slotEditorEl = document.getElementById("slot-editor");
const slotEditorTitleEl = document.getElementById("slot-editor-title");
const slotEditorContentEl = document.getElementById("slot-editor-content");
const slotEditorOnlyThisChatEl = document.getElementById("slot-editor-only-this-chat");
const slotEditorStatusEl = document.getElementById("slot-editor-status");
const slotEditorUseDefaultBtn = document.getElementById("slot-editor-use-default");
const slotEditorTurnOffBtn = document.getElementById("slot-editor-turn-off");

// The slot open in the editor, so its buttons know the conversation/pin_type; null while the panel is closed.
let editingSlot = null;

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
  // Reflects the slot's real state: checked for a per-chat override, unchecked on the shared default (otherwise a
  // customized slot looked like its override was lost).
  slotEditorOnlyThisChatEl.checked = slot.is_override;
  // "Use default"/"Enable" is the same action (POST use-default) with a different label: an active override reverts to
  // the shared default, an off slot is turned back on. Hidden when it's already the active default.
  slotEditorUseDefaultBtn.classList.toggle("hidden", slot.active && !slot.is_override);
  slotEditorUseDefaultBtn.textContent = slot.active ? "Use default" : "Enable";
  // "Turn off" applies only to a slot that is on.
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

/* Fetches the installed models into the picker opened from the model badge: a plain name list (no install here), only to change an existing conversation's model. */
async function openModelPicker() {
  if (!activeConversationId || !canManageActive) return;
  closeSlotEditor();
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
    // Full tag, wrapped not truncated: a pull path like
    // "hf.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF:Ministral-3-3B-Instruct-2512-Q4_K_M" is what an admin needs to
    // tell similar models apart, and `truncate` cut every Ministral variant to the same prefix. `break-all` because a
    // long unbroken segment (the GGUF filename) has no natural wrap point. `title` stays as a hover fallback (as
    // settings.js renderModelRow's pathLine).
    btn.className = "w-full break-all rounded-md px-2 py-1.5 text-start font-mono text-xs leading-snug transition-colors " + (
      model === currentModel ? "bg-slate-800 text-slate-100" : "text-slate-300 hover:bg-slate-800"
    );
    btn.textContent = model;
    btn.title = model;
    btn.addEventListener("click", () => switchModel(model));
    modelPickerListEl.appendChild(btn);
  }
}

/* Switches the active conversation to another model going forward; past messages keep the model that generated them. */
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

// User profile modal (channels only)
// Opened from a message's round avatar (see bubbleFor: wired only for a channel's shared conversation). Enlarges the
// picture and shows what GET /api/account/{id} returns; a place to add fields without touching each message's payload.

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
  // With no name set, the username is the main line rather than an empty line above a duplicate.
  userProfileModalNameEl.textContent = fullName || profile.username;
  userProfileModalUsernameEl.textContent = fullName ? `@${profile.username}` : "";
}

document.getElementById("user-profile-modal-close").addEventListener("click", closeUserProfileModal);
// Closes on a backdrop click only (checks event.target, not currentTarget): a click that started inside the panel and
// bubbled up must not close it.
userProfileModalEl.addEventListener("click", (event) => {
  if (event.target === userProfileModalEl) closeUserProfileModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !userProfileModalEl.classList.contains("hidden")) closeUserProfileModal();
});

/* Shows one clickable icon per persona/rules/skill slot (GET /api/notes/slots/{id}), always all 3 in the same order so the row doesn't jump; an off slot fades but stays clickable to turn it back on (openSlotEditor's "Enable"). Best-effort: a failed lookup shows no icons. */
let pinnedSlots = [];
let activeHasMessages = false;

async function loadPinnedBadges(conversationId) {
  let slots;
  try {
    slots = await api(`/api/notes/slots/${conversationId}`);
  } catch (_) {
    return;
  }
  // Not awaited by the caller (selectConversation): if the user switched conversations meanwhile, the result is stale
  // and must not overwrite the new icons.
  if (conversationId !== activeConversationId) return;
  pinnedSlots = slots;
  renderPinnedBadges();
}

/* Draws the 3 badges from `pinnedSlots`; clickable only for someone who can manage the chat, and only while it has no messages. */
function renderPinnedBadges() {
  const editable = canManageActive && !activeHasMessages;
  const conversationId = activeConversationId;
  pinnedBadgesEl.innerHTML = "";
  for (const slot of pinnedSlots) {
    const badge = document.createElement("button");
    badge.type = "button";
    const look = !editable
      ? "bg-slate-800/60 opacity-60 cursor-not-allowed"
      : slot.active
        ? "bg-slate-800" + (editable ? " hover:bg-slate-700" : " cursor-default")
        : "bg-slate-800/40 opacity-40" + (editable ? " hover:opacity-70" : " cursor-default");
    badge.className = "rounded-full px-1.5 py-0.5 transition-colors " + look;
    const state = !slot.active ? " (off)" : slot.is_override ? " (customized)" : "";
    const readOnlyNote = !canManageActive
      ? " — read-only (admins/managers only)"
      : activeHasMessages ? " — locked: can only change before the first message" : "";
    badge.title = `${PIN_TYPE_LABELS[slot.pin_type] || slot.pin_type}${state}: ${slot.title || ""}${readOnlyNote}`;
    badge.textContent = PIN_TYPE_ICONS[slot.pin_type] || "📌";
    // Reads canManageActive at click time, not render time (see its declaration).
    badge.addEventListener("click", () => {
      if (canManageActive && !activeHasMessages) openSlotEditor(conversationId, slot);
    });
    pinnedBadgesEl.appendChild(badge);
  }
}

// Sidebar / conversation list

/* A "No chats"/"No channels" row for an empty sidebar section, so an empty list doesn't look like loading or broken. */
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
  // text-left, not text-start: the row stays left-aligned in the sidebar's direction whatever the title's language;
  // only the text (via dir below) follows Hebrew/Arabic direction.
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

// Sidebar / channel list

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

  // No delete button: a channel's lifecycle (create/rename/membership/delete) is admin-only, in Admin settings >
  // Channels (app/routers/channels.py).
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
  closeSlotEditor();
  closeModelPicker();
  highlightActiveInSidebar();
  // Closes the connection for the conversation being left if this tab has a send in flight there (see
  // activeStreamAbortController: a channel keeps generating, a personal chat's reply is cancelled).
  activeStreamAbortController?.abort();
  activeStreamAbortController = null;
  stopReplyWatch();
  liveMessageIds = new Set();

  const channelInfo = channelsByConversationId[id] || null;
  canManageActive = !channelInfo || isAdmin || channelInfo.is_manager;
  setModelBadgeManageable(canManageActive);
  askAiBtn.classList.toggle("hidden", !channelInfo);

  const conversation = await api(`/api/conversations/${id}`);
  // A channel conversation's display name is the channel's `name` (Admin settings > Channels), never
  // conversation.title; chat_service.build_reply_stream's title guard leaves that field alone.
  const titleText = channelInfo ? channelInfo.name : (conversation.title || "New chat");
  conversationTitleEl.textContent = titleText;
  conversationTitleEl.dir = detectTextDirection(titleText);
  modelBadgeEl.textContent = conversation.model || "";
  modelBadgeEl.classList.remove("hidden");
  toolToggle.sync(id, conversation, canManageActive);
  activeHasMessages = false;
  loadPinnedBadges(id);

  const messages = await api(`/api/conversations/${id}/messages`);
  renderMessageList(messages, channelInfo);
  if (id === activeConversationId) {
    activeHasMessages = messages.length > 0;
    renderPinnedBadges();
  }

  if (channelInfo) {
    // Watch for other members' activity in this channel, via the mechanism chosen by channelDeliveryMode (see its
    // declaration), for as long as the channel is open.
    if (channelDeliveryMode === "real") startReplyWatch(id);
    else startChannelPolling(id);
  } else {
    // A personal chat has one possible sender and no delivery-mode choice. If its last message is still "streaming",
    // the original connection is gone (disconnect, reload, another tab/device) while generation continued in the
    // background (reply_generation_service). Reconnect and watch that one reply to done/error, then stop
    // (handleReplyLiveEvent's replyWatchIsOneShot); nothing else arrives until the owner sends again, which
    // streamAssistantReply drives.
    const lastMessage = messages[messages.length - 1];
    if (lastMessage && lastMessage.role === "assistant" && lastMessage.status === "streaming") {
      startReplyWatch(id, true);
    }
  }
}

async function createNewConversation() {
  const conversation = await api("/api/conversations", { method: "POST", body: JSON.stringify({}) });
  // Select first, render the sidebar row second: selectConversation sets activeConversationId, which lets messages
  // send. If row rendering breaks, the chat still works and the sidebar is merely stale.
  await selectConversation(conversation.id);
  // The only <p> appended directly here is the "No chats" placeholder (rows are <div data-id>, see
  // renderConversationListItem); clear it before the first row so it doesn't show beside a non-empty list.
  const placeholder = conversationListEl.querySelector("p");
  if (placeholder) placeholder.remove();
  conversationListEl.prepend(renderConversationListItem(conversation));
}

/* Deletes a conversation for good (confirmed first). If it was on screen, falls back to the first remaining chat, or an empty "start a new chat" state if it was the last. */
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

// Sending a message + streaming the reply

/* Grows the textarea with its content up to a max height, and flips its typing direction live for Hebrew/Arabic input. */
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

/* Shows a small red note under the user's bubble for a send that produced no assistant record: a 409 "AI busy" (conversation_service.has_active_reply) or a plain (ask_ai=false) message whose request failed. Distinct from renderReplyBody's REPLY_FAILED_NOTICE, which is for a reply that started and then failed; this matches what reloading would show. */
function showInlineSendError(userWrapper, message) {
  const notice = document.createElement("div");
  notice.className = "mt-1 px-1 text-xs text-red-400";
  notice.textContent = message;
  userWrapper.appendChild(notice);
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  // Plain Send/Enter posts a normal message in a channel, with no AI reply (that is #ask-ai-btn). A personal chat
  // always asks the AI; build_reply_stream's ask_ai coercion enforces this server-side too.
  const activeChannelInfo = activeConversationId ? channelsByConversationId[activeConversationId] : null;
  await sendMessage(!activeChannelInfo);
});

askAiBtn.addEventListener("click", () => sendMessage(true));

/* Sends #chat-input as a normal channel message (askAi=false; the AI isn't invoked, see build_reply_stream's early return) or an explicit "ask AI" request (askAi=true; the only behavior for personal chats). Shared by #chat-form's submit and #ask-ai-btn's click. */
async function sendMessage(askAi) {
  const content = chatInput.value.trim();
  if (!content) return;

  if (!activeConversationId) {
    await createNewConversation();
  }
  // Captured once for this handler (including the reconcile in `finally`): activeConversationId can change mid-stream
  // if the user switches (activeStreamAbortController), and the reconcile/poll helpers no-op once their conversationId
  // stops matching.
  const conversationId = activeConversationId;

  // Captured and cleared from the UI at once, so a file picked during the send is never attached to it.
  const attachmentFiles = pendingAttachments;
  clearPendingAttachment();

  showEmptyState(false);
  chatInput.value = "";
  chatInput.style.height = "auto";
  sendBtn.disabled = true;
  askAiBtn.disabled = true;

  // Shown only for a channel's shared conversation (see bubbleFor); channelsByConversationId is filled by
  // loadChannelList.
  const activeChannelInfo = channelsByConversationId[conversationId] || null;
  // Shown at once via local object URLs/file names, before the server round trip. findOrCreateBubbleForMessage leaves
  // it alone once real URLs are known, and a full renderMessageList rebuild (e.g. reconcileConversationMessages) uses
  // the server's MessageOut fields, so these local URLs serve only this optimistic render.
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
  if (!activeHasMessages) {
    activeHasMessages = true;
    renderPinnedBadges();
  }

  // A normal (askAi=false) message gets no reply and no "thinking" bubble; build_reply_stream's early return never
  // creates an assistant Message.
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
    // AbortError means the user navigated away (activeStreamAbortController); that bubble is already gone
    // (selectConversation/clearMainPane rebuilt #messages), so show nothing.
    if (err.name === "AbortError") {
      // The pane isn't on screen anymore: nothing to re-sync.
    } else if (err.status === 409) {
      // Nothing was created server-side (routers/chat.py stream_reply): someone else's reply is still active, so
      // channelReplyInFlight is left as-is; this rejection confirms it.
      showInlineSendError(userWrapper, err.message);
    } else if (bubble && userWrapper.dataset.messageId) {
      // The user message was persisted (readAssistantReplyStream's "user_message_id" tagging) before generation failed,
      // matching what reopening would show (renderReplyBody): a user message then an assistant message with
      // status="error".
      renderReplyBody(bubble, "", "error");
      if (activeChannelInfo) channelReplyInFlight = false;
    } else if (bubble) {
      // Failed before the server saved anything (e.g. an unsupported/oversized attachment rejected by
      // chat_attachment_service.save_attachments before build_reply_stream; routers/chat.py stream_reply), so no
      // user_message_id event arrived. The "sent" message and failed reply would vanish on reload, so remove both now
      // and show the reason via alert, as no bubble is left for an inline note.
      userWrapper.remove();
      wrapper.remove();
      if (activeChannelInfo) channelReplyInFlight = false;
      alert(err.message);
    } else {
      // askAi was false and the request failed outright (no assistant bubble to show it on).
      showInlineSendError(userWrapper, err.message);
    }
  } finally {
    ownSendInFlight = false;
    sendBtn.disabled = false;
    updateAskAiButtonState();
    chatInput.focus();
  }
}

/* Thin XMLHttpRequest wrapper standing in for fetch + ReadableStream, because fetch has no upload-progress event (updateUploadProgress needs `xhr.upload.onprogress`). Starts the request synchronously so `xhr` is available at once for abort and progress wiring. Returns `{ xhr, chunks }`, where `chunks` is an async generator of raw text deltas, like `response.body.getReader` for readAssistantReplyStream's SSE buffering; `xhr.responseText` is already decoded, so no TextDecoder. */
function startXhrRequest(url, formData) {
  const xhr = new XMLHttpRequest();
  xhr.open("POST", url);
  let seenLength = 0;
  let pending = null;
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

/* Shows/updates a progress bar under the just-sent user message's attachment(s) (optimisticMessage in sendMessage): created on first call, updated on each xhr.upload progress event. A native <progress> element; as the upload is one multipart body, the fraction is the combined progress of all files. */
// Minimum time (ms) the bar stays on screen: over localhost a small attachment can finish before the next paint, so the
// bar would be created and removed (removeUploadProgress) unseen.
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

/* Removes updateUploadProgress's element when the upload phase ends (called after headers arrive, with or without an attachment; a harmless no-op query otherwise), delaying removal to honor MIN_UPLOAD_PROGRESS_VISIBLE_MS. */
function removeUploadProgress(userWrapper) {
  const bar = userWrapper.querySelector(".attachment-upload-progress");
  if (!bar) return;
  const elapsed = performance.now() - Number(bar.dataset.shownAt || 0);
  setTimeout(() => bar.remove(), Math.max(0, MIN_UPLOAD_PROGRESS_VISIBLE_MS - elapsed));
}

/* Wraps readAssistantReplyStream, registering its request as activeStreamAbortController for the call so leaving the conversation (selectConversation/clearMainPane) can abort it instead of leaving it running unseen (see that variable and stream_reply's cancel_on_disconnect in reply_generation_service). An XMLHttpRequest's `.abort` closes the connection as an AbortController would. */
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

/*
 * Sends the user's message and reads the Server-Sent Events response as it arrives, updating `bubble` on every chunk so
 * the reply "types" live.
 * 
 * XMLHttpRequest (startXhrRequest) rather than fetch gives a real upload-progress percentage; EventSource is out since
 * it supports only GET and we POST the body.
 * 
 * `wrapper`/`bubble` are the reply bubble's elements (bubbleFor), `null` when `askAi` is false (a normal message gets
 * no reply bubble; build_reply_stream's early return creates no assistant Message), so branches touching them are
 * guarded. `wrapper` is tagged with the reply's message_id once the first event reveals it and added to liveMessageIds,
 * so poll/live-watch paths leave it alone. `userWrapper` is the user message's wrapper (already showing `content`,
 * added optimistically), tagged from the "user_message_id" event that chat_service.build_reply_stream yields right
 * after saving it. Without that tag, once the ownSendInFlight skip (runChannelPoll/handleReplyLiveEvent) ends, the
 * catch-up fetch would see the user message as unrecognized and duplicate it (a shipped bug).
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
  // FormData, not JSON: the browser sets the multipart boundary in Content-Type, so no header is set (routers/chat.py
  // stream_reply takes Form/File fields). "attachments" repeats per file, matching list[UploadFile].
  const formData = new FormData();
  formData.append("content", content);
  formData.append("ask_ai", askAi);
  for (const file of attachmentFiles) formData.append("attachments", file);

  const { xhr, chunks } = startXhrRequest(`/api/chat/${conversationId}/stream`, formData);
  registerAbortable(xhr);
  if (attachmentFiles.length > 0) {
    // Shown at 0% at once: over localhost a small file can finish before the browser fires a progress event, so the bar
    // would never appear (see MIN_UPLOAD_PROGRESS_VISIBLE_MS).
    updateUploadProgress(userWrapper, 0);
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) updateUploadProgress(userWrapper, e.loaded / e.total);
    });
  }

  // Status/headers are unknown until the response starts; an error response (e.g. 409) arrives through the same
  // `chunks` stream, so the not-ok check runs once, at the first readystatechange past HEADERS_RECEIVED (fetch's
  // `!response.ok` equivalent).
  await new Promise((resolve, reject) => {
    xhr.addEventListener("readystatechange", function onReady() {
      if (xhr.readyState < XMLHttpRequest.HEADERS_RECEIVED) return;
      xhr.removeEventListener("readystatechange", onReady);
      if (xhr.status === 200) {
        resolve();
        return;
      }
      // A 409 ("AI already replying", conversation_service.has_active_reply) carries a {"detail": "..."} body worth
      // showing verbatim near the user's message (sendMessage's catch); anything else gets the generic status message.
      xhr.addEventListener("loadend", () => {
        let message = `Request failed (${xhr.status})`;
        try {
          const errorBody = JSON.parse(xhr.responseText);
          if (errorBody.detail) message = errorBody.detail;
        } catch (_) {
          // Not JSON or no body: keep the generic message.
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
    // SSE frames end with a blank line, and one progress event can hold zero, one or several complete frames, so buffer
    // and split instead of assuming one delta is one frame.
    buffer += newText;
    const frames = buffer.split("\n\n");
    buffer = frames.pop();

    for (const frame of frames) {
      const line = frame.trim();
      if (!line.startsWith("data:")) continue;
      const payload = JSON.parse(line.slice("data:".length).trim());

      // The first event of every send, for every conversation type (build_reply_stream): tags the user message's
      // bubble, which otherwise has no data-message-id (see readAssistantReplyStream's docstring on the duplicate bug).
      // Also calls attachDeleteButtonIfEligible: the sender's tab skips re-processing messages in liveMessageIds
      // (runChannelPoll/handleReplyLiveEvent), so a channel admin/manager would see no delete icon on their own new
      // message until a reconcile.
      if (payload.user_message_id && !userWrapper.dataset.messageId) {
        userWrapper.dataset.messageId = payload.user_message_id;
        liveMessageIds.add(payload.user_message_id);
        attachDeleteButtonIfEligible(userWrapper, payload.user_message_id);
        ChannelShare.decorate(userWrapper, payload.user_message_id, "user");
      }
      // Present on every event (reply_generation_service): tag the bubble the first time it's seen, before the error
      // check, since an instant failure's first event can be the error and a later reconcile must find this node
      // instead of duplicating the message_id. Only present when askAi is true (wrapper/bubble are null otherwise).
      if (bubble && payload.message_id && !wrapper.dataset.messageId) {
        wrapper.dataset.messageId = payload.message_id;
        liveMessageIds.add(payload.message_id);
        attachDeleteButtonIfEligible(wrapper, payload.message_id);
      }
      // A channel conversation never gets an auto-generated title (build_reply_stream); its header and sidebar entry
      // show the channel's name. Declared here because the "deleted" branch below needs it too.
      const isChannel = !!channelsByConversationId[conversationId];
      if (payload.deleted) {
        // A channel admin/manager deleted this reply while it was generating (conversation_service.delete_message).
        // This is the sender's connection and can't wait for a poll/live-watch tick, so remove the bubble now, as
        // deleteChannelMessage does for the clicking tab.
        if (bubble) wrapper.remove();
        if (isChannel) {
          channelReplyInFlight = false;
          updateAskAiButtonState();
        }
        return;
      }
      if (payload.error) {
        // Rendered as a reload would show it (renderReplyBody): streamed text stays visible with the real failure
        // reason below (Message.error_message/MessageOut.error_message; payload.error is the same text arriving live).
        // Shouldn't happen server-side when askAi is false (no reply is attempted; build_reply_stream), but fail safe
        // rather than swallow it.
        if (bubble) {
          renderReplyBody(bubble, fullText, "error", null, payload.error);
          return;
        }
        throw new Error(payload.error);
      }
      // (isChannel is declared above the "deleted" branch.) For a channel, payload.title is the stale "New chat"
      // placeholder and must not overwrite what's showing.
      // Two events can carry a title, handled alike: an early one right after the message is saved (build_reply_stream;
      // "simple" mode already knows the title, so the sidebar needn't wait for the whole reply) and "done" below, whose
      // title is new only for "smart" mode (still "New chat" until the check-back picks it up).
      if (payload.title && !isChannel) {
        updateConversationTitleDisplay(conversationId, payload.title);
      }
      if (bubble && payload.tool) ToolBlocks.add(bubble, payload.tool);
      if (bubble && payload.chunk) {
        fullText += payload.chunk;
        renderReplyBody(bubble, fullText, "streaming");
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
      if (payload.done) {
        if (bubble) renderReplyBody(bubble, fullText, "complete", payload.sources);
        // The reply's time is when it finished, not when its bubble appeared.
        if (wrapper) setBubbleTime(wrapper, new Date().toISOString());
        if (bubble && wrapper.dataset.messageId) ChannelShare.decorate(wrapper, wrapper.dataset.messageId, "assistant");
        if (isChannel && askAi) {
          channelReplyInFlight = false;
          updateAskAiButtonState();
        }
        // Return here instead of looping until the stream closes: the server yields this event before "smart" title
        // mode's model call (Admin settings > System), so the response needn't wait for it.
        if (!isChannel && payload.title === "New chat") {
          pollForGeneratedTitle(conversationId);
        }
        return;
      }
    }
  }
}

/* Updates the header and the sidebar row with a new title; shared by the immediate update above and pollForGeneratedTitle ("smart" mode). No-op if `conversationId` is gone from the sidebar; the header updates only if it is still active. */
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

// How many times, and how far apart, to check for a title still generating when the first reply finished ("smart" mode;
// chat_service.build_reply_stream/title_service.py). Bounded, so a slow model leaves "New chat" until reload rather
// than a timer polling forever.
const TITLE_POLL_ATTEMPTS = 6;
const TITLE_POLL_INTERVAL_MS = 5000;

async function pollForGeneratedTitle(conversationId, attempt = 1) {
  let conversation;
  try {
    conversation = await api(`/api/conversations/${conversationId}`);
  } catch (_) {
    return;
  }
  if (conversation.title && conversation.title !== "New chat") {
    updateConversationTitleDisplay(conversationId, conversation.title);
    return;
  }
  if (attempt < TITLE_POLL_ATTEMPTS) {
    setTimeout(() => pollForGeneratedTitle(conversationId, attempt + 1), TITLE_POLL_INTERVAL_MS);
  }
}

// Boot

/* Loads the admin-configured channel delivery mode (chat_settings_service.get_channel_delivery_mode), readable by every logged-in user since the page needs it in channel chats. Best-effort: a failed fetch keeps the "cheap" default. */
async function loadChannelDeliveryMode() {
  try {
    ({ mode: channelDeliveryMode } = await api("/api/settings/channel-delivery-mode"));
  } catch (_) {
    // Keep the default assigned above.
  }
}

(async function init() {
  // Always land on the empty-state banner, never auto-open the newest chat: every route here (URL, sidebar logo,
  // redirect from Settings/Notes) shows the same neutral screen, and opening a chat is an explicit click.
  await Promise.all([loadChannelList(), loadConversationList(), loadChannelDeliveryMode()]);
  showEmptyState(true);
})();
