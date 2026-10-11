/*
 * "Share to a channel" for private-chat messages, plus the "Shared from a private chat" remark on the channel side
 * (loaded before chat.js, which calls ChannelShare.decorate / ChannelShare.remark in addMessageBubble and
 * streamAssistantReply). The server does the work and all checks (app/services/message_share_service.py); this offers
 * the button, lets the user pick a channel and shows the result. The shared copy is plain text posted as the sharer's
 * own message: no AI reply, no link back to the private chat.
 * 
 * Reads chat.js globals (activeConversationId, channelsByConversationId, api) at call time only.
 */
class ChannelShare {
  static #menu = null;

  /* Adds a share button beside a finished message's timestamp: private chats only, and only if the user is in a channel. Safe to call twice per bubble. */
  static decorate(wrapper, messageId, role) {
    if (!wrapper || !messageId || !["user", "assistant"].includes(role)) return;
    if (channelsByConversationId[activeConversationId]) return;
    if (Object.keys(channelsByConversationId).length === 0) return;
    const time = wrapper.querySelector(".message-time");
    if (!time || wrapper.querySelector(".share-message-btn")) return;

    let footer = wrapper.querySelector(".message-footer");
    if (!footer) {
      footer = document.createElement("div");
      footer.className = "message-footer mt-0.5 flex items-center gap-1.5 px-1";
      time.replaceWith(footer);
      time.classList.remove("mt-0.5", "px-1");
      footer.appendChild(time);
    }
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "share-message-btn text-slate-600 hover:text-brand-400 transition-colors";
    btn.title = "Share this message to a channel";
    btn.innerHTML =
      '<svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" ' +
      'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
      '<path d="M4 12v7a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-7"/><path d="m16 6-4-4-4 4"/><path d="M12 2v13"/></svg>';
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      ChannelShare.#openMenu(btn, messageId);
    });
    footer.appendChild(btn);
  }

  /* The side/style a bubble is drawn with: an AI answer shared into a channel is stored as the sharer's "user" message (so it triggers no reply) but drawn as an AI reply (left, AI colours) with the sharer's name and the remark above it. Other messages keep their role. */
  static visualRole(role, sharedFrom) {
    return sharedFrom === "assistant" ? "assistant" : role;
  }

  /* Adds the "Shared by <name> from a private chat" line above a shared channel bubble (`sharedFrom` is the original's role, "user" or "assistant", falsy for normal messages; `sharerName` is the sender's display name when known). */
  static remark(wrapper, sharedFrom, sharerName) {
    if (!wrapper || !sharedFrom || wrapper.querySelector(".shared-remark")) return;
    const note = document.createElement("div");
    note.className = "shared-remark mb-1 px-1 text-[11px] italic text-slate-500";
    const by = sharerName ? `Shared by ${sharerName}` : "Shared";
    note.textContent = `↗ ${by} from a private chat` + (sharedFrom === "assistant" ? " · AI answer" : "");
    const anchor = wrapper.querySelector(":scope > :not(.message-label-row)");
    wrapper.insertBefore(note, anchor);
  }

  static closeMenuFromOutside() {
    ChannelShare.#closeMenu();
  }

  static #closeMenu() {
    if (ChannelShare.#menu) ChannelShare.#menu.remove();
    ChannelShare.#menu = null;
  }

  static #openMenu(button, messageId) {
    const wasOpenHere = ChannelShare.#menu && ChannelShare.#menu.dataset.messageId === messageId;
    ChannelShare.#closeMenu();
    if (wasOpenHere) return;

    const menu = document.createElement("div");
    menu.dataset.messageId = messageId;
    menu.className = "min-w-[10rem] max-w-xs rounded-lg border border-slate-700 bg-slate-900 py-1 text-sm shadow-lg";
    // Inline styles, not classes: the menu is created on the fly and must be positioned from its first paint (a late
    // class left it as a flex item at the far right).
    menu.style.position = "fixed";
    menu.style.zIndex = "50";
    menu.style.visibility = "hidden";
    const heading = document.createElement("div");
    heading.className = "px-3 py-1 text-xs text-slate-500";
    heading.textContent = "Share to channel";
    menu.appendChild(heading);
    for (const channel of Object.values(channelsByConversationId)) {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "block w-full truncate px-3 py-1.5 text-left text-slate-200 hover:bg-slate-800";
      item.textContent = channel.name;
      item.addEventListener("click", (event) => {
        event.stopPropagation();
        ChannelShare.#closeMenu();
        ChannelShare.#share(button, messageId, channel);
      });
      menu.appendChild(item);
    }
    document.body.appendChild(menu);
    // Viewport coordinates, clamped so the menu is fully visible on either side.
    const rect = button.getBoundingClientRect();
    const margin = 8;
    const left = Math.min(Math.max(margin, rect.left), window.innerWidth - menu.offsetWidth - margin);
    const fitsBelow = rect.bottom + 4 + menu.offsetHeight <= window.innerHeight - margin;
    const top = fitsBelow ? rect.bottom + 4 : Math.max(margin, rect.top - 4 - menu.offsetHeight);
    menu.style.left = `${left}px`;
    menu.style.top = `${top}px`;
    menu.style.visibility = "visible";
    ChannelShare.#menu = menu;
  }

  static async #share(button, messageId, channel) {
    button.disabled = true;
    try {
      await api(`/api/conversations/${activeConversationId}/messages/${messageId}/share`, {
        method: "POST",
        body: JSON.stringify({ channel_id: channel.id }),
      });
      button.title = `Shared to ${channel.name}`;
      button.classList.remove("text-slate-600");
      button.classList.add("text-emerald-400");
    } catch (err) {
      alert(err.message || "Failed to share the message.");
    } finally {
      button.disabled = false;
    }
  }
}

document.addEventListener("click", () => ChannelShare.closeMenuFromOutside());
// A fixed menu would stay put while the chat scrolls under it.
document.addEventListener("scroll", () => ChannelShare.closeMenuFromOutside(), true);
window.addEventListener("resize", () => ChannelShare.closeMenuFromOutside());
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") ChannelShare.closeMenuFromOutside();
});
