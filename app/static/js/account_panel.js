/**
 * Account panel: a quick "edit my name + picture" + "pick a UI theme" shortcut reachable from the small round
 * #account-panel-badge in the header — included on every page except Settings itself (which already has the
 * full Account tab inline, so the shortcut would be redundant there). See app/templates/_account_panel.html for
 * the shared markup this drives, app/routers/account.py for the profile endpoints, and app/static/js/theme.js
 * for ThemeManager (the same class settings.js's own Appearance section uses). Password change and the
 * code-block theme stay Settings-page-only, linked to from the bottom of the panel.
 */

const accountPanelBadgeEl = document.getElementById("account-panel-badge");
const accountPanelEl = document.getElementById("account-panel");
const accountPanelBackdropEl = document.getElementById("account-panel-backdrop");

// The badge itself renders immediately from whatever base.html already
// put on <body> at page load (see app/main.py's chat_page) — no round
// trip needed just to show the icon, same reasoning as chat.js's own
// currentAvatarUrl/currentInitials for the optimistic message bubble.
(function renderAccountPanelBadge() {
  const username = document.body.dataset.username || "";
  const letters = [document.body.dataset.firstName, document.body.dataset.lastName].filter(Boolean).map((n) => n[0]);
  const initials = (letters.length ? letters.join("") : username[0] || "?").toUpperCase();
  renderAvatar(accountPanelBadgeEl, document.body.dataset.avatarUrl || null, initials);
})();

function closeAccountPanel() {
  accountPanelBackdropEl.classList.add("hidden");
  accountPanelEl.classList.add("translate-x-full");
}

async function loadAccountPanelProfile() {
  const statusEl = document.getElementById("account-panel-status");
  statusEl.textContent = "";
  let profile;
  try {
    profile = await api("/api/account/profile");
  } catch (err) {
    statusEl.textContent = err.message;
    return;
  }
  document.getElementById("account-panel-first-name").value = profile.first_name || "";
  document.getElementById("account-panel-last-name").value = profile.last_name || "";
  renderAvatar(document.getElementById("account-panel-avatar-preview"), profile.avatar_url, profile.initials);
  document.getElementById("account-panel-avatar-remove-btn").classList.toggle("hidden", !profile.avatar_url);
}

function openAccountPanel() {
  accountPanelBackdropEl.classList.remove("hidden");
  accountPanelEl.classList.remove("translate-x-full");
  loadAccountPanelProfile();
}

accountPanelBadgeEl.addEventListener("click", openAccountPanel);
document.getElementById("account-panel-close").addEventListener("click", closeAccountPanel);
// "Go to Settings" (for password/theme, not covered by this panel) —
// lands on the Account tab specifically, not Settings' own default tab,
// via the exact same sessionStorage handoff settings.js's own
// restoreTabAfterReload already uses for this (e.g. after an Ollama
// install finishes) — "pairingSettingsActiveTab" here must stay in sync
// with settings.js's own SETTINGS_TAB_STORAGE_KEY constant.
document.getElementById("account-panel-settings-link").addEventListener("click", () => {
  try {
    sessionStorage.setItem("pairingSettingsActiveTab", "account");
  } catch (_err) {
    // Private-browsing/storage-blocked — settings.js falls back to its
    // own default tab, same as if this had never run.
  }
});
// Backdrop click only — a click that started inside the panel and
// bubbled up must not close it (same event.target check as chat.js's
// identical user-profile-modal backdrop).
accountPanelBackdropEl.addEventListener("click", closeAccountPanel);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !accountPanelEl.classList.contains("translate-x-full")) closeAccountPanel();
});

document.getElementById("account-panel-avatar-input").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = ""; // lets picking the exact same file again still fire "change"
  if (!file) return;

  const statusEl = document.getElementById("account-panel-avatar-status");
  statusEl.textContent = "Uploading…";
  statusEl.classList.remove("text-red-400");
  try {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch("/api/account/avatar", { method: "PUT", body: formData });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || "Upload failed.");
    }
    // A full reload (not just re-rendering this panel) — this new
    // picture also needs to show up in the header badge and, on the
    // chat page, in every "user"-role message bubble this tab renders
    // from now on (chat.js reads its own currentAvatarUrl once, at
    // script load, from the same <body> data attributes this reload
    // refreshes) — same reasoning settings.js's own code-theme save
    // already uses for an identical "other things on this page need to
    // pick this up too" case.
    window.location.reload();
  } catch (err) {
    statusEl.textContent = err.message;
    statusEl.classList.add("text-red-400");
  }
});

document.getElementById("account-panel-avatar-remove-btn").addEventListener("click", async () => {
  const statusEl = document.getElementById("account-panel-avatar-status");
  statusEl.textContent = "Removing…";
  statusEl.classList.remove("text-red-400");
  try {
    await api("/api/account/avatar", { method: "DELETE" });
    window.location.reload(); // see the upload handler's own comment above
  } catch (err) {
    statusEl.textContent = err.message;
    statusEl.classList.add("text-red-400");
  }
});

// ---- App-wide UI theme (see app/static/js/theme.js's ThemeManager) ----
// Swatch clicks still live-preview instantly (repaint the whole page, no network call) — only committing the
// choice happens through the single Save button below, alongside the name fields.
const accountPanelThemeManager = new ThemeManager();
let accountPanelPendingUiTheme = accountPanelThemeManager.current;

function highlightSelectedAccountPanelSwatch(themeId) {
  document.querySelectorAll("[data-theme-swatch]").forEach((btn) => {
    btn.classList.toggle("ring-2", btn.dataset.themeId === themeId);
    btn.classList.toggle("ring-brand-500", btn.dataset.themeId === themeId);
  });
}

document.querySelectorAll("[data-theme-swatch]").forEach((btn) => {
  btn.addEventListener("click", () => {
    accountPanelPendingUiTheme = btn.dataset.themeId;
    accountPanelThemeManager.apply(accountPanelPendingUiTheme);
    highlightSelectedAccountPanelSwatch(accountPanelPendingUiTheme);
  });
});

// One combined Save for the whole panel (name + picture's own Upload/Remove save themselves immediately, same
// as always; this button covers first/last name and, if a swatch was clicked, the previewed UI theme) — the
// panel used to have two separate Save buttons (profile, theme), which read as two forms in one drawer for no
// real reason: nothing here is independent enough to justify a second submit.
document.getElementById("account-panel-save-btn").addEventListener("click", async () => {
  const statusEl = document.getElementById("account-panel-status");
  const btn = document.getElementById("account-panel-save-btn");
  const first_name = document.getElementById("account-panel-first-name").value.trim();
  const last_name = document.getElementById("account-panel-last-name").value.trim();

  btn.disabled = true;
  statusEl.textContent = "Saving…";
  try {
    await api("/api/account/profile", { method: "PATCH", body: JSON.stringify({ first_name, last_name }) });
    if (accountPanelPendingUiTheme !== accountPanelThemeManager.saved) {
      await accountPanelThemeManager.save(accountPanelPendingUiTheme);
    }
    window.location.reload(); // see the avatar upload handler's own comment above — other things on this page
    // (message bubbles, the header badge) need to pick up a name/picture change too, so one reload covers both.
  } catch (err) {
    statusEl.textContent = err.message;
    btn.disabled = false;
  }
});
