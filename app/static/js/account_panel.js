/* Account panel: a quick "edit my name + picture" and "pick a UI theme" shortcut from the round #account-panel-badge in the header. Included on every page except Settings (which has the full Account tab). Markup: _account_panel.html; endpoints: app/routers/account.py; ThemeManager: theme.js. Password change and code-block theme stay in Settings, linked from the panel. */

const accountPanelBadgeEl = document.getElementById("account-panel-badge");
const accountPanelEl = document.getElementById("account-panel");
const accountPanelBackdropEl = document.getElementById("account-panel-backdrop");

// The badge renders at once from the data attributes base.html puts on <body> (no round trip), as chat.js's
// currentAvatarUrl/currentInitials.
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
// "Go to Settings" opens the Account tab via the same sessionStorage handoff as settings.js's restoreTabAfterReload;
// "pairingSettingsActiveTab" must match settings.js's SETTINGS_TAB_STORAGE_KEY.
document.getElementById("account-panel-settings-link").addEventListener("click", () => {
  try {
    sessionStorage.setItem("pairingSettingsActiveTab", "account");
  } catch (_err) {
    // Storage blocked: settings.js falls back to its default tab.
  }
});
// Backdrop click only: a click that started inside the panel must not close it (as chat.js's user-profile modal).
accountPanelBackdropEl.addEventListener("click", closeAccountPanel);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !accountPanelEl.classList.contains("translate-x-full")) closeAccountPanel();
});

document.getElementById("account-panel-avatar-input").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
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
    // Full reload, so the header badge and the "user" bubbles chat.js renders (it reads currentAvatarUrl once at load)
    // pick up the new picture.
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
    window.location.reload();
  } catch (err) {
    statusEl.textContent = err.message;
    statusEl.classList.add("text-red-400");
  }
});

// App-wide UI theme (see theme.js's ThemeManager). Swatch clicks preview live; the single Save button below commits.
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

// One combined Save for name fields and the previewed theme (picture Upload/Remove save immediately); two Save buttons
// read as two forms in one drawer.
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
    window.location.reload();
    // Reload so message bubbles and the header badge pick up a name/picture change.
  } catch (err) {
    statusEl.textContent = err.message;
    btn.disabled = false;
  }
});
