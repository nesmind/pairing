/* Admin settings > Update (app/templates/_update_panel.html): checks and applies updates to pAIring and a local Matricxon via /api/settings/update, with the installers' progress bar (startInstallActivity, setInstallBarIndeterminate from settings.js). */
class UpdateCard {
  constructor(panel, key, root) {
    this.panel = panel;
    this.key = key;
    this.root = root;
    this.field = (name) => root.querySelector(`[data-field="${name}"]`);
    this.button = root.querySelector('[data-action="update"]');
    this.button.addEventListener("click", () => this.run());
    this.restartButton = root.querySelector('[data-action="restart"]');
    this.restartButton?.addEventListener("click", () => panel.restartPairing());
  }

  render(status) {
    this.root.classList.toggle("hidden", !status);
    if (!status) return;
    this.field("method").textContent = status.method === "git" ? "git" : status.method ? "download" : "";
    this.field("path").textContent = status.path || "";
    this.field("version").textContent = this.versionText(status);
    const note = this.field("note");
    note.textContent = status.note || "";
    note.classList.toggle("hidden", !status.note);
    const changes = this.field("changes");
    changes.replaceChildren(...status.changes.map((line) => Object.assign(document.createElement("li"), { textContent: line })));
    changes.classList.toggle("hidden", !status.changes.length);
    this.button.disabled = !status.available || status.update_available === false;
    this.button.textContent = status.update_available === false ? "Up to date" : "Update";
  }

  versionText(status) {
    if (!status.current && !status.latest) return "";
    const current = status.current || "unknown";
    if (status.update_available === false) return `Version ${current} — up to date`;
    const behind = status.behind ? ` (${status.behind} new change${status.behind === 1 ? "" : "s"})` : "";
    return `Installed ${current} → latest ${status.latest || "?"}${behind}`;
  }

  appendLog(text) {
    const log = this.field("install-log");
    log.appendChild(Object.assign(document.createElement("div"), { textContent: text }));
    while (log.children.length > 300) log.removeChild(log.firstChild);
    log.scrollTop = log.scrollHeight;
  }

  showProgress(payload, activity) {
    const bar = this.field("install-bar");
    if (payload.step && payload.total_steps) {
      this.field("install-step-label").textContent = `Step ${payload.step} of ${payload.total_steps}: ${payload.step_label || ""}`;
    }
    const mb = (bytes) => (bytes / 1e6).toFixed(1);
    if (payload.completed && payload.total) {
      const pct = Math.round((payload.completed / payload.total) * 100);
      bar.value = pct;
      setInstallBarIndeterminate(bar, false);
      activity.setProgress(payload.unit === "bytes" ? `${pct}% · ${mb(payload.completed)} / ${mb(payload.total)} MB` : `${pct}%`);
    } else {
      setInstallBarIndeterminate(bar, true);
      activity.setProgress(payload.completed && payload.unit === "bytes" ? `${mb(payload.completed)} MB` : "");
    }
    if (payload.status) this.appendLog(payload.status);
  }

  async run() {
    const panelEl = this.field("install-progress-panel");
    const labelEl = this.field("install-step-label");
    const bar = this.field("install-bar");
    panelEl.classList.remove("hidden");
    this.field("install-log").innerHTML = "";
    labelEl.textContent = "Starting…";
    this.panel.setBusy(true);
    const activity = startInstallActivity(labelEl, this.field("install-percent"));
    setInstallBarIndeterminate(bar, true);
    try {
      const done = await this.stream(activity);
      activity.stop();
      setInstallBarIndeterminate(bar, false);
      bar.value = 100;
      labelEl.textContent = done.changed ? "Updated." : "Already up to date.";
      this.field("install-percent").textContent = "100%";
      if (this.key === "pairing" && done.changed) this.panel.offerRestart();
    } catch (err) {
      labelEl.textContent = `Failed: ${err.message}`;
      this.appendLog(err.message);
    } finally {
      activity.stop();
      setInstallBarIndeterminate(bar, false);
      this.panel.setBusy(false);
      await this.panel.refresh(false);
    }
  }

  async stream(activity) {
    const response = await fetch(`${UpdatePanel.URL}/${this.key}`, { method: "POST" });
    if (!response.ok || !response.body) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Update failed (${response.status})`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop();
      for (const frame of frames) {
        const line = frame.trim();
        if (!line.startsWith("data:")) continue;
        const payload = JSON.parse(line.slice(5).trim());
        if (payload.error) throw new Error(payload.error);
        if (payload.done) return payload;
        this.showProgress(payload, activity);
      }
    }
    throw new Error("The connection closed before the update finished.");
  }
}

class UpdatePanel {
  static URL = "/api/settings/update";

  constructor(root) {
    this.root = root;
    this.loading = root.querySelector("#update-loading");
    this.loadingText = this.loading.querySelector('[data-field="loading-text"]');
    this.spinner = this.loading.querySelector('[data-field="spinner"]');
    this.checkButton = root.querySelector("#update-check");
    this.noMatricxon = root.querySelector("#update-no-matricxon");
    this.cards = ["pairing", "matricxon"].map((key) => new UpdateCard(this, key, root.querySelector(`[data-update="${key}"]`)));
    this.restartBox = root.querySelector('[data-field="restart"]');
    this.restartInfo = null;
    this.busy = false;
    this.checkButton.addEventListener("click", () => this.refresh());
    document.querySelector('[data-tab="update"]')?.addEventListener("click", () => this.refresh());
    if (!document.getElementById("tab-update").classList.contains("hidden")) this.refresh();
  }

  setBusy(busy) {
    this.busy = busy;
    this.checkButton.disabled = busy;
    for (const card of this.cards) card.button.disabled = busy || card.button.disabled;
  }

  async refresh(showLoading = true) {
    if (this.busy) return;
    if (showLoading) this.loading.classList.remove("hidden");
    this.loadingText.textContent = "Checking for updates…";
    this.spinner.classList.remove("hidden");
    this.checkButton.disabled = true;
    try {
      const overview = await api(`${UpdatePanel.URL}/status`);
      this.restartInfo = overview.restart;
      this.cards[0].render(overview.pairing);
      this.cards[1].render(overview.matricxon);
      this.noMatricxon.classList.toggle("hidden", Boolean(overview.matricxon));
      this.loading.classList.add("hidden");
    } catch (err) {
      this.loadingText.textContent = `Could not check for updates: ${err.message}`;
      this.spinner.classList.add("hidden");
      this.loading.classList.remove("hidden");
    } finally {
      this.checkButton.disabled = false;
    }
  }

  offerRestart() {
    const info = this.restartInfo;
    if (!info) return;
    this.restartBox.classList.remove("hidden");
    const text = this.restartBox.querySelector('[data-field="restart-text"]');
    const button = this.restartBox.querySelector('[data-action="restart"]');
    text.textContent = info.can_restart
      ? "Restart pAIring to start using the new version."
      : `Restart pAIring to start using the new version: ${info.hint}`;
    button.classList.toggle("hidden", !info.can_restart);
  }

  async restartPairing() {
    const text = this.restartBox.querySelector('[data-field="restart-text"]');
    const button = this.restartBox.querySelector('[data-action="restart"]');
    button.disabled = true;
    try {
      await api(`${UpdatePanel.URL}/pairing/restart`, { method: "POST" });
    } catch (err) {
      text.textContent = `Could not restart: ${err.message}`;
      button.disabled = false;
      return;
    }
    text.textContent = "Restarting pAIring…";
    await this.waitForRestart();
    try {
      sessionStorage.setItem("pairingSettingsActiveTab", "update");
    } catch (_err) {
      // Storage blocked: the reload lands on the default tab.
    }
    window.location.reload();
  }

  // Waits for the server to go down, then come back (any answer below 500 means it's up).
  async waitForRestart() {
    const healthy = async () => {
      try {
        return (await fetch("/login", { cache: "no-store" })).status < 500;
      } catch (_err) {
        return false;
      }
    };
    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
    for (let i = 0; i < 30 && (await healthy()); i++) await sleep(500);
    for (let i = 0; i < 120 && !(await healthy()); i++) await sleep(1000);
  }
}

const updatePanelEl = document.getElementById("update-panel");
if (updatePanelEl) new UpdatePanel(updatePanelEl);
