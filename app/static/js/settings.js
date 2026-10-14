/* Settings page controller: loads the model catalog and the target's (a conversation, or the app-wide defaults) params, renders a slider per tunable value plus the model picker, and saves on demand. Also handles the Knowledge tab's private document upload/list/delete (every chat searches these automatically) and, for admins, the RAG upload limits on the System tab. */

// A small inline "animate-spin" SVG shared by this page's loading indicators (install progress, Admin settings > System
// default-model selects, External servers Start/Stop, the Matricxon panel's fetch div) so they look alike.
const SPINNER_SVG_HTML =
  '<svg class="h-3.5 w-3.5 animate-spin text-brand-500" viewBox="0 0 24 24" fill="none">' +
  '<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>' +
  '<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>';

// A small bare colored chevron (no filled circle; see renderModelRow's `pairing` param) at a paired row's bottom/top
// edge. Down/up shows which row (the model above, or its projector below) each belongs to.
const CHEVRON_DOWN_SVG_HTML =
  '<svg class="h-2.5 w-2.5" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" clip-rule="evenodd" ' +
  'd="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z"/></svg>';
// Sideways ones are a longer stroked arrow (shaft + head) so they read clearly at a compact card's corner.
const CHEVRON_RIGHT_SVG_HTML =
  '<svg class="h-[0.36rem] w-[0.72rem]" viewBox="0 0 24 12" fill="none" stroke="currentColor" stroke-width="1.75" ' +
  'stroke-linecap="round" stroke-linejoin="round"><path d="M1 6h21M17 1.5 22 6l-5 4.5"/></svg>';
const CHEVRON_LEFT_SVG_HTML =
  '<svg class="h-[0.36rem] w-[0.72rem]" viewBox="0 0 24 12" fill="none" stroke="currentColor" stroke-width="1.75" ' +
  'stroke-linecap="round" stroke-linejoin="round"><path d="M23 6H2M7 1.5 2 6l5 4.5"/></svg>';
const CHEVRON_UP_SVG_HTML =
  '<svg class="h-2.5 w-2.5" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" clip-rule="evenodd" ' +
  'd="M14.77 12.79a.75.75 0 01-1.06-.02L10 8.832l-3.71 3.938a.75.75 0 11-1.08-1.04l4.25-4.5a.75.75 0 011.08 0l4.25 4.5a.75.75 0 01-.02 1.06z"/></svg>';

// Inserts a spinner after `anchorEl`; call `.remove` on the result when the fetch or start/stop request settles, in a
// `finally`.
function addInlineSpinner(anchorEl) {
  const spinner = document.createElement("span");
  spinner.className = "inline-flex shrink-0 ml-2 align-middle";
  spinner.innerHTML = SPINNER_SVG_HTML;
  anchorEl.insertAdjacentElement("afterend", spinner);
  return spinner;
}

// Dims `panelEl` (must be `position: relative`) behind a centered spinner while `promise` is pending, for a whole
// section (e.g. External servers' per-engine panel) whose fields are empty/stale during its fetch.
function withPanelSpinner(panelEl, promise) {
  const overlay = document.createElement("div");
  overlay.className = "absolute inset-0 z-10 flex items-center justify-center rounded-lg bg-slate-900/60";
  overlay.innerHTML = SPINNER_SVG_HTML.replace("h-3.5 w-3.5", "h-5 w-5");
  panelEl.appendChild(overlay);
  return promise.finally(() => overlay.remove());
}

// Describes every slider: its GenerationParams key, range/step and a one-line explanation. The single source the UI is
// built from: a new tunable needs one entry here plus the field in app/schemas.py:GenerationParams.
//
// There is no num_ctx slider: the context window is one system-wide value (Settings > System).
const PARAM_DEFS = [
  { key: "temperature", label: "Temperature", min: 0, max: 2, step: 0.05,
    help: "Higher = more creative and varied; lower = more focused and deterministic." },
  { key: "top_p", label: "Top P (nucleus sampling)", min: 0, max: 1, step: 0.01,
    help: "Restricts choices to the smallest set of tokens covering this much probability mass." },
  { key: "top_k", label: "Top K", min: 1, max: 200, step: 1,
    help: "Restricts choices to the K most likely next tokens at each step." },
  { key: "repeat_penalty", label: "Repeat penalty", min: 0, max: 2, step: 0.05,
    help: "Discourages the model from repeating itself. 1.0 = no penalty." },
  { key: "num_predict", label: "Max reply length (tokens)", min: 32, max: 4096, step: 32,
    help: "Upper limit on how many tokens the model may generate in one reply." },
];

const DEFAULTS_TARGET = "__defaults__";

// Whether the user is an admin (data attribute on <body>, see base.html). Only admins can pull/uninstall/hide models
// (app/routers/settings.py endpoints are admin-only); the UI mirrors that so regular users see an "ask an admin" note
// instead of a button that 403s.
const isAdmin = document.body.dataset.isAdmin === "true";
// /admin-settings has no Behavior/Knowledge/Account tabs and /settings no admin tabs, so code for the other page's
// elements must not run or bind here.
const hasUserTabs = document.getElementById("tab-behavior") !== null;

const conversationSelect = document.getElementById("conversation-select");
const modelCatalogEl = document.getElementById("model-catalog");
const installedModelCatalogEl = document.getElementById("installed-model-catalog");
const recommendedModelsHeadingEl = document.getElementById("recommended-models-heading");
const hardwareSummaryEl = document.getElementById("hardware-summary");
const extendedModelCatalogEl = document.getElementById("extended-model-catalog");
const extendedModelCatalogEmptyEl = document.getElementById("extended-model-catalog-empty");
const browseMoreModelsToggleEl = document.getElementById("browse-more-models-toggle");
const browseMoreModelsPanelEl = document.getElementById("browse-more-models-panel");
const browseMoreModelsChevronEl = document.getElementById("browse-more-models-chevron");
const hfSearchInputEl = document.getElementById("hf-search-input");
const hfSearchBtn = document.getElementById("hf-search-btn");
const hfSearchStatusEl = document.getElementById("hf-search-status");
const hfSearchResultsEl = document.getElementById("hf-search-results");
const hfRepoFilesEl = document.getElementById("hf-repo-files");
const hfRepoFilesStatusEl = document.getElementById("hf-repo-files-status");
const paramSlidersEl = document.getElementById("param-sliders");
const paramsSectionHintEl = document.getElementById("params-section-hint");
const ragUnavailableEl = document.getElementById("rag-unavailable");
const ragUnavailableReasonEl = document.getElementById("rag-unavailable-reason");
// Set by loadRagAvailability, read by applyRagAvailabilityToKnowledgeTab and renderDocumentRow's Delete button. Starts
// true so Knowledge controls don't flash disabled before the first check.
let ragAvailable = true;
const embeddingModelCatalogEl = document.getElementById("embedding-model-catalog");
const ragTopKInput = document.getElementById("rag-top-k");
const ragTopKValueEl = document.getElementById("rag-top-k-value");
const myDocsSummaryEl = document.getElementById("my-docs-summary");
const saveBtn = document.getElementById("save-btn");
const saveStatusEl = document.getElementById("save-status");

let currentParams = null;
let currentCatalog = null;
let currentExtendedCatalog = null;
let currentEmbeddingCatalog = null;
let selectedModelTag = null;
let modelsAvailable = true;
let modelTabLoading = false;
// The active engine (app.services.engine_service), read from /health (public) rather than admin-only GET
// /api/settings/engine, since the Model tab's catalog and badges are visible to every user. Used only to decide whether
// the "Not supported" badge applies; hidden when Ollama (which runs every catalog entry) is active.
let currentActiveEngine = null;

// Every engine app.services.engines.registry knows (see EngineOption). Static for the page's life, so fetched once and
// reused (Active engine picker, its display-name map, Stats page engine dropdown).
let _engineOptionsPromise = null;
function engineOptionsList() {
  if (!_engineOptionsPromise) {
    _engineOptionsPromise = api("/api/settings/engine/options").then((r) => r.engines);
  }
  return _engineOptionsPromise;
}

function renderSliders(params) {
  paramSlidersEl.innerHTML = "";
  for (const def of PARAM_DEFS) {
    const row = document.createElement("div");
    row.innerHTML = `
      <div class="flex items-center justify-between mb-1">
        <label class="text-xs font-medium text-slate-300">${def.label}</label>
        <span class="text-xs font-mono text-slate-400" data-value-for="${def.key}">${params[def.key]}</span>
      </div>
      <input type="range" data-param="${def.key}" min="${def.min}" max="${def.max}" step="${def.step}"
        value="${params[def.key]}" class="slim-slider slim-slider-ring block w-full">
      <p class="mt-1 text-xs text-slate-500">${def.help}</p>
    `;
    paramSlidersEl.appendChild(row);
  }

  paramSlidersEl.querySelectorAll("input[type=range]").forEach((input) => {
    input.addEventListener("input", () => {
      const key = input.dataset.param;
      const isInt = Number.isInteger(PARAM_DEFS.find((d) => d.key === key).step);
      currentParams[key] = isInt ? parseInt(input.value, 10) : parseFloat(input.value);
      paramSlidersEl.querySelector(`[data-value-for="${key}"]`).textContent = currentParams[key];
    });
  });
}

// Model catalog

async function loadCatalog() {
  if (!currentCatalog) {
    currentCatalog = await api("/api/settings/model-catalog");
  }
  return currentCatalog;
}

/* The admin-managed "browse more models" catalog (app.services.extended_model_catalog_service), fetched fresh each call (not cached like loadCatalog) since it loads only on demand. */
async function loadExtendedCatalog() {
  currentExtendedCatalog = await api("/api/settings/model-catalog/extended");
  return currentExtendedCatalog;
}

function findCatalogEntry(tag) {
  return (
    currentCatalog?.entries.find((entry) => entry.tag === tag) ||
    currentExtendedCatalog?.entries.find((entry) => entry.tag === tag)
  );
}

function formatContextLength(length) {
  if (!length) return null;
  return length >= 1000 ? `${Math.round(length / 1000)}K` : `${length}`;
}

function formatSize(gb) {
  return gb ? `${gb} GB` : "—";
}

/* Real per-engine RAM figure (CatalogEntry.engine_support/EngineModelSupport), not the flat hand-typed min_ram_gb alone: engines' requirements for one file differ (Matricxon dequantizes to bf16 first; Ollama doesn't). Shows the active engine's number, falling back to the static estimate when it hasn't reported one (e.g. Matricxon, not installed there; never a figure borrowed from another engine, see EngineModelSupport.min_ram_gb). */
function formatRam(entry) {
  const ram = entry.engine_support?.[currentActiveEngine]?.min_ram_gb ?? entry.min_ram_gb;
  return ram ? `${ram}GB RAM` : null; // 0 = unknown (e.g. a connector model), not "needs no RAM"
}

/* Persists the selected model as the default for new chats at once. The Model tab has no Save button (nothing to batch); an existing conversation's model is changed from that chat's model badge (chat.js switchModel), so this always means "the default." */
async function saveModelSelection() {
  try {
    await api("/api/settings/default-model", {
      method: "PUT", body: JSON.stringify({ model: selectedModelTag }),
    });
    // Reloading (not redirecting to chat) confirms the update and keeps whatever else was being edited.
    window.location.reload();
  } catch (err) {
    saveStatusEl.textContent = `Failed to update model: ${err.message}`;
  }
}

function selectModel(entry) {
  selectedModelTag = entry.tag;
  renderModelCatalog();
  saveModelSelection();
}

/* Re-fetches and re-renders both catalogs after Pull/Uninstall/Hide/Remove, since a renderModelRow row can come from either (same shape, renderCatalogGroup) and installed/hidden/hardware_ok can change in both. The extended one is refreshed only if it has been loaded (browsing it is opt-in, toggleBrowseMoreModels). */
async function refreshCatalogs() {
  document.dispatchEvent(new Event("pairing:models-changed"));
  currentCatalog = null;
  await loadCatalog();
  renderModelCatalog();
  if (currentExtendedCatalog) {
    await loadExtendedCatalog();
    renderExtendedModelCatalog();
  }
}

async function pullModel(entry, button, progressEl, onDone = refreshCatalogs, confirmDuplicate = false) {
  button.disabled = true;
  button.textContent = "Pulling…";
  progressEl.classList.remove("hidden");
  progressEl.textContent = "Starting…";

  try {
    const response = await fetch(entry.is_image ? "/api/settings/image-models/pull" : "/api/settings/pull-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tag: entry.tag, confirm_duplicate: confirmDuplicate }),
    });
    // DuplicateInstallDetector found an installed model that looks like the same file under another name
    // (routers/settings.py pull_model). The detail is structured, not the usual plain string, so it is handled before
    // the generic error path. Declining is a cancellation: reset the button, don't throw.
    if (response.status === 409) {
      const body = await response.json().catch(() => ({}));
      if (body.detail?.duplicate_of && confirm(body.detail.message)) {
        return pullModel(entry, button, progressEl, onDone, true);
      }
      progressEl.textContent = "Cancelled.";
      button.disabled = false;
      button.textContent = "Pull";
      return;
    }
    if (!response.ok || !response.body) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Pull failed (${response.status})`);
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
        const payload = JSON.parse(line.slice("data:".length).trim());

        if (payload.error) throw new Error(payload.error);
        if (payload.done) continue;

        if (payload.total && payload.completed) {
          const pct = Math.round((payload.completed / payload.total) * 100);
          progressEl.textContent = `${payload.status || "Downloading"} — ${pct}%`;
        } else if (payload.status) {
          progressEl.textContent = payload.status;
        }
      }
    }

    progressEl.textContent = "Done.";
    await onDone();
  } catch (err) {
    progressEl.textContent = `Failed: ${err.message}`;
    button.disabled = false;
    button.textContent = "Pull";
  }
}

async function uninstallModel(entry, button, progressEl, onDone = refreshCatalogs) {
  if (!confirm(`Uninstall ${entry.tag}? This removes it from the engine for every user.`)) return;

  button.disabled = true;
  button.textContent = "Uninstalling…";
  progressEl.classList.remove("hidden");
  progressEl.textContent = "";

  try {
    await api(entry.is_image ? "/api/settings/image-models/delete" : "/api/settings/delete-model", {
      method: "POST",
      body: JSON.stringify({ tag: entry.tag }),
    });
    // If the removed model was selected for the current target, fall back to no selection instead of a phantom choice.
    if (selectedModelTag === entry.tag) selectedModelTag = null;
    await onDone();
  } catch (err) {
    progressEl.textContent = `Failed: ${err.message}`;
    button.disabled = false;
    button.textContent = "Uninstall";
  }
}

/* Disables or enables a model for regular users (hidden from their picker, blocked in chats); admin only (POST /api/settings/hide-model). A visibility toggle, so no confirmation like Uninstall. */
async function toggleHideModel(entry, button, progressEl) {
  const newHidden = !entry.hidden;
  button.disabled = true;
  progressEl.classList.remove("hidden");
  progressEl.textContent = newHidden ? "Disabling…" : "Enabling…";

  try {
    await api("/api/settings/hide-model", {
      method: "POST",
      body: JSON.stringify({ tag: entry.tag, hidden: newHidden }),
    });
    await refreshCatalogs();
  } catch (err) {
    progressEl.textContent = `Failed: ${err.message}`;
    button.disabled = false;
  }
}

/* Removes a not-yet-installed extended-catalog entry from the list (admin only; an installed one never reaches this, see CatalogEntry.removable). Catalog curation like toggleHideModel: nothing to uninstall, so no confirmation. */
async function removeExtendedModel(entry, button, progressEl) {
  button.disabled = true;
  progressEl.classList.remove("hidden");
  progressEl.textContent = "Removing…";

  try {
    await api("/api/settings/model-catalog/extended", { method: "DELETE", body: JSON.stringify({ tag: entry.tag }) });
    await refreshCatalogs();
  } catch (err) {
    progressEl.textContent = `Failed: ${err.message}`;
    button.disabled = false;
  }
}

/* One model's card in the picker: name/meta left, actions right, wrapping to their own line if needed. Plain flex, not a fixed grid, so it can't overflow however many buttons a row has (Select, Pull, Uninstall, Hide, Remove). */
function renderModelRow(entry, pairing = null, compact = false) {
  const isSelected = entry.tag !== null && entry.tag === selectedModelTag;
  const row = document.createElement("div");
  row.className =
    "relative rounded-lg border px-3 text-sm " +
    (compact ? "py-2 flex flex-col justify-between gap-1.5 " : "py-2.5 ") +
    (isSelected ? "border-brand-500 bg-[rgb(var(--color-brand-500)/0.1)]" : "border-slate-800 bg-slate-900") +
    (entry.hidden ? " opacity-60" : "");
  // Looked up after a fresh add/pull (addBtn's click handler) to scroll to the affected row instead of wherever hiding
  // the search UI left the scroll.
  if (entry.tag) row.dataset.tag = entry.tag;

  const top = document.createElement("div");
  top.className = compact
    ? "flex flex-col gap-1.5"
    : "flex flex-wrap items-center justify-between gap-x-3 gap-y-1.5";

  const label = document.createElement("div");
  label.className = "min-w-0";
  // A tag like "hf.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF:Meta-Llama-3-8B-Instruct.Q4_K_M" is the pull path,
  // not a headline: as the bold title it was so long it pushed the buttons onto a wrapped line. "family +
  // parameter_size" (e.g. "Llama 3 8B") is short, always populated (build_model_catalog and the "installed but not in
  // the static catalog" branch) and reads like a name; the tag still shows below.
  const modelName = [entry.family, entry.parameter_size].filter(Boolean).join(" ");
  // Only the facts the engine knows: a connector model (e.g. Cloudflare) has no download size or RAM figure.
  const metaParts = [
    entry.download_gb ? formatSize(entry.download_gb) : null,
    formatContextLength(entry.context_length) ? `${formatContextLength(entry.context_length)} ctx` : null,
    formatRam(entry),
  ].filter(Boolean);
  // "+Vision" = a chat model that can also see an image attachment (e.g. gemma4:12b). Plain "Vision" (no "+") would
  // mean a vision-only model with no text chat (hypothetical). "Vision projector" is the live case of an installed
  // mmproj sidecar (entry.is_projector), not a model, so plain "Vision" would mislead; same pill style as the others (a
  // badge, not plain text).
  const visionBadgeText = entry.is_projector ? "Vision projector" : entry.text_capable ? "+Vision" : "Vision";
  const visionBadge = entry.vision || entry.is_projector
    ? ` <span class="inline-flex items-center rounded-full bg-[rgb(var(--color-brand-500)/0.15)] border border-[rgb(var(--color-brand-500)/0.4)] px-1.5 py-0.5 text-[10px] font-medium text-brand-500 align-middle">${visionBadgeText}</span>`
    : "";
  // Shown for the active engine (see the "Active engine" picker above External servers). Most engines (e.g. Ollama) run
  // every catalog entry, so it would be noise for them. Also skipped for a vision projector (entry.is_projector): "Not
  // supported" would read as "this engine can't run it" for a file never meant as a chat model; its vision badge
  // (visionBadge) says what it is. See CatalogEntry.engine_support/EngineModelSupport.
  const activeSupport = entry.engine_support?.[currentActiveEngine];
  const matricxonBadge = !activeSupport || activeSupport.supported || entry.is_projector
    ? ""
    : ` <span class="inline-flex items-center rounded-full bg-slate-800 border border-slate-700 px-1.5 py-0.5 text-[10px] font-medium text-slate-400 align-middle" title="${escapeHtml(activeSupport.reason || "Not supported by the active engine")}">Not supported</span>`;
  // Only for an installed entry missing from app/model_catalog.py's hand-curated list
  // (CatalogEntry.is_auto_discovered). Not-yet-installed extended entries already sit under "Browse more models";
  // projectors are auto-discovered by definition, so a per-row note would be noise.
  const notInCatalogBadge = entry.installed && entry.is_auto_discovered && !entry.is_projector
    ? ` <span class="inline-flex items-center rounded-full bg-slate-800 border border-slate-700 px-1.5 py-0.5 text-[10px] font-medium text-slate-400 align-middle" title="Installed, but not part of the hand-curated model list">Not in catalog</span>`
    : "";
  // Matricxon's "chat_format_unverified" capability (CatalogEntry.chat_format_unverified): a Hebrew Mistral fine-tune
  // gave incoherent non-chat output with any prompt format, with nothing warning that its instruction
  // format was unconfirmed. Amber (a caution, not "broken") rather than +Vision's brand color or neutral slate.
  const chatFormatUnverifiedBadge = entry.chat_format_unverified
    ? ` <span class="inline-flex items-center rounded-full bg-warning-500/15 border border-warning-500/40 px-1.5 py-0.5 text-[10px] font-medium text-warning-400 align-middle" title="Matricxon has no confirmed chat template for this model and is falling back to a best-effort guess — it may not reliably follow instructions or stay in character.">Chat format unverified</span>`
    : "";
  label.innerHTML =
    `<span class="block truncate font-medium text-slate-100">${escapeHtml(modelName || entry.tag)}` +
    (entry.hidden ? ` <span class="text-xs font-normal text-slate-500">(disabled for users)</span>` : "") +
    visionBadge +
    (compact ? "" : matricxonBadge + notInCatalogBadge + chatFormatUnverifiedBadge) +
    `</span>` +
    `<span class="block truncate text-xs text-slate-500">${escapeHtml(metaParts.join(" · "))}</span>`;

  // flex-wrap: buttons that don't fit on one line drop to a second line instead of overflowing the card.
  const action = document.createElement("div");
  action.className = compact ? "flex flex-wrap items-center gap-1.5" : "flex flex-wrap items-center justify-end gap-1.5 shrink-0";

  const progressRow = document.createElement("p");
  progressRow.className = "mt-1.5 text-xs text-slate-500 hidden";

  // False for a connector engine (RunPod) that manages its models remotely (ModelCatalogResponse.model_management):
  // Pull/Uninstall/Hide/Remove would fail or do nothing, so they don't render. Select still does: choosing which
  // reported model is the default still matters.
  const managementEnabled = currentCatalog?.model_management ?? true;

  if (!entry.hardware_ok) {
    const badge = document.createElement("span");
    badge.className = "text-xs font-medium text-warning-400";
    badge.textContent = "RAM shortage";
    action.appendChild(badge);
  } else if (entry.installed) {
    // A vision projector is never a selectable chat model (CatalogEntry.is_projector): shown as confirmation that the
    // install took, not an action.
    if (entry.is_projector) {
      const badge = document.createElement("span");
      badge.className = "text-xs font-medium text-slate-500";
      badge.textContent = "Installed";
      action.appendChild(badge);
    } else {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.disabled = isSelected;
      btn.className = isSelected
        ? "rounded-md bg-brand-600 px-2.5 py-1 text-xs font-medium text-white cursor-default"
        : "rounded-md border border-slate-700 px-2.5 py-1 text-xs text-slate-200 hover:bg-slate-800 transition-colors";
      btn.textContent = isSelected ? "Selected" : "Select";
      btn.addEventListener("click", () => selectModel(entry));
      action.appendChild(btn);
    }

    // Uninstalling frees disk space but affects every user, so it is admin-only like pulling (also server-side: POST
    // /api/settings/delete-model, routers/settings.py).
    if (isAdmin && managementEnabled) {
      const uninstallBtn = document.createElement("button");
      uninstallBtn.type = "button";
      uninstallBtn.title = "Uninstall — removes this model from the engine for everyone";
      uninstallBtn.className =
        "rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-400 hover:border-red-500 hover:text-red-400 transition-colors";
      uninstallBtn.textContent = "Uninstall";
      uninstallBtn.addEventListener("click", () => uninstallModel(entry, uninstallBtn, progressRow));
      action.appendChild(uninstallBtn);
    }
  } else if (isAdmin && managementEnabled) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "rounded-md border border-slate-700 px-2.5 py-1 text-xs text-slate-200 hover:bg-slate-800 transition-colors";
    btn.textContent = "Pull";
    btn.addEventListener("click", () => pullModel(entry, btn, progressRow));
    action.appendChild(btn);
  } else if (!managementEnabled) {
    const note = document.createElement("span");
    note.className = "text-xs text-slate-500";
    note.textContent = "Not offered by this connector";
    action.appendChild(note);
  } else {
    const note = document.createElement("span");
    note.className = "text-xs text-slate-500";
    note.textContent = "Not installed — ask an admin";
    action.appendChild(note);
  }

  // Disable/Enable: admin-only, independent of install state, so an admin can declutter the picker for regular users
  // even before a model is installed.
  // Only for an installed model; the not-installed "Recommended" list has no Disable/Enable.
  if (isAdmin && managementEnabled && entry.installed) {
    const hideBtn = document.createElement("button");
    hideBtn.type = "button";
    hideBtn.title = entry.hidden
      ? "Enable — let regular users use this model again"
      : "Disable — regular users can no longer pick or chat with this model";
    hideBtn.className =
      "rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-400 hover:bg-slate-800 transition-colors";
    hideBtn.textContent = entry.hidden ? "Enable" : "Disable";
    hideBtn.addEventListener("click", () => toggleHideModel(entry, hideBtn, progressRow));
    action.appendChild(hideBtn);
  }

  // Remove from list: admin-only, for a not-yet-installed extended-catalog entry (server-computed,
  // CatalogEntry.removable). A default app/model_catalog.py entry never has it (nothing stored), and an installed one
  // gets Uninstall instead.
  if (entry.removable) {
    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.title = "Remove from this list — doesn't affect Ollama, since it was never installed";
    removeBtn.className =
      "rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-400 hover:border-red-500 hover:text-red-400 transition-colors";
    removeBtn.textContent = "Remove";
    removeBtn.addEventListener("click", () => removeExtendedModel(entry, removeBtn, progressRow));
    action.appendChild(removeBtn);
  }

  top.appendChild(label);
  top.appendChild(action);
  row.appendChild(top);

  // The pull path sits below the title/buttons row so its length never competes with them (see modelName's comment);
  // truncated with the full value in a native title="" tooltip.
  if (compact) {
    // Compact card: only name + sizes show; the full path and details the full row prints inline
    // (support/catalog/chat-format notes, unavailable reason) are in the hover tooltip.
    row.title = [
      entry.tag,
      activeSupport && !activeSupport.supported && !entry.is_projector
        ? `Not supported by the active engine${activeSupport.reason ? `: ${activeSupport.reason}` : ""}`
        : null,
      entry.installed && entry.is_auto_discovered && !entry.is_projector
        ? "Installed, but not part of the hand-curated model list"
        : null,
      entry.chat_format_unverified ? "Chat format unverified — best-effort guess, may not follow instructions reliably" : null,
      !entry.hardware_ok ? entry.unavailable_reason : null,
    ]
      .filter(Boolean)
      .join("\n");
  } else if (entry.tag) {
    const pathLine = document.createElement("p");
    pathLine.className = "mt-1 truncate font-mono text-[11px] text-slate-600";
    pathLine.textContent = entry.tag;
    pathLine.title = entry.tag;
    row.appendChild(pathLine);
  }

  if (!entry.hardware_ok && !compact) {
    const reason = document.createElement("p");
    reason.className = "mt-1.5 text-xs text-warning-400";
    reason.textContent = entry.unavailable_reason
      || `Requires ~${entry.engine_support?.[currentActiveEngine]?.min_ram_gb ?? entry.min_ram_gb}GB RAM/VRAM — this machine has ~${currentCatalog.hardware.total_gb}GB.`;
    row.appendChild(reason);
  }

  row.appendChild(progressRow);

  // "Bound together" indicator for a model+projector pair pulled from the same Hugging Face repo
  // (reorderProjectorsNextToTheirModel/renderCatalogGroup): at the model's bottom edge (pointing down at its projector)
  // or the projector's top edge (pointing up). On the right near the action buttons, a bare colored glyph (no filled
  // circle) so it reads as an accent rather than a shape on top of the Hide button.
  if (pairing) {
    const badge = document.createElement("span");
    const sideways = pairing.direction === "right" || pairing.direction === "left";
    // Compact two-per-row cards: the model card points right at its projector's card, which points back left, in the
    // upper corner of each card's side edge, clear of the text. Shown only from the two-column breakpoint up; in one
    // column the projector sits below its model and a sideways arrow would point at nothing.
    badge.className = sideways
      ? "absolute top-0.5 hidden sm:block text-brand-500 " +
        (pairing.direction === "right" ? "right-0.5" : "left-0.5")
      : "absolute right-2 text-brand-500 " + (pairing.direction === "down" ? "bottom-0.5" : "top-0.5");
    badge.title = pairing.tooltip;
    badge.innerHTML = {
      down: CHEVRON_DOWN_SVG_HTML,
      up: CHEVRON_UP_SVG_HTML,
      right: CHEVRON_RIGHT_SVG_HTML,
      left: CHEVRON_LEFT_SVG_HTML,
    }[pairing.direction];
    row.appendChild(badge);
  }

  return row;
}

/* "hf.co/<org>/<repo>:<file>" -> "<org>/<repo>", or null for a plain tag; the same extraction as InstalledProjectorCatalog._vendor_from_tag/_display_name server-side. Used to line up a projector row with its parent model (reorderProjectorsNextToTheirModel): family strings often differ between the two (a projector's family is its repo id; the model's is its extended-catalog family or a coarser architecture name), but files from the same Hugging Face repo share the repo id. */
function repoIdFromTag(tag) {
  if (!tag || !tag.startsWith("hf.co/")) return null;
  return tag.slice("hf.co/".length).split(":", 1)[0];
}

/* Moves each installed projector row directly after its parent model row (same repo id, repoIdFromTag) within one vendor's family-sorted list. Seen under the ggml-org heading, where model and projector families don't sort adjacently. A projector with no installed parent in the group (not pulled, or hidden from a non-admin) stays in sorted position, unlinked. */
function reorderProjectorsNextToTheirModel(vendorEntries) {
  const projectors = vendorEntries.filter((e) => e.is_projector);
  const claimed = new Set();
  const ordered = [];
  for (const entry of vendorEntries) {
    if (entry.is_projector) continue;
    ordered.push(entry);
    const repoId = repoIdFromTag(entry.tag);
    if (!repoId) continue;
    for (const projector of projectors) {
      if (!claimed.has(projector) && repoIdFromTag(projector.tag) === repoId) {
        ordered.push(projector);
        claimed.add(projector);
      }
    }
  }
  for (const projector of projectors) {
    if (!claimed.has(projector)) ordered.push(projector);
  }
  return ordered;
}

/* Shared by the default catalog (#model-catalog) and the admin-extensible one behind "Browse more models" (#extended-model-catalog); both endpoints return the same CatalogEntry shape. One heading per vendor, all its models under it regardless of family, each row's bold title (renderModelRow: family + parameter_size, e.g. "Llama-3.2-3B-Instruct 3.2B") carrying the family detail. It used to be one heading per vendor+family; showed `family` is often the exact model name, so nearly every admin-added model got its own single-entry heading. Entries still sort by family within a vendor, so same-family variants (e.g. Gemma 4 e2b/e4b/12b) stay adjacent. */
function renderCatalogGroup(containerEl, entries, compact = false) {
  containerEl.innerHTML = "";
  const vendors = [...new Set(entries.map((entry) => entry.vendor))];
  // Compact list where every vendor has one model: a heading + single card per vendor would leave rows half empty, so
  // lay all in one shared two-per-row grid, each card under its vendor heading. Once any vendor has 2+ models, the
  // per-vendor grouping below applies.
  if (compact && vendors.length > 1 && vendors.every((v) => entries.filter((e) => e.vendor === v).length === 1)) {
    const grid = document.createElement("div");
    grid.className = "grid grid-cols-1 sm:grid-cols-2 gap-x-2 gap-y-3";
    for (const entry of entries) {
      const cell = document.createElement("div");
      cell.className = "space-y-1.5";
      const heading = document.createElement("p");
      heading.className = "text-xs font-semibold text-slate-300";
      heading.textContent = entry.vendor;
      cell.appendChild(heading);
      cell.appendChild(renderModelRow(entry, null, true));
      grid.appendChild(cell);
    }
    containerEl.appendChild(grid);
    return;
  }
  for (const vendor of vendors) {
    const sortedEntries = [...entries.filter((e) => e.vendor === vendor)].sort((a, b) =>
      a.family.localeCompare(b.family)
    );
    const vendorEntries = reorderProjectorsNextToTheirModel(sortedEntries);

    const group = document.createElement("div");
    group.className = "space-y-1.5";
    // Compact (installed) cards sit two to a row under the vendor heading.
    const grid = compact ? document.createElement("div") : group;
    if (compact) grid.className = "grid grid-cols-1 sm:grid-cols-2 gap-2";

    const heading = document.createElement("p");
    heading.className = "text-xs font-semibold text-slate-300";
    heading.textContent = vendor;
    group.appendChild(heading);

    for (let i = 0; i < vendorEntries.length; i++) {
      const entry = vendorEntries[i];
      const prev = vendorEntries[i - 1];
      const next = vendorEntries[i + 1];
      let pairing = null;
      if (
        !entry.is_projector &&
        next?.is_projector &&
        repoIdFromTag(entry.tag) &&
        repoIdFromTag(entry.tag) === repoIdFromTag(next.tag)
      ) {
        pairing = { direction: "down", tooltip: `Linked with its vision projector, right below (${next.family})` };
      } else if (
        entry.is_projector &&
        prev &&
        repoIdFromTag(entry.tag) &&
        repoIdFromTag(entry.tag) === repoIdFromTag(prev.tag)
      ) {
        pairing = { direction: "up", tooltip: `Linked with its model, right above (${prev.family})` };
      }
      // Compact grid: "down"/"up" become "right"/"left", and a model with a projector is pinned to the left column so
      // its projector lands beside it in the same row.
      if (compact && pairing) {
        pairing = {
          direction: pairing.direction === "down" ? "right" : "left",
          tooltip: pairing.tooltip.replace("right below", "next to it").replace("right above", "next to it"),
        };
      }
      const row = renderModelRow(entry, pairing, compact);
      if (compact && pairing?.direction === "right") row.classList.add("sm:col-start-1");
      grid.appendChild(row);
    }
    if (compact) group.appendChild(grid);
    containerEl.appendChild(group);
  }
}

function renderModelCatalog() {
  const { hardware, entries } = currentCatalog;
  hardwareSummaryEl.textContent = hardware.vram_gb > 0
    ? `${hardware.ram_gb}GB RAM + ${hardware.vram_gb}GB VRAM`
    : `${hardware.ram_gb}GB RAM, no GPU detected`;
  // Installed models on top as compact two-per-row cards; the rest (recommended, not installed) below in the regular
  // one-per-row list.
  renderCatalogGroup(installedModelCatalogEl, entries.filter((e) => e.installed), true);
  document.getElementById("installed-model-catalog-section").classList.toggle("hidden", !entries.some((e) => e.installed));
  // The not-installed "Recommended" list is admin-only; regular users see only installed models. A connector engine
  // (model_management false) has nothing to pull, so it shows none.
  const listRecommended = isAdmin && (currentCatalog.model_management ?? true);
  const showRecommended = listRecommended && entries.some((e) => !e.installed);
  renderCatalogGroup(modelCatalogEl, listRecommended ? entries.filter((e) => !e.installed) : [], true);
  recommendedModelsHeadingEl.classList.toggle("hidden", !showRecommended);
  // Catches every way the stored default can point at nothing real: cleared by an engine switch
  // (engine_switch_service.clear_stale_default_models), the model uninstalled, or never chosen. A stale tag looks the
  // same to this picker whatever the cause. findCatalogEntry also checks the lazily loaded extended catalog, like
  // renderModelRow's isSelected, so no false positive for a default only in the unopened panel. Requiring `.installed`
  // makes the "uninstalled" case work: settings_service.get_default_model falls back to app.config.DEFAULT_MODEL (a
  // curated-catalog tag) even when nothing is installed, so an uninstalled fallback would otherwise pass as valid.
  const defaultEntry = selectedModelTag !== null ? findCatalogEntry(selectedModelTag) : null;
  const hasValidDefault = Boolean(defaultEntry?.installed);
  // Two cases: nothing installed (banner above the lists) vs. models installed but none chosen as default (banner at
  // the top of the installed list).
  const anyInstalled = entries.some((e) => e.installed);
  document.getElementById("model-no-default-banner").classList.toggle("hidden", anyInstalled);
  document.getElementById("model-no-default-selected-banner").classList.toggle("hidden", !anyInstalled || hasValidDefault);
}

function renderExtendedModelCatalog() {
  const entries = currentExtendedCatalog?.entries || [];
  extendedModelCatalogEmptyEl.classList.toggle("hidden", entries.length > 0);
  renderCatalogGroup(extendedModelCatalogEl, entries);
}

/* The panel starts open (see settings.html) and initModelTab has already preloaded its catalog, so this matters only after the user collapses it and wants it back, or if the preload failed (still guarded by currentExtendedCatalog, so never fetched twice). */
async function toggleBrowseMoreModels() {
  const opening = browseMoreModelsPanelEl.classList.contains("hidden");
  browseMoreModelsPanelEl.classList.toggle("hidden", !opening);
  browseMoreModelsChevronEl.classList.toggle("rotate-180", opening);
  if (opening && !currentExtendedCatalog) {
    await loadExtendedCatalog();
    renderExtendedModelCatalog();
  }
  if (opening) {
    // When re-expanding, scroll the toggle (not the panel) into view so the "Browse more models" heading is at the top
    // with the content below, as on the first, already-open render.
    browseMoreModelsToggleEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

browseMoreModelsToggleEl.addEventListener("click", toggleBrowseMoreModels);

// Three steps: search, pick a repo, add a file. Admin-only: hfSearchBtn/hfSearchResultsEl/hfRepoFilesEl don't exist for
// regular users (settings.html's {% if is_admin %}), so the block is skipped instead of throwing on a null target.
// Search and the file list are read-only (app/services/huggingface_client.py); nothing is saved until a file's "Add" is
// clicked, which re-verifies it server-side (extended_model_catalog_service.add_model).
function hideHfRepoFiles() {
  hfRepoFilesEl.classList.add("hidden");
  hfRepoFilesEl.innerHTML = "";
  hfRepoFilesStatusEl.textContent = "";
  hfRepoFilesStatusEl.classList.remove("text-red-400");
}

function renderHfSearchResults(results) {
  hfSearchResultsEl.innerHTML = "";
  hideHfRepoFiles();
  if (results.length === 0) {
    hfSearchResultsEl.classList.add("hidden");
    return;
  }

  for (const repo of results) {
    const row = document.createElement("button");
    row.type = "button";
    row.className =
      "block w-full text-left rounded-md border border-slate-800 px-2.5 py-1.5 text-sm hover:bg-slate-800 transition-colors";
    const metaParts = [`${repo.downloads.toLocaleString()} downloads`, `${repo.likes.toLocaleString()} likes`];
    if (repo.gated) metaParts.push("gated");
    if (repo.is_image) metaParts.unshift("Image model → Diffusion models");
    row.innerHTML =
      `<span class="block truncate font-medium text-slate-100">${escapeHtml(repo.repo_id)}</span>` +
      `<span class="block truncate text-xs text-slate-500">${escapeHtml(metaParts.join(" · "))}</span>`;
    row.addEventListener("click", () => loadHfRepoFiles(repo.repo_id));
    hfSearchResultsEl.appendChild(row);
  }
  hfSearchResultsEl.classList.remove("hidden");
}

// Ollama's library models default to Q4_K_M, the widely agreed balance of quality and size, so a file matching it
// (case-insensitively; HF filenames vary) gets a "Recommended" badge and sorts first, rather than leaving every quant
// looking arbitrary to someone who doesn't know Q4_K_M/Q8_0/IQ2_XS.
function isRecommendedQuantFile(filename) {
  return /q4_k_m/i.test(filename);
}

function renderHfRepoFiles(repo) {
  hfRepoFilesEl.innerHTML = "";

  const metaParts = [repo.parameter_size, formatContextLength(repo.context_length) && `${formatContextLength(repo.context_length)} ctx`, repo.license]
    .filter(Boolean);
  if (repo.is_image) metaParts.unshift("Image model — added under Diffusion models (installs into the stable-diffusion.cpp engine)");
  if (repo.gated) metaParts.push("gated — may need a Hugging Face token Ollama doesn't have, could fail to pull");
  const header = document.createElement("div");
  header.innerHTML =
    `<span class="block truncate text-sm font-medium text-slate-100">${escapeHtml(repo.repo_id)}</span>` +
    `<span class="block truncate text-xs text-slate-500">${escapeHtml([repo.family, ...metaParts].filter(Boolean).join(" · "))}</span>`;
  hfRepoFilesEl.appendChild(header);

  if (repo.files.length === 0) {
    const empty = document.createElement("p");
    empty.className = "text-xs text-slate-500 pt-1";
    empty.textContent = "No single-file GGUF variants found in this repo.";
    hfRepoFilesEl.appendChild(empty);
  } else {
    const help = document.createElement("p");
    help.className = "text-xs text-slate-500 pt-1";
    help.textContent =
      "Q4_K_M is usually the best balance of quality and size. Q5_K_M/Q6_K/Q8_0/fp16 are larger and closer to " +
      "full quality; Q2_K/Q3_K are smaller and noticeably lower quality.";
    hfRepoFilesEl.appendChild(help);
  }

  // Stable sort (guaranteed since ES2019): only the recommended file(s) move to the front.
  const byRecommendedFirst = (a, b) =>
    Number(isRecommendedQuantFile(b.filename)) - Number(isRecommendedQuantFile(a.filename));
  // Two separate sections, not one interleaved list: a repo with several chat-model quants and
  // several mmproj quants (e.g. ggml-org/SmolVLM2-2.2B-Instruct-GGUF: three main files, two mmproj) made it hard to
  // tell which "+Vision" row each projector belonged with. There is no strict 1:1 pairing by quant (mmproj files come
  // in fewer quants), so it groups by kind rather than guessing a match.
  const modelFiles = repo.files.filter((f) => !f.is_projector).sort(byRecommendedFirst);
  const projectorFiles = repo.files.filter((f) => f.is_projector).sort(byRecommendedFirst);

  // Shared by both groups: a projector row looks the same, just filtered into the second section.
  function buildFileRow(file) {
    const fileRow = document.createElement("div");
    fileRow.className = "flex items-center justify-between gap-3 pt-1";

    const recommendedBadge = isRecommendedQuantFile(file.filename)
      ? ` <span class="inline-flex items-center rounded-full bg-emerald-500/15 border border-emerald-500/40 px-1.5 py-0.5 text-[10px] font-medium text-emerald-400 align-middle">Recommended</span>`
      : "";
    // The repo-level family/parameter_size in the header above (repo.family/repo.parameter_size) describes the file
    // Hugging Face treats as the repo's primary `gguf` metadata block, almost never a vision-projector sidecar
    // (huggingface_client._is_projector_file). Flagged so picking one doesn't read as the header's chat model; this
    // once caused an mmproj file to be added and pulled as a standalone chat model.
    const projectorBadge = file.is_projector
      ? ` <span class="inline-flex items-center rounded-full bg-warning-500/15 border border-warning-500/40 px-1.5 py-0.5 text-[10px] font-medium text-warning-400 align-middle">Vision projector — not a chat model on its own</span>`
      : "";
    // The "+Vision" pill from renderModelRow: this repo also ships an mmproj sidecar that gets paired with it.
    const visionBadge = file.vision
      ? ` <span class="inline-flex items-center rounded-full bg-[rgb(var(--color-brand-500)/0.15)] border border-[rgb(var(--color-brand-500)/0.4)] px-1.5 py-0.5 text-[10px] font-medium text-brand-500 align-middle">+Vision</span>`
      : "";
    const label = document.createElement("span");
    label.className = "truncate text-xs text-slate-300";
    label.innerHTML = `${escapeHtml(`${file.filename} (${formatSize(file.download_gb)})`)}${recommendedBadge}${visionBadge}${projectorBadge}`;
    fileRow.appendChild(label);

    const addBtnGroup = document.createElement("span");
    addBtnGroup.className = "shrink-0 flex items-center gap-2";

    // A wrapping <span> with innerHTML set to the full <svg>...</svg>, not document.createElement("svg"), which makes a
    // plain HTMLUnknownElement in the wrong namespace and never renders. innerHTML goes through the
    // HTML parser, which enters the SVG namespace, as the Model tab's static spinner does.
    const addSpinner = document.createElement("span");
    addSpinner.className = "hidden";
    addSpinner.innerHTML =
      '<svg class="h-4 w-4 animate-spin text-slate-400" viewBox="0 0 24 24" fill="none">' +
      '<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>' +
      '<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>' +
      "</svg>";

    const addBtn = document.createElement("button");
    addBtn.type = "button";
    addBtn.className =
      "rounded-md bg-brand-600 px-3 py-1 text-xs font-medium text-white hover:bg-brand-500 transition-colors disabled:opacity-60";
    addBtn.textContent = "Add";
    addBtn.addEventListener("click", async () => {
      addBtn.disabled = true;
      addSpinner.classList.remove("hidden");
      // Shown below the file list (hf-repo-files-status), not up by the search box (hf-search-status): adding a file
      // (the GGUF-header probe can take seconds, see app.services.gguf_probe) left "Adding…" where an admin who had
      // scrolled down couldn't see it. The spinner beside the button (addSpinner) serves whoever is still looking at
      // it.
      hfRepoFilesStatusEl.textContent = "Adding…";
      hfRepoFilesStatusEl.classList.remove("text-red-400");
      try {
        // already_installed (ExtendedModelCatalogResponse): an already-installed tag never shows in "browse more
        // models" to pull, so a flat "Added" read as "go find it in the list", which no longer had it.
        const { already_installed } = await api("/api/settings/model-catalog/extended", {
          method: "POST",
          body: JSON.stringify({ repo_id: repo.repo_id, filename: file.filename }),
        });
        const message = already_installed
          ? `"${file.filename}" is already installed — nothing to pull.`
          : `Added "${file.filename}".`;
        hideHfRepoFiles();
        hfSearchResultsEl.classList.add("hidden");
        hfSearchStatusEl.textContent = message;
        // refreshCatalogs (not just loadExtendedCatalog/renderExtendedModelCatalog): a bug. When
        // already_installed is true, the tag can now show in the main catalog too (ChatModelCatalogBuilder.build's
        // is_auto_discovered handling), but only the extended one was refreshed, so it vanished from "Browse more
        // models" (ExtendedModelCatalog.build excludes installed tags) yet never appeared above until a reload.
        await refreshCatalogs();
        // Hiding the search results/file picker shrinks the page and leaves the scroll somewhere odd ("jumping to the
        // wrong place"). Scroll to the row this add affected (renderModelRow's data-tag): the main
        // list if installed, Browse more models otherwise. Same build_tag as ExtendedModelCatalog.build_tag.
        const addedTag = `hf.co/${repo.repo_id}:${file.filename.replace(/\.gguf$/, "")}`;
        document.querySelector(`[data-tag="${CSS.escape(addedTag)}"]`)?.scrollIntoView({
          behavior: "smooth",
          block: "center",
        });
      } catch (err) {
        hfRepoFilesStatusEl.textContent = err.message;
        hfRepoFilesStatusEl.classList.add("text-red-400");
        addBtn.disabled = false;
        addSpinner.classList.add("hidden");
      }
    });
    addBtnGroup.appendChild(addSpinner);
    addBtnGroup.appendChild(addBtn);
    fileRow.appendChild(addBtnGroup);
    return fileRow;
  }

  for (const file of modelFiles) {
    hfRepoFilesEl.appendChild(buildFileRow(file));
  }

  if (projectorFiles.length > 0) {
    const divider = document.createElement("div");
    divider.className = "pt-2 mt-1 border-t border-slate-700 text-[11px] font-medium text-slate-400";
    divider.textContent =
      modelFiles.length > 0
        ? "Vision projector (mmproj) files — pull one of these too, alongside a model above, for image support:"
        : "Vision projector (mmproj) files only — this repo has no standalone chat model file:";
    hfRepoFilesEl.appendChild(divider);

    for (const file of projectorFiles) {
      hfRepoFilesEl.appendChild(buildFileRow(file));
    }
  }

  hfRepoFilesEl.classList.remove("hidden");
}

async function loadHfRepoFiles(repoId) {
  hideHfRepoFiles();
  hfSearchStatusEl.textContent = "Loading files…";
  hfSearchStatusEl.classList.remove("text-red-400");
  try {
    const repo = await api("/api/settings/model-catalog/extended/hf-files", {
      method: "POST",
      body: JSON.stringify({ repo_id: repoId }),
    });
    hfSearchStatusEl.textContent = "";
    renderHfRepoFiles(repo);
    // The repo list can hold 20 rows; without this, clicking a top one leaves the new file picker below the fold with
    // no cue (as toggleBrowseMoreModels' scrollIntoView).
    hfRepoFilesEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    hfSearchStatusEl.textContent = err.message;
    hfSearchStatusEl.classList.add("text-red-400");
  }
}

async function runHfSearch() {
  const query = hfSearchInputEl.value.trim();
  if (!query) return;

  hideHfRepoFiles();
  hfSearchBtn.disabled = true;
  hfSearchStatusEl.textContent = "Searching…";
  hfSearchStatusEl.classList.remove("text-red-400");
  try {
    const { results } = await api("/api/settings/model-catalog/extended/search-hf", {
      method: "POST",
      body: JSON.stringify({ query }),
    });
    hfSearchStatusEl.textContent = results.length === 0 ? "No matching repos found." : "";
    renderHfSearchResults(results);
  } catch (err) {
    hfSearchStatusEl.textContent = err.message;
    hfSearchStatusEl.classList.add("text-red-400");
  } finally {
    hfSearchBtn.disabled = false;
  }
}

if (hfSearchBtn) {
  hfSearchBtn.addEventListener("click", runHfSearch);
  hfSearchInputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter") runHfSearch();
  });
}

/* Loads the Model tab's default-model picker, independent of Behavior's target picker (the model here is the default for new chats, see saveModelSelection). Runs at page load and each time the tab is re-activated (loadTabData). Guarded against overlap (modelTabLoading): a bug, when a reload restoring the "Model" tab ran it twice concurrently (boot sequence and restoreTabAfterReload's loadTabData), and two runs racing to show/hide the spinner could leave it stuck visible. */
async function initModelTab() {
  if (modelTabLoading) return;
  modelTabLoading = true;
  // Shown on every call, not just the first load: this fetch re-runs on each re-activation and the previous run hid it.
  // Matricxon's /api/tags can take seconds (MatricxonSupportChecker.load: catalog builders share one fetch), and
  // without this the section looked empty and broken.
  const loadingEl = document.getElementById("model-loading");
  loadingEl.classList.remove("hidden");
  document.getElementById("model-unavailable").classList.add("hidden");
  document.getElementById("model-field-normal").classList.add("hidden");
  try {
    currentActiveEngine = (await api("/health")).active_engine;
    await loadCatalog();
    selectedModelTag = (await api("/api/settings/default-model")).model;
    renderModelCatalog();
    modelsAvailable = true;
    // Hidden entirely for a connector (RunPod) that manages its models remotely
    // (ModelCatalogResponse.model_management): nothing to search Hugging Face for or pull.
    const browseMoreModelsSection = document.getElementById("browse-more-models-section");
    browseMoreModelsSection.classList.toggle("hidden", !(currentCatalog.model_management ?? true));
    // Same for the embedding and diffusion pull lists: a connector's models come from its provider.
    for (const id of ["embedding-models-section", "diffusion-models-section"]) {
      document.getElementById(id)?.classList.toggle("hidden", !(currentCatalog.model_management ?? true));
    }
    // "Browse more models" starts open (settings.html); preload it so the search box is usable at once. Own try/catch
    // so a failure (e.g. no internet) doesn't block the main picker. Guarded by currentExtendedCatalog like
    // toggleBrowseMoreModels' lazy load, so re-activating the tab doesn't re-fetch.
    if ((currentCatalog.model_management ?? true) && !currentExtendedCatalog) {
      try {
        await loadExtendedCatalog();
        renderExtendedModelCatalog();
      } catch (err) {
        console.error("Failed to preload the extended model catalog:", err);
      }
    }
  } catch (err) {
    modelsAvailable = false;
    // Seen: this always showed "System is not configured for models yet, ask your admin.", whether the cause
    // was a fresh install or (more often) a transient failure reaching the active engine (Matricxon can answer slowly
    // mid-generation, see MatricxonSupportChecker). Naming the engine is clearer than err.message alone: a slow
    // Matricxon request that times out in some layer doesn't always forward GET /api/settings/model-catalog's clean 503
    // detail (routers/settings.py get_model_catalog), leaving a bare "Service Unavailable". currentActiveEngine doesn't
    // depend on that. err.message is still appended when present, as it can be more specific (e.g. "connection
    // refused").
    const engineLabel = { ollama: "Ollama", matricxon: "Matricxon" }[currentActiveEngine] || "the active engine";
    const detail = err.message ? ` (${err.message})` : "";
    document.getElementById("model-unavailable-reason").textContent =
      `Couldn't reach ${engineLabel} to load the model list${detail} — it may be unreachable, still starting up, ` +
      "or slow to respond right now. Try refreshing in a moment.";
  }
  loadingEl.classList.add("hidden");
  document.getElementById("model-unavailable").classList.toggle("hidden", modelsAvailable);
  document.getElementById("model-field-normal").classList.toggle("hidden", !modelsAvailable);
  modelTabLoading = false;
}

// RAG availability + knowledge base (private documents)
// The shared/"global" scope still exists server-side (app/rag.py, app/routers/documents.py: scope="global" on upload,
// owner_id NULL on Document) but has no UI yet; only "My documents" (private) is exposed until it gets a management
// pass.

async function loadRagAvailability() {
  if (!hasUserTabs) return;
  const { available, reason } = await api("/api/settings/rag-availability");
  ragAvailable = available;
  ragUnavailableEl.classList.toggle("hidden", available);
  // The `hidden` attribute too: the tab's space-y spacing skips only [hidden] siblings, so with just the class a hidden
  // banner left a blank gap.
  ragUnavailableEl.hidden = available;
  if (reason) ragUnavailableReasonEl.textContent = reason;
  applyRagAvailabilityToKnowledgeTab();
}

/* Grays out (never hides; the banner explains why) the Knowledge tab's file controls when RAG is off: Upload, the top-k slider, each document's Delete (renderDocumentRow reads ragAvailable, so re-rendering the page picks up a change) and the shared Save button, the last only while on the Knowledge tab, since that button also saves Behavior's params, unrelated to RAG. Called after each availability check (loadRagAvailability) and tab switch (activateTab), as either can change this. */
function applyRagAvailabilityToKnowledgeTab() {
  document.getElementById("my-docs-upload-btn").disabled = !ragAvailable;
  document.getElementById("rag-top-k").disabled = !ragAvailable;
  renderMyDocsPage();
  document.getElementById("save-btn").disabled = activeTabName === "knowledge" && !ragAvailable;
}

async function loadEmbeddingCatalog() {
  if (!embeddingModelCatalogEl) return;
  currentEmbeddingCatalog = await api("/api/settings/embedding-model-catalog");
  renderEmbeddingCatalog();
}

/* One embedding model's row, deliberately not renderModelRow: there is nothing to select (no live UI switch yet, see EMBEDDING_MODEL in app/config.py) and nothing to hide/remove (a fixed two-entry list, not the extended catalog). Visible to every user (install status isn't secret, as the #rag-unavailable banner); only Pull is admin-gated, as renderModelRow. */
function renderEmbeddingCatalog() {
  embeddingModelCatalogEl.innerHTML = "";
  if (!currentEmbeddingCatalog) return;

  for (const entry of currentEmbeddingCatalog.entries) {
    const isDefault = entry.tag === currentEmbeddingCatalog.default_tag;
    const row = document.createElement("div");
    row.className = "rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-sm flex flex-col justify-between gap-1.5";

    const top = document.createElement("div");
    top.className = "flex flex-col gap-1.5";

    const label = document.createElement("div");
    label.className = "min-w-0";
    const modelName = [entry.family, entry.parameter_size].filter(Boolean).join(" ");
    const metaParts = [
      entry.download_gb ? formatSize(entry.download_gb) : null,
      entry.embedding_dim ? `${entry.embedding_dim}-dim` : null,
      formatContextLength(entry.context_length) ? `${formatContextLength(entry.context_length)} ctx` : null,
    ].filter(Boolean);
    const defaultBadge = isDefault
      ? ` <span class="inline-flex items-center rounded-full bg-[rgb(var(--color-brand-500)/0.15)] border border-[rgb(var(--color-brand-500)/0.4)] px-1.5 py-0.5 text-[10px] font-medium text-brand-500 align-middle">Default</span>`
      : "";
    // See renderModelRow's matricxonBadge.
    const activeSupport = entry.engine_support?.[currentActiveEngine];
    const matricxonBadge = !activeSupport || activeSupport.supported
      ? ""
      : ` <span class="inline-flex items-center rounded-full bg-slate-800 border border-slate-700 px-1.5 py-0.5 text-[10px] font-medium text-slate-400 align-middle" title="${escapeHtml(activeSupport.reason || "Not supported by the active engine")}">Not supported</span>`;
    label.innerHTML =
      `<span class="block truncate font-medium text-slate-100">${escapeHtml(modelName || entry.tag)}${defaultBadge}${matricxonBadge}</span>` +
      `<span class="block truncate text-xs text-slate-500">${escapeHtml(metaParts.join(" · "))}</span>`;

    const action = document.createElement("div");
    action.className = "flex flex-wrap items-center gap-1.5";
    const progressRow = document.createElement("p");
    progressRow.className = "mt-1.5 text-xs text-slate-500 hidden";

    // Pull and Uninstall refresh the embedding catalog + RAG-availability banner, not the chat-model catalogs
    // pullModel/uninstallModel refresh by default.
    const refreshEmbedding = async () => {
      await loadEmbeddingCatalog();
      await loadRagAvailability();
    };
    // See renderModelRow's managementEnabled: the active engine, not the catalog type, decides this.
    const managementEnabled = currentCatalog?.model_management ?? true;

    if (entry.installed) {
      const badge = document.createElement("span");
      badge.className = "text-xs font-medium text-emerald-400";
      badge.textContent = "Installed";
      action.appendChild(badge);

      // Uninstalling frees disk space but affects every user, so it is admin-only like pulling (also server-side: POST
      // /api/settings/delete-model).
      if (isAdmin && managementEnabled) {
        const uninstallBtn = document.createElement("button");
        uninstallBtn.type = "button";
        uninstallBtn.title = "Uninstall — removes this model from the engine for everyone";
        uninstallBtn.className =
          "rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-400 hover:border-red-500 hover:text-red-400 transition-colors";
        uninstallBtn.textContent = "Uninstall";
        uninstallBtn.addEventListener("click", () => uninstallModel(entry, uninstallBtn, progressRow, refreshEmbedding));
        action.appendChild(uninstallBtn);
      }
    } else if (isAdmin && managementEnabled) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "rounded-md border border-slate-700 px-2.5 py-1 text-xs text-slate-200 hover:bg-slate-800 transition-colors";
      btn.textContent = "Pull";
      btn.addEventListener("click", () => pullModel(entry, btn, progressRow, refreshEmbedding));
      action.appendChild(btn);
    } else if (!managementEnabled) {
      const note = document.createElement("span");
      note.className = "text-xs text-slate-500";
      note.textContent = "Not offered by this connector";
      action.appendChild(note);
    } else {
      const note = document.createElement("span");
      note.className = "text-xs text-slate-500";
      note.textContent = "Not installed — ask an admin";
      action.appendChild(note);
    }

    top.appendChild(label);
    top.appendChild(action);
    row.appendChild(top);

    row.title = entry.tag;
    row.appendChild(progressRow);

    embeddingModelCatalogEl.appendChild(row);
  }
}

// Cached from the last summary fetch so handleUpload can check a file's size against the limit before sending (the
// server enforces the real limit; this is a faster, friendlier rejection).
let currentUploadLimits = null;

function renderKnowledgeSummary(summary) {
  currentUploadLimits = summary.limits;
  myDocsSummaryEl.textContent =
    `${summary.mine.document_count} doc${summary.mine.document_count === 1 ? "" : "s"}, ` +
    `${summary.mine.chunk_count} chunk${summary.mine.chunk_count === 1 ? "" : "s"} — ` +
    `${summary.mine.used_mb}MB of ${summary.limits.max_user_space_mb}MB used`;
  renderRagQuota(summary);
}

/* More decimals the smaller the share, so a small upload against a big quota never reads "0.0%". */
function formatPercent(percent) {
  if (percent === 0) return "0%";
  if (percent < 0.001) return "<0.001%";
  return `${percent.toFixed(percent < 1 ? 3 : percent < 10 ? 2 : 1)}%`;
}

/* The Knowledge tab's "Your quota" box: storage used against the admin's per-user limit, plus the other limits on this account's uploads. */
function renderRagQuota(summary) {
  const usageEl = document.getElementById("rag-quota-usage");
  if (!usageEl) return;
  const { used_mb: used, document_count: docs } = summary.mine;
  const { max_user_space_mb: space, max_file_mb: perFile, enabled } = summary.limits;
  const percent = space > 0 ? Math.min(100, (used / space) * 100) : 0;
  usageEl.textContent = `${used}MB of ${space}MB (${formatPercent(percent)})`;
  const bar = document.getElementById("rag-quota-bar");
  bar.style.width = `${percent}%`;
  bar.classList.toggle("bg-red-500", percent >= 90);
  bar.classList.toggle("bg-brand-600", percent < 90);
  const details = [
    `${Math.max(0, +(space - used).toFixed(2))}MB of storage left for your private documents`,
    `Largest single file: ${perFile}MB`,
    "Up to 5 files per upload",
    `${docs} document${docs === 1 ? "" : "s"} stored`,
  ];
  if (!enabled) details.unshift("The knowledge base is turned off by an admin");
  document.getElementById("rag-quota-details").replaceChildren(
    ...details.map((text) => Object.assign(document.createElement("li"), { textContent: text })),
  );
}

async function loadKnowledgeSummary() {
  renderKnowledgeSummary(await api("/api/documents/summary"));
}

/* One row in a document list: name, chunk count and, for whoever may remove it, a Delete button. Follows the model catalog's uninstall pattern (confirm, remove, refresh). */
function renderDocumentRow(doc, canDelete) {
  const row = document.createElement("div");
  row.className = "flex items-center justify-between gap-3 px-3 py-2";

  const info = document.createElement("div");
  info.className = "min-w-0";
  info.innerHTML =
    `<span class="block truncate text-slate-200">${escapeHtml(doc.filename)}</span>` +
    `<span class="block text-xs text-slate-500">${doc.chunk_count} chunk${doc.chunk_count === 1 ? "" : "s"}</span>`;
  row.appendChild(info);

  if (canDelete) {
    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className =
      "shrink-0 rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-400 hover:border-red-500 hover:text-red-400 transition-colors disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:border-slate-700 disabled:hover:text-slate-400";
    // Still rendered when RAG is off (applyRagAvailabilityToKnowledgeTab), disabled not hidden, like the Upload button,
    // top-k slider and Save button.
    delBtn.disabled = !ragAvailable;
    delBtn.textContent = "Delete";
    delBtn.addEventListener("click", async () => {
      if (!confirm(`Delete "${doc.filename}"?`)) return;
      delBtn.disabled = true;
      try {
        await api(`/api/documents/${doc.id}`, { method: "DELETE" });
        await Promise.all([loadDocumentLists(), loadKnowledgeSummary()]);
      } catch (err) {
        alert(`Failed to delete: ${err.message}`);
        delBtn.disabled = false;
      }
    });
    row.appendChild(delBtn);
  }

  return row;
}

// Own documents shown per page (renderMyDocsPage). The full list is fetched in one request (no server paging at this
// scale) but not all rendered at once.
const MY_DOCS_PAGE_SIZE = 10;
let myDocsAll = [];
let myDocsPage = 1;

function renderMyDocsPage() {
  const containerEl = document.getElementById("my-docs-list");
  const paginationEl = document.getElementById("my-docs-pagination");

  const totalPages = Math.max(1, Math.ceil(myDocsAll.length / MY_DOCS_PAGE_SIZE));
  // Clamped, since the list can shrink (a delete) between renders and leave the page past the new last page.
  myDocsPage = Math.min(Math.max(1, myDocsPage), totalPages);

  const start = (myDocsPage - 1) * MY_DOCS_PAGE_SIZE;
  const pageItems = myDocsAll.slice(start, start + MY_DOCS_PAGE_SIZE);

  containerEl.innerHTML = "";
  if (pageItems.length === 0) {
    containerEl.innerHTML = `<p class="px-3 py-2 text-xs text-slate-500">No documents yet.</p>`;
  } else {
    for (const doc of pageItems) containerEl.appendChild(renderDocumentRow(doc, true));
  }

  paginationEl.classList.toggle("hidden", myDocsAll.length <= MY_DOCS_PAGE_SIZE);
  document.getElementById("my-docs-page-info").textContent = `Page ${myDocsPage} of ${totalPages}`;
  document.getElementById("my-docs-prev-btn").disabled = myDocsPage <= 1;
  document.getElementById("my-docs-next-btn").disabled = myDocsPage >= totalPages;
}

document.getElementById("my-docs-prev-btn")?.addEventListener("click", () => {
  myDocsPage -= 1;
  renderMyDocsPage();
});
document.getElementById("my-docs-next-btn")?.addEventListener("click", () => {
  myDocsPage += 1;
  renderMyDocsPage();
});

async function loadDocumentLists() {
  const docs = await api("/api/documents");
  myDocsAll = docs.filter((d) => d.scope === "mine");
  renderMyDocsPage();
}

// How often to poll an upload job's progress. Plain GET/POST + polling instead of a streamed response: reading a
// streaming fetch body incrementally depends on browser buffering, and a stuck-looking upload is the failure to
// avoid. Complete request/response cycles work everywhere.
const UPLOAD_POLL_MS = 1500;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/*
 * Starts an upload job (POST, returns quickly with a job id and per-file state) and polls it to completion, calling
 * `onUpdate(job)` after every poll, including the POST response, so the caller can paint per-file progress. Resolves
 * with the job's final `result` once `done`.
 * 
 * A raw fetch for the POST, not api: api always sends Content-Type: application/json, wrong for
 * multipart/form-data, where the browser must set the header with the boundary.
 */
async function uploadFiles(formData, onUpdate) {
  const response = await fetch("/api/documents/upload", { method: "POST", body: formData });
  if (response.status === 401) {
    window.location.href = "/login";
    throw new Error("Not authenticated");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Upload failed (${response.status})`);
  }

  let job = await response.json();
  onUpdate(job);

  // The upload keeps running server-side whether or not a poll succeeds, so one dropped connection while polling isn't
  // a failed upload. Retry a few times before giving up.
  let consecutiveFailures = 0;
  while (!job.done) {
    await sleep(UPLOAD_POLL_MS);
    try {
      job = await api(`/api/documents/upload/${job.job_id}`);
      consecutiveFailures = 0;
      onUpdate(job);
    } catch (err) {
      consecutiveFailures += 1;
      if (consecutiveFailures >= 5) throw err;
    }
  }

  if (!job.result) throw new Error("Upload job finished without a result.");
  return job.result;
}

/* Reads the 5 upload slots (app/templates/settings.html) into { slotEl, file, statusEl, barWrapEl, barEl } for the slots that have a file; empty slots are skipped. */
function readUploadSlots() {
  const entries = [];
  for (const slotEl of document.querySelectorAll("#my-docs-upload-slots > [data-slot]")) {
    const file = slotEl.querySelector(".upload-slot-input").files[0];
    if (!file) continue;
    entries.push({
      slotEl,
      file,
      statusEl: slotEl.querySelector(".upload-slot-status"),
      barWrapEl: slotEl.querySelector(".upload-slot-bar-wrap"),
      barEl: slotEl.querySelector(".upload-slot-bar"),
    });
  }
  return entries;
}

function resetUploadSlot(entry) {
  entry.slotEl.querySelector(".upload-slot-input").value = "";
  entry.statusEl.textContent = "—";
  entry.statusEl.classList.remove("text-warning-400", "text-emerald-400");
  entry.barWrapEl.classList.add("hidden");
  entry.barEl.classList.remove("bg-red-500");
  entry.barEl.classList.add("bg-brand-600");
  entry.barEl.style.width = "0%";
}

async function handleUpload() {
  const statusEl = document.getElementById("my-docs-status");
  const overallWrapEl = document.getElementById("my-docs-overall-wrap");
  const overallBarEl = document.getElementById("my-docs-overall-bar");
  const overallPctEl = document.getElementById("my-docs-overall-pct");
  const uploadBtn = document.getElementById("my-docs-upload-btn");

  const entries = readUploadSlots();
  if (entries.length === 0) {
    statusEl.textContent = "Choose at least one file.";
    return;
  }

  // Client-side size pre-check per slot: reject just that slot (shown in its row) instead of blocking the batch; the
  // server would reject only that file anyway.
  const valid = [];
  for (const entry of entries) {
    if (currentUploadLimits && entry.file.size > currentUploadLimits.max_file_mb * 1024 * 1024) {
      entry.statusEl.textContent = "Too large";
      entry.statusEl.classList.add("text-warning-400");
      continue;
    }
    valid.push(entry);
  }
  if (valid.length === 0) {
    statusEl.textContent = "No files small enough to upload — see the size limit above.";
    statusEl.classList.add("text-warning-400");
    return;
  }

  const formData = new FormData();
  for (const entry of valid) formData.append("files", entry.file);
  formData.append("scope", "mine");

  // Progress events are matched to a slot by filename, not position: the server processes files in its own order
  // (alphabetical in the embedding pass, not upload order; app.rag.sync_folder_stream).
  const slotByFilename = new Map();
  for (const entry of valid) {
    slotByFilename.set(entry.file.name, entry);
    entry.statusEl.textContent = "Waiting…";
    entry.statusEl.classList.remove("text-warning-400", "text-emerald-400");
    entry.barWrapEl.classList.remove("hidden");
    entry.barEl.style.width = "0%";
  }

  const totalFiles = valid.length;

  function setOverall(pct) {
    overallBarEl.style.width = `${pct}%`;
    overallPctEl.textContent = `${pct}%`;
  }

  // Applied to every poll response (uploadFiles' onUpdate): a full snapshot of every tracked file, repainted from
  // scratch, so a missed or repeated poll is harmless.
  function applyJobState(job) {
    let totalFraction = 0;
    for (const [filename, entry] of slotByFilename) {
      const fileState = job.files[filename];
      if (!fileState) continue;

      let fraction = 0;
      if (fileState.status === "processing" && fileState.chunks_total) {
        // Embedding one chunk can take long on slow hardware and a document can be dozens of chunks; without this a
        // file's bar would freeze for its whole processing time.
        const pct = Math.round((fileState.chunk / fileState.chunks_total) * 100);
        entry.barEl.style.width = `${pct}%`;
        entry.statusEl.textContent = `${pct}% (${fileState.chunk}/${fileState.chunks_total})`;
        fraction = fileState.chunk / fileState.chunks_total;
      } else if (fileState.status === "processing") {
        entry.statusEl.textContent = "Processing…";
      } else if (fileState.status === "waiting") {
        entry.statusEl.textContent = "Waiting…";
      } else if (fileState.status === "error") {
        entry.barEl.style.width = "100%";
        entry.statusEl.textContent = fileState.error;
        entry.statusEl.classList.add("text-warning-400");
        entry.barEl.classList.remove("bg-brand-600");
        entry.barEl.classList.add("bg-red-500");
        fraction = 1;
      } else {
        // added / updated / skipped
        entry.barEl.style.width = "100%";
        entry.statusEl.textContent = fileState.status;
        entry.statusEl.classList.add("text-emerald-400");
        fraction = 1;
      }
      totalFraction += fraction;
    }
    setOverall(Math.min(100, Math.round((totalFraction / totalFiles) * 100)));
  }

  uploadBtn.disabled = true;
  overallWrapEl.classList.remove("hidden");
  setOverall(0);
  statusEl.textContent = `Uploading ${totalFiles} file${totalFiles === 1 ? "" : "s"}…`;
  statusEl.classList.remove("text-warning-400");

  try {
    const result = await uploadFiles(formData, applyJobState);

    setOverall(100);
    const parts = [];
    if (result.uploaded.length) parts.push(`Uploaded ${result.uploaded.length}.`);
    if (result.errors.length) {
      parts.push(...result.errors);
      statusEl.classList.add("text-warning-400");
    }
    statusEl.textContent = parts.join(" ") || "Done.";
    renderKnowledgeSummary(result.summary);
    await loadDocumentLists();
  } catch (err) {
    statusEl.textContent = `Upload failed: ${err.message}`;
    statusEl.classList.add("text-warning-400");
  } finally {
    uploadBtn.disabled = false;
    setTimeout(() => {
      overallWrapEl.classList.add("hidden");
      valid.forEach(resetUploadSlot);
    }, 2500);
  }
}

document.getElementById("my-docs-upload-btn")?.addEventListener("click", handleUpload);

// Loading a target's current settings

async function loadTarget(targetId) {
  let params, model;
  if (targetId === DEFAULTS_TARGET) {
    params = await api("/api/settings/defaults");
    model = (await api("/api/settings/default-model")).model;
    paramsSectionHintEl.textContent =
      "Sets the starting point for every new chat you create from now on — leaves every chat you already have " +
      "untouched. Applies the same regardless of which model is picked on the Model tab — these values aren't " +
      "tied to any one model, and switching models there never changes or resets them.";
  } else {
    const conversation = await api(`/api/conversations/${targetId}`);
    params = conversation.params;
    model = conversation.model;
    paramsSectionHintEl.textContent =
      "Saving here changes this existing conversation's behavior directly — its very next reply uses these " +
      "values, not just a future new chat. They aren't tied to this conversation's model specifically — the " +
      "same values would carry over even if its model were changed (from that chat's own model badge, not here).";
  }

  currentParams = { ...params };
  ragTopKInput.value = currentParams.rag_top_k || 4;
  ragTopKValueEl.textContent = ragTopKInput.value;

  // No model picker on this tab (see the Model tab's initModelTab); the catalog is read only for the "no models
  // installed" banner below. A failed fetch (Ollama unreachable) counts as nothing usable.
  let modelsAvailableHere = false;
  try {
    const catalog = await loadCatalog();
    modelsAvailableHere = catalog.entries.some((entry) => entry.installed);
  } catch (err) {
    // Same as "no models": nothing usable now.
  }
  renderSliders(currentParams);
  // Same banner/wording as the Model tab's #model-unavailable (initModelTab). The sliders render either way (generic
  // numeric params), but nothing told the admin why a new chat still gets no reply.
  document.getElementById("behavior-model-unavailable").classList.toggle("hidden", modelsAvailableHere);
  updateSaveBarVisibility();
}

async function populateConversationSelect() {
  const conversations = await api("/api/conversations");

  conversationSelect.innerHTML = "";
  const defaultsOption = document.createElement("option");
  defaultsOption.value = DEFAULTS_TARGET;
  defaultsOption.textContent = "Defaults for new chats";
  conversationSelect.appendChild(defaultsOption);

  for (const conversation of conversations) {
    const option = document.createElement("option");
    option.value = conversation.id;
    option.textContent = conversation.title || "New chat";
    conversationSelect.appendChild(option);
  }

  // Channels this user may reconfigure: admins and that channel's managers (also enforced in
  // conversation_service.update_conversation). Plain members never get the group.
  const known = new Set(conversations.map((c) => c.id));
  let channels = [];
  try {
    channels = (await api("/api/channels")).filter((c) => c.can_manage && !known.has(c.conversation_id));
  } catch (_err) {
    // Channels unavailable: personal chats above still work.
  }
  if (channels.length) {
    const group = document.createElement("optgroup");
    group.label = "Channels";
    for (const channel of channels) {
      const option = document.createElement("option");
      option.value = channel.conversation_id;
      option.textContent = `#${channel.name}`;
      group.appendChild(option);
    }
    conversationSelect.appendChild(group);
  }

  // Always opens on "Defaults for new chats", not the last conversation open in chat; a conversation's settings are one
  // click away in the dropdown.
  conversationSelect.value = DEFAULTS_TARGET;
  await loadTarget(DEFAULTS_TARGET);
}

conversationSelect?.addEventListener("change", () => loadTarget(conversationSelect.value));

ragTopKInput?.addEventListener("input", () => {
  currentParams.rag_top_k = parseInt(ragTopKInput.value, 10);
  ragTopKValueEl.textContent = ragTopKInput.value;
});

// Saving

saveBtn?.addEventListener("click", async () => {
  const targetId = conversationSelect.value;
  saveBtn.disabled = true;
  saveStatusEl.textContent = "Saving…";
  try {
    if (targetId === DEFAULTS_TARGET) {
      await api("/api/settings/defaults", { method: "PUT", body: JSON.stringify(currentParams) });
    } else {
      // Model is left out: an existing conversation's model is changed only from that chat's model badge (chat.js
      // switchModel).
      await api(`/api/conversations/${targetId}`, {
        method: "PATCH",
        body: JSON.stringify({ params: currentParams }),
      });
    }
    // Reload to apply the saved values, returning to the tab being edited (Behavior or Knowledge) rather than the
    // default Model tab.
    try {
      sessionStorage.setItem(SETTINGS_TAB_STORAGE_KEY, activeTabName);
    } catch (_err) {
      // Storage blocked: the reload still happens, landing on the default tab.
    }
    window.location.reload();
  } catch (err) {
    saveStatusEl.textContent = `Failed to save: ${err.message}`;
    saveBtn.disabled = false;
  }
});

// Tabs
// "My Settings" and "Knowledge" exist for every user; "System" and "Users" are admin-only extras that settings.html
// renders (with their nav buttons) only when Jinja's `is_admin` is true, so tabButtons and the admin-only elements
// below don't exist in the DOM for regular users.

const tabButtons = document.querySelectorAll(".settings-tab-btn");
const saveBarEl = document.getElementById("save-bar");
const knowledgeBaseSectionEl = document.getElementById("knowledge-base-section");
// Where the save-bar sits in the markup (below My Settings), remembered at load so it can move back when leaving the
// Knowledge tab.
const saveBarHomeParent = saveBarEl?.parentNode;
const saveBarHomeNextSibling = saveBarEl?.nextSibling;
const TAB_CONTENT_IDS = ["model", "behavior", "knowledge", "account", "system", "external-servers", "mcp-servers", "users", "channels"];

let activeTabName = "model";

// The shared save-bar is relevant only on Behavior and Knowledge, which edit the same currentParams for the target
// chosen on Behavior's conversation-select. Model has no save-bar (selection saves at once, saveModelSelection), so no
// modelsAvailable gating applies. activateTab and loadTarget both call this rather than toggling "hidden", so neither
// clobbers the other's reason for hiding it.
function updateSaveBarVisibility() {
  const tabAllowsSave = activeTabName === "behavior" || activeTabName === "knowledge";
  saveBarEl?.classList.toggle("hidden", !tabAllowsSave);
}

function activateTab(name) {
  for (const btn of tabButtons) {
    const active = btn.dataset.tab === name;
    btn.classList.toggle("border-brand-500", active);
    btn.classList.toggle("text-slate-100", active);
    btn.classList.toggle("border-transparent", !active);
    btn.classList.toggle("text-slate-500", !active);
  }
  for (const id of TAB_CONTENT_IDS) {
    const el = document.getElementById(`tab-${id}`);
    if (el) el.classList.toggle("hidden", id !== name);
  }
  // The Save button applies to Behavior + Knowledge (same currentParams); Model saves at once and System/Users save
  // inline. It is one shared element, moved above the "Knowledge base" section while that tab is active and back below
  // Behavior otherwise.
  if (hasUserTabs) {
    if (name === "knowledge") {
      knowledgeBaseSectionEl.parentNode.insertBefore(saveBarEl, knowledgeBaseSectionEl);
    } else if (saveBarEl.parentNode !== saveBarHomeParent || saveBarEl.nextSibling !== saveBarHomeNextSibling) {
      saveBarHomeParent.insertBefore(saveBarEl, saveBarHomeNextSibling);
    }
  }
  activeTabName = name;
  updateSaveBarVisibility();
  if (hasUserTabs) applyRagAvailabilityToKnowledgeTab();
}

// Extracted from the click handler so the per-tab data load also runs when a reload restores a saved tab
// (SETTINGS_TAB_STORAGE_KEY); otherwise a reload lands on the default "Model" tab and discards where the admin was
// (seen: reloading right after starting Ollama on External Servers left it blank until re-clicked, as its data loads
// only on that tab's click).
function loadTabData(name) {
  if (name === "model") {
    initModelTab();
  }
  if (name === "account") {
    loadAccountProfileIntoEditor();
    loadDefaultNotesEnabledIntoEditor();
  }
  if (name === "external-servers") {
    loadExternalServersIntoEditor();
  }
  if (name === "system") {
    loadSystemInfo();
    loadDbConfigIntoEditor();
    loadInstancesIntoEditor();
    loadProxyModeIntoEditor();
    loadHttpProxyIntoEditor();
    loadRagLimitsIntoEditor();
    loadDefaultModelForNewUsersEditor();
    loadDefaultVisionModelEditor();
    loadDefaultEmbeddingModelEditor();
    loadTitleModeIntoEditor();
    loadChannelDeliveryModeIntoEditor();
    loadReplyTimeoutIntoEditor();
    loadVisionReplyTimeoutIntoEditor();
    loadRetentionSettingsIntoEditor();
  }
  if (name === "users") {
    loadUsers();
    refreshAddUserGate();
  }
  if (name === "channels") {
    loadChannels();
  }
}

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    activateTab(btn.dataset.tab);
    loadTabData(btn.dataset.tab);
  });
});

// See loadTabData: set before a reload that should land on a specific tab. Restored at the bottom of this file
// (restoreTabAfterReload), not here: loadTabData("external-servers") reaches EXTERNAL_SERVERS, a const declared later,
// and calling it this early hits its temporal dead zone ("Cannot access 'EXTERNAL_SERVERS' before initialization").
const SETTINGS_TAB_STORAGE_KEY = "pairingSettingsActiveTab";

async function loadSystemInfo() {
  const info = await api("/api/settings/system-info");
  document.getElementById("system-app-version").textContent = `v${info.app_version}`;
  document.getElementById("system-hardware").textContent = info.hardware.vram_gb > 0
    ? `${info.hardware.ram_gb}GB RAM + ${info.hardware.vram_gb}GB VRAM`
    : `${info.hardware.ram_gb}GB RAM, no GPU detected`;
}

// Database (admin only)

const dbTypeSelect = document.getElementById("db-type");
const dbFieldsSqliteEl = document.getElementById("db-fields-sqlite");
const dbFieldsMysqlEl = document.getElementById("db-fields-mysql");
const dbStatusEl = document.getElementById("db-status");
const dbRestartBannerEl = document.getElementById("db-restart-banner");

function showDbFieldsForType(type) {
  dbFieldsSqliteEl.classList.toggle("hidden", type !== "sqlite");
  dbFieldsMysqlEl.classList.toggle("hidden", type !== "mysql");
}

if (dbTypeSelect) {
  dbTypeSelect.addEventListener("change", () => showDbFieldsForType(dbTypeSelect.value));
}

async function loadDbConfigIntoEditor() {
  if (!dbTypeSelect) return;
  const cfg = await api("/api/settings/database");
  dbTypeSelect.value = cfg.db_type;
  showDbFieldsForType(cfg.db_type);
  if (cfg.sqlite) {
    document.getElementById("db-sqlite-directory").value = cfg.sqlite.directory;
    document.getElementById("db-sqlite-filename").value = cfg.sqlite.filename;
  }
  if (cfg.mysql) {
    document.getElementById("db-mysql-host").value = cfg.mysql.host;
    document.getElementById("db-mysql-port").value = cfg.mysql.port;
    document.getElementById("db-mysql-user").value = cfg.mysql.user;
    document.getElementById("db-mysql-password").value = "";
    document.getElementById("db-mysql-password").placeholder = cfg.has_password
      ? "(unchanged if left blank)"
      : "(none set)";
    document.getElementById("db-mysql-database").value = cfg.mysql.database;
  }
  dbRestartBannerEl.classList.toggle("hidden", !cfg.restart_required);
  dbStatusEl.textContent = "";
}

// Mirrors DbConfigUpdate (app/schemas/db_config.py): the same body for save/test-connection/build-schema, so the three
// actions below send one payload to three endpoints.
function readDbConfigForm() {
  const dbType = dbTypeSelect.value;
  if (dbType === "sqlite") {
    return {
      db_type: "sqlite",
      sqlite: {
        directory: document.getElementById("db-sqlite-directory").value.trim(),
        filename: document.getElementById("db-sqlite-filename").value.trim() || "pAIchat.db",
      },
    };
  }
  return {
    db_type: "mysql",
    mysql: {
      host: document.getElementById("db-mysql-host").value.trim(),
      port: Number(document.getElementById("db-mysql-port").value) || 3306,
      user: document.getElementById("db-mysql-user").value.trim(),
      password: document.getElementById("db-mysql-password").value,
      database: document.getElementById("db-mysql-database").value.trim(),
    },
  };
}

function setDbStatus(text, kind) {
  dbStatusEl.textContent = text;
  dbStatusEl.classList.remove("text-slate-500", "text-emerald-400", "text-red-400");
  dbStatusEl.classList.add(kind === "ok" ? "text-emerald-400" : kind === "error" ? "text-red-400" : "text-slate-500");
}

function wireDbAction(buttonId, path, { onSuccess, confirmMessage } = {}) {
  const btn = document.getElementById(buttonId);
  if (!btn) return;
  btn.addEventListener("click", async () => {
    if (confirmMessage && !confirm(confirmMessage)) return;
    btn.disabled = true;
    setDbStatus("Working…", "muted");
    try {
      const result = await api(path, { method: path === "/api/settings/database" ? "PUT" : "POST", body: JSON.stringify(readDbConfigForm()) });
      setDbStatus(result.message, result.ok ? "ok" : "error");
      if (result.ok && onSuccess) onSuccess();
    } catch (err) {
      setDbStatus(err.message, "error");
    } finally {
      btn.disabled = false;
    }
  });
}

wireDbAction("db-test-btn", "/api/settings/database/test-connection");
wireDbAction("db-build-schema-btn", "/api/settings/database/build-schema");
wireDbAction("db-save-btn", "/api/settings/database", {
  onSuccess: loadDbConfigIntoEditor,
  confirmMessage:
    "Switching does not move anything from the current database — every existing conversation, message, " +
    "channel, non-admin user, document, and admin setting stays behind and won't be visible here anymore. " +
    "The new database only gets fresh starter data. Continue?",
});

// Local instances (admin only)

const instanceCountSaveBtn = document.getElementById("instance-count-save-btn");

/* Renders the plain-text "Instance N (primary) — port P — status" rows as a `<ul>`, with no progress bar or colored pill, per this section's "simple, well understood" brief (unlike the RAG upload progress bar). */
function renderInstanceList(config) {
  const listEl = document.getElementById("instance-list");
  listEl.innerHTML = config.instances
    .map((inst) => {
      const label = inst.index === 0 ? "primary" : "sibling";
      const status = inst.healthy === null ? "checking…" : inst.healthy ? "healthy" : "not responding";
      return `<li>Instance ${inst.index} (${label}) — port ${inst.port} — ${status}</li>`;
    })
    .join("");
}

async function loadInstancesIntoEditor() {
  if (!instanceCountSaveBtn) return;
  const config = await api("/api/settings/instances");
  document.getElementById("instance-count").value = config.count;
  renderInstanceList(config);
}

if (instanceCountSaveBtn) {
  instanceCountSaveBtn.addEventListener("click", async () => {
    const statusEl = document.getElementById("instance-count-status");
    const count = parseInt(document.getElementById("instance-count").value, 10);

    instanceCountSaveBtn.disabled = true;
    statusEl.textContent = "Saving…";
    try {
      let config = await api("/api/settings/instances", { method: "PUT", body: JSON.stringify({ count }) });
      renderInstanceList(config);
      // A short settle-and-recheck, not a progress bar: spawning/stopping takes a beat, so re-poll status a few times
      // for the count to catch up.
      for (let attempt = 0; attempt < 3 && config.instances.length !== count; attempt++) {
        statusEl.textContent = `Starting… (${config.instances.length} of ${count})`;
        await new Promise((resolve) => setTimeout(resolve, 500));
        config = await api("/api/settings/instances");
        renderInstanceList(config);
      }
      statusEl.textContent = `Saved — ${config.instances.length} of ${count} running.`;
    } catch (err) {
      statusEl.textContent = `Failed to save: ${err.message}`;
    } finally {
      instanceCountSaveBtn.disabled = false;
      setTimeout(() => (statusEl.textContent = ""), 2500);
    }
  });
}

// Proxy mode (admin only)

const proxyModeSaveBtn = document.getElementById("proxy-mode-save-btn");

async function loadProxyModeIntoEditor() {
  if (!proxyModeSaveBtn) return;
  const { mode } = await api("/api/settings/proxy-mode");
  document.getElementById("proxy-mode").value = mode;
}

if (proxyModeSaveBtn) {
  proxyModeSaveBtn.addEventListener("click", async () => {
    const statusEl = document.getElementById("proxy-mode-status");
    const mode = document.getElementById("proxy-mode").value;

    proxyModeSaveBtn.disabled = true;
    statusEl.textContent = "Saving…";
    try {
      await api("/api/settings/proxy-mode", { method: "PUT", body: JSON.stringify({ mode }) });
      statusEl.textContent = "Saved.";
    } catch (err) {
      statusEl.textContent = `Failed to save: ${err.message}`;
    } finally {
      proxyModeSaveBtn.disabled = false;
      setTimeout(() => (statusEl.textContent = ""), 2500);
    }
  });
}

// Outbound HTTP proxy (admin only)
// Unrelated to "Proxy mode" above (renamed "Local pAIring server mode" in the UI to avoid the clash): this routes the
// app's own internet downloads (app.services.http_proxy_service) through an admin-configured proxy, not incoming
// requests.

const httpProxySaveBtn = document.getElementById("http-proxy-save-btn");

async function loadHttpProxyIntoEditor() {
  if (!httpProxySaveBtn) return;
  const config = await api("/api/settings/http-proxy");
  document.getElementById("http-proxy-enabled").checked = config.enabled;
  document.getElementById("http-proxy-host").value = config.host ?? "";
  document.getElementById("http-proxy-port").value = config.port ?? "";
  document.getElementById("http-proxy-username").value = config.username ?? "";
  // Never pre-filled with the saved password (the API doesn't send it,
  // http_proxy_service.get_http_proxy_config_for_display); same "(unchanged if left blank)" pattern as the Database
  // section's MySQL password.
  const passwordInput = document.getElementById("http-proxy-password");
  passwordInput.value = "";
  passwordInput.placeholder = config.has_password ? "(unchanged if left blank)" : "(none set)";
}

if (httpProxySaveBtn) {
  httpProxySaveBtn.addEventListener("click", async () => {
    const statusEl = document.getElementById("http-proxy-status");
    const enabled = document.getElementById("http-proxy-enabled").checked;
    const host = document.getElementById("http-proxy-host").value.trim() || null;
    const portRaw = document.getElementById("http-proxy-port").value;
    const port = portRaw ? parseInt(portRaw, 10) : null;
    const username = document.getElementById("http-proxy-username").value.trim() || null;
    const password = document.getElementById("http-proxy-password").value || null;

    httpProxySaveBtn.disabled = true;
    statusEl.textContent = "Saving…";
    try {
      await api("/api/settings/http-proxy", {
        method: "PUT",
        body: JSON.stringify({ enabled, host, port, username, password }),
      });
      statusEl.textContent = "Saved.";
      await loadHttpProxyIntoEditor();
    } catch (err) {
      statusEl.textContent = `Failed to save: ${err.message}`;
    } finally {
      httpProxySaveBtn.disabled = false;
      setTimeout(() => (statusEl.textContent = ""), 2500);
    }
  });
}

// External servers: Ollama & ComfyUI (admin only)
//
// Both sections share one DOM shape (settings.html's [data-server="ollama"|"comfyui"]): a mode select, a "local" div
// (install/start/stop + tunable parameters) and a "remote" div (up to 12 host rows), so one generic controller drives
// both. installServer is where their local-mode "Install" buttons differ (what happens after each install).

const EXTERNAL_SERVERS = ["ollama", "matricxon", "comfyui", "sdcpp"];
const MAX_REMOTE_HOSTS = 120;

function serverSectionEl(server) {
  return document.querySelector(`[data-server="${server}"]`);
}

const HOST_CHECK_STATUS_IDLE_CLASS = "shrink-0 text-xs text-slate-500 w-24 text-center";

// Only for this row's placeholder/title example: each server's usual default port and display name.
const REMOTE_HOST_EXAMPLE = {
  ollama: { port: "11434", name: "Ollama" },
  matricxon: { port: "8420", name: "Matricxon" },
  comfyui: { port: "8188", name: "ComfyUI" },
  sdcpp: { port: "8189", name: "stable-diffusion.cpp" },
};

function addRemoteHostRow(section, value = "") {
  const listEl = section.querySelector('[data-field="remote-hosts-list"]');
  if (listEl.children.length >= MAX_REMOTE_HOSTS) return;
  const example = REMOTE_HOST_EXAMPLE[section.dataset.server] || REMOTE_HOST_EXAMPLE.ollama;
  const row = document.createElement("div");
  row.className = "flex items-center gap-2";
  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = `http://10.0.0.5:${example.port}`;
  input.title = `URL of a remote ${example.name} server to load-balance across, including its port, e.g. http://10.0.0.5:${example.port}.`;
  input.value = value;
  input.className =
    "flex-1 rounded-md border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm focus:outline-none focus:border-brand-500";

  // Reachability check for this row's URL: a real ping (check-host endpoint in
  // app/routers/ollama_admin.py|comfyui_admin.py), not limited to saved hosts, so an admin can verify a remote
  // candidate before Save. Cleared on edit, since a stale "✓ healthy" beside a changed URL would mislead.
  const statusEl = document.createElement("span");
  statusEl.className = HOST_CHECK_STATUS_IDLE_CLASS;
  input.addEventListener("input", () => {
    statusEl.textContent = "";
    statusEl.className = HOST_CHECK_STATUS_IDLE_CLASS;
  });

  const checkBtn = document.createElement("button");
  checkBtn.type = "button";
  checkBtn.textContent = "Check";
  checkBtn.className =
    "shrink-0 rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-200 hover:bg-slate-800 transition-colors disabled:opacity-40 disabled:cursor-not-allowed";
  checkBtn.addEventListener("click", async () => {
    const host = input.value.trim();
    if (!host) {
      statusEl.textContent = "no URL";
      statusEl.className = HOST_CHECK_STATUS_IDLE_CLASS;
      return;
    }
    checkBtn.disabled = true;
    statusEl.textContent = "checking…";
    statusEl.className = HOST_CHECK_STATUS_IDLE_CLASS;
    try {
      const server = section.dataset.server;
      const result = await api(`/api/settings/${server}/check-host?host=${encodeURIComponent(host)}`);
      statusEl.textContent = result.healthy ? "✓ healthy" : "✗ unreachable";
      statusEl.className = `shrink-0 text-xs w-24 text-center ${result.healthy ? "text-emerald-400" : "text-red-400"}`;
    } catch (err) {
      statusEl.textContent = "✗ error";
      statusEl.className = "shrink-0 text-xs w-24 text-center text-red-400";
    } finally {
      checkBtn.disabled = false;
    }
  });

  const removeBtn = document.createElement("button");
  removeBtn.type = "button";
  removeBtn.textContent = "×";
  removeBtn.className = "shrink-0 text-slate-500 hover:text-red-400 px-2";
  removeBtn.addEventListener("click", () => row.remove());
  row.append(input, checkBtn, statusEl, removeBtn);
  listEl.appendChild(row);
}

function renderRemoteHosts(section, hosts) {
  const listEl = section.querySelector('[data-field="remote-hosts-list"]');
  listEl.innerHTML = "";
  for (const host of hosts.length ? hosts : [""]) addRemoteHostRow(section, host);
}

function collectRemoteHosts(section) {
  // In cluster mode the nodes belong to the Cluster page; the form hands back the list it was shown (the server ignores
  // it, but a remote-mode save with no hosts would fall back to local mode).
  if (section.dataset.clusterHosts) return JSON.parse(section.dataset.clusterHosts);
  return Array.from(section.querySelectorAll('[data-field="remote-hosts-list"] input'))
    .map((input) => input.value.trim())
    .filter(Boolean);
}

// The last status fetched per server, kept current by renderServerStatus, so the mode <select>'s "change" event and the
// manual-path toggle can recompute visibility without a round trip.
const lastServerStatus = {};

/*
 * Local mode with nothing installed shows only the install flow (no local parameter fields, Start/Stop or Save):
 * half-filled params for a server that doesn't exist are more confusing than useful. Both servers have a manual
 * override link for an admin pointing at their own Ollama/ComfyUI install (see "manual-path-entry" actions below).
 * Remote mode shows the full form but never Start/Stop, since this app doesn't supervise a process it doesn't own.
 * 
 * install-controls (Install/Reinstall button + version override) stay visible either way; hiding them once installed
 * left no way to see which repo/version was installed or reinstall another. Hidden only in manual-path-entry mode,
 * where reinstalling would overwrite the admin's own path with the managed install location (installServer's completion
 * handler always saves the installed path).
 */
function applyServerVisibility(server, section) {
  const status = lastServerStatus[server];
  if (!status) return;
  const mode = section.querySelector('[data-field="mode"]').value;
  const manualOverride = section.dataset.manualPathEntry === "true";
  const showLocalParams = status.installed || manualOverride;

  section.querySelector('[data-mode-section="local"]').classList.toggle("hidden", mode !== "local");
  section.querySelector('[data-mode-section="remote"]').classList.toggle("hidden", mode !== "remote");

  const installControls = section.querySelector('[data-field="install-controls"]');
  if (installControls) installControls.classList.toggle("hidden", mode !== "local" || manualOverride);
  const installBtn = section.querySelector('[data-action="install"]');
  if (installBtn) installBtn.textContent = status.installed ? "Reinstall from GitHub" : "Install from GitHub";
  const sourceBtn = section.querySelector('[data-action="build-source"]');
  if (sourceBtn) {
    const missing = status.source_build_missing ?? [];
    sourceBtn.textContent = status.installed ? "Rebuild from source" : "Build from source";
    sourceBtn.disabled = missing.length > 0;
    if (missing.length) sourceBtn.title = `Needs ${missing.join(", ")} - on Debian/Ubuntu: sudo apt install git cmake build-essential`;
  }

  const notInstalledPanel = section.querySelector('[data-field="local-not-installed"]');
  const installedPanel = section.querySelector('[data-field="local-installed"]');
  if (notInstalledPanel) notInstalledPanel.classList.toggle("hidden", showLocalParams);
  if (installedPanel) installedPanel.classList.toggle("hidden", !showLocalParams);

  const backLink = section.querySelector('[data-action="manual-path-entry-back"]');
  if (backLink) backLink.classList.toggle("hidden", !(manualOverride && !status.installed));

  const actionsBar = section.querySelector('[data-field="actions-bar"]');
  const startBtn = section.querySelector('[data-action="start"]');
  const stopBtn = section.querySelector('[data-action="stop"]');
  const hideEverything = mode === "local" && !showLocalParams;
  if (actionsBar) actionsBar.classList.toggle("hidden", hideEverything);
  if (startBtn) startBtn.classList.toggle("hidden", mode !== "local");
  if (stopBtn) stopBtn.classList.toggle("hidden", mode !== "local");
  // Same disabled condition for both: nothing to start (or stop) until something is installed.
  const disabledUntilInstalled = mode === "local" && !status.installed;
  if (startBtn) startBtn.disabled = disabledUntilInstalled;
  if (stopBtn) stopBtn.disabled = disabledUntilInstalled;
}

function renderServerStatus(section, status) {
  const server = section.dataset.server;
  lastServerStatus[server] = status;
  const line = section.querySelector('[data-field="status-line"]');
  if (!status.running) {
    line.textContent = status.installed ? "Not running." : "Not installed.";
  } else {
    const health = status.healthy === null ? "checking…" : status.healthy ? "healthy" : "not responding";
    line.textContent = status.pid ? `Running (pid ${status.pid}) — ${health}` : `Running — ${health}`;
  }
  applyServerVisibility(server, section);
}

/* Styles a server section's status text as a persistent error (wrapping, red) or back to plain. */
function setServerStatusError(statusEl, isError) {
  statusEl.classList.toggle("text-red-400", isError);
  statusEl.classList.toggle("text-slate-500", !isError);
  statusEl.classList.toggle("min-w-0", isError);
  statusEl.classList.toggle("break-words", isError);
}

function intFieldOrNull(section, field) {
  const raw = section.querySelector(`[data-field="${field}"]`).value.trim();
  return raw === "" ? null : parseInt(raw, 10);
}

function floatFieldOrNull(section, field) {
  const raw = section.querySelector(`[data-field="${field}"]`).value.trim();
  return raw === "" ? null : parseFloat(raw);
}

function collectServerConfig(server, section) {
  const base = {
    mode: section.querySelector('[data-field="mode"]').value,
    remote_hosts: collectRemoteHosts(section),
    // Empty means "use the built-in pinned default" (as OllamaServerConfig in app.schemas.ollama_server_config).
    install_repo: section.querySelector('[data-field="install_repo"]')?.value.trim() || null,
    install_version: section.querySelector('[data-field="install_version"]')?.value.trim() || null,
  };
  // Local-mode-only fields live in inputs that stay in the DOM, hidden (applyServerVisibility), after switching to
  // remote; reading them would resend a stale local value (e.g. models_path) on every remote-mode Save, which reapplies
  // if the admin switches back. So they are nulled/defaulted in remote mode.
  const isRemote = base.mode === "remote";
  // models_path is the exception: the remote engine manages its own storage (downloads go through its API), but the
  // saved local path is kept so switching back to local still finds the models.
  const keptModelsPath = () => section.querySelector('[data-field="models_path"]').dataset.loadedValue || null;
  const keptCachePath = () => section.querySelector('[data-field="prompt_cache_path"]')?.dataset.loadedValue || null;
  if (server === "ollama") {
    return {
      ...base,
      binary_path: isRemote ? null : section.querySelector('[data-field="binary_path"]').value.trim() || null,
      models_path: isRemote ? keptModelsPath() : section.querySelector('[data-field="models_path"]').value.trim() || null,
      num_parallel: isRemote ? null : intFieldOrNull(section, "num_parallel"),
      keep_alive: isRemote ? null : section.querySelector('[data-field="keep_alive"]').value.trim() || null,
      max_loaded_models: isRemote ? null : intFieldOrNull(section, "max_loaded_models"),
      context_length: isRemote ? null : intFieldOrNull(section, "context_length"),
    };
  }
  if (server === "matricxon") {
    return {
      ...base,
      project_dir: isRemote ? null : section.querySelector('[data-field="project_dir"]').value.trim() || null,
      models_path: isRemote ? keptModelsPath() : section.querySelector('[data-field="models_path"]').value.trim() || null,
      prompt_cache_path: isRemote
        ? keptCachePath()
        : section.querySelector('[data-field="prompt_cache_path"]').value.trim() || null,
      // Not editable here (the Cluster page owns it) but must be sent back as loaded, or every Save would reset cluster
      // mode to "off".
      cluster_role: section.dataset.clusterRole || "off",
      max_loaded_models: isRemote ? null : intFieldOrNull(section, "max_loaded_models"),
      keep_alive_minutes: isRemote ? null : intFieldOrNull(section, "keep_alive_minutes"),
      prompt_cache_slots: isRemote ? null : intFieldOrNull(section, "prompt_cache_slots"),
      prompt_cache_budget_mb: isRemote ? null : intFieldOrNull(section, "prompt_cache_budget_mb"),
      max_decode_batch: isRemote ? null : intFieldOrNull(section, "max_decode_batch"),
      memory_safety_margin: isRemote ? null : floatFieldOrNull(section, "memory_safety_margin"),
      log_level: isRemote ? 0 : parseInt(section.querySelector('[data-field="log_level"]').value, 10),
      enable_quantized_native_compute:
        isRemote || section.querySelector('[data-field="enable_quantized_native_compute"]').checked,
      gemv_backend: isRemote ? "native" : section.querySelector('[data-field="gemv_backend"]').value,
      torch_threads: isRemote ? null : intFieldOrNull(section, "torch_threads"),
      device: isRemote ? "cpu" : matricxonDeviceValue(section),
      gpu_weight_mode: isRemote ? "dequantized" : section.querySelector('[data-field="gpu_weight_mode"]').value,
    };
  }
  if (server === "sdcpp") {
    return {
      ...base,
      binary_path: section.querySelector('[data-field="binary_path"]').value.trim() || null,
      model_path: section.querySelector('[data-field="model_path"]').value.trim() || null,
      models_path: section.querySelector('[data-field="models_path"]').value.trim() || null,
      images_path: section.querySelector('[data-field="images_path"]').value.trim() || null,
      timeout_minutes: Math.min(1440, Math.max(1, parseInt(section.querySelector('[data-field="timeout_minutes"]').value, 10) || 30)),
      extra_args: section.querySelector('[data-field="extra_args"]').value.trim() || null,
      build: section.querySelector('[data-field="build"]').value,
    };
  }
  return {
    ...base,
    python_path: section.querySelector('[data-field="python_path"]').value.trim() || null,
    main_py_path: section.querySelector('[data-field="main_py_path"]').value.trim() || null,
    extra_args: section.querySelector('[data-field="extra_args"]').value.trim() || null,
  };
}

/* stable-diffusion.cpp's build choice (CPU/Vulkan/ROCm) rides along as a query param, so it applies without a prior Save. */
function installQuery(server, section, buildOverride = null) {
  const build = buildOverride || (server === "sdcpp" ? section.querySelector('[data-field="build"]')?.value : null);
  return build ? `?build=${encodeURIComponent(build)}` : "";
}

/* Fills stable-diffusion.cpp's Model dropdown from the local models folder (app/services/image_model_service.py); a saved path not in the folder (set before this dropdown existed) stays selectable so it isn't lost. */
async function populateSdcppModels(section, savedPath) {
  const select = section.querySelector('[data-field="model_path"]');
  const hint = section.querySelector('[data-field="model-hint"]');
  const { models, folder } = await api("/api/settings/image-models");
  section.querySelector('[data-field="models_path"]').placeholder = folder;
  const options = models.map((m) => ({ value: m.path, label: `${m.name} (${m.size_gb} GB)` }));
  if (savedPath && !options.some((o) => o.value === savedPath)) options.push({ value: savedPath, label: savedPath });
  select.innerHTML = `<option value="">— choose a model —</option>` +
    options.map((o) => `<option value="${escapeHtml(o.value)}">${escapeHtml(o.label)}</option>`).join("");
  select.value = savedPath ?? "";
  hint.classList.toggle("hidden", models.length > 0 || !!savedPath);
}

function renderInstallSource(section, config, defaults) {
  const repoInput = section.querySelector('[data-field="install_repo"]');
  const versionInput = section.querySelector('[data-field="install_version"]');
  if (repoInput) {
    repoInput.value = config.install_repo ?? "";
    repoInput.placeholder = defaults.repo;
  }
  if (versionInput) {
    versionInput.value = config.install_version ?? "";
    versionInput.placeholder = defaults.version;
  }
  const label = section.querySelector('[data-field="install-version-label"]');
  if (label) label.textContent = `${config.install_repo || defaults.repo} @ ${config.install_version || defaults.version}`;
}

async function loadServerSection(server) {
  const section = serverSectionEl(server);
  if (!section) return;
  const fetchPanel = section.querySelector('[data-field="fetch-panel"]');
  const [config, installDefaults] = await withPanelSpinner(
    fetchPanel,
    Promise.all([api(`/api/settings/${server}/config`), api(`/api/settings/${server}/install-defaults`)]),
  );
  section.querySelector('[data-field="mode"]').value = config.mode;
  section.dataset.loadedMode = config.mode;
  renderInstallSource(section, config, installDefaults);

  if (server === "ollama") {
    const binaryPathInput = section.querySelector('[data-field="binary_path"]');
    binaryPathInput.value = config.binary_path ?? "";
    // Shows what a blank value resolves to (PATH, then the usual fixed locations), a real current value rather than
    // "auto-detected from PATH".
    const autoDetected = await api("/api/settings/ollama/auto-detected-path");
    binaryPathInput.placeholder = autoDetected.path || "/var/lib/Ollama";
    const modelsPathInput = section.querySelector('[data-field="models_path"]');
    modelsPathInput.value = config.models_path ?? "";
    modelsPathInput.dataset.loadedValue = config.models_path ?? "";
    modelsPathInput.placeholder = autoDetected.models_path;
    section.querySelector('[data-field="num_parallel"]').value = config.num_parallel ?? "";
    section.querySelector('[data-field="keep_alive"]').value = config.keep_alive ?? "";
    section.querySelector('[data-field="max_loaded_models"]').value = config.max_loaded_models ?? "";
    section.querySelector('[data-field="context_length"]').value = config.context_length ?? "";
  } else if (server === "matricxon") {
    const projectDirInput = section.querySelector('[data-field="project_dir"]');
    projectDirInput.value = config.project_dir ?? "";
    // Same "show what blank resolves to" idea as Ollama's binary_path: here, the sibling checkout
    // app.services.matricxon_process._find_project_dir falls back to.
    const autoDetected = await api("/api/settings/matricxon/auto-detected-path");
    projectDirInput.placeholder = autoDetected.path || "/var/lib/matricxon";
    const modelsPathInput = section.querySelector('[data-field="models_path"]');
    modelsPathInput.value = config.models_path ?? "";
    modelsPathInput.dataset.loadedValue = config.models_path ?? "";
    modelsPathInput.placeholder = autoDetected.models_path || "/var/lib/models";
    const cachePathInput = section.querySelector('[data-field="prompt_cache_path"]');
    cachePathInput.value = config.prompt_cache_path ?? "";
    cachePathInput.dataset.loadedValue = config.prompt_cache_path ?? "";
    section.dataset.clusterRole = config.cluster_role ?? "off";
    section.querySelector('[data-field="max_loaded_models"]').value = config.max_loaded_models ?? "";
    section.querySelector('[data-field="keep_alive_minutes"]').value = config.keep_alive_minutes ?? "";
    for (const field of ["prompt_cache_slots", "prompt_cache_budget_mb", "max_decode_batch"]) {
      section.querySelector(`[data-field="${field}"]`).value = config[field] ?? "";
    }
    section.querySelector('[data-field="memory_safety_margin"]').value = config.memory_safety_margin ?? "";
    section.querySelector('[data-field="log_level"]').value = String(config.log_level ?? 0);
    section.querySelector('[data-field="enable_quantized_native_compute"]').checked =
      config.enable_quantized_native_compute ?? true;
    section.querySelector('[data-field="gemv_backend"]').value = config.gemv_backend ?? "native";
    section.querySelector('[data-field="torch_threads"]').value = config.torch_threads ?? "";
    // A hand-set "cuda:1" has no option of its own; show it as generic CUDA and remember it (matricxonDeviceValue) so
    // saving doesn't change the GPU index.
    section.dataset.savedDevice = config.device ?? "cpu";
    section.querySelector('[data-field="device"]').value = (config.device ?? "cpu").startsWith("cuda") ? "cuda" : "cpu";
    section.querySelector('[data-field="gpu_weight_mode"]').value = config.gpu_weight_mode ?? "dequantized";
    syncMatricxonKernelSelect(section);
  } else if (server === "sdcpp") {
    section.querySelector('[data-field="binary_path"]').value = config.binary_path ?? "";
    section.querySelector('[data-field="models_path"]').value = config.models_path ?? "";
    section.querySelector('[data-field="images_path"]').value = config.images_path ?? "";
    section.querySelector('[data-field="timeout_minutes"]').value = config.timeout_minutes ?? 30;
    await populateSdcppModels(section, config.model_path);
    section.querySelector('[data-field="build"]').value = config.build ?? "auto";
    section.querySelector('[data-field="extra_args"]').value = config.extra_args ?? "";
  } else {
    section.querySelector('[data-field="python_path"]').value = config.python_path ?? "";
    section.querySelector('[data-field="main_py_path"]').value = config.main_py_path ?? "";
    section.querySelector('[data-field="extra_args"]').value = config.extra_args ?? "";
  }
  renderRemoteHosts(section, config.remote_hosts || []);
  renderServerStatus(section, await api(`/api/settings/${server}/status`));
}

/* Grays out "Install from GitHub" with an explanation on any OS the auto-installer can't run on (app.hardware.local_install_supported: Ollama's pinned release asset and this app's deployment assume Linux), so the admin doesn't wait minutes for a doomed attempt. ollama_installer/comfyui_installer install_stream enforce the same check server-side; this is a courtesy. */
function applyLocalInstallSupport(server, section, supported) {
  if (!section) return;
  const installBtn = section.querySelector('[data-action="install"]');
  const noteEl = section.querySelector('[data-field="install-os-note"]');
  if (installBtn) installBtn.disabled = !supported;
  if (installBtn) installBtn.classList.toggle("opacity-40", !supported);
  if (installBtn) installBtn.classList.toggle("cursor-not-allowed", !supported);
  if (noteEl) {
    noteEl.classList.toggle("hidden", supported);
    if (!supported) {
      noteEl.textContent =
        "Auto-install needs Linux — this server isn't running it. Install it yourself and enter its path(s) " +
        "manually, or use Remote mode.";
    }
  }
}

// Ollama and Matricxon are mutually exclusive chat backends (see the "Active engine" picker); ComfyUI (image
// generation) is separate and always visible.
const ENGINE_SERVERS = ["ollama", "matricxon"];

/* Shows only the selected engine's configuration section, hiding the other so an admin doesn't edit the form of a backend that isn't live. Re-run on load and on the select's "change" event. */
function applyActiveEngineVisibility() {
  const selectEl = document.getElementById("active-engine-select");
  if (!selectEl) return;
  for (const server of ENGINE_SERVERS) {
    const section = serverSectionEl(server);
    if (section) section.classList.toggle("hidden", server !== selectEl.value);
  }
  // A connector engine has no form here; settings_connector_summary.js shows its summary instead.
  document.dispatchEvent(new CustomEvent("active-engine-shown", { detail: selectEl.value }));
  // The saved-chat-caches box is a Matricxon feature (its API answers 409 for other engines): shown only while
  // Matricxon is both the picked and the saved engine; hidden at once when another is picked.
  const cachePanel = document.getElementById("matricxon-cache-panel");
  if (cachePanel && savedActiveEngine !== null) {
    const wanted = selectEl.value === "matricxon" && savedActiveEngine === "matricxon";
    // In cluster mode the Cluster page owns these limits (settings_cache_persistence.js sets the flag).
    cachePanel.classList.toggle("hidden", !wanted || cachePanel.dataset.clusterManaged === "1");
  }
}

// The engine saved as of the last load/save, compared with the select's live value on each "change" so the switch note
// shows only for a pending switch, not on page load.
let savedActiveEngine = null;

async function loadActiveEngineIntoEditor() {
  const selectEl = document.getElementById("active-engine-select");
  if (!selectEl) return;
  const [{ active_engine: activeEngine }, engineOptions] = await Promise.all([
    api("/api/settings/engine"),
    engineOptionsList(),
  ]);
  selectEl.innerHTML = "";
  // Only a ready engine (EngineOption.ready, app/schemas/common.py) can be picked: Ollama/Matricxon always are; a
  // connector-backed engine (e.g. "runpod") only after an admin configures, enables and tests it on the Connectors
  // page. The active engine is always shown, even if it became not-ready (e.g. its connector was disabled), so the
  // picker reflects reality.
  for (const { name, display_name: displayName, ready } of engineOptions) {
    if (!ready && name !== activeEngine) continue;
    const option = document.createElement("option");
    option.value = name;
    option.textContent = displayName;
    selectEl.appendChild(option);
  }
  selectEl.value = activeEngine;
  savedActiveEngine = activeEngine;
  applyActiveEngineVisibility();
  applyActiveEngineSwitchNote();
}

/* Ollama and Matricxon have separate model storage (see the External Servers page), which the picker alone doesn't show. Shown the moment the selection differs from the saved one (not after Save, to inform the decision) and hidden again if changed back. */
function applyActiveEngineSwitchNote() {
  const selectEl = document.getElementById("active-engine-select");
  const noteEl = document.getElementById("active-engine-switch-note");
  if (!selectEl || !noteEl) return;
  noteEl.classList.toggle("hidden", selectEl.value === savedActiveEngine);
}

async function loadExternalServersIntoEditor() {
  await Promise.all([loadActiveEngineIntoEditor(), ...EXTERNAL_SERVERS.map(loadServerSection)]);
  const { local_install_supported: supported } = await api("/api/settings/system-info");
  for (const server of EXTERNAL_SERVERS) {
    applyLocalInstallSupport(server, serverSectionEl(server), supported);
  }
}

const activeEngineSelectEl = document.getElementById("active-engine-select");
if (activeEngineSelectEl) {
  activeEngineSelectEl.addEventListener("change", () => {
    applyActiveEngineVisibility();
    applyActiveEngineSwitchNote();
  });
}

const activeEngineSaveBtn = document.getElementById("active-engine-save-btn");
if (activeEngineSaveBtn) {
  activeEngineSaveBtn.addEventListener("click", async () => {
    const statusEl = document.getElementById("active-engine-status");
    const selectEl = document.getElementById("active-engine-select");
    statusEl.textContent = "Saving…";
    try {
      await api("/api/settings/engine", { method: "PUT", body: JSON.stringify({ active_engine: selectEl.value }) });
      // A full reload, not a refetch: the Model tab's catalog (currentCatalog) and other model fetches (installed
      // vision/embedding models, RAG availability,...) cache for the page load and go stale on an engine switch. Same
      // reload-instead-of-invalidating pattern as the Ollama/Matricxon Start button below. Stays on External servers so
      // the admin can keep configuring the new engine.
      try {
        sessionStorage.setItem(SETTINGS_TAB_STORAGE_KEY, "external-servers");
      } catch (_err) {
        // Storage blocked: the reload still happens, landing on the default tab.
      }
      window.location.reload();
      return;
    } catch (err) {
      statusEl.textContent = `Failed to save: ${err.message}`;
      setTimeout(() => (statusEl.textContent = ""), 2500);
    }
  });
}

/* Fills a server's version <datalist> with tags fetched live from GitHub (app.services.github_releases; one endpoint per server, GET /api/settings/{server}/available-versions), so the "Version / tag" field becomes pick-or-type instead of guessing exact spelling (a typo otherwise fails an install several steps in). A convenience over the free-text input: an empty or failed fetch (GitHub unreachable, rate-limited, no tags) leaves the datalist empty. Re-run when the repo field changes, since a fork's tags differ from the default repo's. */
async function loadAvailableVersions(server, section) {
  const datalist = section.querySelector('[data-field="install-version-datalist"]');
  if (!datalist) return;
  const repo = section.querySelector('[data-field="install_repo"]')?.value.trim() || "";
  try {
    const { versions } = await api(
      `/api/settings/${server}/available-versions${repo ? `?repo=${encodeURIComponent(repo)}` : ""}`,
    );
    datalist.innerHTML = "";
    for (const version of versions) {
      const option = document.createElement("option");
      option.value = version;
      datalist.appendChild(option);
    }
  } catch (_err) {
    // Leaves what was there (or nothing); the text input still works (see the docstring).
  }
}

/* Switching Ollama/Matricxon between local and remote changes which models exist (a remote server has its own storage, managed via its API), so every model cache on this page is stale: reload as for the active-engine switch and return to External servers. Returns true if it reloaded. */
function reloadIfModeChanged(section, body, force = false) {
  const server = section.dataset.server;
  if (!["ollama", "matricxon"].includes(server) || (!force && body.mode === section.dataset.loadedMode)) return false;
  try {
    sessionStorage.setItem(SETTINGS_TAB_STORAGE_KEY, "external-servers");
  } catch (_err) {
    // Storage blocked: the reload still happens, landing on the default tab.
  }
  window.location.reload();
  return true;
}

for (const server of EXTERNAL_SERVERS) {
  const section = serverSectionEl(server);
  if (!section) continue;

  section.querySelector('[data-field="mode"]').addEventListener("change", () => {
    applyServerVisibility(server, section);
  });
  section.querySelector('[data-action="add-host"]').addEventListener("click", () => addRemoteHostRow(section));

  // Both servers (settings.html): an admin pointing at their own Ollama/ComfyUI install, not one this app's installer
  // manages.
  const manualPathBtn = section.querySelector('[data-action="manual-path-entry"]');
  if (manualPathBtn) {
    manualPathBtn.addEventListener("click", () => {
      section.dataset.manualPathEntry = "true";
      applyServerVisibility(server, section);
    });
  }
  const manualPathBackBtn = section.querySelector('[data-action="manual-path-entry-back"]');
  if (manualPathBackBtn) {
    manualPathBackBtn.addEventListener("click", () => {
      section.dataset.manualPathEntry = "false";
      applyServerVisibility(server, section);
    });
  }

  const toggleSourceBtn = section.querySelector('[data-action="toggle-install-source"]');
  if (toggleSourceBtn) {
    toggleSourceBtn.addEventListener("click", () => {
      const opening = section.querySelector('[data-field="install-source-panel"]').classList.contains("hidden");
      section.querySelector('[data-field="install-source-panel"]').classList.toggle("hidden");
      if (opening) loadAvailableVersions(server, section);
    });
  }

  // Re-fetches when the admin finishes typing a different repo: a fork's tags differ from the default repo's (see
  // loadAvailableVersions).
  const repoInput = section.querySelector('[data-field="install_repo"]');
  if (repoInput) {
    repoInput.addEventListener("change", () => loadAvailableVersions(server, section));
  }

  const saveSourceBtn = section.querySelector('[data-action="save-install-source"]');
  if (saveSourceBtn) {
    saveSourceBtn.addEventListener("click", async () => {
      const statusEl = section.querySelector('[data-field="install-source-status"]');
      statusEl.textContent = "Saving…";
      try {
        await api(`/api/settings/${server}/config`, {
          method: "PUT",
          body: JSON.stringify(collectServerConfig(server, section)),
        });
        statusEl.textContent = "Saved.";
        await loadServerSection(server);
      } catch (err) {
        statusEl.textContent = `Failed to save: ${err.message}`;
      } finally {
        setTimeout(() => (statusEl.textContent = ""), 2500);
      }
    });
  }

  const resetSourceBtn = section.querySelector('[data-action="reset-install-source"]');
  if (resetSourceBtn) {
    resetSourceBtn.addEventListener("click", () => {
      section.querySelector('[data-field="install_repo"]').value = "";
      section.querySelector('[data-field="install_version"]').value = "";
      saveSourceBtn.click();
    });
  }

  section.querySelector('[data-action="save"]').addEventListener("click", async () => {
    const statusEl = section.querySelector('[data-field="status"]');
    const body = collectServerConfig(server, section);
    // Seen: clearing this field silently drops the override, so Matricxon/Ollama falls back to its default
    // models directory on its next restart and models visible only at the custom path vanish with no warning at save
    // time. Destructive enough to ask first (like the duplicate-install confirm in pullModel and the "this restarts the
    // engine" confirm below).
    const modelsPathInput = section.querySelector('[data-field="models_path"]');
    if (modelsPathInput?.dataset.loadedValue && !body.models_path) {
      const confirmed = confirm(
        `Clearing the models directory override means ${server === "ollama" ? "Ollama" : "Matricxon"} will fall ` +
          `back to its own default location on its next restart — models currently at "${modelsPathInput.dataset.loadedValue}" ` +
          "will stop showing up as installed (they aren't deleted, just no longer where it's looking). Continue?",
      );
      if (!confirmed) return;
    }
    statusEl.textContent = "Saving…";
    setServerStatusError(statusEl, false);
    let failed = false;
    let failedPrefix = "Failed to save";
    try {
      // Local-mode parameters of Ollama (and Matricxon, see app.services.matricxon_process.apply_local_config) can't
      // apply live: both read their env-based config at process start, so saving restarts a running one. Confirm first,
      // as the old "Ollama concurrency" save button did.
      const restartableEngines = { ollama: "Ollama", matricxon: "Matricxon", sdcpp: "stable-diffusion.cpp" };
      const engineLabel = restartableEngines[server];
      const localStatus =
        engineLabel && body.mode === "local" ? await api(`/api/settings/${server}/status`) : null;
      // Matricxon changes most settings live (app/services/matricxon_live_settings.py): restart only when needed.
      let restartReason = "";
      if (localStatus?.running && server === "matricxon") {
        const plan = await api("/api/settings/matricxon/plan", { method: "POST", body: JSON.stringify(body) });
        if (!plan.restart_required) {
          try {
            const live = await api("/api/settings/matricxon/apply-live", {
              method: "POST",
              body: JSON.stringify(body),
            });
            statusEl.textContent = live.applied_next_load.length
              ? "Saved — applied without a restart. Kernel and weight changes apply when a model next loads."
              : "Saved — applied without a restart.";
            reloadIfModeChanged(section, body);
            return;
          } catch (err) {
            // 409: this Matricxon can't change settings live (older version), so fall back to a restart.
            if (err.status !== 409) throw err;
          }
        } else {
          restartReason = ` (changed: ${plan.restart_labels.join(", ")})`;
        }
      }
      if (localStatus?.running) {
        const confirmed = confirm(
          `Saving restarts ${engineLabel}${restartReason} — every ${server === "sdcpp" ? "image" : "reply"} ` +
            "currently being generated, on every local instance, will be interrupted. Continue?",
        );
        if (!confirmed) {
          statusEl.textContent = "";
          return;
        }
        statusEl.textContent = `Restarting ${engineLabel}…`;
        failedPrefix = "Not saved";
        renderServerStatus(
          section,
          await api(`/api/settings/${server}/apply`, { method: "POST", body: JSON.stringify(body) }),
        );
        statusEl.textContent = `Saved — ${engineLabel} restarted.`;
        reloadIfModeChanged(section, body, true);
        return;
      }
      await api(`/api/settings/${server}/config`, { method: "PUT", body: JSON.stringify(body) });
      statusEl.textContent = "Saved.";
      // A new Models folder changes what the Model dropdown lists: re-read it.
      if (server === "sdcpp") await loadServerSection(server);
      // Not running (stopped by hand, or a previous restart failed, e.g. a GPU device that isn't there): saving alone
      // shows nothing, which reads as "nothing happened"; offer to start it with the new settings.
      if (localStatus && localStatus.installed && !localStatus.running) {
        if (confirm(`${engineLabel} is not running. Start it now with the saved settings?`)) {
          failedPrefix = "Saved, but failed to start";
          statusEl.textContent = `Saved — starting ${engineLabel}…`;
          const started = await api(`/api/settings/${server}/start`, { method: "POST" });
          renderServerStatus(section, started);
          statusEl.textContent = `Saved — ${engineLabel} started.`;
          try {
            sessionStorage.setItem(SETTINGS_TAB_STORAGE_KEY, "external-servers");
          } catch (_err) {
            // Storage blocked: the reload still happens, landing on the default tab.
          }
          window.location.reload();
          return;
        }
      }
      reloadIfModeChanged(section, body);
    } catch (err) {
      // Stays put (no timeout or redirect) so the admin can read why and fix the setting (e.g. a CUDA device on a
      // machine without a GPU fails the restart). The failed restart left the engine stopped, so refresh the status
      // line.
      failed = true;
      statusEl.textContent = `${failedPrefix}: ${err.message}`;
      setServerStatusError(statusEl, true);
      try {
        renderServerStatus(section, await api(`/api/settings/${server}/status`));
      } catch (_statusErr) {
        // The error above is what matters; the status line keeps its last value.
      }
    } finally {
      if (!failed) setTimeout(() => (statusEl.textContent = ""), 3000);
    }
  });

  for (const action of ["start", "stop"]) {
    const actionBtn = section.querySelector(`[data-action="${action}"]`);
    actionBtn.addEventListener("click", async () => {
      const statusEl = section.querySelector('[data-field="status"]');
      // Anchored to the Save button (last in the row, before the status text), not whichever of Start/Stop was clicked,
      // so the spinner stays in one place.
      const saveBtn = section.querySelector('[data-action="save"]');
      statusEl.textContent = "Working…";
      setServerStatusError(statusEl, false);
      let failed = false;
      actionBtn.disabled = true;
      const spinner = addInlineSpinner(saveBtn);
      try {
        const result = await api(`/api/settings/${server}/${action}`, { method: "POST" });
        renderServerStatus(section, result);
        statusEl.textContent = "Done.";
        // Ollama (or Matricxon) coming up can newly make models available for "Model & behavior", which loads the
        // catalog once at page load and would keep saying "System is not configured for models yet". A full reload is
        // the simplest reliable fix for an infrequent admin action, saved so it returns to External Servers (otherwise
        // the status line went blank until re-clicked, which read as stuck).
        if ((server === "ollama" || server === "matricxon") && (action === "stop" || result.running)) {
          try {
            sessionStorage.setItem(SETTINGS_TAB_STORAGE_KEY, "external-servers");
          } catch (_err) {
            // Storage blocked: the reload still happens, landing on the default tab.
          }
          window.location.reload();
          return;
        }
      } catch (err) {
        // Stays visible (see the Save handler), e.g. Start failing because the saved device is a GPU that isn't there;
        // the admin needs to read it and change the setting.
        failed = true;
        statusEl.textContent = `Failed: ${err.message}`;
        setServerStatusError(statusEl, true);
      } finally {
        spinner.remove();
        actionBtn.disabled = false;
        if (!failed) setTimeout(() => (statusEl.textContent = ""), 2500);
      }
    });
  }

  const installBtn = section.querySelector('[data-action="install"]');
  const sourceBtn = section.querySelector('[data-action="build-source"]');
  // Reinstalling overwrites the managed install directory, which would leave a running process half-updated, so stop it
  // first if it is up (confirmed, like Save).
  async function startInstall(button, buildOverride = null) {
    const status = lastServerStatus[server];
    const wasRunning = !!status?.running;
    if (wasRunning) {
      const engineLabel = { ollama: "Ollama", matricxon: "Matricxon", comfyui: "ComfyUI", sdcpp: "stable-diffusion.cpp" }[server] || server;
      const buildNote = buildOverride === "source" ? " Building from source can take 10-30 minutes, and it stays stopped meanwhile." : "";
      if (!confirm(`${engineLabel} is currently running — reinstalling stops it first.${buildNote} Continue?`)) return;
      button.disabled = true;
      try {
        renderServerStatus(section, await api(`/api/settings/${server}/stop`, { method: "POST" }));
      } catch (err) {
        alert(`Could not stop ${engineLabel} first: ${err.message}`);
        button.disabled = false;
        return;
      }
      button.disabled = false;
    }
    // Restart afterward only if this click stopped it: a first-time install leaves it stopped on purpose, matching
    // Ollama/ComfyUI's separate first Start click.
    await installServer(server, section, wasRunning, buildOverride);
  }
  installBtn?.addEventListener("click", () => startInstall(installBtn));
  sourceBtn?.addEventListener("click", () => startInstall(sourceBtn, "source"));
}

/*
 * Downloads/installs Ollama or ComfyUI from GitHub (app.services.ollama_installer/comfyui_installer), streaming
 * progress over SSE as pullModel reads a model pull (fetch + reader, not EventSource, since it is a POST). It can
 * take many minutes (Ollama's release is multi-GB; ComfyUI's pip install pulls torch), so every installer event
 * (step/total_steps/step_label, byte progress, raw git/pip lines) is shown live: a step label, a progress bar
 * (determinate once byte counts are known) and a scrolling log of raw status lines. The mode <select> is disabled
 * meanwhile so switching mid-install can't desync the UI from the server.
 * 
 * `restartAfter` (true only for reinstalling something that was running; the install button's handler stops it first)
 * restarts it after a successful install. Without it a reinstall left a stopped engine with no sign that a manual Start
 * was needed.
 */
// Spinner + elapsed timer beside an install's step label, so a long step without numbers (pip's install phase, a slow
// mirror) doesn't look frozen (reported: "3/3 at 100%" through a ~900 MB PyTorch download). The spinner sits
// in a small wrapper so the header row's label-left / percent-right layout is unchanged.
function startInstallActivity(stepLabelEl, percentEl) {
  let wrapper = stepLabelEl.parentElement;
  if (!wrapper.dataset.installActivity) {
    wrapper = document.createElement("span");
    wrapper.dataset.installActivity = "1";
    wrapper.className = "flex min-w-0 items-center gap-2";
    stepLabelEl.before(wrapper);
    wrapper.appendChild(stepLabelEl);
  }
  const spinner = document.createElement("span");
  spinner.className = "inline-flex shrink-0";
  spinner.innerHTML =
    '<svg class="h-3.5 w-3.5 animate-spin text-brand-500" viewBox="0 0 24 24" fill="none">' +
    '<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>' +
    '<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>';
  wrapper.prepend(spinner);

  const startedAt = Date.now();
  let progressText = "";
  const render = () => {
    const seconds = Math.floor((Date.now() - startedAt) / 1000);
    const elapsed = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
    percentEl.textContent = progressText ? `${progressText} · ${elapsed}` : elapsed;
  };
  const timer = setInterval(render, 1000);
  render();
  return {
    setProgress(text) {
      progressText = text;
      render();
    },
    stop() {
      clearInterval(timer);
      spinner.remove();
    },
  };
}

// A "working, no numbers yet" bar instead of a value-less <progress>, which Firefox draws as a full static bar ("100%,
// stuck" through pip's install). A short segment slides across a track the size of the progress bar, via the Web
// Animations API (no extra CSS).
function setInstallBarIndeterminate(barEl, indeterminate) {
  let track = barEl.nextElementSibling;
  if (!track?.dataset.installIndeterminate) {
    track = document.createElement("div");
    track.dataset.installIndeterminate = "1";
    track.className = "hidden relative h-1.5 w-full overflow-hidden rounded-full bg-slate-800";
    const segment = document.createElement("div");
    segment.className = "absolute inset-y-0 w-1/3 rounded-full bg-brand-500";
    track.appendChild(segment);
    barEl.after(track);
    segment.animate([{ left: "-33%" }, { left: "100%" }], {
      duration: 1400,
      iterations: Infinity,
      easing: "ease-in-out",
    });
  }
  track.classList.toggle("hidden", !indeterminate);
  barEl.classList.toggle("hidden", indeterminate);
}

async function installServer(server, section, restartAfter = false, buildOverride = null) {
  const installIdle = section.querySelector('[data-field="install-idle"]');
  const progressPanel = section.querySelector('[data-field="install-progress-panel"]');
  const stepLabelEl = section.querySelector('[data-field="install-step-label"]');
  const percentEl = section.querySelector('[data-field="install-percent"]');
  const barEl = section.querySelector('[data-field="install-bar"]');
  const logEl = section.querySelector('[data-field="install-log"]');
  const modeSelect = section.querySelector('[data-field="mode"]');

  function appendLog(text) {
    const line = document.createElement("div");
    line.textContent = text;
    logEl.appendChild(line);
    while (logEl.children.length > 300) logEl.removeChild(logEl.firstChild);
    logEl.scrollTop = logEl.scrollHeight;
  }

  installIdle.classList.add("hidden");
  progressPanel.classList.remove("hidden");
  logEl.innerHTML = "";
  barEl.removeAttribute("value");
  percentEl.textContent = "";
  stepLabelEl.textContent = "Starting…";
  modeSelect.disabled = true;
  const activity = startInstallActivity(stepLabelEl, percentEl);
  setInstallBarIndeterminate(barEl, true);

  try {
    const response = await fetch(`/api/settings/${server}/install${installQuery(server, section, buildOverride)}`, { method: "POST" });
    if (!response.ok || !response.body) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Install failed (${response.status})`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let doneEvent = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop();

      for (const frame of frames) {
        const line = frame.trim();
        if (!line.startsWith("data:")) continue;
        const payload = JSON.parse(line.slice("data:".length).trim());

        if (payload.error) throw new Error(payload.error);
        if (payload.done) {
          doneEvent = payload;
          continue;
        }
        if (payload.step && payload.total_steps) {
          stepLabelEl.textContent = `Step ${payload.step} of ${payload.total_steps}: ${payload.step_label || ""}`;
        }
        if (payload.total && payload.completed) {
          const pct = Math.round((payload.completed / payload.total) * 100);
          barEl.value = pct;
          // pip download progress is in bytes (app.services.pip_progress), shown as MB; git clone's completed/total are
          // object counts, where only the percentage matters.
          const mb = (bytes) => (bytes / 1e6).toFixed(0);
          const size = payload.unit === "bytes" ? ` · ${mb(payload.completed)} / ${mb(payload.total)} MB` : "";
          activity.setProgress(`${pct}%${size}`);
          setInstallBarIndeterminate(barEl, false);
        } else {
          barEl.removeAttribute("value");
          setInstallBarIndeterminate(barEl, true);
          activity.setProgress("");
        }
        if (payload.status) appendLog(payload.status);
      }
    }

    activity.stop();
    stepLabelEl.textContent = "Installed.";
    percentEl.textContent = "100%";
    setInstallBarIndeterminate(barEl, false);
    barEl.value = 100;
    // Ollama's managed binary is in ollama_process._FALLBACK_BIN_PATHS, so a first install with no override is
    // auto-detected. The gap this closes: an explicit binary_path override (_find_binary checks it first and
    // exclusively if valid) would make every Reinstall install a binary that is never used, since the stale override
    // keeps winning (seen: "looks ok" after reinstall because the OLD binary still started). Saving the installed path,
    // like ComfyUI/Matricxon below, makes a reinstall take effect.
    if (server === "ollama" && doneEvent) {
      section.querySelector('[data-field="binary_path"]').value = doneEvent.binary_path;
      await api("/api/settings/ollama/config", {
        method: "PUT",
        body: JSON.stringify(collectServerConfig("ollama", section)),
      });
    }
    // ComfyUI has no fixed path to auto-discover, unlike Ollama's managed binary (ollama_process._FALLBACK_BIN_PATHS);
    // the installed paths are filled in and saved so Start works at once. extra_args is "--cpu" on a machine without an
    // NVIDIA GPU (comfyui_installer), since the default CUDA torch build crashes at startup there.
    if (server === "comfyui" && doneEvent) {
      section.querySelector('[data-field="python_path"]').value = doneEvent.python_path;
      section.querySelector('[data-field="main_py_path"]').value = doneEvent.main_py_path;
      if (doneEvent.extra_args) section.querySelector('[data-field="extra_args"]').value = doneEvent.extra_args;
      await api("/api/settings/comfyui/config", {
        method: "PUT",
        body: JSON.stringify(collectServerConfig("comfyui", section)),
      });
    }
    // stable-diffusion.cpp: as ComfyUI above, save the installed sd-server path so Start works. The model file is the
    // admin's own (see the models folder in the log) and stays as entered.
    if (server === "sdcpp" && doneEvent) {
      section.querySelector('[data-field="binary_path"]').value = doneEvent.binary_path;
      appendLog(`Installed the ${doneEvent.build} build.`);
      appendLog("Next: download an image model under Settings > Model > Browse more models, then pick it below.");
      await api("/api/settings/sdcpp/config", {
        method: "PUT",
        body: JSON.stringify(collectServerConfig("sdcpp", section)),
      });
    }
    // Matricxon has no fixed auto-discovered path either; as ComfyUI above, with project_dir instead of
    // python_path/main_py_path (matricxon_process._find_project_dir treats this saved value as an override ahead of its
    // sibling-directory default).
    if (server === "matricxon" && doneEvent) {
      section.querySelector('[data-field="project_dir"]').value = doneEvent.project_dir;
      await api("/api/settings/matricxon/config", {
        method: "PUT",
        body: JSON.stringify(collectServerConfig("matricxon", section)),
      });
    }
    if (restartAfter) {
      stepLabelEl.textContent = "Starting…";
      try {
        await api(`/api/settings/${server}/start`, { method: "POST" });
      } catch (err) {
        // Surfaced via the status line loadServerSection renders next, not thrown: the install succeeded, and reporting
        // it as a failure would re-show the "Install from GitHub" idle panel via the catch block below.
        appendLog(`Installed, but could not start it back up: ${err.message}`);
      }
    }
    // A full reload of this section, not a status re-fetch: Ollama's "auto-detected" binary_path placeholder
    // (loadServerSection) reflects the new binary only once re-fetched, and a restartAfter start shows as "Running"
    // without a manual refresh.
    await loadServerSection(server);
    // Swap back to the idle Install/Reinstall button when there's nothing left to show; otherwise "Installed. 100%"
    // lingered, indistinguishable days later from an install in progress.
    progressPanel.classList.add("hidden");
    installIdle.classList.remove("hidden");
  } catch (err) {
    stepLabelEl.textContent = `Failed: ${err.message}`;
    appendLog(err.message);
    // progressPanel stays visible here (unlike success): it is the only place the failure reason and log appear, and
    // hiding it would bury what the admin needs to retry.
    installIdle.classList.remove("hidden");
  } finally {
    activity.stop();
    setInstallBarIndeterminate(barEl, false);
    modeSelect.disabled = false;
  }
}

// Default model for new users (admin only)

async function loadDefaultModelForNewUsersEditor() {
  const selectEl = document.getElementById("default-model-for-new-users");
  const emptyEl = document.getElementById("default-model-for-new-users-empty");
  const saveBtn = document.getElementById("default-model-for-new-users-save-btn");
  if (!selectEl) return;

  const spinner = addInlineSpinner(selectEl);
  try {
    await loadCatalog();
  } finally {
    spinner.remove();
  }
  const installed = currentCatalog.entries.filter((entry) => entry.installed);

  selectEl.innerHTML = "";
  const hasInstalled = installed.length > 0;
  emptyEl.classList.toggle("hidden", hasInstalled);
  selectEl.disabled = !hasInstalled;
  saveBtn.disabled = !hasInstalled;
  if (!hasInstalled) return;

  for (const entry of installed) {
    const option = document.createElement("option");
    option.value = entry.tag;
    option.textContent = entry.tag;
    selectEl.appendChild(option);
  }
  const current = await api("/api/settings/default-model-for-new-users");
  if (current.model) selectEl.value = current.model;
}

// Admin-only (settings.html): this element and the rest of the block don't exist for regular users, as other admin-only
// sections guarded like this. The unguarded call threw for every non-admin on page load and, running at top level,
// halted the whole script, breaking every other tab including My Settings.
const defaultModelForNewUsersSaveBtn = document.getElementById("default-model-for-new-users-save-btn");
if (defaultModelForNewUsersSaveBtn) {
  defaultModelForNewUsersSaveBtn.addEventListener("click", async () => {
    const selectEl = document.getElementById("default-model-for-new-users");
    const statusEl = document.getElementById("default-model-for-new-users-status");
    const btn = defaultModelForNewUsersSaveBtn;

    btn.disabled = true;
    statusEl.textContent = "Saving…";
    try {
      await api("/api/settings/default-model-for-new-users", {
        method: "PUT", body: JSON.stringify({ model: selectEl.value }),
      });
      statusEl.textContent = "Saved.";
    } catch (err) {
      statusEl.textContent = `Failed to save: ${err.message}`;
    } finally {
      btn.disabled = false;
      setTimeout(() => (statusEl.textContent = ""), 2500);
    }
  });
}

// Default vision model (admin only)
// The model that answers any message with an image attached, for that reply only (chat_service.build_reply_stream). A
// separate picker from "Default model for new users" since a vision model differs from an ordinary chat model
// (model_catalog_service.installed_vision_models).

async function loadDefaultVisionModelEditor() {
  const selectEl = document.getElementById("default-vision-model");
  const emptyEl = document.getElementById("default-vision-model-empty");
  const saveBtn = document.getElementById("default-vision-model-save-btn");
  if (!selectEl) return;

  const spinner = addInlineSpinner(selectEl);
  let models;
  try {
    ({ models } = await api("/api/settings/installed-vision-models"));
  } finally {
    spinner.remove();
  }

  selectEl.innerHTML = "";
  const hasInstalled = models.length > 0;
  emptyEl.classList.toggle("hidden", hasInstalled);
  selectEl.disabled = !hasInstalled;
  saveBtn.disabled = !hasInstalled;
  if (!hasInstalled) return;

  for (const tag of models) {
    const option = document.createElement("option");
    option.value = tag;
    option.textContent = tag;
    selectEl.appendChild(option);
  }
  const current = await api("/api/settings/default-vision-model");
  if (current.model) selectEl.value = current.model;
}

const defaultVisionModelSaveBtn = document.getElementById("default-vision-model-save-btn");
if (defaultVisionModelSaveBtn) {
  defaultVisionModelSaveBtn.addEventListener("click", async () => {
    const selectEl = document.getElementById("default-vision-model");
    const statusEl = document.getElementById("default-vision-model-status");
    const btn = defaultVisionModelSaveBtn;

    btn.disabled = true;
    statusEl.textContent = "Saving…";
    try {
      await api("/api/settings/default-vision-model", {
        method: "PUT", body: JSON.stringify({ model: selectEl.value }),
      });
      statusEl.textContent = "Saved.";
    } catch (err) {
      statusEl.textContent = `Failed to save: ${err.message}`;
    } finally {
      btn.disabled = false;
      setTimeout(() => (statusEl.textContent = ""), 2500);
    }
  });
}

// Default embedding model (admin only)
// The model app.services.inference_client.embed uses for the Knowledge base (RAG); same select/empty-state/save pattern
// as "Default vision model", against the embedding-capable list.

async function loadDefaultEmbeddingModelEditor() {
  const selectEl = document.getElementById("default-embedding-model");
  const emptyEl = document.getElementById("default-embedding-model-empty");
  const saveBtn = document.getElementById("default-embedding-model-save-btn");
  if (!selectEl) return;

  const spinner = addInlineSpinner(selectEl);
  let models;
  try {
    ({ models } = await api("/api/settings/installed-embedding-models"));
  } finally {
    spinner.remove();
  }

  selectEl.innerHTML = "";
  const hasInstalled = models.length > 0;
  emptyEl.classList.toggle("hidden", hasInstalled);
  selectEl.disabled = !hasInstalled;
  saveBtn.disabled = !hasInstalled;
  if (!hasInstalled) return;

  for (const tag of models) {
    const option = document.createElement("option");
    option.value = tag;
    option.textContent = tag;
    selectEl.appendChild(option);
  }
  const current = await api("/api/settings/default-embedding-model");
  if (current.model) selectEl.value = current.model;
}

const defaultEmbeddingModelSaveBtn = document.getElementById("default-embedding-model-save-btn");
if (defaultEmbeddingModelSaveBtn) {
  defaultEmbeddingModelSaveBtn.addEventListener("click", async () => {
    const selectEl = document.getElementById("default-embedding-model");
    const statusEl = document.getElementById("default-embedding-model-status");
    const btn = defaultEmbeddingModelSaveBtn;

    btn.disabled = true;
    statusEl.textContent = "Saving…";
    try {
      await api("/api/settings/default-embedding-model", {
        method: "PUT", body: JSON.stringify({ model: selectEl.value }),
      });
      statusEl.textContent = "Saved.";
    } catch (err) {
      statusEl.textContent = `Failed to save: ${err.message}`;
    } finally {
      btn.disabled = false;
      setTimeout(() => (statusEl.textContent = ""), 2500);
    }
  });
}

// Account (everyone)

// Profile: name + picture (app/routers/account.py). `has_avatar` lets "Remove" show/hide without re-deriving from
// avatar_url, like has_password elsewhere on this page.
async function loadAccountProfileIntoEditor() {
  const profile = await api("/api/account/profile");
  document.getElementById("account-first-name").value = profile.first_name || "";
  document.getElementById("account-last-name").value = profile.last_name || "";
  renderAvatar(document.getElementById("account-avatar-preview"), profile.avatar_url, profile.initials);
  document.getElementById("account-avatar-remove-btn").classList.toggle("hidden", !profile.avatar_url);
  fillTimezoneSelect(profile.timezone);
}

// Timezone: the browser's IANA list (UTC via the fallback) with the saved one selected.
function fillTimezoneSelect(selected) {
  const select = document.getElementById("account-timezone");
  const zones = new Set(typeof Intl.supportedValuesOf === "function" ? Intl.supportedValuesOf("timeZone") : []);
  zones.add("UTC");
  zones.add(selected);
  select.replaceChildren(
    ...[...zones].sort().map((zone) => Object.assign(document.createElement("option"), { value: zone, textContent: zone }))
  );
  select.value = selected;
}

document.getElementById("account-timezone-detect-btn")?.addEventListener("click", () => {
  const select = document.getElementById("account-timezone");
  const detected = Intl.DateTimeFormat().resolvedOptions().timeZone;
  if (![...select.options].some((o) => o.value === detected)) fillTimezoneSelect(detected);
  select.value = detected;
});

document.getElementById("account-timezone-save-btn")?.addEventListener("click", async () => {
  const statusEl = document.getElementById("account-timezone-status");
  const btn = document.getElementById("account-timezone-save-btn");
  btn.disabled = true;
  statusEl.textContent = "Saving…";
  try {
    const timezone = document.getElementById("account-timezone").value;
    await api("/api/account/profile", { method: "PATCH", body: JSON.stringify({ timezone }) });
    statusEl.textContent = "Saved.";
  } catch (err) {
    statusEl.textContent = err.message;
  } finally {
    btn.disabled = false;
    setTimeout(() => (statusEl.textContent = ""), 2500);
  }
});

document.getElementById("account-avatar-input")?.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;

  const statusEl = document.getElementById("account-avatar-status");
  statusEl.textContent = "Uploading…";
  statusEl.classList.remove("text-red-400");
  try {
    const formData = new FormData();
    formData.append("file", file);
    // A raw fetch without the JSON Content-Type (api's default), as chat.js's attachment upload: the browser must set
    // the multipart boundary.
    const response = await fetch("/api/account/avatar", { method: "PUT", body: formData });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || "Upload failed.");
    }
    statusEl.textContent = "Saved.";
    await loadAccountProfileIntoEditor();
  } catch (err) {
    statusEl.textContent = err.message;
    statusEl.classList.add("text-red-400");
  } finally {
    setTimeout(() => {
      statusEl.textContent = "";
      statusEl.classList.remove("text-red-400");
    }, 2500);
  }
});

document.getElementById("account-avatar-remove-btn")?.addEventListener("click", async () => {
  const statusEl = document.getElementById("account-avatar-status");
  statusEl.textContent = "Removing…";
  try {
    await api("/api/account/avatar", { method: "DELETE" });
    statusEl.textContent = "";
    await loadAccountProfileIntoEditor();
  } catch (err) {
    statusEl.textContent = err.message;
    statusEl.classList.add("text-red-400");
    setTimeout(() => {
      statusEl.textContent = "";
      statusEl.classList.remove("text-red-400");
    }, 2500);
  }
});

document.getElementById("account-profile-save-btn")?.addEventListener("click", async () => {
  const statusEl = document.getElementById("account-profile-status");
  const btn = document.getElementById("account-profile-save-btn");
  const first_name = document.getElementById("account-first-name").value.trim();
  const last_name = document.getElementById("account-last-name").value.trim();

  btn.disabled = true;
  statusEl.textContent = "Saving…";
  try {
    await api("/api/account/profile", { method: "PATCH", body: JSON.stringify({ first_name, last_name }) });
    statusEl.textContent = "Saved.";
  } catch (err) {
    statusEl.textContent = err.message;
  } finally {
    btn.disabled = false;
    setTimeout(() => (statusEl.textContent = ""), 2500);
  }
});

document.getElementById("account-password-toggle")?.addEventListener("click", () => {
  const form = document.getElementById("account-password-form");
  const toggle = document.getElementById("account-password-toggle");
  const opening = form.classList.contains("hidden");
  form.classList.toggle("hidden");
  toggle.textContent = opening ? "Cancel" : "Update password";
});

document.getElementById("account-password-save-btn")?.addEventListener("click", async () => {
  const currentInput = document.getElementById("account-current-password");
  const newInput = document.getElementById("account-new-password");
  const confirmInput = document.getElementById("account-confirm-password");
  const statusEl = document.getElementById("account-password-status");
  const btn = document.getElementById("account-password-save-btn");

  const current_password = currentInput.value;
  const new_password = newInput.value;
  const confirm = confirmInput.value;

  statusEl.classList.remove("text-warning-400");

  if (!current_password || !new_password) {
    statusEl.textContent = "Fill in both your current and new password.";
    statusEl.classList.add("text-warning-400");
    return;
  }
  // Mirrors ChangePasswordRequest's constraint (app/schemas.py), so a too-short password shows a plain message instead
  // of a raw FastAPI validation error.
  if (new_password.length < 3 || new_password.length > 200) {
    statusEl.textContent = "New password must be 3-200 characters.";
    statusEl.classList.add("text-warning-400");
    return;
  }
  if (new_password !== confirm) {
    statusEl.textContent = "New password and confirmation don't match.";
    statusEl.classList.add("text-warning-400");
    return;
  }

  btn.disabled = true;
  statusEl.textContent = "Saving…";
  try {
    await api("/api/settings/change-password", {
      method: "POST",
      body: JSON.stringify({ current_password, new_password }),
    });
    currentInput.value = "";
    newInput.value = "";
    confirmInput.value = "";
    statusEl.textContent = "Password changed.";
  } catch (err) {
    statusEl.textContent = err.message;
    statusEl.classList.add("text-warning-400");
  } finally {
    btn.disabled = false;
    setTimeout(() => {
      statusEl.textContent = "";
      statusEl.classList.remove("text-warning-400");
    }, 3000);
  }
});

// App-wide UI theme (ThemeManager in theme.js)
// A swatch click previews live (repaints the page via ThemeManager.apply, no network call or reload); only "Save
// theme" persists. The same picker and ThemeManager class are in the account panel on other pages (account_panel.js).
const themeManager = new ThemeManager();
let pendingUiTheme = themeManager.current;

function highlightSelectedSwatch(themeId) {
  document.querySelectorAll("[data-theme-swatch]").forEach((btn) => {
    btn.classList.toggle("ring-2", btn.dataset.themeId === themeId);
    btn.classList.toggle("ring-brand-500", btn.dataset.themeId === themeId);
  });
}

document.querySelectorAll("[data-theme-swatch]").forEach((btn) => {
  btn.addEventListener("click", () => {
    pendingUiTheme = btn.dataset.themeId;
    themeManager.apply(pendingUiTheme);
    highlightSelectedSwatch(pendingUiTheme);
  });
});

document.getElementById("ui-theme-save-btn")?.addEventListener("click", async () => {
  const statusEl = document.getElementById("ui-theme-status");
  const btn = document.getElementById("ui-theme-save-btn");

  btn.disabled = true;
  statusEl.textContent = "Saving…";
  try {
    await themeManager.save(pendingUiTheme);
    statusEl.textContent = "Saved";
  } catch (err) {
    statusEl.textContent = err.message;
    statusEl.classList.add("text-warning-400");
  } finally {
    btn.disabled = false;
    setTimeout(() => {
      statusEl.textContent = "";
      statusEl.classList.remove("text-warning-400");
    }, 3000);
  }
});

// Default persona/rules/skill notes on new chats/channels (every user)
// A plain on/off toggle, not a Save button: there is nothing to preview, so flipping it saves at once. It affects only
// chats/channels this user starts afterward (note_service.seed_initial_disabled_notes).

async function loadDefaultNotesEnabledIntoEditor() {
  const toggleEl = document.getElementById("default-notes-enabled-toggle");
  if (!toggleEl) return;
  const { enabled } = await api("/api/notes/default-notes-enabled");
  toggleEl.checked = enabled;
}

document.getElementById("default-notes-enabled-toggle")?.addEventListener("change", async (e) => {
  const statusEl = document.getElementById("default-notes-enabled-status");
  const enabled = e.target.checked;
  e.target.disabled = true;
  statusEl.textContent = "Saving…";
  try {
    await api("/api/notes/default-notes-enabled", { method: "PUT", body: JSON.stringify({ enabled }) });
    statusEl.textContent = "Saved";
  } catch (err) {
    e.target.checked = !enabled;
    statusEl.textContent = err.message;
    statusEl.classList.add("text-warning-400");
  } finally {
    e.target.disabled = false;
    setTimeout(() => {
      statusEl.textContent = "";
      statusEl.classList.remove("text-warning-400");
    }, 2500);
  }
});

// RAG upload limits (admin only; these elements don't exist for non-admins, like the rest of the System tab)

const ragLimitsSaveBtn = document.getElementById("rag-limits-save-btn");
const ragEnabledToggle = document.getElementById("rag-enabled");

async function loadRagLimitsIntoEditor() {
  if (!ragLimitsSaveBtn) return;
  const limits = await api("/api/settings/rag-limits");
  ragEnabledToggle.checked = limits.enabled;
  document.getElementById("rag-limit-max-file").value = limits.max_file_mb;
  document.getElementById("rag-limit-max-space").value = limits.max_user_space_mb;
}

// PUT /api/settings/rag-limits takes the full RagLimits object (unlike default-notes-enabled's single-field endpoint),
// so the toggle's auto-save and the numeric limits' Save button both use this function, each sending all three fields.
async function saveRagLimits(statusEl) {
  const enabled = ragEnabledToggle.checked;
  const max_file_mb = parseFloat(document.getElementById("rag-limit-max-file").value);
  const max_user_space_mb = parseFloat(document.getElementById("rag-limit-max-space").value);
  if (!(max_file_mb > 0) || !(max_user_space_mb > 0)) {
    statusEl.textContent = "Both limits must be positive numbers.";
    return false;
  }
  await api("/api/settings/rag-limits", {
    method: "PUT",
    body: JSON.stringify({ enabled, max_file_mb, max_user_space_mb }),
  });
  currentUploadLimits = null;
  // Reflects the new on/off state at once if the Knowledge tab is already open in this page load, like other cross-tab
  // refreshes here.
  await loadRagAvailability();
  return true;
}

if (ragEnabledToggle) {
  // A plain on/off toggle, not tied to the Save button below (as default-notes-enabled-toggle): flipping it saves at
  // once instead of silently waiting for the numeric limits to be saved.
  ragEnabledToggle.addEventListener("change", async () => {
    const statusEl = document.getElementById("rag-limits-status");
    const wasChecked = !ragEnabledToggle.checked;
    ragEnabledToggle.disabled = true;
    statusEl.textContent = "Saving…";
    try {
      const ok = await saveRagLimits(statusEl);
      statusEl.textContent = ok ? "Saved." : statusEl.textContent;
      if (!ok) ragEnabledToggle.checked = wasChecked;
    } catch (err) {
      ragEnabledToggle.checked = wasChecked;
      statusEl.textContent = `Failed to save: ${err.message}`;
    } finally {
      ragEnabledToggle.disabled = false;
      setTimeout(() => (statusEl.textContent = ""), 2500);
    }
  });
}

if (ragLimitsSaveBtn) {
  ragLimitsSaveBtn.addEventListener("click", async () => {
    const statusEl = document.getElementById("rag-limits-status");
    ragLimitsSaveBtn.disabled = true;
    statusEl.textContent = "Saving…";
    try {
      const ok = await saveRagLimits(statusEl);
      if (ok) statusEl.textContent = "Saved.";
    } catch (err) {
      statusEl.textContent = `Failed to save: ${err.message}`;
    } finally {
      ragLimitsSaveBtn.disabled = false;
      setTimeout(() => (statusEl.textContent = ""), 2500);
    }
  });
}

// Chat title mode (admin only)

const titleModeSaveBtn = document.getElementById("title-mode-save-btn");

async function loadTitleModeIntoEditor() {
  if (!titleModeSaveBtn) return;
  const { mode } = await api("/api/settings/title-mode");
  document.getElementById("title-mode").value = mode;
}

if (titleModeSaveBtn) {
  titleModeSaveBtn.addEventListener("click", async () => {
    const statusEl = document.getElementById("title-mode-status");
    const mode = document.getElementById("title-mode").value;

    titleModeSaveBtn.disabled = true;
    statusEl.textContent = "Saving…";
    try {
      await api("/api/settings/title-mode", { method: "PUT", body: JSON.stringify({ mode }) });
      statusEl.textContent = "Saved.";
    } catch (err) {
      statusEl.textContent = `Failed to save: ${err.message}`;
    } finally {
      titleModeSaveBtn.disabled = false;
      setTimeout(() => (statusEl.textContent = ""), 2500);
    }
  });
}

// Channel reply delivery mode (read: any user; save: admin only)
// See chat_settings_service.get_channel_delivery_mode/set_channel_delivery_mode and chat.js, which reads the same GET
// endpoint to decide how to behave in a channel chat.

const channelDeliveryModeSaveBtn = document.getElementById("channel-delivery-mode-save-btn");

async function loadChannelDeliveryModeIntoEditor() {
  if (!channelDeliveryModeSaveBtn) return;
  const { mode } = await api("/api/settings/channel-delivery-mode");
  document.getElementById("channel-delivery-mode").value = mode;
  const { include } = await api("/api/settings/channel-plain-messages");
  document.getElementById("channel-plain-messages").value = include ? "include" : "skip";
}

if (channelDeliveryModeSaveBtn) {
  channelDeliveryModeSaveBtn.addEventListener("click", async () => {
    const statusEl = document.getElementById("channel-delivery-mode-status");
    const mode = document.getElementById("channel-delivery-mode").value;

    channelDeliveryModeSaveBtn.disabled = true;
    statusEl.textContent = "Saving…";
    try {
      await api("/api/settings/channel-delivery-mode", { method: "PUT", body: JSON.stringify({ mode }) });
      const include = document.getElementById("channel-plain-messages").value === "include";
      await api("/api/settings/channel-plain-messages", { method: "PUT", body: JSON.stringify({ include }) });
      statusEl.textContent = "Saved.";
    } catch (err) {
      statusEl.textContent = `Failed to save: ${err.message}`;
    } finally {
      channelDeliveryModeSaveBtn.disabled = false;
      setTimeout(() => (statusEl.textContent = ""), 2500);
    }
  });
}

// Reply timeouts (admin only)
// See chat_settings_service.get_reply_timeout_seconds/get_vision_reply_timeout_seconds and reply_generation_service,
// the only reader; no frontend code branches on them. Two independent settings with the same {timeout_seconds} shape
// and save flow, so one helper wires both.

function wireTimeoutSetting(inputId, saveBtnId, statusId, apiPath) {
  const saveBtn = document.getElementById(saveBtnId);
  if (!saveBtn) return null;

  saveBtn.addEventListener("click", async () => {
    const statusEl = document.getElementById(statusId);
    const timeout_seconds = Number(document.getElementById(inputId).value);

    saveBtn.disabled = true;
    statusEl.textContent = "Saving…";
    try {
      await api(apiPath, { method: "PUT", body: JSON.stringify({ timeout_seconds }) });
      statusEl.textContent = "Saved.";
    } catch (err) {
      statusEl.textContent = `Failed to save: ${err.message}`;
    } finally {
      saveBtn.disabled = false;
      setTimeout(() => (statusEl.textContent = ""), 2500);
    }
  });

  return async function loadIntoEditor() {
    const { timeout_seconds } = await api(apiPath);
    document.getElementById(inputId).value = timeout_seconds;
  };
}

const loadReplyTimeoutIntoEditor =
  wireTimeoutSetting("reply-timeout", "reply-timeout-save-btn", "reply-timeout-status", "/api/settings/reply-timeout") ||
  (async () => {});
const loadVisionReplyTimeoutIntoEditor =
  wireTimeoutSetting(
    "vision-reply-timeout",
    "vision-reply-timeout-save-btn",
    "vision-reply-timeout-status",
    "/api/settings/vision-reply-timeout"
  ) || (async () => {});

// Data retention (admin only)
// See retention_settings_service and retention_poller, the only reader; no frontend code outside this section branches
// on it.

const retentionSaveBtn = document.getElementById("retention-save-btn");

async function loadRetentionSettingsIntoEditor() {
  if (!retentionSaveBtn) return;
  const { telemetry_days, system_metrics_days } = await api("/api/settings/retention");
  document.getElementById("retention-telemetry-days").value = telemetry_days;
  document.getElementById("retention-system-metrics-days").value = system_metrics_days;
}

if (retentionSaveBtn) {
  retentionSaveBtn.addEventListener("click", async () => {
    const statusEl = document.getElementById("retention-status");
    const telemetry_days = Number(document.getElementById("retention-telemetry-days").value);
    const system_metrics_days = Number(document.getElementById("retention-system-metrics-days").value);

    retentionSaveBtn.disabled = true;
    statusEl.textContent = "Saving…";
    try {
      await api("/api/settings/retention", {
        method: "PUT",
        body: JSON.stringify({ telemetry_days, system_metrics_days }),
      });
      statusEl.textContent = "Saved.";
    } catch (err) {
      statusEl.textContent = `Failed to save: ${err.message}`;
    } finally {
      retentionSaveBtn.disabled = false;
      setTimeout(() => (statusEl.textContent = ""), 2500);
    }
  });
}

// Users (admin only)

// The logged-in user (data attribute on <body>, see base.html), so their own row can be marked "(you)".
const currentUsername = document.body.dataset.username || "";

// Mirrors UserCreate/UserUpdate constraints (app/schemas.py), so a bad username/password shows a plain message instead
// of FastAPI's raw 422 list of field errors.
const USERNAME_RE = /^[A-Za-z0-9_.-]{3,50}$/;
function validUsername(value) {
  return USERNAME_RE.test(value);
}
function validPassword(value) {
  return value.length >= 3 && value.length <= 200;
}

/* While one row of a list is edited (or an "add" form is open), shows only that row/form: hides the other rows and any `chrome` elements (Add button, pagination). `row` null restores everything. */
function focusListRow(listEl, row, ...chrome) {
  if (!listEl) return;
  for (const child of listEl.children) child.style.display = row && child !== row ? "none" : "";
  for (const el of chrome) if (el) el.style.display = row ? "none" : "";
}

const userListEl = document.getElementById("system-user-list");
const addUserBtn = document.getElementById("add-user-btn");
const addUserForm = document.getElementById("add-user-form");
const addUserStatusEl = document.getElementById("add-user-status");
const createUserBtn = document.getElementById("create-user-btn");

/* Renders one row of the Users list: a compact "view" state (username, role, status, Edit button) that swaps into an inline form (password/role/status) on Edit, as the app has no modals. */
function renderUserRow(user) {
  const row = document.createElement("div");
  row.className = "px-4 py-2.5";

  const userChrome = () => [addUserBtn, document.getElementById("system-user-pagination")];

  function renderView() {
    row.innerHTML = "";
    if (row.parentElement) focusListRow(row.parentElement, null, ...userChrome());
    const wrap = document.createElement("div");
    wrap.className = "flex items-center justify-between gap-3";
    const fullName = [user.first_name, user.last_name].filter(Boolean).join(" ");
    wrap.innerHTML = `
      <span class="min-w-0 truncate text-slate-200">
        ${escapeHtml(user.username)}
        ${fullName ? `<span class="text-slate-500">— ${escapeHtml(fullName)}</span>` : ""}
        ${user.username === currentUsername ? '<span class="text-slate-500">(you)</span>' : ""}
      </span>
      <span class="flex shrink-0 items-center gap-2">
        <span class="text-xs text-slate-500 uppercase tracking-wide">${escapeHtml(user.role)}</span>
        <span class="text-xs uppercase tracking-wide ${user.status === "active" ? "text-emerald-400" : "text-slate-500"}">${escapeHtml(user.status)}</span>
        <button type="button" class="edit-user-btn rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800 transition-colors">Edit</button>
        ${user.username === "admin" ? "" : '<button type="button" class="delete-user-btn rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-400 hover:border-red-500 hover:text-red-400 transition-colors">Delete</button>'}
      </span>
    `;
    wrap.querySelector(".edit-user-btn").addEventListener("click", renderEdit);

    // In the main list, no need to open Edit. Permanent and cascading (DELETE /api/settings/users/{username}): removes
    // the account's conversations, documents and notes, with no undo; disabling (in Edit) is the reversible way to
    // revoke access. Not rendered for "admin" (see the ternary above), which can't be deleted
    // (app.routers.settings.delete_user enforces it), so a button could only fail.
    const deleteBtn = wrap.querySelector(".delete-user-btn");
    if (deleteBtn) {
      deleteBtn.addEventListener("click", async () => {
        if (!confirm(
          `Permanently delete "${user.username}"? This also deletes all of their conversations, ` +
          "documents, and notes. This cannot be undone.",
        )) return;

        deleteBtn.disabled = true;
        try {
          await api(`/api/settings/users/${encodeURIComponent(user.username)}`, { method: "DELETE" });
          allUsers = allUsers.filter((u) => u.username !== user.username);
          renderUserPage();
        } catch (err) {
          alert(`Failed to delete: ${err.message}`);
          deleteBtn.disabled = false;
        }
      });
    }

    row.appendChild(wrap);
  }

  function renderEdit() {
    row.innerHTML = "";
    focusListRow(row.parentElement, row, ...userChrome());
    const form = document.createElement("div");
    form.className = "space-y-2";
    form.innerHTML = `
      <p class="text-slate-200">${escapeHtml(user.username)}</p>
      <div class="grid grid-cols-2 gap-2">
        <input type="password" placeholder="New password (leave blank to keep)" autocomplete="new-password"
          class="edit-password col-span-2 rounded-md border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs focus:outline-none focus:border-brand-500">
        <input type="text" placeholder="First name" autocomplete="off" value="${escapeHtml(user.first_name || "")}"
          class="edit-first-name rounded-md border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs focus:outline-none focus:border-brand-500">
        <input type="text" placeholder="Last name" autocomplete="off" value="${escapeHtml(user.last_name || "")}"
          class="edit-last-name rounded-md border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs focus:outline-none focus:border-brand-500">
        <select class="edit-role rounded-md border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs focus:outline-none focus:border-brand-500">
          <option value="user" ${user.role === "user" ? "selected" : ""}>User</option>
          <option value="channel_manager" ${user.role === "channel_manager" ? "selected" : ""}>Channel manager</option>
          <option value="admin" ${user.role === "admin" ? "selected" : ""}>Admin</option>
        </select>
        <select class="edit-status rounded-md border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs focus:outline-none focus:border-brand-500">
          <option value="active" ${user.status === "active" ? "selected" : ""}>Active</option>
          <option value="disabled" ${user.status === "disabled" ? "selected" : ""}>Disabled</option>
        </select>
      </div>
      <div class="flex items-center gap-2">
        <button type="button" class="save-user-btn rounded-md bg-brand-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-brand-500 transition-colors">Save</button>
        <button type="button" class="cancel-user-btn rounded-md border border-slate-700 px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-800 transition-colors">Cancel</button>
        <span class="save-user-status text-xs text-slate-500"></span>
      </div>
    `;
    row.appendChild(form);

    form.querySelector(".cancel-user-btn").addEventListener("click", renderView);
    form.querySelector(".save-user-btn").addEventListener("click", async () => {
      const password = form.querySelector(".edit-password").value;
      const first_name = form.querySelector(".edit-first-name").value.trim();
      const last_name = form.querySelector(".edit-last-name").value.trim();
      const role = form.querySelector(".edit-role").value;
      const status = form.querySelector(".edit-status").value;
      const msgEl = form.querySelector(".save-user-status");
      if (password && !validPassword(password)) {
        msgEl.textContent = "Password must be 3-200 characters.";
        return;
      }
      // first_name/last_name are always sent (even as "") so clearing a name clears it; user_service.update_user's "or
      // None" handling needs "" to tell "cleared" from "untouched".
      const body = { first_name, last_name, role, status };
      if (password) body.password = password;

      msgEl.textContent = "Saving…";
      try {
        const updated = await api(`/api/settings/users/${encodeURIComponent(user.username)}`, {
          method: "PATCH",
          body: JSON.stringify(body),
        });
        Object.assign(user, updated);
        renderView();
      } catch (err) {
        msgEl.textContent = err.message;
      }
    });
  }

  renderView();
  return row;
}

// Users shown per page. The full list is fetched in one request (no server paging at this scale, as MY_DOCS_PAGE_SIZE),
// but not all rendered at once.
const USER_PAGE_SIZE = 20;
let allUsers = [];
let userPage = 1;

function renderUserPage() {
  const paginationEl = document.getElementById("system-user-pagination");
  const totalPages = Math.max(1, Math.ceil(allUsers.length / USER_PAGE_SIZE));
  // Clamped, as a delete can shrink the list between renders and leave the page past the last.
  userPage = Math.min(Math.max(1, userPage), totalPages);

  const start = (userPage - 1) * USER_PAGE_SIZE;
  const pageItems = allUsers.slice(start, start + USER_PAGE_SIZE);

  userListEl.innerHTML = "";
  for (const user of pageItems) {
    userListEl.appendChild(renderUserRow(user));
  }

  paginationEl.classList.toggle("hidden", allUsers.length <= USER_PAGE_SIZE);
  document.getElementById("system-user-page-info").textContent = `Page ${userPage} of ${totalPages}`;
  document.getElementById("system-user-prev-btn").disabled = userPage <= 1;
  document.getElementById("system-user-next-btn").disabled = userPage >= totalPages;
}

// Guarded like other admin-only elements (userListEl is null for non-admins, whose page has no Users tab). These two
// were missed when pagination was added, so a non-admin's page crashed with "Cannot read properties of null", silently
// killing the rest of the script, including the Add User form wiring below.
if (userListEl) {
  document.getElementById("system-user-prev-btn").addEventListener("click", () => {
    userPage -= 1;
    renderUserPage();
  });
  document.getElementById("system-user-next-btn").addEventListener("click", () => {
    userPage += 1;
    renderUserPage();
  });
}

async function loadUsers() {
  allUsers = await api("/api/settings/users");
  renderUserPage();
}

/* Disables "+ Add user" (with a note) when no model is installed, matching the server-side guard in POST /api/settings/users (400 otherwise); surfaces the reason before the form opens. */
async function refreshAddUserGate() {
  if (!addUserBtn) return;
  await loadCatalog();
  const hasInstalled = currentCatalog.entries.some((entry) => entry.installed);
  addUserBtn.disabled = !hasInstalled;
  document.getElementById("add-user-no-models-hint").classList.toggle("hidden", hasInstalled);
}

function resetAddUserForm() {
  document.getElementById("new-user-username").value = "";
  document.getElementById("new-user-password").value = "";
  document.getElementById("new-user-first-name").value = "";
  document.getElementById("new-user-last-name").value = "";
  document.getElementById("new-user-role").value = "user";
  document.getElementById("new-user-status").value = "active";
  addUserStatusEl.textContent = "";
}

if (addUserBtn) {
  const showAddUser = (open) => {
    addUserForm.classList.toggle("hidden", !open);
    focusListRow(userListEl, open ? addUserForm : null, addUserBtn, document.getElementById("system-user-pagination"));
  };
  addUserBtn.addEventListener("click", () => {
    showAddUser(addUserForm.classList.contains("hidden"));
    if (!addUserForm.classList.contains("hidden")) {
      resetAddUserForm();
      document.getElementById("new-user-username").focus();
    }
  });

  document.getElementById("cancel-add-user-btn").addEventListener("click", () => showAddUser(false));

  createUserBtn.addEventListener("click", async () => {
    const username = document.getElementById("new-user-username").value.trim();
    const password = document.getElementById("new-user-password").value;
    const first_name = document.getElementById("new-user-first-name").value.trim();
    const last_name = document.getElementById("new-user-last-name").value.trim();
    const role = document.getElementById("new-user-role").value;
    const status = document.getElementById("new-user-status").value;

    if (!validUsername(username)) {
      addUserStatusEl.textContent = "Username must be 3-50 characters: letters, numbers, _ . -";
      return;
    }
    if (!validPassword(password)) {
      addUserStatusEl.textContent = "Password must be 3-200 characters.";
      return;
    }

    createUserBtn.disabled = true;
    addUserStatusEl.textContent = "Creating…";
    try {
      await api("/api/settings/users", {
        method: "POST",
        body: JSON.stringify({ username, password, first_name, last_name, role, status }),
      });
      showAddUser(false);
      // list_users orders by created_at, so a new user is on the last page: jump there (renderUserPage's clamp settles
      // it) rather than re-fetching behind the shown page, which could look like nothing happened.
      userPage = Number.MAX_SAFE_INTEGER;
      await loadUsers();
    } catch (err) {
      addUserStatusEl.textContent = err.message;
    } finally {
      createUserBtn.disabled = false;
    }
  });
}

// Channels (admin only; these elements don't exist for non-admins, like System/Users)
//
// The member/manager picker makes channel creation usable with 1000+ users. A fixed-height <select multiple> becomes an
// unscrollable wall, and server-side paging goes against this app's convention (USER_PAGE_SIZE/MY_DOCS_PAGE_SIZE: fetch
// the full list once, do the rest client-side). So it is a search box over the fetched user list plus a capped checkbox
// list that always shows the current selection (buildChannelForm).

// How many unselected matching rows to paint at once: rendering every row at 1000+ users is what breaks. Selected
// members always show in full, so narrowing the search never hides a pick.
const MEMBER_PICKER_RENDER_LIMIT = 50;

function renderUserCheckboxRow(user, checked, onToggle) {
  const row = document.createElement("label");
  row.className = "flex items-center gap-2 px-2 py-1.5 text-xs text-slate-200 hover:bg-slate-950/60 cursor-pointer";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = checked;
  checkbox.className = "accent-brand-600";
  checkbox.addEventListener("change", onToggle);
  const label = document.createElement("span");
  label.className = "min-w-0 truncate";
  label.textContent = user.username + (user.role !== "user" ? ` (${user.role})` : "");
  row.append(checkbox, label);
  return row;
}

/* Builds the name + member/manager picker form shared by "+ Add channel" and each row's inline Edit (as the Users tab's renderView/renderEdit swap); `channel` is null when creating. `onSave(body)` gets {name, member_user_ids, manager_user_ids} shaped for POST/PATCH /api/settings/channels; `onCancel` closes the form unsaved. */
function buildChannelForm(users, channel, { onSave, onCancel }) {
  const selectedMemberIds = new Set((channel?.members || []).map((m) => m.user_id));
  const selectedManagerIds = new Set((channel?.members || []).filter((m) => m.is_manager).map((m) => m.user_id));

  // No border/bg of its own: both callers already sit in one. #add-channel-form carries it in settings.html (add flow),
  // and edit-in-place appends this into a channel row inside #channel-list's bordered box, where a second border would
  // nest.
  const form = document.createElement("div");
  form.className = "space-y-3";
  form.innerHTML = `
    <div>
      <label class="block text-xs font-medium text-slate-400 mb-1">Channel name</label>
      <input type="text" class="channel-name w-full rounded-md border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm focus:outline-none focus:border-brand-500" value="${channel ? escapeHtml(channel.name) : ""}">
    </div>
    <div>
      <label class="block text-xs font-medium text-slate-400 mb-1">Members</label>
      <input type="text" placeholder="Search users…" class="member-search w-full rounded-md border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs mb-1.5 focus:outline-none focus:border-brand-500">
      <div class="member-list max-h-48 overflow-y-auto rounded-md border border-slate-700 bg-slate-950 divide-y divide-slate-800"></div>
      <p class="member-count mt-1 text-xs text-slate-500"></p>
    </div>
    <div>
      <label class="block text-xs font-medium text-slate-400 mb-1">Managers</label>
      <p class="text-xs text-slate-500 mb-1.5">Every user holding the "Channel manager" role (see the Users tab) — checking one here also adds them as a member.</p>
      <div class="manager-list max-h-32 overflow-y-auto rounded-md border border-slate-700 bg-slate-950 divide-y divide-slate-800"></div>
      <p class="manager-empty-hint hidden px-2 py-3 text-xs text-slate-500">No users hold the Channel manager role yet — set that on the Users tab first.</p>
    </div>
    <div class="flex items-center gap-2 pt-1">
      <button type="button" class="channel-save-btn rounded-md bg-brand-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-brand-500 transition-colors">Save</button>
      <button type="button" class="channel-cancel-btn rounded-md border border-slate-700 px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-800 transition-colors">Cancel</button>
      <span class="channel-form-status text-xs text-slate-500"></span>
    </div>
  `;

  const memberListEl = form.querySelector(".member-list");
  const memberCountEl = form.querySelector(".member-count");
  const managerListEl = form.querySelector(".manager-list");
  const managerHintEl = form.querySelector(".manager-empty-hint");
  const searchInput = form.querySelector(".member-search");

  function renderMembers() {
    const query = searchInput.value.trim().toLowerCase();
    const matches = users.filter((u) => u.username.toLowerCase().includes(query));
    // Selected members first, in full, so narrowing the search never hides a pick; the other matches are capped
    // (MEMBER_PICKER_RENDER_LIMIT).
    const selectedMatches = matches.filter((u) => selectedMemberIds.has(u.id));
    const restMatches = matches.filter((u) => !selectedMemberIds.has(u.id));
    const shown = [...selectedMatches, ...restMatches.slice(0, MEMBER_PICKER_RENDER_LIMIT)];

    memberListEl.innerHTML = "";
    if (shown.length === 0) {
      memberListEl.innerHTML = `<p class="px-2 py-3 text-xs text-slate-500">No users match.</p>`;
    }
    for (const u of shown) {
      memberListEl.appendChild(renderUserCheckboxRow(u, selectedMemberIds.has(u.id), () => {
        if (selectedMemberIds.has(u.id)) {
          selectedMemberIds.delete(u.id);
          selectedManagerIds.delete(u.id);
        } else {
          selectedMemberIds.add(u.id);
        }
        renderMembers();
        renderManagers();
      }));
    }
    const omitted = restMatches.length - (shown.length - selectedMatches.length);
    memberCountEl.textContent = `${selectedMemberIds.size} selected` +
      (omitted > 0 ? ` — ${omitted} more match "${searchInput.value}", refine your search to see them` : "");
  }

  function renderManagers() {
    // Every channel_manager-role user is listed regardless of membership: checking one binds them to the channel
    // (member_user_ids must be a superset of manager_user_ids, here and server-side,
    // channel_service._validate_membership), with no trip to the Members list first. Unchecking revokes manager rights
    // only and keeps membership.
    const eligible = users.filter((u) => u.role === "channel_manager");
    managerListEl.innerHTML = "";
    managerHintEl.classList.toggle("hidden", eligible.length > 0);
    for (const u of eligible) {
      managerListEl.appendChild(renderUserCheckboxRow(u, selectedManagerIds.has(u.id), () => {
        if (selectedManagerIds.has(u.id)) {
          selectedManagerIds.delete(u.id);
        } else {
          selectedManagerIds.add(u.id);
          selectedMemberIds.add(u.id);
        }
        renderMembers();
        renderManagers();
      }));
    }
  }

  searchInput.addEventListener("input", renderMembers);
  renderMembers();
  renderManagers();

  form.querySelector(".channel-cancel-btn").addEventListener("click", onCancel);
  form.querySelector(".channel-save-btn").addEventListener("click", async () => {
    const saveBtn = form.querySelector(".channel-save-btn");
    const statusEl = form.querySelector(".channel-form-status");
    const name = form.querySelector(".channel-name").value.trim();
    if (!name) {
      statusEl.textContent = "Channel name is required.";
      return;
    }
    if (selectedMemberIds.size === 0) {
      statusEl.textContent = "A channel needs at least one member.";
      return;
    }
    if (selectedManagerIds.size === 0) {
      statusEl.textContent = "A channel needs at least one manager — check one in the Managers list above.";
      return;
    }

    saveBtn.disabled = true;
    statusEl.textContent = "Saving…";
    try {
      await onSave({
        name, member_user_ids: [...selectedMemberIds], manager_user_ids: [...selectedManagerIds],
      });
    } catch (err) {
      statusEl.textContent = err.message;
      saveBtn.disabled = false;
    }
  });

  return form;
}

const channelListEl = document.getElementById("channel-list");
const addChannelBtn = document.getElementById("add-channel-btn");
const addChannelFormEl = document.getElementById("add-channel-form");
let allChannels = [];

/* One channel's row: a compact view (name, member/manager counts, Edit/Delete) that swaps to buildChannelForm's inline editor on Edit, as renderUserRow. It fetches a fresh user list (not the Users tab's cached `allUsers`) so a role change just made there (e.g. granting Channel manager) shows at once. */
function renderChannelRow(channel) {
  const row = document.createElement("div");
  row.className = "px-4 py-2.5";

  function renderView() {
    row.innerHTML = "";
    if (row.parentElement) focusListRow(row.parentElement, null, addChannelBtn);
    const managerCount = channel.members.filter((m) => m.is_manager).length;
    const wrap = document.createElement("div");
    wrap.className = "flex items-center justify-between gap-3";
    wrap.innerHTML = `
      <span class="min-w-0 truncate text-slate-200">#${escapeHtml(channel.name)}</span>
      <span class="flex shrink-0 items-center gap-2">
        <span class="text-xs text-slate-500">
          ${channel.members.length} member${channel.members.length === 1 ? "" : "s"},
          ${managerCount} manager${managerCount === 1 ? "" : "s"}
        </span>
        <button type="button" class="edit-channel-btn rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800 transition-colors">Edit</button>
        <button type="button" class="reset-channel-btn rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-400 hover:border-red-500 hover:text-red-400 transition-colors" title="Clear this channel's chat history (members and notes stay)">Reset</button>
        <button type="button" class="delete-channel-btn rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-400 hover:border-red-500 hover:text-red-400 transition-colors">Delete</button>
      </span>
    `;
    wrap.querySelector(".edit-channel-btn").addEventListener("click", async () => {
      const users = await api("/api/settings/users");
      row.innerHTML = "";
      focusListRow(row.parentElement, row, addChannelBtn);
      row.appendChild(buildChannelForm(users, channel, {
        onCancel: renderView,
        onSave: async (body) => {
          const updated = await api(`/api/settings/channels/${channel.id}`, {
            method: "PATCH", body: JSON.stringify(body),
          });
          Object.assign(channel, updated);
          renderView();
        },
      }));
    });
    wrap.querySelector(".reset-channel-btn").addEventListener("click", async () => {
      if (!confirm(`Reset "#${channel.name}"? This permanently deletes its whole chat history for every member. Members and notes are kept. This cannot be undone.`)) return;
      try {
        await api(`/api/settings/channels/${channel.id}/reset`, { method: "POST" });
        alert(`"#${channel.name}" was reset.`);
      } catch (err) {
        alert(`Failed to reset: ${err.message}`);
      }
    });
    wrap.querySelector(".delete-channel-btn").addEventListener("click", async () => {
      if (!confirm(`Permanently delete "#${channel.name}"? This deletes its chat history for every member. This cannot be undone.`)) return;
      try {
        await api(`/api/settings/channels/${channel.id}`, { method: "DELETE" });
        allChannels = allChannels.filter((c) => c.id !== channel.id);
        renderChannelList();
      } catch (err) {
        alert(`Failed to delete: ${err.message}`);
      }
    });
    row.appendChild(wrap);
  }

  renderView();
  return row;
}

function renderChannelList() {
  channelListEl.innerHTML = "";
  if (allChannels.length === 0) {
    channelListEl.innerHTML = `<p class="px-4 py-3 text-xs text-slate-500">No channels yet.</p>`;
    return;
  }
  for (const channel of allChannels) {
    channelListEl.appendChild(renderChannelRow(channel));
  }
}

async function loadChannels() {
  if (!channelListEl) return;
  allChannels = await api("/api/settings/channels");
  renderChannelList();
}

if (addChannelBtn) {
  addChannelBtn.addEventListener("click", async () => {
    // Acts as a toggle: clicking while the form is open closes it unsaved (the form's own Cancel also does).
    const closeAdd = () => {
      addChannelFormEl.classList.add("hidden");
      addChannelFormEl.innerHTML = "";
      focusListRow(channelListEl, null, addChannelBtn);
    };
    if (!addChannelFormEl.classList.contains("hidden")) {
      closeAdd();
      return;
    }
    const users = await api("/api/settings/users");
    addChannelFormEl.innerHTML = "";
    addChannelFormEl.appendChild(buildChannelForm(users, null, {
      onCancel: closeAdd,
      onSave: async (body) => {
        await api("/api/settings/channels", { method: "POST", body: JSON.stringify(body) });
        closeAdd();
        await loadChannels();
      },
    }));
    addChannelFormEl.classList.remove("hidden");
    focusListRow(channelListEl, addChannelFormEl, addChannelBtn);
  });
}

// Boot

(async function init() {
  // Each step runs independently: a failure in one (e.g. Ollama unreachable breaking populateConversationSelect's
  // model-catalog fetch) must not stop the unrelated steps after it, as a sequential await chain would.
  const steps = hasUserTabs
    ? [initModelTab, populateConversationSelect, loadRagAvailability, loadEmbeddingCatalog, loadKnowledgeSummary, loadDocumentLists]
    : [initModelTab, loadEmbeddingCatalog];
  for (const step of steps) {
    try {
      await step();
    } catch (err) {
      console.error("Settings page init step failed:", err);
    }
  }
})();

// Restores the tab a reload was asked to land on (SETTINGS_TAB_STORAGE_KEY), at the end of the file so every
// function/const its loaders reach (e.g. loadTabData("external-servers") needs EXTERNAL_SERVERS) is defined.
(function restoreTabAfterReload() {
  let savedTab = null;
  try {
    savedTab = sessionStorage.getItem(SETTINGS_TAB_STORAGE_KEY);
    sessionStorage.removeItem(SETTINGS_TAB_STORAGE_KEY);
  } catch (_err) {
    // Storage blocked: falls back to the default tab.
  }
  if (savedTab && document.querySelector(`[data-tab="${savedTab}"]`)) {
    activateTab(savedTab);
    loadTabData(savedTab);
  }
})();

function matricxonDeviceValue(section) {
  const chosen = section.querySelector('[data-field="device"]').value;
  const saved = section.dataset.savedDevice ?? "";
  return chosen === "cuda" && saved.startsWith("cuda") ? saved : chosen;
}

// Matricxon's "Kernels" choice only applies with quantized native compute on (see settings.html's combined
// "Quantized native compute" row) — disabled, not hidden, while it's off so the saved choice stays visible.
function syncMatricxonKernelSelect(section) {
  const checkbox = section.querySelector('[data-field="enable_quantized_native_compute"]');
  const select = section.querySelector('[data-field="gemv_backend"]');
  if (checkbox && select) select.disabled = !checkbox.checked;
  // "GPU weights" only applies with a CUDA device selected.
  const device = section.querySelector('[data-field="device"]');
  const weights = section.querySelector('[data-field="gpu_weight_mode"]');
  if (device && weights) weights.disabled = device.value === "cpu";
}

document.addEventListener("change", (event) => {
  if (event.target.matches('[data-field="enable_quantized_native_compute"], [data-field="device"]')) {
    syncMatricxonKernelSelect(event.target.closest("[data-server]") || document);
  }
});
