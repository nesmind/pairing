/**
 * Settings page controller: loads the model catalog and the selected
 * target's (a conversation, or the app-wide defaults) current params,
 * renders one slider per tunable value plus the model picker, and saves
 * changes back on demand. Also handles the Knowledge tab's private
 * document upload/list/delete (every chat searches these automatically
 * — nothing to sync or resync) and, for admin, the RAG upload limits on
 * the System tab.
 */

// Describes every slider on the page: which key in GenerationParams it
// edits, its range/step, and a one-line explanation shown under it. This
// is the single source of truth the UI is built from, so adding a new
// tunable parameter later only means adding one entry here (plus the
// matching field in app/schemas.py::GenerationParams).
//
// num_ctx's `max` is mutated in place by applyContextLimitForModel()
// below whenever the selected model changes — Gemma 4's 256K context
// window and Llama 3's 8K one shouldn't share a single hardcoded slider
// ceiling, so this starts at a generic fallback and gets narrowed (or
// widened) to match whatever model is actually active.
const PARAM_DEFS = [
  { key: "temperature", label: "Temperature", min: 0, max: 2, step: 0.05,
    help: "Higher = more creative and varied; lower = more focused and deterministic." },
  { key: "top_p", label: "Top P (nucleus sampling)", min: 0, max: 1, step: 0.01,
    help: "Restricts choices to the smallest set of tokens covering this much probability mass." },
  { key: "top_k", label: "Top K", min: 1, max: 200, step: 1,
    help: "Restricts choices to the K most likely next tokens at each step." },
  { key: "repeat_penalty", label: "Repeat penalty", min: 0, max: 2, step: 0.05,
    help: "Discourages the model from repeating itself. 1.0 = no penalty." },
  { key: "num_ctx", label: "Context window (tokens)", min: 256, max: 32768, step: 256,
    help: "How much conversation history the model can see at once. Capped by the selected model's own limit." },
  { key: "num_predict", label: "Max reply length (tokens)", min: 32, max: 4096, step: 32,
    help: "Upper limit on how many tokens the model may generate in one reply." },
];

// A model's true context window can be huge (Gemma 4 goes up to 256K);
// the slider is capped here purely so the range input stays a
// reasonable number of steps, not because larger values are unsafe.
const MAX_UI_CONTEXT = 131072;

const DEFAULTS_TARGET = "__defaults__";

// Whether the logged-in user is an admin — set server-side as a data
// attribute on <body> (see app/templates/base.html) since only admin can
// actually pull/uninstall/hide models (the matching endpoints in
// app/routers/settings.py are all admin-only); the UI mirrors that here
// so a regular user sees an honest "ask an admin" note instead of a
// button that would just 403 when clicked.
const isAdmin = document.body.dataset.isAdmin === "true";

const conversationSelect = document.getElementById("conversation-select");
const modelCatalogEl = document.getElementById("model-catalog");
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
const embeddingModelCatalogEl = document.getElementById("embedding-model-catalog");
const ragTopKInput = document.getElementById("rag-top-k");
const ragTopKValueEl = document.getElementById("rag-top-k-value");
const myDocsSummaryEl = document.getElementById("my-docs-summary");
const saveBtn = document.getElementById("save-btn");
const saveStatusEl = document.getElementById("save-status");

let currentParams = null; // the GenerationParams object currently shown/edited
let currentCatalog = null; // cached GET /api/settings/model-catalog response
let currentExtendedCatalog = null; // cached GET /api/settings/model-catalog/extended response — null until "Browse more models" is opened at least once
let currentEmbeddingCatalog = null; // cached GET /api/settings/embedding-model-catalog response
let selectedModelTag = null; // the default model for new chats (Model tab), or the target being edited's own model (Behavior tab's num_ctx narrowing only — see loadTarget)
let modelsAvailable = true; // false when the model catalog itself failed to load (see initModelTab)
// Which engine (see app.services.engine_service) is currently active — read from /health (public, no admin
// gate) rather than GET /api/settings/engine (admin-only), since the Model tab's own catalog/badges are visible
// to every user, not just admins. Used only to decide whether the "Not supported" badge below is relevant right
// now — irrelevant, so hidden, whenever Ollama (which runs every catalog entry regardless) is the active one.
let currentActiveEngine = null;

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
        value="${params[def.key]}" class="w-full accent-brand-600">
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

/** Narrows (or widens) the num_ctx slider to match the selected model's
 * real context window, and pulls the current value down if it now
 * exceeds that limit — this is what makes fine-tuning "reflect the
 * chosen model" rather than showing the same fixed range for every
 * model regardless of what it can actually support. */
function applyContextLimitForModel(entry) {
  const numCtxDef = PARAM_DEFS.find((d) => d.key === "num_ctx");
  numCtxDef.max = entry && entry.context_length
    ? Math.min(entry.context_length, MAX_UI_CONTEXT)
    : 32768;
  if (currentParams.num_ctx > numCtxDef.max) {
    currentParams.num_ctx = numCtxDef.max;
  }
  renderSliders(currentParams);
}

// ---- Model catalog ------------------------------------------------------

async function loadCatalog() {
  if (!currentCatalog) {
    currentCatalog = await api("/api/settings/model-catalog");
  }
  return currentCatalog;
}

/** The admin-managed "browse more models" catalog (see app.services.extended_model_catalog_service) — fetched
 * fresh every call (not cached the way loadCatalog() is) since it's only ever loaded on demand, when "Browse
 * more models" is actually opened, so a stale-cache concern isn't worth the extra bookkeeping. */
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

/** Real, per-engine RAM figure (see CatalogEntry.min_ram_gb_ollama/min_ram_gb_matricxon's own
 * docstrings) — never the old flat, hand-typed min_ram_gb alone, since Ollama and Matricxon's real
 * requirements for the same file aren't proportional to each other (Matricxon dequantizes to bf16
 * before computing; Ollama doesn't). Shows only whichever engine is actually active right now — that's
 * the number that matters for "can I chat with this today" — falling back to Ollama's/the static
 * estimate when Matricxon is active but hasn't reported its own real figure yet (not installed there,
 * so min_ram_gb_matricxon is still unknown — see _get_matricxon_installed_info_or_none's own
 * docstring). */
function formatRam(entry) {
  const ram =
    currentActiveEngine === "matricxon"
      ? (entry.min_ram_gb_matricxon ?? entry.min_ram_gb_ollama ?? entry.min_ram_gb)
      : (entry.min_ram_gb_ollama ?? entry.min_ram_gb);
  return ram != null ? `${ram}GB RAM` : null;
}

/** Persists whichever model is currently selected as the default for new chats right away — the Model tab has
 * no "Save changes" button of its own at all (unlike Behavior/Knowledge), since there's nothing to batch: an
 * existing conversation's model is changed from that chat's own model badge instead (see chat.js: switchModel),
 * never from Settings, so this only ever means "the default." */
async function saveModelSelection() {
  try {
    await api("/api/settings/default-model", {
      method: "PUT", body: JSON.stringify({ model: selectedModelTag }),
    });
    // Reloading (rather than the old redirect-to-chat-page) is what
    // actually confirms the update, and staying on Settings means
    // whatever else was being edited here isn't lost.
    window.location.reload();
  } catch (err) {
    saveStatusEl.textContent = `Failed to update model: ${err.message}`;
  }
}

function selectModel(entry) {
  selectedModelTag = entry.tag;
  applyContextLimitForModel(entry);
  renderModelCatalog();
  saveModelSelection();
}

/** Re-fetches and re-renders both catalogs after an action (Pull/Uninstall/Hide/Remove) that could affect either one —
 * a row rendered via renderModelRow can come from the default catalog or the extended one (same shape, see
 * renderCatalogGroup), and either could change any of installed/hidden/hardware_ok. Extended is only refreshed
 * if it's actually been loaded at least once (browsing it is opt-in — see toggleBrowseMoreModels), not fetched
 * just to immediately discard an unopened panel's result. */
async function refreshCatalogs() {
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
    const response = await fetch("/api/settings/pull-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tag: entry.tag, confirm_duplicate: confirmDuplicate }),
    });
    // DuplicateInstallDetector found an already-installed model that looks like the same file under a
    // different name (see app/routers/settings.py's pull_model) — a structured detail, not the plain-string
    // one every other error here uses, so it's handled before the generic error path below. Declining is a
    // cancellation, not a failure: reset the button instead of throwing.
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
    await onDone(); // stale now that this model is installed
  } catch (err) {
    progressEl.textContent = `Failed: ${err.message}`;
    button.disabled = false;
    button.textContent = "Pull";
  }
}

async function uninstallModel(entry, button, progressEl, onDone = refreshCatalogs) {
  if (!confirm(`Uninstall ${entry.tag}? This removes it from Ollama for every user.`)) return;

  button.disabled = true;
  button.textContent = "Uninstalling…";
  progressEl.classList.remove("hidden");
  progressEl.textContent = "";

  try {
    await api("/api/settings/delete-model", {
      method: "POST",
      body: JSON.stringify({ tag: entry.tag }),
    });
    // If the model just removed was selected for the current target,
    // fall back to no selection rather than leaving a phantom choice
    // pointing at a model that no longer exists.
    if (selectedModelTag === entry.tag) selectedModelTag = null;
    await onDone(); // stale now that this model is gone
  } catch (err) {
    progressEl.textContent = `Failed: ${err.message}`;
    button.disabled = false;
    button.textContent = "Uninstall";
  }
}

/** Hides or unhides a model from regular users' picker (admin only —
 * see POST /api/settings/hide-model). Purely a visibility toggle, so
 * there's no confirmation dialog the way Uninstall gets. */
async function toggleHideModel(entry, button, progressEl) {
  const newHidden = !entry.hidden;
  button.disabled = true;
  progressEl.classList.remove("hidden");
  progressEl.textContent = newHidden ? "Hiding…" : "Unhiding…";

  try {
    await api("/api/settings/hide-model", {
      method: "POST",
      body: JSON.stringify({ tag: entry.tag, hidden: newHidden }),
    });
    await refreshCatalogs(); // stale — force a refresh so `hidden` reflects the change
  } catch (err) {
    progressEl.textContent = `Failed: ${err.message}`;
    button.disabled = false;
  }
}

/** Removes a not-yet-installed extended-catalog entry from the list (admin only — see CatalogEntry.removable's
 * own docstring for why an installed one never reaches this at all). Purely catalog curation, same as
 * toggleHideModel above — nothing to uninstall from Ollama since it was never pulled, so no confirmation dialog
 * the way uninstallModel gets. */
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

/** One model's card in the picker: name/meta on the left, actions on
 * the right, wrapping onto their own line if they don't fit — a plain
 * flex layout rather than a fixed-column grid, so it can't overflow its
 * container no matter how many action buttons a row ends up with
 * (Select, Pull, Uninstall, Hide, Remove — not every row has all of them). */
function renderModelRow(entry) {
  const isSelected = entry.tag !== null && entry.tag === selectedModelTag;
  const row = document.createElement("div");
  row.className =
    "rounded-lg border px-3 py-2.5 text-sm " +
    (isSelected ? "border-brand-500 bg-[rgb(var(--color-brand-500)/0.1)]" : "border-slate-800 bg-slate-900") +
    (entry.hidden ? " opacity-60" : "");

  const top = document.createElement("div");
  top.className = "flex flex-wrap items-center justify-between gap-x-3 gap-y-1.5";

  const label = document.createElement("div");
  label.className = "min-w-0";
  // A tag like "hf.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF:Meta-Llama-3-8B-Instruct.Q4_K_M" is the real
  // pull path, not something anyone wants as a row's headline — it used to be the bold title here and, being far
  // longer than the old plain Ollama tags ("llama3:latest"), pushed the action buttons onto their own wrapped
  // line even on a normal-width screen. "family + parameter_size" (e.g. "Llama 3 8B") is short, always present
  // (see build_model_catalog/the "installed but not in the static catalog" branch — both always populate these),
  // and reads like an actual model name; the real tag still shows below, just no longer competing with the
  // buttons for space.
  const modelName = [entry.family, entry.parameter_size].filter(Boolean).join(" ");
  const metaParts = [
    formatSize(entry.download_gb),
    formatContextLength(entry.context_length) ? `${formatContextLength(entry.context_length)} ctx` : null,
    formatRam(entry),
  ].filter(Boolean);
  // "+Vision" = a normal chat model that can *also* see an image
  // attachment (e.g. gemma4:12b) — the only case this catalog can
  // actually produce today, since build_model_catalog only ever lists
  // "completion"-capable models to begin with. "Vision" alone would
  // mean a vision-only model with no text-chat ability at all — kept
  // here for a correct/honest label if one ever does appear, not
  // because one can right now.
  const visionBadge = entry.vision
    ? ` <span class="inline-flex items-center rounded-full bg-[rgb(var(--color-brand-500)/0.15)] border border-[rgb(var(--color-brand-500)/0.4)] px-1.5 py-0.5 text-[10px] font-medium text-brand-500 align-middle">${entry.text_capable ? "+Vision" : "Vision"}</span>`
    : "";
  // Only shown while Matricxon is the currently active engine (see the "Active engine" picker above the
  // External servers tab) — Ollama runs every catalog entry regardless, so this distinction is meaningless
  // (and would just read as confusing noise) whenever it's the one actually serving requests. See
  // CatalogEntry.matricxon_supported's own docstring. Also skipped for a vision projector (entry.is_projector)
  // regardless of engine — "Not supported" reads as "Matricxon can't run this," which is backwards for a file
  // that was never meant to run as a standalone chat model at all; its own "(vision projector)" family label
  // already says what it actually is.
  const matricxonBadge = entry.matricxon_supported || entry.is_projector || currentActiveEngine !== "matricxon"
    ? ""
    : ` <span class="inline-flex items-center rounded-full bg-slate-800 border border-slate-700 px-1.5 py-0.5 text-[10px] font-medium text-slate-400 align-middle" title="${escapeHtml(entry.matricxon_unsupported_reason || "Not supported by Matricxon")}">Not supported</span>`;
  // Only for an *installed* entry that isn't in app/model_catalog.py's hand-curated list (see
  // CatalogEntry.is_auto_discovered's own docstring) — a not-yet-installed extended-catalog entry is already
  // clearly organized under its own "Browse more models" section, so it doesn't need this too. Projectors are
  // also auto-discovered by definition but already read clearly via their own "(vision projector)" family
  // suffix, so this would just be redundant noise there.
  const notInCatalogBadge = entry.installed && entry.is_auto_discovered && !entry.is_projector
    ? ` <span class="inline-flex items-center rounded-full bg-slate-800 border border-slate-700 px-1.5 py-0.5 text-[10px] font-medium text-slate-400 align-middle" title="Installed, but not part of the hand-curated model list">Not in catalog</span>`
    : "";
  label.innerHTML =
    `<span class="block truncate font-medium text-slate-100">${escapeHtml(modelName || entry.tag)}` +
    (entry.hidden ? ` <span class="text-xs font-normal text-slate-500">(hidden from users)</span>` : "") +
    visionBadge +
    matricxonBadge +
    notInCatalogBadge +
    `</span>` +
    `<span class="block truncate text-xs text-slate-500">${escapeHtml(metaParts.join(" · "))}</span>`;

  // flex-wrap: if every button that applies to this row doesn't fit on
  // one line, the extras drop to a second line instead of overflowing
  // past the card's edge.
  const action = document.createElement("div");
  action.className = "flex flex-wrap items-center justify-end gap-1.5 shrink-0";

  const progressRow = document.createElement("p");
  progressRow.className = "mt-1.5 text-xs text-slate-500 hidden";

  if (!entry.hardware_ok) {
    const badge = document.createElement("span");
    badge.className = "text-xs font-medium text-amber-400";
    badge.textContent = "Unavailable";
    action.appendChild(badge);
  } else if (entry.installed) {
    // A vision projector is never itself a selectable chat model (see CatalogEntry.is_projector's own
    // docstring) — shown as plain confirmation that the install actually took, not an action.
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

    // Uninstalling frees disk space but affects every user, so it's
    // gated the same way pulling is (admin-only server-side too — see
    // POST /api/settings/delete-model in app/routers/settings.py).
    if (isAdmin) {
      const uninstallBtn = document.createElement("button");
      uninstallBtn.type = "button";
      uninstallBtn.title = "Uninstall — removes this model from Ollama for everyone";
      uninstallBtn.className =
        "rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-400 hover:border-red-500 hover:text-red-400 transition-colors";
      uninstallBtn.textContent = "Uninstall";
      uninstallBtn.addEventListener("click", () => uninstallModel(entry, uninstallBtn, progressRow));
      action.appendChild(uninstallBtn);
    }
  } else if (isAdmin) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "rounded-md border border-slate-700 px-2.5 py-1 text-xs text-slate-200 hover:bg-slate-800 transition-colors";
    btn.textContent = "Pull";
    btn.addEventListener("click", () => pullModel(entry, btn, progressRow));
    action.appendChild(btn);
  } else {
    const note = document.createElement("span");
    note.className = "text-xs text-slate-500";
    note.textContent = "Not installed — ask an admin";
    action.appendChild(note);
  }

  // Hide/Unhide: admin-only, independent of install state — an admin
  // can declutter the picker for regular users even for a model that
  // isn't installed yet.
  if (isAdmin) {
    const hideBtn = document.createElement("button");
    hideBtn.type = "button";
    hideBtn.title = entry.hidden
      ? "Unhide — show this model to regular users again"
      : "Hide — remove this model from regular users' picker";
    hideBtn.className =
      "rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-400 hover:bg-slate-800 transition-colors";
    hideBtn.textContent = entry.hidden ? "Unhide" : "Hide";
    hideBtn.addEventListener("click", () => toggleHideModel(entry, hideBtn, progressRow));
    action.appendChild(hideBtn);
  }

  // Remove from list: admin-only, and only for a not-yet-installed extended-catalog entry (server-computed —
  // see CatalogEntry.removable's own docstring). A default app/model_catalog.py entry never has this (nothing
  // stored to delete), and an installed one gets Uninstall above instead, never both.
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

  // The real pull path, below the title/buttons row so its own length never competes with them for space (see
  // modelName's own comment above) — truncated with the full value in a native title="" tooltip on hover rather
  // than just cut off with no way to see the rest.
  if (entry.tag) {
    const pathLine = document.createElement("p");
    pathLine.className = "mt-1 truncate font-mono text-[11px] text-slate-600";
    pathLine.textContent = entry.tag;
    pathLine.title = entry.tag;
    row.appendChild(pathLine);
  }

  if (!entry.hardware_ok) {
    const reason = document.createElement("p");
    reason.className = "mt-1.5 text-xs text-amber-400";
    reason.textContent = entry.unavailable_reason
      || `Requires ~${entry.min_ram_gb_ollama ?? entry.min_ram_gb}GB RAM/VRAM — this machine has ~${currentCatalog.hardware.total_gb}GB.`;
    row.appendChild(reason);
  }

  row.appendChild(progressRow);

  return row;
}

/** Shared by both the default catalog (#model-catalog) and the admin-extensible one behind "Browse more models"
 * (#extended-model-catalog) — same CatalogEntry shape from either endpoint, so one grouping/rendering routine
 * covers both. Groups by vendor first, family second — "Google" / "Gemma 4" reads more like a real product
 * catalog than a flat list of tags, and answers "who makes this" at a glance; every tag sharing one family
 * (e.g. Gemma 4's e2b/e4b/12b/26b/31b sizes) renders together as that family's own "versions." */
function renderCatalogGroup(containerEl, entries) {
  containerEl.innerHTML = "";
  const vendors = [...new Set(entries.map((entry) => entry.vendor))];
  for (const vendor of vendors) {
    const vendorEntries = entries.filter((e) => e.vendor === vendor);
    const families = [...new Set(vendorEntries.map((e) => e.family))];

    for (const family of families) {
      const group = document.createElement("div");
      group.className = "space-y-1.5";

      const heading = document.createElement("p");
      heading.className = "text-xs font-semibold text-slate-300";
      heading.innerHTML = `${escapeHtml(vendor)} <span class="font-normal text-slate-500">— ${escapeHtml(family)}</span>`;
      group.appendChild(heading);

      for (const entry of vendorEntries.filter((e) => e.family === family)) {
        group.appendChild(renderModelRow(entry));
      }
      containerEl.appendChild(group);
    }
  }
}

function renderModelCatalog() {
  const { hardware, entries } = currentCatalog;
  hardwareSummaryEl.textContent = hardware.vram_gb > 0
    ? `${hardware.ram_gb}GB RAM + ${hardware.vram_gb}GB VRAM`
    : `${hardware.ram_gb}GB RAM, no GPU detected`;
  renderCatalogGroup(modelCatalogEl, entries);
  // Catches every way the stored default can end up pointing at nothing real — cleared out from under it by an
  // engine switch (see app.services.engine_switch_service.clear_stale_default_models), the model being
  // uninstalled, or nothing ever having been chosen at all — not just the engine-switch case specifically,
  // since a stale tag reads the same way to this picker regardless of why it went stale. findCatalogEntry also
  // checks the (lazily loaded) extended catalog, matching renderModelRow's own isSelected check exactly, so
  // this banner can't fire a false positive for a default that's only in that not-yet-opened panel. Requiring
  // `.installed` (not just "some catalog entry exists with this tag") is what actually makes the "uninstalled"
  // case above work: app.services.settings_service.get_default_model falls back to app.config.DEFAULT_MODEL —
  // a real curated-catalog tag — even when nothing has ever been installed, so an uninstalled fallback tag
  // would otherwise still be found in the catalog and silently pass as "valid".
  const defaultEntry = selectedModelTag !== null ? findCatalogEntry(selectedModelTag) : null;
  const hasValidDefault = Boolean(defaultEntry?.installed);
  document.getElementById("model-no-default-banner").classList.toggle("hidden", hasValidDefault);
}

function renderExtendedModelCatalog() {
  const entries = currentExtendedCatalog?.entries || [];
  extendedModelCatalogEmptyEl.classList.toggle("hidden", entries.length > 0);
  renderCatalogGroup(extendedModelCatalogEl, entries);
}

/** Lazily loads the extended catalog the first time it's opened (see currentExtendedCatalog's own comment) —
 * subsequent toggles just show/hide the already-rendered panel, no refetch, matching the plain accordion
 * pattern the rest of Settings doesn't otherwise need. */
async function toggleBrowseMoreModels() {
  const opening = browseMoreModelsPanelEl.classList.contains("hidden");
  browseMoreModelsPanelEl.classList.toggle("hidden", !opening);
  browseMoreModelsChevronEl.classList.toggle("rotate-180", opening);
  if (opening && !currentExtendedCatalog) {
    await loadExtendedCatalog();
    renderExtendedModelCatalog();
  }
  if (opening) {
    // The panel (search box + list) is hidden by default and sits at the bottom of a long tab — without this,
    // expanding it leaves the newly-revealed content below the fold with no visual cue anything changed.
    // Scrolling the toggle itself (not the panel) into view keeps the "Browse more models" heading visible at
    // the top of the viewport with the panel's content right below it, whether the list ends up empty or not.
    browseMoreModelsToggleEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

browseMoreModelsToggleEl.addEventListener("click", toggleBrowseMoreModels);

// Three-step "Search, pick a repo, Add a file" (admin-only — hfSearchBtn/hfSearchResultsEl/hfRepoFilesEl don't
// exist in the DOM at all for a regular user, see settings.html's own {% if is_admin %} guard, so this whole
// block is skipped rather than throwing on a null addEventListener target). Search and the repo file list are
// both read-only (see app/services/huggingface_client.py); nothing is saved until a specific file's own "Add"
// button is clicked, which re-verifies it from scratch server-side rather than trusting this search/file-list
// round trip (see extended_model_catalog_service.add_model's own docstring).
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
    row.innerHTML =
      `<span class="block truncate font-medium text-slate-100">${escapeHtml(repo.repo_id)}</span>` +
      `<span class="block truncate text-xs text-slate-500">${escapeHtml(metaParts.join(" · "))}</span>`;
    row.addEventListener("click", () => loadHfRepoFiles(repo.repo_id));
    hfSearchResultsEl.appendChild(row);
  }
  hfSearchResultsEl.classList.remove("hidden");
}

// Ollama's own library models default to a Q4_K_M quant when no tag is given — the widely-agreed "best balance
// of quality and size" for a general-purpose pull, so a file matching it (case-insensitively; HF filenames use
// every casing) gets a "Recommended" badge and is sorted first, instead of leaving every quant looking equally
// arbitrary to someone who doesn't already know what Q4_K_M/Q8_0/IQ2_XS etc. mean.
function isRecommendedQuantFile(filename) {
  return /q4_k_m/i.test(filename);
}

function renderHfRepoFiles(repo) {
  hfRepoFilesEl.innerHTML = "";

  const metaParts = [repo.parameter_size, formatContextLength(repo.context_length) && `${formatContextLength(repo.context_length)} ctx`, repo.license]
    .filter(Boolean);
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

  // Stable sort (spec-guaranteed order for equal keys since ES2019) — only reorders the recommended file(s) to
  // the front, every other file keeps its original relative order.
  const sortedFiles = [...repo.files].sort(
    (a, b) => Number(isRecommendedQuantFile(b.filename)) - Number(isRecommendedQuantFile(a.filename))
  );

  for (const file of sortedFiles) {
    const fileRow = document.createElement("div");
    fileRow.className = "flex items-center justify-between gap-3 pt-1";

    const recommendedBadge = isRecommendedQuantFile(file.filename)
      ? ` <span class="inline-flex items-center rounded-full bg-emerald-500/15 border border-emerald-500/40 px-1.5 py-0.5 text-[10px] font-medium text-emerald-400 align-middle">Recommended</span>`
      : "";
    // The repo-level family/parameter_size shown once in the header above (repo.family/repo.parameter_size)
    // describes whichever file Hugging Face treats as this repo's one primary `gguf` metadata block — almost
    // never a same-repo vision-projector sidecar (see app.services.huggingface_client._is_projector_file's own
    // docstring). Flagged here so picking one doesn't read as "the phi2/whatever chat model" the header
    // implies — confirmed live: this is exactly what caused a mmproj file to get added and pulled as if it
    // were a real, standalone chat model.
    const projectorBadge = file.is_projector
      ? ` <span class="inline-flex items-center rounded-full bg-amber-500/15 border border-amber-500/40 px-1.5 py-0.5 text-[10px] font-medium text-amber-400 align-middle">Vision projector — not a chat model on its own</span>`
      : "";
    const label = document.createElement("span");
    label.className = "truncate text-xs text-slate-300";
    label.innerHTML = `${escapeHtml(`${file.filename} (${formatSize(file.download_gb)})`)}${recommendedBadge}${projectorBadge}`;
    fileRow.appendChild(label);

    const addBtn = document.createElement("button");
    addBtn.type = "button";
    addBtn.className =
      "shrink-0 rounded-md bg-brand-600 px-3 py-1 text-xs font-medium text-white hover:bg-brand-500 transition-colors";
    addBtn.textContent = "Add";
    addBtn.addEventListener("click", async () => {
      addBtn.disabled = true;
      // Shown right below the file list itself (hf-repo-files-status), not hf-search-status up near the
      // search box — confirmed live: adding a file (this GGUF-header probe can take a few real seconds, see
      // app.services.gguf_probe's own docstring) left "Adding…" sitting somewhere an admin who'd already
      // scrolled down to click this exact button couldn't see at all.
      hfRepoFilesStatusEl.textContent = "Adding…";
      hfRepoFilesStatusEl.classList.remove("text-red-400");
      try {
        // already_installed (see ExtendedModelCatalogResponse's own docstring): this exact tag already being
        // installed means it was never going to show up in "browse more models" to pull — confirmed live, a
        // flat "Added" here read as "go find it in the list," which no longer existed anywhere to click.
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
        await loadExtendedCatalog();
        renderExtendedModelCatalog();
      } catch (err) {
        hfRepoFilesStatusEl.textContent = err.message;
        hfRepoFilesStatusEl.classList.add("text-red-400");
        addBtn.disabled = false;
      }
    });
    fileRow.appendChild(addBtn);
    hfRepoFilesEl.appendChild(fileRow);
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
    // The repo list above can run to 20 rows — without this, clicking one near the top leaves the newly-opened
    // file picker below the fold with no visual cue anything happened (same reasoning as
    // toggleBrowseMoreModels' own scrollIntoView).
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

/** Loads the Model tab's own default-model picker — independent of Behavior's target-picker (the model here
 * only ever means "the default for new chats", see saveModelSelection's own docstring), so this runs once at
 * page load and again each time the Model tab is re-activated (see loadTabData), not tied to loadTarget at all. */
async function initModelTab() {
  try {
    currentActiveEngine = (await api("/health")).active_engine;
    await loadCatalog();
    selectedModelTag = (await api("/api/settings/default-model")).model;
    renderModelCatalog();
    modelsAvailable = true;
  } catch (err) {
    modelsAvailable = false;
  }
  document.getElementById("model-unavailable").classList.toggle("hidden", modelsAvailable);
  document.getElementById("model-field-normal").classList.toggle("hidden", !modelsAvailable);
}

// ---- RAG availability + knowledge base (private documents) ---------------
// The shared/"global" scope still exists server-side (see app/rag.py and
// app/routers/documents.py — scope="global" on upload, owner_id NULL on
// Document) but has no UI here for now; only "My documents" (private) is
// exposed until that gets a proper management pass.

async function loadRagAvailability() {
  const { available } = await api("/api/settings/rag-availability");
  ragUnavailableEl.classList.toggle("hidden", available);
}

async function loadEmbeddingCatalog() {
  currentEmbeddingCatalog = await api("/api/settings/embedding-model-catalog");
  renderEmbeddingCatalog();
}

/** One embedding model's row — deliberately not renderModelRow (its Select/Uninstall/Hide/Remove logic doesn't
 * apply to an embedding model: nothing to "select" since there's no live UI switch this pass — see
 * EMBEDDING_MODEL's own comment in app/config.py — and nothing to hide/remove since this is a fixed two-entry
 * list, not the admin-curated extended catalog). Visible to every user (installed status isn't admin-secret,
 * same as the #rag-unavailable banner above it); only the Pull button itself is admin-gated, same convention
 * renderModelRow already uses for its own Pull button. */
function renderEmbeddingCatalog() {
  embeddingModelCatalogEl.innerHTML = "";
  if (!currentEmbeddingCatalog) return;

  for (const entry of currentEmbeddingCatalog.entries) {
    const isDefault = entry.tag === currentEmbeddingCatalog.default_tag;
    const row = document.createElement("div");
    row.className = "rounded-lg border border-slate-800 bg-slate-900 px-3 py-2.5 text-sm";

    const top = document.createElement("div");
    top.className = "flex flex-wrap items-center justify-between gap-x-3 gap-y-1.5";

    const label = document.createElement("div");
    label.className = "min-w-0";
    const modelName = [entry.family, entry.parameter_size].filter(Boolean).join(" ");
    const metaParts = [
      formatSize(entry.download_gb),
      `${entry.embedding_dim}-dim`,
      formatContextLength(entry.context_length) ? `${formatContextLength(entry.context_length)} ctx` : null,
    ].filter(Boolean);
    const defaultBadge = isDefault
      ? ` <span class="inline-flex items-center rounded-full bg-[rgb(var(--color-brand-500)/0.15)] border border-[rgb(var(--color-brand-500)/0.4)] px-1.5 py-0.5 text-[10px] font-medium text-brand-500 align-middle">Default</span>`
      : "";
    // See renderModelRow's identical matricxonBadge for the full reasoning.
    const matricxonBadge = entry.matricxon_supported || currentActiveEngine !== "matricxon"
      ? ""
      : ` <span class="inline-flex items-center rounded-full bg-slate-800 border border-slate-700 px-1.5 py-0.5 text-[10px] font-medium text-slate-400 align-middle" title="${escapeHtml(entry.matricxon_unsupported_reason || "Not supported by Matricxon")}">Not supported</span>`;
    label.innerHTML =
      `<span class="block truncate font-medium text-slate-100">${escapeHtml(modelName || entry.tag)}${defaultBadge}${matricxonBadge}</span>` +
      `<span class="block truncate text-xs text-slate-500">${escapeHtml(metaParts.join(" · "))}</span>`;

    const action = document.createElement("div");
    action.className = "flex flex-wrap items-center justify-end gap-1.5 shrink-0";
    const progressRow = document.createElement("p");
    progressRow.className = "mt-1.5 text-xs text-slate-500 hidden";

    // Both Pull and Uninstall need the embedding catalog + RAG-availability banner refreshed
    // afterward, not the chat-model catalogs pullModel()/uninstallModel() default to.
    const refreshEmbedding = async () => {
      await loadEmbeddingCatalog();
      await loadRagAvailability();
    };

    if (entry.installed) {
      const badge = document.createElement("span");
      badge.className = "text-xs font-medium text-emerald-400";
      badge.textContent = "Installed";
      action.appendChild(badge);

      // Uninstalling frees disk space but affects every user, so it's gated the same way
      // pulling is (admin-only server-side too — see POST /api/settings/delete-model).
      if (isAdmin) {
        const uninstallBtn = document.createElement("button");
        uninstallBtn.type = "button";
        uninstallBtn.title = "Uninstall — removes this model from Ollama for everyone";
        uninstallBtn.className =
          "rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-400 hover:border-red-500 hover:text-red-400 transition-colors";
        uninstallBtn.textContent = "Uninstall";
        uninstallBtn.addEventListener("click", () => uninstallModel(entry, uninstallBtn, progressRow, refreshEmbedding));
        action.appendChild(uninstallBtn);
      }
    } else if (isAdmin) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "rounded-md border border-slate-700 px-2.5 py-1 text-xs text-slate-200 hover:bg-slate-800 transition-colors";
      btn.textContent = "Pull";
      btn.addEventListener("click", () => pullModel(entry, btn, progressRow, refreshEmbedding));
      action.appendChild(btn);
    } else {
      const note = document.createElement("span");
      note.className = "text-xs text-slate-500";
      note.textContent = "Not installed — ask an admin";
      action.appendChild(note);
    }

    top.appendChild(label);
    top.appendChild(action);
    row.appendChild(top);

    const pathLine = document.createElement("p");
    pathLine.className = "mt-1 truncate font-mono text-[11px] text-slate-600";
    pathLine.textContent = entry.tag;
    pathLine.title = entry.tag;
    row.appendChild(pathLine);
    row.appendChild(progressRow);

    embeddingModelCatalogEl.appendChild(row);
  }
}

// Cached from the last summary fetch so handleUpload() can check a
// file's size against the current limit before even sending it (the
// server enforces the real limit regardless — this is just a faster,
// friendlier rejection for the common case).
let currentUploadLimits = null;

function renderKnowledgeSummary(summary) {
  currentUploadLimits = summary.limits;
  myDocsSummaryEl.textContent =
    `${summary.mine.document_count} doc${summary.mine.document_count === 1 ? "" : "s"}, ` +
    `${summary.mine.chunk_count} chunk${summary.mine.chunk_count === 1 ? "" : "s"} — ` +
    `${summary.mine.used_mb}MB of ${summary.limits.max_user_space_mb}MB used`;
}

async function loadKnowledgeSummary() {
  renderKnowledgeSummary(await api("/api/documents/summary"));
}

/** One row in a document list: name, chunk count, and — for whoever's
 * allowed to remove it — a Delete button. Mirrors the model catalog's
 * uninstall pattern (confirm, then remove and refresh). */
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
      "shrink-0 rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-400 hover:border-red-500 hover:text-red-400 transition-colors";
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

// How many of the user's own documents to show per page (see
// renderMyDocsPage) — the full list is still fetched in one request
// (this app's scale doesn't need server-side paging), just not all
// rendered/shown at once, so the list stays manageable as it grows.
const MY_DOCS_PAGE_SIZE = 10;
let myDocsAll = [];
let myDocsPage = 1;

function renderMyDocsPage() {
  const containerEl = document.getElementById("my-docs-list");
  const paginationEl = document.getElementById("my-docs-pagination");

  const totalPages = Math.max(1, Math.ceil(myDocsAll.length / MY_DOCS_PAGE_SIZE));
  // Clamped rather than trusted as-is: the underlying list can shrink
  // (a delete) between renders, which could otherwise leave the page
  // number pointing past the new last page.
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

document.getElementById("my-docs-prev-btn").addEventListener("click", () => {
  myDocsPage -= 1;
  renderMyDocsPage();
});
document.getElementById("my-docs-next-btn").addEventListener("click", () => {
  myDocsPage += 1;
  renderMyDocsPage();
});

async function loadDocumentLists() {
  const docs = await api("/api/documents");
  myDocsAll = docs.filter((d) => d.scope === "mine");
  renderMyDocsPage();
}

// How often to poll an upload job's progress. Plain GET/POST + polling
// rather than a streamed response: reading a streaming fetch() body
// incrementally depends on how a given browser buffers it, which isn't
// consistent across every browser/version, and a slow, silently-stuck-
// looking upload is exactly the failure mode that's not worth risking
// here. Ordinary complete request/response cycles have no such
// dependency — they work the same everywhere.
const UPLOAD_POLL_MS = 1500;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Starts an upload job (POST, returns almost immediately with a job
 * id and the initial per-file state) and polls it to completion,
 * calling `onUpdate(job)` after every poll — including the first,
 * synchronous one from the POST response itself — so the caller can
 * paint per-file progress as it comes in. Resolves with the job's
 * final `result` once `done` is true.
 *
 * A raw fetch rather than the shared api() helper for the POST: api()
 * always sends Content-Type: application/json, which is wrong for a
 * multipart/form-data body — the browser needs to set that header
 * itself, with the boundary. */
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

  // The upload itself keeps running server-side regardless of whether
  // any one poll succeeds — so a single dropped connection or transient
  // network hiccup while polling shouldn't be treated as the upload
  // having failed. Retry a few times before actually giving up.
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

/** Reads the 5 upload slots (see app/templates/settings.html) into a
 * list of { slotEl, file, statusEl, barWrapEl, barEl } for whichever
 * ones actually have a file picked — an empty slot is just skipped. */
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
  entry.statusEl.classList.remove("text-amber-400", "text-emerald-400");
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

  // Client-side size pre-check per slot — reject just that one slot
  // (shown right in its own row) rather than blocking the whole batch;
  // the server would only ever reject that same file anyway.
  const valid = [];
  for (const entry of entries) {
    if (currentUploadLimits && entry.file.size > currentUploadLimits.max_file_mb * 1024 * 1024) {
      entry.statusEl.textContent = "Too large";
      entry.statusEl.classList.add("text-amber-400");
      continue;
    }
    valid.push(entry);
  }
  if (valid.length === 0) {
    statusEl.textContent = "No files small enough to upload — see the size limit above.";
    statusEl.classList.add("text-amber-400");
    return;
  }

  const formData = new FormData();
  for (const entry of valid) formData.append("files", entry.file);
  formData.append("scope", "mine");

  // Progress events are matched back to a slot by filename rather than
  // position: the server processes files in its own order (alphabetical
  // during the embedding pass, not upload order — see
  // app.rag.sync_folder_stream), so slot index alone can't be trusted.
  const slotByFilename = new Map();
  for (const entry of valid) {
    slotByFilename.set(entry.file.name, entry);
    entry.statusEl.textContent = "Waiting…";
    entry.statusEl.classList.remove("text-amber-400", "text-emerald-400");
    entry.barWrapEl.classList.remove("hidden");
    entry.barEl.style.width = "0%";
  }

  const totalFiles = valid.length;

  function setOverall(pct) {
    overallBarEl.style.width = `${pct}%`;
    overallPctEl.textContent = `${pct}%`;
  }

  // Applied to every poll response (see uploadFiles' onUpdate callback):
  // a full snapshot of every tracked file's current state, not a single
  // incremental event — simpler to reason about and naturally tolerant
  // of a missed or repeated poll, since it just re-paints from scratch
  // each time rather than accumulating deltas.
  function applyJobState(job) {
    let totalFraction = 0;
    for (const [filename, entry] of slotByFilename) {
      const fileState = job.files[filename];
      if (!fileState) continue;

      let fraction = 0;
      if (fileState.status === "processing" && fileState.chunks_total) {
        // Embedding one chunk can itself take a long time on slow
        // hardware, and a real document can be dozens of chunks —
        // without this, a file's bar would sit frozen for its entire
        // processing time instead of visibly moving.
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
        entry.statusEl.classList.add("text-amber-400");
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
  statusEl.classList.remove("text-amber-400");

  try {
    const result = await uploadFiles(formData, applyJobState);

    setOverall(100);
    const parts = [];
    if (result.uploaded.length) parts.push(`Uploaded ${result.uploaded.length}.`);
    if (result.errors.length) {
      parts.push(...result.errors);
      statusEl.classList.add("text-amber-400");
    }
    statusEl.textContent = parts.join(" ") || "Done.";
    renderKnowledgeSummary(result.summary);
    await loadDocumentLists();
  } catch (err) {
    statusEl.textContent = `Upload failed: ${err.message}`;
    statusEl.classList.add("text-amber-400");
  } finally {
    uploadBtn.disabled = false;
    setTimeout(() => {
      overallWrapEl.classList.add("hidden");
      valid.forEach(resetUploadSlot);
    }, 2500);
  }
}

document.getElementById("my-docs-upload-btn").addEventListener("click", handleUpload);

// ---- Loading a target's current settings ---------------------------------

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

  // No model picker on this tab at all anymore (see the Model tab's own initModelTab) — the only reason this
  // still touches the catalog is to narrow the num_ctx slider to whichever model this target actually uses. A
  // failed catalog fetch (Ollama unreachable) just means no narrowing happens — num_ctx keeps its generic
  // fallback max — not something worth its own error state here the way the Model tab's own picker needs one.
  let modelsAvailableHere = false;
  try {
    const catalog = await loadCatalog();
    applyContextLimitForModel(findCatalogEntry(model));
    modelsAvailableHere = catalog.entries.some((entry) => entry.installed);
  } catch (err) {
    // Nothing to narrow against — num_ctx stays at its default max. Same as "no models" for the banner below:
    // either way, there's genuinely nothing usable right now.
  }
  // Same banner/wording as the Model tab's own #model-unavailable (see initModelTab) — sliders here still
  // render fine either way (they're generic numeric params, not tied to a model existing), but without this
  // there was nothing on this tab telling the admin *why* a new chat still won't get a reply.
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

  // Always opens on "Defaults for new chats", not whichever conversation
  // was last open in chat — a specific conversation's own settings are
  // just as reachable one click away in the dropdown.
  conversationSelect.value = DEFAULTS_TARGET;
  await loadTarget(DEFAULTS_TARGET);
}

conversationSelect.addEventListener("change", () => loadTarget(conversationSelect.value));

ragTopKInput.addEventListener("input", () => {
  currentParams.rag_top_k = parseInt(ragTopKInput.value, 10);
  ragTopKValueEl.textContent = ragTopKInput.value;
});

// ---- Saving ---------------------------------------------------------------

saveBtn.addEventListener("click", async () => {
  const targetId = conversationSelect.value;
  saveBtn.disabled = true;
  saveStatusEl.textContent = "Saving…";
  try {
    if (targetId === DEFAULTS_TARGET) {
      await api("/api/settings/defaults", { method: "PUT", body: JSON.stringify(currentParams) });
    } else {
      // Model is deliberately left out here — an existing conversation's
      // model is only ever changed from that chat's own model badge (see
      // chat.js: switchModel), not from Settings.
      await api(`/api/conversations/${targetId}`, {
        method: "PATCH",
        body: JSON.stringify({ params: currentParams }),
      });
    }
    window.location.reload();
  } catch (err) {
    saveStatusEl.textContent = `Failed to save: ${err.message}`;
    saveBtn.disabled = false;
  }
});

// ---- Tabs -------------------------------------------------------------
// "My Settings" and "Knowledge" exist for every user; "System" and
// "Users" are admin-only extras that app/templates/settings.html only
// renders (and only adds nav buttons for) when Jinja's `is_admin` is
// true — so tabButtons/the admin-only elements below simply don't exist
// in the DOM at all for a regular user.

const tabButtons = document.querySelectorAll(".settings-tab-btn");
const saveBarEl = document.getElementById("save-bar");
const knowledgeBaseSectionEl = document.getElementById("knowledge-base-section");
// Where save-bar naturally sits in the markup (below My Settings) —
// remembered once at load so it can be moved back here when leaving
// the Knowledge tab, rather than needing a second hardcoded position.
const saveBarHomeParent = saveBarEl.parentNode;
const saveBarHomeNextSibling = saveBarEl.nextSibling;
const TAB_CONTENT_IDS = ["model", "behavior", "knowledge", "account", "system", "external-servers", "users", "channels"];

let activeTabName = "model";

// The shared save-bar (see its own comment above) is only relevant on Behavior and Knowledge — both edit the
// same currentParams for the target picked on Behavior's own conversation-select. Model has no save-bar of its
// own at all (model selection saves itself immediately — see saveModelSelection), so there's no modelsAvailable
// gating to apply here anymore, unlike before Model became its own tab. Both activateTab and loadTarget call
// this rather than toggling "hidden" directly, so neither one can clobber the other's reason for hiding it.
function updateSaveBarVisibility() {
  const tabAllowsSave = activeTabName === "behavior" || activeTabName === "knowledge";
  saveBarEl.classList.toggle("hidden", !tabAllowsSave);
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
  // The Save button applies to Behavior + Knowledge (both edit the
  // same currentParams) — Model saves itself immediately, and System/
  // Users save their own changes inline; none of those have any use
  // for it. It's one shared element (not a duplicate per tab),
  // physically relocated to sit just above the "Knowledge base"
  // section while that tab's active, and moved back to its normal spot
  // below Behavior otherwise.
  if (name === "knowledge") {
    knowledgeBaseSectionEl.parentNode.insertBefore(saveBarEl, knowledgeBaseSectionEl);
  } else if (saveBarEl.parentNode !== saveBarHomeParent || saveBarEl.nextSibling !== saveBarHomeNextSibling) {
    saveBarHomeParent.insertBefore(saveBarEl, saveBarHomeNextSibling);
  }
  activeTabName = name;
  updateSaveBarVisibility();
}

// Extracted from the click handler below so the same per-tab data load
// can also run right after a reload restores a saved tab (see
// SETTINGS_TAB_STORAGE_KEY) — a plain page reload always lands back on
// the default "Model" tab otherwise, discarding whatever the admin
// was looking at (confirmed live: reloading straight after starting
// Ollama on the External Servers tab left it blank until manually
// re-clicked, since its own data only ever loads on that tab's click).
function loadTabData(name) {
  if (name === "model") {
    initModelTab();
  }
  if (name === "account") {
    loadAccountProfileIntoEditor();
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

// See loadTabData's own comment — set right before a reload that should
// land back on a specific tab instead of the default. Actually restored
// at the very bottom of this file (see restoreTabAfterReload), not here
// — loadTabData("external-servers") reaches EXTERNAL_SERVERS, a const
// declared later in the file, and calling that this early hits its
// temporal dead zone (confirmed live: "Cannot access 'EXTERNAL_SERVERS'
// before initialization").
const SETTINGS_TAB_STORAGE_KEY = "pairingSettingsActiveTab";

async function loadSystemInfo() {
  const info = await api("/api/settings/system-info");
  document.getElementById("system-app-version").textContent = `v${info.app_version}`;
  document.getElementById("system-hardware").textContent = info.hardware.vram_gb > 0
    ? `${info.hardware.ram_gb}GB RAM + ${info.hardware.vram_gb}GB VRAM`
    : `${info.hardware.ram_gb}GB RAM, no GPU detected`;
}

// ---- Database (admin only) -------------------------------------------

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

// Mirrors app/schemas/db_config.py's DbConfigUpdate — same body shape
// for save/test-connection/build-schema, so the three actions below
// just point the same payload at three different endpoints.
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

// ---- Local instances (admin only) ------------------------------------

const instanceCountSaveBtn = document.getElementById("instance-count-save-btn");

/** Renders the plain-text "Instance N (primary) — port P — status" rows
 * under the select — deliberately just a `<ul>`, no per-row progress
 * bar or color-coded pill, matching this section's "simple, well
 * understood" brief (contrast app/static/js/settings.js's RAG upload
 * progress bar, which is what NOT to copy here). */
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
      // A short settle-and-recheck, not a progress bar: spawning/
      // stopping a process takes a beat, so re-poll status a few times
      // to let the count shown here catch up before declaring done.
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

// ---- Proxy mode (admin only) ------------------------------------------

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

// ---- Outbound HTTP proxy (admin only) -----------------------------------
// Unrelated to "Proxy mode" above (renamed "Local pAIring server mode" in
// the UI specifically to avoid this exact collision) — this one routes
// this app's own internet downloads (see app.services.http_proxy_service)
// through an admin-configured proxy, not incoming requests to this app.

const httpProxySaveBtn = document.getElementById("http-proxy-save-btn");

async function loadHttpProxyIntoEditor() {
  if (!httpProxySaveBtn) return;
  const config = await api("/api/settings/http-proxy");
  document.getElementById("http-proxy-enabled").checked = config.enabled;
  document.getElementById("http-proxy-host").value = config.host ?? "";
  document.getElementById("http-proxy-port").value = config.port ?? "";
  document.getElementById("http-proxy-username").value = config.username ?? "";
  // Never pre-filled with the real saved password (the API never sends
  // it back — see http_proxy_service.get_http_proxy_config_for_display)
  // — same "(unchanged if left blank)" pattern the Database section's
  // own MySQL password field already uses.
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
      await loadHttpProxyIntoEditor(); // refreshes the password placeholder's has_password state
    } catch (err) {
      statusEl.textContent = `Failed to save: ${err.message}`;
    } finally {
      httpProxySaveBtn.disabled = false;
      setTimeout(() => (statusEl.textContent = ""), 2500);
    }
  });
}

// ---- External servers: Ollama & ComfyUI (admin only) -------------------
//
// Both sections share the exact same DOM shape (see settings.html's
// [data-server="ollama"|"comfyui"] sections) — mode select, a "local"
// div (install/start/stop + a few tunable parameters) and a "remote" div
// (up to 12 host rows), so one generic controller drives both instead of
// duplicating the same wiring twice. See installServer() below for the
// one place their local-mode "Install" buttons genuinely differ (what
// happens once each one's own install finishes).

const EXTERNAL_SERVERS = ["ollama", "matricxon", "comfyui"];
const MAX_REMOTE_HOSTS = 120; // mirrors app.schemas.common.MAX_REMOTE_HOSTS

function serverSectionEl(server) {
  return document.querySelector(`[data-server="${server}"]`);
}

const HOST_CHECK_STATUS_IDLE_CLASS = "shrink-0 text-xs text-slate-500 w-24 text-center";

// Just for this row's own placeholder/title example — each server's usual default port and display name, so
// the example URL and tooltip at least look right for whichever section it's shown in.
const REMOTE_HOST_EXAMPLE = {
  ollama: { port: "11434", name: "Ollama" },
  matricxon: { port: "8420", name: "Matricxon" },
  comfyui: { port: "8188", name: "ComfyUI" },
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

  // Reachability check for this one row's own URL — a real ping (see
  // app/routers/ollama_admin.py|comfyui_admin.py's check-host endpoint),
  // not limited to a host already saved to the config, so an admin can
  // verify a remote candidate works *before* clicking the form's own
  // Save. Cleared on edit since a stale "✓ healthy" next to a since-
  // changed URL would be actively misleading.
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
  return Array.from(section.querySelectorAll('[data-field="remote-hosts-list"] input'))
    .map((input) => input.value.trim())
    .filter(Boolean);
}

// The last status fetched for each server — the mode <select>'s own
// "change" event and the manual-path-entry toggle below both need to
// recompute visibility without a fresh round-trip, so this is the
// write-through cache renderServerStatus keeps current.
const lastServerStatus = {};

/**
 * Local mode with nothing installed yet shows *only* the install flow —
 * no local-mode parameter fields, no Start/Stop/Save — per the explicit
 * ask: half-filled-in params for a server that doesn't exist yet is
 * more confusing than useful. Both servers have a manual override link
 * for an admin pointing at an already-installed Ollama/ComfyUI of their
 * own (not one this app's own installer manages) — see the
 * "manual-path-entry" actions below. Remote mode shows the full form but
 * never Start/Stop, since this app doesn't supervise a process it
 * doesn't own.
 *
 * install-controls (the Install/Reinstall button + version-override panel) stays visible either way, unlike an
 * earlier version of this page that hid it entirely once something was installed — a real, confirmed-live gap:
 * there was no way to even see which repo/version had been installed, let alone reinstall a different one,
 * once the not-installed panel disappeared. Hidden only in manual-path-entry mode: reinstalling there would
 * silently overwrite an admin's own deliberately-entered path with this app's own managed install location
 * instead (see installServer's own completion handler, which always saves the just-installed path).
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
  // Same disabled condition for both — nothing to start (or stop) until
  // something's actually installed.
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
    // Empty means "use the built-in pinned default" — see
    // app.schemas.ollama_server_config.OllamaServerConfig's identical pair.
    install_repo: section.querySelector('[data-field="install_repo"]')?.value.trim() || null,
    install_version: section.querySelector('[data-field="install_version"]')?.value.trim() || null,
  };
  if (server === "ollama") {
    return {
      ...base,
      binary_path: section.querySelector('[data-field="binary_path"]').value.trim() || null,
      models_path: section.querySelector('[data-field="models_path"]').value.trim() || null,
      num_parallel: intFieldOrNull(section, "num_parallel"),
      keep_alive: section.querySelector('[data-field="keep_alive"]').value.trim() || null,
      max_loaded_models: intFieldOrNull(section, "max_loaded_models"),
      context_length: intFieldOrNull(section, "context_length"),
    };
  }
  if (server === "matricxon") {
    return {
      ...base,
      project_dir: section.querySelector('[data-field="project_dir"]').value.trim() || null,
      models_path: section.querySelector('[data-field="models_path"]').value.trim() || null,
      max_loaded_models: intFieldOrNull(section, "max_loaded_models"),
      memory_safety_margin: floatFieldOrNull(section, "memory_safety_margin"),
      log_level: parseInt(section.querySelector('[data-field="log_level"]').value, 10),
      enable_quantized_native_compute: section.querySelector('[data-field="enable_quantized_native_compute"]').checked,
    };
  }
  return {
    ...base,
    python_path: section.querySelector('[data-field="python_path"]').value.trim() || null,
    main_py_path: section.querySelector('[data-field="main_py_path"]').value.trim() || null,
    extra_args: section.querySelector('[data-field="extra_args"]').value.trim() || null,
  };
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
  const [config, installDefaults] = await Promise.all([
    api(`/api/settings/${server}/config`),
    api(`/api/settings/${server}/install-defaults`),
  ]);
  section.querySelector('[data-field="mode"]').value = config.mode;
  renderInstallSource(section, config, installDefaults);

  if (server === "ollama") {
    const binaryPathInput = section.querySelector('[data-field="binary_path"]');
    binaryPathInput.value = config.binary_path ?? "";
    // Shows what leaving this blank would actually resolve to (PATH,
    // then the usual fixed locations) — a real, current value rather
    // than just the generic words "auto-detected from PATH".
    const autoDetected = await api("/api/settings/ollama/auto-detected-path");
    binaryPathInput.placeholder = autoDetected.path || "not found — install or enter a path";
    const modelsPathInput = section.querySelector('[data-field="models_path"]');
    modelsPathInput.value = config.models_path ?? "";
    modelsPathInput.placeholder = autoDetected.models_path;
    section.querySelector('[data-field="num_parallel"]').value = config.num_parallel ?? "";
    section.querySelector('[data-field="keep_alive"]').value = config.keep_alive ?? "";
    section.querySelector('[data-field="max_loaded_models"]').value = config.max_loaded_models ?? "";
    section.querySelector('[data-field="context_length"]').value = config.context_length ?? "";
  } else if (server === "matricxon") {
    const projectDirInput = section.querySelector('[data-field="project_dir"]');
    projectDirInput.value = config.project_dir ?? "";
    // Same "show what blank actually resolves to" idea as Ollama's binary_path above — here, the sibling
    // checkout app.services.matricxon_process._find_project_dir falls back to.
    const autoDetected = await api("/api/settings/matricxon/auto-detected-path");
    projectDirInput.placeholder = autoDetected.path || "not found — enter a path";
    const modelsPathInput = section.querySelector('[data-field="models_path"]');
    modelsPathInput.value = config.models_path ?? "";
    modelsPathInput.placeholder = autoDetected.models_path || "<project directory>/data/models";
    section.querySelector('[data-field="max_loaded_models"]').value = config.max_loaded_models ?? "";
    section.querySelector('[data-field="memory_safety_margin"]').value = config.memory_safety_margin ?? "";
    section.querySelector('[data-field="log_level"]').value = String(config.log_level ?? 0);
    section.querySelector('[data-field="enable_quantized_native_compute"]').checked =
      config.enable_quantized_native_compute ?? false;
  } else {
    section.querySelector('[data-field="python_path"]').value = config.python_path ?? "";
    section.querySelector('[data-field="main_py_path"]').value = config.main_py_path ?? "";
    section.querySelector('[data-field="extra_args"]').value = config.extra_args ?? "";
  }
  renderRemoteHosts(section, config.remote_hosts || []);
  renderServerStatus(section, await api(`/api/settings/${server}/status`));
}

/**
 * Grays out "Install from GitHub" and explains why, on any OS the auto-installer can't actually run on (see
 * app.hardware.local_install_supported — both Ollama's pinned release asset and this app's own deployment
 * assumptions are Linux-only). The button staying clickable would just mean the admin waits for a multi-minute
 * "attempt" that was always going to fail — app.services.ollama_installer/comfyui_installer's own install_stream
 * enforces this same check server-side regardless, so this is a courtesy, not the only guard.
 */
function applyLocalInstallSupport(server, section, supported) {
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

// Ollama and Matricxon are mutually exclusive chat backends (see the "Active engine" picker) — ComfyUI is a
// wholly separate concern (image generation) and always stays visible regardless of this choice.
const ENGINE_SERVERS = ["ollama", "matricxon"];

/**
 * Only the currently-selected engine's own configuration section is shown — the other one is fully hidden
 * rather than shown side-by-side, so an admin isn't looking at (and could accidentally edit) the form for a
 * backend that isn't even live right now. Re-run on load and every time the select's own "change" event fires.
 */
function applyActiveEngineVisibility() {
  const selectEl = document.getElementById("active-engine-select");
  if (!selectEl) return;
  for (const server of ENGINE_SERVERS) {
    const section = serverSectionEl(server);
    if (section) section.classList.toggle("hidden", server !== selectEl.value);
  }
}

// The engine actually saved right now (as of the last load/save) — compared against the select's own live value
// on every "change" so the switch note below only shows for a genuine *pending* switch, not just on page load.
let savedActiveEngine = null;

async function loadActiveEngineIntoEditor() {
  const selectEl = document.getElementById("active-engine-select");
  if (!selectEl) return;
  const { active_engine: activeEngine } = await api("/api/settings/engine");
  selectEl.value = activeEngine;
  savedActiveEngine = activeEngine;
  applyActiveEngineVisibility();
  applyActiveEngineSwitchNote();
}

/** Ollama and Matricxon each have their own separate model storage (see the External Servers page's own
 * explanation) — an admin switching engines has no way to know that just from this picker alone. Shown the
 * instant the selection actually changes to something other than what's currently saved (not after Save is
 * clicked — the point is to inform the decision, not just explain it after the fact), and hidden again if they
 * change it back to match. */
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
      // A full reload, not just a client-side refetch: the Model tab's own catalog (currentCatalog) and every
      // other model-related fetch on this page (installed vision/embedding models, RAG availability, ...) cache
      // themselves for the lifetime of this page load — switching engines here makes every one of those stale,
      // so this is the same "reload rather than try to selectively invalidate every cache" pattern the
      // Ollama/Matricxon Start button already uses right below for the identical reason. Confirmed live: without
      // this, the Model tab kept showing whichever engine's models were loaded *before* the switch until a hard
      // refresh. Lands on the Model tab itself (not back on System, where this picker lives) — that's the one
      // place an admin actually needs to look right after a switch, to see the new engine's own models and pick
      // a default (see model_catalog_service.compute_matricxon_support and the "no default selected yet" gap
      // this same switch can leave behind).
      try {
        sessionStorage.setItem(SETTINGS_TAB_STORAGE_KEY, "model");
      } catch (_err) {
        // Private-browsing/storage-blocked — reload still happens, just lands on the default tab.
      }
      window.location.reload();
      return;
    } catch (err) {
      statusEl.textContent = `Failed to save: ${err.message}`;
      setTimeout(() => (statusEl.textContent = ""), 2500);
    }
  });
}

/** Populates a server's version <datalist> with real tags fetched live from GitHub (see
 * app.services.github_releases — one shared endpoint per server, GET /api/settings/{server}/available-versions),
 * so the plain-text "Version / tag" field becomes pick-from-a-list-or-type-your-own instead of "guess the exact
 * tag spelling" — a typo there today only ever surfaces as a failed install several steps in. Always a
 * convenience layered on the existing free-text input, never a replacement: an empty/failed fetch (GitHub
 * unreachable, rate-limited, or the repo genuinely has no tags) just leaves the datalist empty, same as before
 * this existed. Re-run whenever the repo field changes (see its own "change" listener below) since a fork's
 * tags are a completely different list from the default repo's. */
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
    // Leaves whatever was already there (or nothing) — the plain text input still works regardless, see this
    // function's own docstring.
  }
}

for (const server of EXTERNAL_SERVERS) {
  const section = serverSectionEl(server);
  if (!section) continue;

  section.querySelector('[data-field="mode"]').addEventListener("change", () => {
    applyServerVisibility(server, section);
  });
  section.querySelector('[data-action="add-host"]').addEventListener("click", () => addRemoteHostRow(section));

  // Both servers (see settings.html) — an admin pointing at an already-
  // installed Ollama/ComfyUI of their own, not one this app's installer
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

  // Re-fetches once the admin finishes typing a different repo — a fork's own tags are a different list
  // entirely from the maintainer's default repo's tags (see loadAvailableVersions's own docstring).
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
        await loadServerSection(server); // refreshes the version label + placeholders
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
    statusEl.textContent = "Saving…";
    try {
      // Ollama's (and Matricxon's — see app.services.matricxon_process.apply_local_config) local-mode parameters
      // can't apply live: both only read their env-based config at their own process startup, so if either is
      // already running, saving must restart it — this confirms first, same as the old "Ollama concurrency" save
      // button already did.
      const restartableEngines = { ollama: "Ollama", matricxon: "Matricxon" };
      const engineLabel = restartableEngines[server];
      if (engineLabel && body.mode === "local" && (await api(`/api/settings/${server}/status`)).running) {
        const confirmed = confirm(
          `Saving restarts ${engineLabel} — every reply currently being generated, on every local ` +
            "instance, will be interrupted. Continue?",
        );
        if (!confirmed) {
          statusEl.textContent = "";
          return;
        }
        statusEl.textContent = `Restarting ${engineLabel}…`;
        renderServerStatus(
          section,
          await api(`/api/settings/${server}/apply`, { method: "POST", body: JSON.stringify(body) }),
        );
        statusEl.textContent = `Saved — ${engineLabel} restarted.`;
        return;
      }
      await api(`/api/settings/${server}/config`, { method: "PUT", body: JSON.stringify(body) });
      statusEl.textContent = "Saved.";
    } catch (err) {
      statusEl.textContent = `Failed to save: ${err.message}`;
    } finally {
      setTimeout(() => (statusEl.textContent = ""), 3000);
    }
  });

  for (const action of ["start", "stop"]) {
    section.querySelector(`[data-action="${action}"]`).addEventListener("click", async () => {
      const statusEl = section.querySelector('[data-field="status"]');
      statusEl.textContent = "Working…";
      try {
        const result = await api(`/api/settings/${server}/${action}`, { method: "POST" });
        renderServerStatus(section, result);
        statusEl.textContent = "Done.";
        // Ollama coming up can newly make models available for "Model &
        // behavior" elsewhere on this page — that section only ever
        // loads the model catalog once, at page load, so without this it
        // kept showing "System is not configured for models yet" even
        // once Ollama was genuinely running. A full reload is the
        // simplest reliable fix for an admin-triggered, infrequent action
        // like this (not the hot chat-generation path). Saved so the
        // reload lands back on External Servers instead of the default
        // "My Settings" tab — confirmed live: without this, the status
        // line here just went blank (this tab's own data only loads on
        // its own click) until manually re-clicked, which read as a
        // stuck/broken status rather than what it actually was. Same reload for Matricxon coming up, for the
        // identical reason — either one starting can newly make models available.
        if ((server === "ollama" || server === "matricxon") && action === "start" && result.running) {
          try {
            sessionStorage.setItem(SETTINGS_TAB_STORAGE_KEY, "external-servers");
          } catch (_err) {
            // Private-browsing/storage-blocked — reload still happens,
            // just lands on the default tab, same as before this fix.
          }
          window.location.reload();
          return;
        }
      } catch (err) {
        statusEl.textContent = `Failed: ${err.message}`;
      } finally {
        setTimeout(() => (statusEl.textContent = ""), 2500);
      }
    });
  }

  const installBtn = section.querySelector('[data-action="install"]');
  if (installBtn) {
    installBtn.addEventListener("click", async () => {
      // Reinstalling overwrites the managed install directory outright (see app.services.ollama_installer/
      // comfyui_installer/matricxon_installer's own shutil.rmtree on the existing target_dir) — doing that out
      // from under an actively-running process is exactly the kind of thing that leaves it in a broken half-
      // updated state, so this always stops it first if it's up, same confirm-before-interrupting convention
      // the Save button already uses for a local-mode restart.
      const status = lastServerStatus[server];
      const wasRunning = !!status?.running;
      if (wasRunning) {
        const engineLabel = { ollama: "Ollama", matricxon: "Matricxon", comfyui: "ComfyUI" }[server] || server;
        if (!confirm(`${engineLabel} is currently running — reinstalling stops it first. Continue?`)) return;
        installBtn.disabled = true;
        try {
          renderServerStatus(section, await api(`/api/settings/${server}/stop`, { method: "POST" }));
        } catch (err) {
          alert(`Could not stop ${engineLabel} first: ${err.message}`);
          installBtn.disabled = false;
          return;
        }
        installBtn.disabled = false;
      }
      // Only restart afterward if *this* click is the reason it's stopped — a first-time install (nothing was
      // running before) leaves it stopped on purpose, matching Ollama/ComfyUI's existing convention of a
      // separate, deliberate first Start click.
      await installServer(server, section, wasRunning);
    });
  }
}

/**
 * Downloads/installs Ollama or ComfyUI from GitHub (see
 * app.services.ollama_installer/comfyui_installer), streaming progress
 * over SSE the same way pullModel() above reads a model pull — manual
 * fetch + reader, not EventSource, since this is a POST. A real,
 * potentially many-minutes operation (Ollama's release is multi-GB;
 * ComfyUI's own pip install pulls torch) — every event the installer
 * itself yields (its step/total_steps/step_label, byte-level download
 * progress, or a raw git/pip stdout line) is shown live: a step label,
 * a progress bar (determinate once byte counts are known, indeterminate
 * otherwise), and a scrolling detail log of every raw status line — not
 * summarized away, so nothing about a multi-minute install happens
 * somewhere the admin can't see it. The mode <select> is disabled for
 * the duration so switching away mid-install can't desync the UI from
 * what's actually still running server-side.
 *
 * `restartAfter` (only ever true for a reinstall of something that was running — see the install button's own
 * click handler, which stops it first and passes this through) starts it back up once the install succeeds — a
 * real, confirmed-live gap without this: reinstalling left the admin with a freshly-updated but stopped engine
 * and no visible signal that a manual Start was now needed, reading as "stuck" until a page reload happened to
 * show the true (stopped) status. */
async function installServer(server, section, restartAfter = false) {
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

  try {
    const response = await fetch(`/api/settings/${server}/install`, { method: "POST" });
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
          percentEl.textContent = `${pct}%`;
        } else {
          barEl.removeAttribute("value");
          percentEl.textContent = "";
        }
        if (payload.status) appendLog(payload.status);
      }
    }

    stepLabelEl.textContent = "Installed.";
    percentEl.textContent = "100%";
    barEl.value = 100;
    // Ollama's own managed binary IS in ollama_process._FALLBACK_BIN_PATHS, so a *first-ever* install with no
    // override set already auto-detects it correctly with nothing saved here. The real gap this closes: an
    // admin who previously set an explicit binary_path override (_find_binary checks it "first and,
    // exclusively" if valid — see that function's own docstring) would have every future Reinstall silently
    // install a fresh binary that never actually gets used, since the stale override keeps winning and nothing
    // updated it — confirmed live: "the service looks ok" after reinstalling, because the OLD binary was still
    // what actually started. Saving the just-installed path here, same as ComfyUI/Matricxon below, is what
    // makes a reinstall actually take effect regardless of whatever was configured before it.
    if (server === "ollama" && doneEvent) {
      section.querySelector('[data-field="binary_path"]').value = doneEvent.binary_path;
      await api("/api/settings/ollama/config", {
        method: "PUT",
        body: JSON.stringify(collectServerConfig("ollama", section)),
      });
    }
    // ComfyUI's own install has no fixed path to auto-discover the way
    // Ollama's managed binary does (see ollama_process._FALLBACK_BIN_PATHS)
    // — the paths it just installed to are filled in and saved so
    // Start works immediately without the admin copying them by hand.
    // extra_args is "--cpu" on a machine with no NVIDIA GPU (see
    // comfyui_installer's own reasoning) — without it, ComfyUI's default
    // CUDA-enabled torch build crashes at startup on a GPU-less machine.
    if (server === "comfyui" && doneEvent) {
      section.querySelector('[data-field="python_path"]').value = doneEvent.python_path;
      section.querySelector('[data-field="main_py_path"]').value = doneEvent.main_py_path;
      if (doneEvent.extra_args) section.querySelector('[data-field="extra_args"]').value = doneEvent.extra_args;
      await api("/api/settings/comfyui/config", {
        method: "PUT",
        body: JSON.stringify(collectServerConfig("comfyui", section)),
      });
    }
    // Matricxon has no fixed path to auto-discover the way Ollama's managed binary does either — same reasoning
    // as ComfyUI just above, project_dir in place of python_path/main_py_path (see
    // app.services.matricxon_process._find_project_dir, which treats this saved value as an override ahead of
    // its own sibling-directory default).
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
        // Surfaced via the status line loadServerSection renders next, not thrown — the install itself
        // genuinely succeeded, so this must not be reported as an install failure (which would incorrectly
        // re-show the "Install from GitHub" idle panel via the catch block below).
        appendLog(`Installed, but could not start it back up: ${err.message}`);
      }
    }
    // A full reload of this section, not just a status re-fetch: Ollama's
    // own "auto-detected" binary_path placeholder (see loadServerSection)
    // only reflects the newly-installed binary once re-fetched — without
    // this it stayed blank/stale until the next unrelated action (e.g.
    // clicking Start) happened to trigger a reload some other way. Also
    // what makes a restartAfter start actually show up as "Running" instead of needing a manual refresh.
    await loadServerSection(server);
    // Swaps back to the idle Install/Reinstall button once there's nothing left to show progress for — left
    // showing "Installed. 100%" indefinitely otherwise, with no way to tell a finished install from one still
    // in progress at a glance days later.
    progressPanel.classList.add("hidden");
    installIdle.classList.remove("hidden");
  } catch (err) {
    stepLabelEl.textContent = `Failed: ${err.message}`;
    appendLog(err.message);
    // progressPanel stays visible here (unlike the success path above) — it's the only place the actual
    // failure reason and log are shown, so hiding it on error would bury exactly what the admin needs to see
    // to retry successfully.
    installIdle.classList.remove("hidden");
  } finally {
    modeSelect.disabled = false;
  }
}

// ---- Default model for new users (admin only) -----------------------------

async function loadDefaultModelForNewUsersEditor() {
  const selectEl = document.getElementById("default-model-for-new-users");
  const emptyEl = document.getElementById("default-model-for-new-users-empty");
  const saveBtn = document.getElementById("default-model-for-new-users-save-btn");
  if (!selectEl) return;

  await loadCatalog();
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

// Admin-only (see app/templates/settings.html) — this element and the
// rest of this block don't exist in the DOM at all for a regular user,
// same as the other admin-only sections guarded this way elsewhere in
// this file. The unguarded version of this call used to throw for
// every non-admin user on page load, which — since it ran at the top
// level, not inside a function — halted this whole script's execution
// right there, breaking every *other* tab too (including My Settings).
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

// ---- Default vision model (admin only) -------------------------------------
// The model any message with an image attached is answered by, for that
// one reply only (see app.services.chat_service.build_reply_stream) — a
// separate picker from "Default model for new users" above since a
// vision-capable model is a different kind of thing from an ordinary
// chat model (see model_catalog_service.installed_vision_models).

async function loadDefaultVisionModelEditor() {
  const selectEl = document.getElementById("default-vision-model");
  const emptyEl = document.getElementById("default-vision-model-empty");
  const saveBtn = document.getElementById("default-vision-model-save-btn");
  if (!selectEl) return;

  const { models } = await api("/api/settings/installed-vision-models");

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

// ---- Default embedding model (admin only) ---------------------------------
// The model app.services.inference_client.embed uses for the Knowledge base
// (RAG) feature — same select/empty-state/save pattern as "Default vision
// model" above, just against the embedding-capable model list instead.

async function loadDefaultEmbeddingModelEditor() {
  const selectEl = document.getElementById("default-embedding-model");
  const emptyEl = document.getElementById("default-embedding-model-empty");
  const saveBtn = document.getElementById("default-embedding-model-save-btn");
  if (!selectEl) return;

  const { models } = await api("/api/settings/installed-embedding-models");

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

// ---- Account (everyone) ---------------------------------------------------

// Profile: name + picture (see app/routers/account.py). `has_avatar` lets
// the "Remove" button show/hide without re-deriving it from avatar_url
// itself, matching the has_password precedent elsewhere on this page.
async function loadAccountProfileIntoEditor() {
  const profile = await api("/api/account/profile");
  document.getElementById("account-first-name").value = profile.first_name || "";
  document.getElementById("account-last-name").value = profile.last_name || "";
  renderAvatar(document.getElementById("account-avatar-preview"), profile.avatar_url, profile.initials);
  document.getElementById("account-avatar-remove-btn").classList.toggle("hidden", !profile.avatar_url);
}

document.getElementById("account-avatar-input").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = ""; // lets picking the exact same file again still fire "change"
  if (!file) return;

  const statusEl = document.getElementById("account-avatar-status");
  statusEl.textContent = "Uploading…";
  statusEl.classList.remove("text-red-400");
  try {
    const formData = new FormData();
    formData.append("file", file);
    // No JSON Content-Type here (see api()'s default) — a raw fetch
    // instead, same reasoning as chat.js's attachment upload: the
    // browser must set the multipart boundary itself.
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

document.getElementById("account-avatar-remove-btn").addEventListener("click", async () => {
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

document.getElementById("account-profile-save-btn").addEventListener("click", async () => {
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

document.getElementById("account-password-toggle").addEventListener("click", () => {
  const form = document.getElementById("account-password-form");
  const toggle = document.getElementById("account-password-toggle");
  const opening = form.classList.contains("hidden");
  form.classList.toggle("hidden");
  toggle.textContent = opening ? "Cancel" : "Update password";
});

document.getElementById("account-password-save-btn").addEventListener("click", async () => {
  const currentInput = document.getElementById("account-current-password");
  const newInput = document.getElementById("account-new-password");
  const confirmInput = document.getElementById("account-confirm-password");
  const statusEl = document.getElementById("account-password-status");
  const btn = document.getElementById("account-password-save-btn");

  const current_password = currentInput.value;
  const new_password = newInput.value;
  const confirm = confirmInput.value;

  statusEl.classList.remove("text-amber-400");

  if (!current_password || !new_password) {
    statusEl.textContent = "Fill in both your current and new password.";
    statusEl.classList.add("text-amber-400");
    return;
  }
  // Mirrors app/schemas.py's ChangePasswordRequest constraint, so a
  // too-short password shows a plain message here instead of a raw
  // FastAPI validation error.
  if (new_password.length < 3 || new_password.length > 200) {
    statusEl.textContent = "New password must be 3-200 characters.";
    statusEl.classList.add("text-amber-400");
    return;
  }
  if (new_password !== confirm) {
    statusEl.textContent = "New password and confirmation don't match.";
    statusEl.classList.add("text-amber-400");
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
    statusEl.classList.add("text-amber-400");
  } finally {
    btn.disabled = false;
    setTimeout(() => {
      statusEl.textContent = "";
      statusEl.classList.remove("text-amber-400");
    }, 3000);
  }
});

// ---- App-wide UI theme (see app/static/js/theme.js's ThemeManager) ----
// Clicking a swatch live-previews instantly — repaints the whole page (no network call, no reload) via
// ThemeManager.apply(); only "Save theme" persists it. Same picker (and the same ThemeManager class) also lives
// in the account panel reachable from every other page — see app/static/js/account_panel.js.
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

document.getElementById("ui-theme-save-btn").addEventListener("click", async () => {
  const statusEl = document.getElementById("ui-theme-status");
  const btn = document.getElementById("ui-theme-save-btn");

  btn.disabled = true;
  statusEl.textContent = "Saving…";
  try {
    await themeManager.save(pendingUiTheme);
    statusEl.textContent = "Saved";
  } catch (err) {
    statusEl.textContent = err.message;
    statusEl.classList.add("text-amber-400");
  } finally {
    btn.disabled = false;
    setTimeout(() => {
      statusEl.textContent = "";
      statusEl.classList.remove("text-amber-400");
    }, 3000);
  }
});

// ---- RAG upload limits (admin only — these elements don't exist in the
// DOM at all for a non-admin user, same as the rest of the System tab) ----

const ragLimitsSaveBtn = document.getElementById("rag-limits-save-btn");

async function loadRagLimitsIntoEditor() {
  if (!ragLimitsSaveBtn) return;
  const limits = await api("/api/settings/rag-limits");
  document.getElementById("rag-limit-max-file").value = limits.max_file_mb;
  document.getElementById("rag-limit-max-space").value = limits.max_user_space_mb;
}

if (ragLimitsSaveBtn) {
  ragLimitsSaveBtn.addEventListener("click", async () => {
    const statusEl = document.getElementById("rag-limits-status");
    const max_file_mb = parseFloat(document.getElementById("rag-limit-max-file").value);
    const max_user_space_mb = parseFloat(document.getElementById("rag-limit-max-space").value);
    if (!(max_file_mb > 0) || !(max_user_space_mb > 0)) {
      statusEl.textContent = "Both limits must be positive numbers.";
      return;
    }

    ragLimitsSaveBtn.disabled = true;
    statusEl.textContent = "Saving…";
    try {
      await api("/api/settings/rag-limits", {
        method: "PUT",
        body: JSON.stringify({ max_file_mb, max_user_space_mb }),
      });
      statusEl.textContent = "Saved.";
      currentUploadLimits = null; // stale — the Knowledge tab will refetch next time it loads
    } catch (err) {
      statusEl.textContent = `Failed to save: ${err.message}`;
    } finally {
      ragLimitsSaveBtn.disabled = false;
      setTimeout(() => (statusEl.textContent = ""), 2500);
    }
  });
}

// ---- Chat title mode (admin only) -----------------------------------

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

// ---- Channel reply delivery mode (read: any user; save: admin only) ------
// See app.services.chat_settings_service.get_channel_delivery_mode/
// set_channel_delivery_mode and app/static/js/chat.js, which reads the
// same GET endpoint to decide how to behave in a channel chat.

const channelDeliveryModeSaveBtn = document.getElementById("channel-delivery-mode-save-btn");

async function loadChannelDeliveryModeIntoEditor() {
  if (!channelDeliveryModeSaveBtn) return;
  const { mode } = await api("/api/settings/channel-delivery-mode");
  document.getElementById("channel-delivery-mode").value = mode;
}

if (channelDeliveryModeSaveBtn) {
  channelDeliveryModeSaveBtn.addEventListener("click", async () => {
    const statusEl = document.getElementById("channel-delivery-mode-status");
    const mode = document.getElementById("channel-delivery-mode").value;

    channelDeliveryModeSaveBtn.disabled = true;
    statusEl.textContent = "Saving…";
    try {
      await api("/api/settings/channel-delivery-mode", { method: "PUT", body: JSON.stringify({ mode }) });
      statusEl.textContent = "Saved.";
    } catch (err) {
      statusEl.textContent = `Failed to save: ${err.message}`;
    } finally {
      channelDeliveryModeSaveBtn.disabled = false;
      setTimeout(() => (statusEl.textContent = ""), 2500);
    }
  });
}

// ---- Reply timeouts (admin only) -------------------------------------------
// See app.services.chat_settings_service.get_reply_timeout_seconds/
// get_vision_reply_timeout_seconds and app.services.reply_generation_service,
// the only reader of either — no frontend code branches on them, unlike
// channel delivery mode above. Two independent settings, same {timeout_seconds}
// shape and save flow, so one small helper wires both instead of duplicating it.

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

// ---- Data retention (admin only) ------------------------------------------
// See app.services.retention_settings_service and app.services.retention_poller,
// the only reader of this — no frontend code outside this section branches
// on it, same as reply timeout above.

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

// ---- Users (admin only) --------------------------------------------------

// Whoever's logged in right now — set server-side as a data attribute on
// <body> (see app/templates/base.html) — just so their own row in the
// list below can be marked "(you)" rather than looking like any other.
const currentUsername = document.body.dataset.username || "";

// Mirrors app/schemas.py's UserCreate/UserUpdate constraints, so a bad
// username/password shows a plain-English message here instead of the
// raw list-of-field-errors FastAPI's own 422 response would otherwise
// surface through the generic api() error handling.
const USERNAME_RE = /^[A-Za-z0-9_.-]{3,50}$/;
function validUsername(value) {
  return USERNAME_RE.test(value);
}
function validPassword(value) {
  return value.length >= 3 && value.length <= 200;
}

const userListEl = document.getElementById("system-user-list");
const addUserBtn = document.getElementById("add-user-btn");
const addUserForm = document.getElementById("add-user-form");
const addUserStatusEl = document.getElementById("add-user-status");
const createUserBtn = document.getElementById("create-user-btn");

/** Renders one row in the Users list. Starts in a compact "view" state
 * (username, role, status, an Edit button); clicking Edit swaps the
 * same row into an inline form (password/role/status) rather than
 * opening a separate modal — there's no modal machinery elsewhere in
 * this app, so this matches everything else here. */
function renderUserRow(user) {
  const row = document.createElement("div");
  row.className = "px-4 py-2.5";

  function renderView() {
    row.innerHTML = "";
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

    // Right in the main list — no need to open Edit first. Permanent
    // and cascading (see DELETE /api/settings/users/{username}): this
    // removes the account's conversations, documents, and notes along
    // with it, with no undo. Disabling (inside Edit) is the reversible
    // way to just revoke access instead. Not rendered at all for
    // "admin" (see the ternary above) — that account can't be deleted
    // (app.routers.settings.delete_user enforces this server-side
    // regardless), so there's no point offering a button that can only
    // ever fail.
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
      // first_name/last_name are always sent (even as "") so clearing a
      // name actually clears it — see user_service.update_user's "or
      // None" handling, which relies on receiving "" rather than the
      // field being left out entirely to tell "cleared" apart from
      // "untouched".
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

// How many users to show per page — the full list is still fetched in
// one request (this app's scale doesn't need server-side paging, same
// reasoning as MY_DOCS_PAGE_SIZE above), just not all rendered at once.
const USER_PAGE_SIZE = 20;
let allUsers = [];
let userPage = 1;

function renderUserPage() {
  const paginationEl = document.getElementById("system-user-pagination");
  const totalPages = Math.max(1, Math.ceil(allUsers.length / USER_PAGE_SIZE));
  // Clamped rather than trusted as-is: a delete can shrink the list
  // between renders, which could otherwise leave the page number
  // pointing past the new last page.
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

// Guarded like every other admin-only element in this file (userListEl
// is null for a non-admin, since the whole Users tab isn't rendered for
// them) — these two were missed when pagination was added, which meant
// a non-admin's page crashed here with "Cannot read properties of null",
// silently killing every script line after this one (the rest of this
// file, including the Add User form wiring below).
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

/** Disables "+ Add user" (with an explanatory note) when no model is
 * installed at all — matches the server-side guard in
 * POST /api/settings/users, which would otherwise 400 on submit; this
 * just surfaces the same reason before the form is even opened. */
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
  addUserBtn.addEventListener("click", () => {
    addUserForm.classList.toggle("hidden");
    if (!addUserForm.classList.contains("hidden")) {
      resetAddUserForm();
      document.getElementById("new-user-username").focus();
    }
  });

  document.getElementById("cancel-add-user-btn").addEventListener("click", () => {
    addUserForm.classList.add("hidden");
  });

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
      addUserForm.classList.add("hidden");
      // list_users orders by created_at, so the user just created is always on the *last* page — jump there
      // (renderUserPage's own clamp settles this to whatever that turns out to be) rather than silently
      // re-fetching behind whichever page happened to be showing, which could leave it looking like nothing
      // happened at all.
      userPage = Number.MAX_SAFE_INTEGER;
      await loadUsers();
    } catch (err) {
      addUserStatusEl.textContent = err.message;
    } finally {
      createUserBtn.disabled = false;
    }
  });
}

// ---- Channels (admin only — these elements don't exist in the DOM at
// all for a non-admin user, same as System/Users above) --------------
//
// The member/manager picker below is what makes channel creation usable
// with 1000+ users: rather than a fixed-height <select multiple> (which
// becomes an unscrollable wall of options at that size) or fetching
// pages of users from the server (this app's established convention —
// see USER_PAGE_SIZE/MY_DOCS_PAGE_SIZE above — is to fetch the full
// list once and do everything else client-side), it's a search box over
// the already-fetched user list plus a capped, always-shows-your-
// current-selection checkbox list. See buildChannelForm below.

// How many *unselected* matching rows to paint into the member picker
// at once — the thing that actually breaks at 1000+ users is rendering
// every row simultaneously, not the lack of a fancier widget. Already-
// selected members are always shown in full regardless of this cap, so
// narrowing the search never hides what's already picked.
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

/** Builds the name + member/manager picker form shared by "+ Add
 * channel" and each row's inline Edit (mirrors the Users tab's
 * renderView/renderEdit row-swap pattern) — `channel` is null when
 * creating. `onSave(body)` receives a {name, member_user_ids,
 * manager_user_ids} object shaped for POST/PATCH
 * /api/settings/channels; `onCancel()` closes the form unsaved. */
function buildChannelForm(users, channel, { onSave, onCancel }) {
  const selectedMemberIds = new Set((channel?.members || []).map((m) => m.user_id));
  const selectedManagerIds = new Set((channel?.members || []).filter((m) => m.is_manager).map((m) => m.user_id));

  // No border/bg of its own — the two callers below already sit inside one: #add-channel-form carries it
  // directly in settings.html (for the "add" flow, which has no other frame around it), and the "edit in place"
  // flow appends this straight into a channel row already inside #channel-list's own bordered box, where a
  // second border here would nest visibly inside the first.
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
    // Selected members first, in full, so narrowing the search can never
    // make an already-picked member disappear from view; the rest of
    // the matches are capped (see MEMBER_PICKER_RENDER_LIMIT above).
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
          selectedManagerIds.delete(u.id); // can't manage a channel you're not a member of
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
    // Every channel_manager-role user is listed here regardless of
    // whether they're currently a member — checking one below binds
    // them to the channel automatically (member_user_ids must be a
    // superset of manager_user_ids, both here and server-side — see
    // channel_service._validate_membership), rather than requiring a
    // separate trip to the Members list first. Unchecking a manager,
    // by contrast, only revokes their manager rights and leaves their
    // membership alone — turning a manager back into a plain member,
    // not removing them from the channel.
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

/** One channel's row: a compact view (name, member/manager counts, Edit/
 * Delete) that swaps to buildChannelForm's inline editor on Edit — same
 * row-swap pattern as renderUserRow above. Always fetches a fresh user
 * list for the picker (not the Users tab's cached `allUsers`) so a role
 * change made moments ago on the Users tab (e.g. granting someone
 * Channel manager) is reflected immediately rather than needing a page
 * reload. */
function renderChannelRow(channel) {
  const row = document.createElement("div");
  row.className = "px-4 py-2.5";

  function renderView() {
    row.innerHTML = "";
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
        <button type="button" class="delete-channel-btn rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-400 hover:border-red-500 hover:text-red-400 transition-colors">Delete</button>
      </span>
    `;
    wrap.querySelector(".edit-channel-btn").addEventListener("click", async () => {
      const users = await api("/api/settings/users");
      row.innerHTML = "";
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
    // Acts as a toggle: clicking again while the form's open closes it
    // unsaved, same as there being no separate "Cancel" affordance next
    // to the button itself (the form's own Cancel button covers that).
    if (!addChannelFormEl.classList.contains("hidden")) {
      addChannelFormEl.classList.add("hidden");
      addChannelFormEl.innerHTML = "";
      return;
    }
    const users = await api("/api/settings/users");
    addChannelFormEl.innerHTML = "";
    addChannelFormEl.appendChild(buildChannelForm(users, null, {
      onCancel: () => {
        addChannelFormEl.classList.add("hidden");
        addChannelFormEl.innerHTML = "";
      },
      onSave: async (body) => {
        await api("/api/settings/channels", { method: "POST", body: JSON.stringify(body) });
        addChannelFormEl.classList.add("hidden");
        addChannelFormEl.innerHTML = "";
        await loadChannels();
      },
    }));
    addChannelFormEl.classList.remove("hidden");
  });
}

// ---- Boot -------------------------------------------------------------

(async function init() {
  // Each step runs independently — a failure in one (e.g. Ollama being
  // unreachable breaks populateConversationSelect's own model-catalog
  // fetch) must not silently prevent the unrelated ones after it from
  // ever running, the way a single sequential await chain would.
  for (const step of [initModelTab, populateConversationSelect, loadRagAvailability, loadEmbeddingCatalog, loadKnowledgeSummary, loadDocumentLists]) {
    try {
      await step();
    } catch (err) {
      console.error("Settings page init step failed:", err);
    }
  }
})();

// Restores the tab a reload was asked to land back on (see
// SETTINGS_TAB_STORAGE_KEY's own comment) — placed at the very end of
// the file, after every function/const a restored tab's own loaders
// might reach (e.g. loadTabData("external-servers") needs
// EXTERNAL_SERVERS, declared earlier but still after this file's
// halfway point) has actually been defined.
(function restoreTabAfterReload() {
  let savedTab = null;
  try {
    savedTab = sessionStorage.getItem(SETTINGS_TAB_STORAGE_KEY);
    sessionStorage.removeItem(SETTINGS_TAB_STORAGE_KEY);
  } catch (_err) {
    // Private-browsing/storage-blocked — falls back to the default tab,
    // same as if nothing had been saved.
  }
  if (savedTab && document.querySelector(`[data-tab="${savedTab}"]`)) {
    activateTab(savedTab);
    loadTabData(savedTab);
  }
})();
