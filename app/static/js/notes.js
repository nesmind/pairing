/**
 * Notes page controller: list/create/edit/delete the current user's
 * notes, and — on the same Create/Save action, not a separate step —
 * choose every conversation the note should be pinned to and under
 * which role (see app/notes.py + app/routers/chat.py for how a pin
 * actually reaches the model). Follows the same inline view-state/
 * edit-state row pattern as the Users tab in settings.js — there's no
 * modal machinery anywhere else in this app.
 */

const PIN_TYPE_LABELS = { persona: "🎭 Persona", rules: "📏 Rules", skill: "🛠️ Skill" };

const notesListEl = document.getElementById("notes-list");
const notesEmptyEl = document.getElementById("notes-empty");
const notesSearchEl = document.getElementById("notes-search");
const addNoteBtn = document.getElementById("add-note-btn");
const newNoteFormEl = document.getElementById("new-note-form");
const addNoteStatusEl = document.getElementById("add-note-status");

// Same "fetch the whole list once, page/filter it client-side" pattern as
// settings.js's USER_PAGE_SIZE/MY_DOCS_PAGE_SIZE — this app's scale
// doesn't need a server-side ?page=/?q= endpoint, and there's no
// precedent for one anywhere else in the codebase.
const NOTES_PAGE_SIZE = 20;
let notesCache = [];
let notesPage = 1;
let notesSearchQuery = "";
let conversationsCache = null; // lazily loaded — only needed once a note editor is open

async function loadConversations() {
  if (!conversationsCache) {
    conversationsCache = await api("/api/conversations");
  }
  return conversationsCache;
}

function formatUpdatedAt(isoString) {
  return new Date(isoString).toLocaleDateString([], { month: "short", day: "numeric" });
}

/** Read-only pin chips shown in a note's view state — "what's this note
 * currently affecting" scannable without opening the editor. */
function renderPinChips(note) {
  if (note.pins.length === 0) {
    return `<span class="text-xs text-slate-500">Not pinned to any chat</span>`;
  }
  return note.pins
    .map((pin) => {
      const label = `${PIN_TYPE_LABELS[pin.pin_type] || pin.pin_type} → ${escapeHtml(pin.conversation_title)}`;
      return `<span class="inline-flex items-center rounded-full bg-slate-800 px-2 py-0.5 text-xs text-slate-300">${label}</span>`;
    })
    .join(" ");
}

/**
 * One checkbox+type-select row per conversation the user has — shared
 * by both the "+ New note" form and each note's edit form, so pinning
 * works identically (and to any number of chats at once) whether the
 * note is brand new or already exists. A checked row's type select
 * saves along with everything else on the surrounding Create/Save
 * click; nothing here calls the API by itself.
 */
function renderPinPickerRows(conversations, pinsByConversationId) {
  if (conversations.length === 0) {
    return `<p class="text-xs text-slate-500">Start a chat first — there's nothing to pin to yet.</p>`;
  }
  return conversations
    .map((conversation) => {
      const existing = pinsByConversationId.get(conversation.id);
      const checked = !!existing;
      const type = existing ? existing.pin_type : "persona";
      return `
        <label class="flex items-center gap-2 py-0.5">
          <input type="checkbox" class="pin-picker-checkbox" data-conv-id="${conversation.id}" ${checked ? "checked" : ""}>
          <span dir="auto" class="flex-1 min-w-0 truncate text-xs text-slate-300">${escapeHtml(conversation.title || "New chat")}</span>
          <select class="pin-picker-type rounded-md border border-slate-700 bg-slate-900 px-1.5 py-1 text-xs focus:outline-none focus:border-brand-500"
            data-conv-id="${conversation.id}" ${checked ? "" : "disabled"}>
            <option value="persona" ${type === "persona" ? "selected" : ""}>Persona</option>
            <option value="rules" ${type === "rules" ? "selected" : ""}>Rules</option>
            <option value="skill" ${type === "skill" ? "selected" : ""}>Skill</option>
          </select>
        </label>`;
    })
    .join("");
}

/** Wires up each checkbox to enable/disable its own type select —
 * pure UI state, no API calls (see renderPinPickerRows above). */
function attachPinPickerHandlers(containerEl) {
  containerEl.querySelectorAll(".pin-picker-checkbox").forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      const select = containerEl.querySelector(`.pin-picker-type[data-conv-id="${checkbox.dataset.convId}"]`);
      if (select) select.disabled = !checkbox.checked;
    });
  });
}

/** Reads the picker back into the `pins` shape NoteCreate/NoteUpdate
 * expect — every checked conversation, with its selected type. */
function collectPins(containerEl) {
  const pins = [];
  containerEl.querySelectorAll(".pin-picker-checkbox").forEach((checkbox) => {
    if (!checkbox.checked) return;
    const select = containerEl.querySelector(`.pin-picker-type[data-conv-id="${checkbox.dataset.convId}"]`);
    pins.push({ conversation_id: checkbox.dataset.convId, pin_type: select.value });
  });
  return pins;
}

function renderNoteRow(note) {
  const row = document.createElement("div");
  row.className = "rounded-lg bg-slate-900 border border-slate-800 px-4 py-3 space-y-2";

  function renderView() {
    row.innerHTML = "";
    const top = document.createElement("div");
    top.className = "flex items-start justify-between gap-3";
    top.innerHTML = `
      <div class="min-w-0">
        <p dir="auto" class="truncate text-sm font-medium text-slate-100">
          ${escapeHtml(note.title)}
          ${note.is_default ? '<span class="ml-1.5 rounded-full bg-sky-400 px-1.5 py-0.5 text-xs font-semibold text-slate-900 align-middle">Default</span>' : ""}
        </p>
        <p class="text-xs text-slate-500 mt-0.5">Updated ${formatUpdatedAt(note.updated_at)}</p>
      </div>
      <span class="shrink-0 flex items-center gap-2">
        <button type="button" class="edit-note-btn rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800 transition-colors">Edit</button>
        ${
          note.is_default
            ? ""
            : '<button type="button" class="delete-note-btn rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-400 hover:border-red-500 hover:text-red-400 transition-colors">Delete</button>'
        }
      </span>
    `;
    const pinsRow = document.createElement("div");
    pinsRow.className = "flex flex-wrap gap-1.5";
    pinsRow.innerHTML = renderPinChips(note);

    top.querySelector(".edit-note-btn").addEventListener("click", renderEdit);
    const deleteBtn = top.querySelector(".delete-note-btn");
    if (deleteBtn) {
      deleteBtn.addEventListener("click", async () => {
        if (!confirm(`Delete "${note.title}"?`)) return;
        try {
          await api(`/api/notes/${note.id}`, { method: "DELETE" });
          await loadNotes();
        } catch (err) {
          alert(`Failed to delete: ${err.message}`);
        }
      });
    }

    row.appendChild(top);
    row.appendChild(pinsRow);
  }

  async function renderEdit() {
    row.innerHTML = `<p class="text-xs text-slate-500">Loading…</p>`;
    const conversations = await loadConversations();
    const pinsByConversationId = new Map(note.pins.map((pin) => [pin.conversation_id, pin]));

    row.innerHTML = "";
    const form = document.createElement("div");
    form.className = "space-y-3";
    form.innerHTML = `
      <input class="edit-title w-full rounded-md border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm focus:outline-none focus:border-brand-500"
        type="text" dir="auto" value="${escapeHtml(note.title)}">
      <textarea class="edit-content w-full resize-y rounded-md border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm focus:outline-none focus:border-brand-500"
        rows="5" dir="auto">${escapeHtml(note.content)}</textarea>

      <div class="rounded-lg bg-slate-950 border border-slate-800 px-3 py-2.5 space-y-1">
        <p class="text-xs font-semibold uppercase tracking-wide text-slate-400">Pin to chats</p>
        <div class="pin-picker space-y-0.5 max-h-40 overflow-y-auto">${renderPinPickerRows(conversations, pinsByConversationId)}</div>
      </div>

      <div class="flex items-center gap-2">
        <button type="button" class="save-note-btn rounded-md bg-brand-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-brand-500 transition-colors">Save</button>
        <button type="button" class="cancel-note-btn rounded-md border border-slate-700 px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-800 transition-colors">Cancel</button>
        <span class="save-note-status text-xs text-slate-500"></span>
      </div>
    `;
    row.appendChild(form);
    attachPinPickerHandlers(form.querySelector(".pin-picker"));

    form.querySelector(".cancel-note-btn").addEventListener("click", renderView);
    form.querySelector(".save-note-btn").addEventListener("click", async () => {
      const title = form.querySelector(".edit-title").value.trim() || "Untitled note";
      const content = form.querySelector(".edit-content").value;
      const pins = collectPins(form.querySelector(".pin-picker"));
      const statusEl = form.querySelector(".save-note-status");
      statusEl.textContent = "Saving…";
      try {
        const updated = await api(`/api/notes/${note.id}`, {
          method: "PATCH", body: JSON.stringify({ title, content, pins }),
        });
        Object.assign(note, updated);
        renderView();
      } catch (err) {
        statusEl.textContent = err.message;
      }
    });
  }

  renderView();
  return row;
}

/** notesCache filtered by the current search box, title or content,
 * case-insensitive substring match — same plain `.filter()` approach as
 * settings.js's channel member-search box, the only other search
 * anywhere in this app (no debounce there either: it's an in-memory
 * array, not a network call per keystroke). Defaults-first ordering
 * (see loadNotes) is preserved since this only filters, never re-sorts. */
function filteredNotes() {
  const query = notesSearchQuery.trim().toLowerCase();
  if (!query) return notesCache;
  return notesCache.filter(
    (note) => note.title.toLowerCase().includes(query) || note.content.toLowerCase().includes(query)
  );
}

/** Renders one page of (possibly search-filtered) notes — mirrors
 * settings.js's renderUserPage/renderMyDocsPage exactly: clamp the page
 * number (a delete, or a new search, can shrink the list out from under
 * it), slice, render, then show/hide the Prev/Next controls. */
function renderNotesPage() {
  const notes = filteredNotes();
  const paginationEl = document.getElementById("notes-pagination");
  const totalPages = Math.max(1, Math.ceil(notes.length / NOTES_PAGE_SIZE));
  notesPage = Math.min(Math.max(1, notesPage), totalPages);

  const start = (notesPage - 1) * NOTES_PAGE_SIZE;
  const pageItems = notes.slice(start, start + NOTES_PAGE_SIZE);

  notesListEl.innerHTML = "";
  for (const note of pageItems) {
    notesListEl.appendChild(renderNoteRow(note));
  }

  // Two different reasons the list could be empty — "you have no notes
  // at all" (create one above) vs "none of your notes match this
  // search" — need different copy so the first one doesn't wrongly
  // suggest there's nothing here when there's just nothing *matching*.
  notesEmptyEl.classList.toggle("hidden", notes.length > 0);
  notesEmptyEl.textContent =
    notesCache.length === 0 ? "No notes yet — create one above." : "No notes match your search.";

  paginationEl.classList.toggle("hidden", notes.length <= NOTES_PAGE_SIZE);
  document.getElementById("notes-page-info").textContent = `Page ${notesPage} of ${totalPages}`;
  document.getElementById("notes-prev-btn").disabled = notesPage <= 1;
  document.getElementById("notes-next-btn").disabled = notesPage >= totalPages;
}

document.getElementById("notes-prev-btn").addEventListener("click", () => {
  notesPage -= 1;
  renderNotesPage();
});
document.getElementById("notes-next-btn").addEventListener("click", () => {
  notesPage += 1;
  renderNotesPage();
});
notesSearchEl.addEventListener("input", () => {
  notesSearchQuery = notesSearchEl.value;
  notesPage = 1; // a new search is a new result set — always start from its own page 1
  renderNotesPage();
});

async function loadNotes() {
  notesCache = await api("/api/notes");
  // Defaults first (stable sort keeps each group's own updated-at-desc order).
  notesCache.sort((a, b) => Number(b.is_default) - Number(a.is_default));
  renderNotesPage();
}

function resetNewNoteForm() {
  document.getElementById("new-note-title").value = "";
  document.getElementById("new-note-content").value = "";
  addNoteStatusEl.textContent = "";
}

addNoteBtn.addEventListener("click", async () => {
  newNoteFormEl.classList.toggle("hidden");
  if (!newNoteFormEl.classList.contains("hidden")) {
    resetNewNoteForm();
    const pickerEl = document.getElementById("new-note-pin-picker");
    pickerEl.innerHTML = `<p class="text-xs text-slate-500">Loading…</p>`;
    const conversations = await loadConversations();
    pickerEl.innerHTML = renderPinPickerRows(conversations, new Map());
    attachPinPickerHandlers(pickerEl);
    document.getElementById("new-note-title").focus();
  }
});

document.getElementById("cancel-add-note-btn").addEventListener("click", () => {
  newNoteFormEl.classList.add("hidden");
});

document.getElementById("create-note-btn").addEventListener("click", async () => {
  const title = document.getElementById("new-note-title").value.trim() || "Untitled note";
  const content = document.getElementById("new-note-content").value;
  const pins = collectPins(document.getElementById("new-note-pin-picker"));
  const btn = document.getElementById("create-note-btn");

  btn.disabled = true;
  addNoteStatusEl.textContent = "Creating…";
  try {
    await api("/api/notes", { method: "POST", body: JSON.stringify({ title, content, pins }) });
    newNoteFormEl.classList.add("hidden");
    await loadNotes();
  } catch (err) {
    addNoteStatusEl.textContent = err.message;
  } finally {
    btn.disabled = false;
  }
});

loadNotes();
