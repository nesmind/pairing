/* Notes page controller: list/create/edit/delete the user's notes and, in the same Create/Save action, choose which conversations the note is pinned to and under which role (see app/notes.py + app/routers/chat.py for how a pin reaches the model). Uses the inline view/edit row pattern of settings.js's Users tab; the app has no modals. */

const PIN_TYPE_LABELS = { persona: "🎭 Persona", rules: "📏 Rules", skill: "🛠️ Skill" };

const notesListEl = document.getElementById("notes-list");
const notesEmptyEl = document.getElementById("notes-empty");
const notesSearchEl = document.getElementById("notes-search");
const addNoteBtn = document.getElementById("add-note-btn");
const newNoteFormEl = document.getElementById("new-note-form");
const addNoteStatusEl = document.getElementById("add-note-status");

// Fetch the whole list once, then page/filter client-side (as settings.js's USER_PAGE_SIZE/MY_DOCS_PAGE_SIZE); at this
// scale no server-side ?page=/?q= is needed.
const NOTES_PAGE_SIZE = 20;
let notesCache = [];
let notesPage = 1;
let notesSearchQuery = "";
let conversationsCache = null;

async function loadConversations() {
  if (!conversationsCache) {
    conversationsCache = await api("/api/conversations");
  }
  return conversationsCache;
}

function formatUpdatedAt(isoString) {
  return new Date(isoString).toLocaleDateString([], { month: "short", day: "numeric" });
}

/* Read-only pin chips in a note's view state: shows what the note affects without opening the editor. */
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

/* One checkbox + type-select row per conversation, shared by the "+ New note" form and each note's edit form, so pinning works the same for new and existing notes and any number of chats. A checked row's type saves with the surrounding Create/Save click; nothing here calls the API. */
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

/* Wires each checkbox to enable/disable its type select (UI state only; see renderPinPickerRows). */
function attachPinPickerHandlers(containerEl) {
  containerEl.querySelectorAll(".pin-picker-checkbox").forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      const select = containerEl.querySelector(`.pin-picker-type[data-conv-id="${checkbox.dataset.convId}"]`);
      if (select) select.disabled = !checkbox.checked;
    });
  });
}

/* Reads the picker into the `pins` shape NoteCreate/NoteUpdate expect: every checked conversation with its type. */
function collectPins(containerEl) {
  const pins = [];
  containerEl.querySelectorAll(".pin-picker-checkbox").forEach((checkbox) => {
    if (!checkbox.checked) return;
    const select = containerEl.querySelector(`.pin-picker-type[data-conv-id="${checkbox.dataset.convId}"]`);
    pins.push({ conversation_id: checkbox.dataset.convId, pin_type: select.value });
  });
  return pins;
}

/* While a note is being edited (or the new-note form is open), shows only it: hides the other rows, search box, pagination, empty-state text and "+ New note". `row` null restores everything. */
function focusNoteRow(row) {
  for (const child of notesListEl.children) child.style.display = row && child !== row ? "none" : "";
  const chrome = [addNoteBtn, notesSearchEl, notesEmptyEl, document.getElementById("notes-pagination")];
  for (const el of chrome) el.style.display = row ? "none" : "";
}

function renderNoteRow(note) {
  const row = document.createElement("div");
  row.className = "rounded-lg bg-slate-900 border border-slate-800 px-4 py-3 space-y-2";

  function renderView() {
    row.innerHTML = "";
    if (row.parentElement) focusNoteRow(null);
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
    focusNoteRow(row);
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

/* notesCache filtered by the search box (title or content, case-insensitive substring), a plain `.filter` as settings.js's channel member search (no debounce: it is an in-memory array). Defaults-first order (loadNotes) is kept since this never re-sorts. */
function filteredNotes() {
  const query = notesSearchQuery.trim().toLowerCase();
  if (!query) return notesCache;
  return notesCache.filter(
    (note) => note.title.toLowerCase().includes(query) || note.content.toLowerCase().includes(query)
  );
}

/* Renders one page of the (filtered) notes, as settings.js's renderUserPage/renderMyDocsPage: clamp the page (a delete or new search can shrink the list), slice, render, then show/hide Prev/Next. */
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

  // The list can be empty for two reasons, "no notes at all" (create one above) vs "none match this search", which need
  // different copy.
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
  notesPage = 1;
  renderNotesPage();
});

async function loadNotes() {
  notesCache = await api("/api/notes");
  // Defaults first (stable sort keeps each group's updated-at-desc order).
  notesCache.sort((a, b) => Number(b.is_default) - Number(a.is_default));
  renderNotesPage();
}

function resetNewNoteForm() {
  document.getElementById("new-note-title").value = "";
  document.getElementById("new-note-content").value = "";
  addNoteStatusEl.textContent = "";
}

function showNewNoteForm(open) {
  newNoteFormEl.classList.toggle("hidden", !open);
  focusNoteRow(open ? newNoteFormEl : null);
}

addNoteBtn.addEventListener("click", async () => {
  showNewNoteForm(newNoteFormEl.classList.contains("hidden"));
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

document.getElementById("cancel-add-note-btn").addEventListener("click", () => showNewNoteForm(false));

document.getElementById("create-note-btn").addEventListener("click", async () => {
  const title = document.getElementById("new-note-title").value.trim() || "Untitled note";
  const content = document.getElementById("new-note-content").value;
  const pins = collectPins(document.getElementById("new-note-pin-picker"));
  const btn = document.getElementById("create-note-btn");

  btn.disabled = true;
  addNoteStatusEl.textContent = "Creating…";
  try {
    await api("/api/notes", { method: "POST", body: JSON.stringify({ title, content, pins }) });
    showNewNoteForm(false);
    await loadNotes();
  } catch (err) {
    addNoteStatusEl.textContent = err.message;
  } finally {
    btn.disabled = false;
  }
});

loadNotes();
