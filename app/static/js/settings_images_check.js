/** Warns under the image engine's "Images folder" field when cluster mode is on and the folder is not on shared
 * storage (GET /api/settings/sdcpp/images-check). It is pAIring's own concern: it never blocks the Matricxon
 * cluster. Checked on load and again after the field changes or the form is saved. */

async function refreshImagesFolderWarning() {
  const note = document.querySelector('[data-field="images-warning"]');
  if (!note) return;
  try {
    const check = await api("/api/settings/sdcpp/images-check");
    note.textContent = check ? `${check.what}. ${check.hint}` : "";
    note.classList.toggle("hidden", !check);
  } catch (_err) {
    /* not an admin: no warning */
  }
}

document.addEventListener("DOMContentLoaded", () => {
  setTimeout(refreshImagesFolderWarning, 1500); // after the form has loaded its saved folder
  const field = document.querySelector('[data-field="images_path"]');
  if (field) field.addEventListener("change", () => setTimeout(refreshImagesFolderWarning, 2500));
});
