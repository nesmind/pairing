/**
 * Gallery extras for the Images page: the finished-image tile (picture + a Save link under it) and a click-to-enlarge
 * viewer. Exposes window.completeTileHtml for image_generation.js.
 */
(() => {
  const fileName = (job) => `${job.prompt.replace(/[^\w]+/g, "_").slice(0, 40) || "image"}_${job.id.slice(0, 6)}.png`;
  const saveClass = "shrink-0 text-xs text-brand-400 hover:text-brand-300";

  window.completeTileHtml = (job) => `<div class="h-full w-full flex flex-col">
      <img src="${job.url}" alt="${escapeHtml(job.prompt)}" data-enlarge title="Click to enlarge"
        class="min-h-0 flex-1 w-full object-cover cursor-zoom-in">
      <div class="flex items-center justify-end border-t border-slate-800 bg-slate-950 px-2 py-1">
        <a href="${job.url}" download="${escapeHtml(fileName(job))}" class="${saveClass}">Save</a>
      </div>
    </div>`;

  let viewer = null;
  function buildViewer() {
    const el = document.createElement("div");
    el.className = "hidden fixed inset-0 z-50 flex flex-col items-center justify-center gap-3 bg-black/80 p-4";
    el.innerHTML = `<p data-viewer-caption dir="auto" class="max-h-[12vh] max-w-3xl overflow-y-auto text-center text-sm text-slate-200"></p>
      <img data-viewer-img alt="" class="max-h-[70vh] max-w-full rounded-md object-contain">
      <button type="button" data-viewer-close class="text-sm text-slate-300 hover:text-white">Close</button>`;
    el.addEventListener("click", (event) => {
      if (event.target === el || event.target.closest("[data-viewer-close]")) closeViewer();
    });
    document.body.appendChild(el);
    return el;
  }

  function openViewer(img) {
    viewer ??= buildViewer();
    const big = viewer.querySelector("[data-viewer-img]");
    const caption = viewer.querySelector("[data-viewer-caption]");
    big.onload = () => (caption.textContent = `${img.alt} (${big.naturalWidth}×${big.naturalHeight})`);
    big.src = img.src;
    big.alt = img.alt;
    caption.textContent = img.alt;
    viewer.classList.remove("hidden");
  }

  function closeViewer() {
    viewer?.classList.add("hidden");
  }

  document.getElementById("gen-gallery").addEventListener("click", (event) => {
    const img = event.target.closest("[data-enlarge]");
    if (img) openViewer(img);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeViewer();
  });
})();
