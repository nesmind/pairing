/**
 * Image generation page controller: submit a prompt to the active image engine (see
 * app/routers/image_generation.py), then poll each job's own status
 * until it's complete/errored/cancelled, rendering a gallery tile per generation.
 * Plain polling (no websocket): an in-progress tile shows a live progress bar and a Cancel button.
 */

const galleryEl = document.getElementById("gen-gallery");
const galleryEmptyEl = document.getElementById("gen-gallery-empty");
const generateBtn = document.getElementById("gen-generate-btn");
const statusEl = document.getElementById("gen-status");
const checkpointSelect = document.getElementById("gen-checkpoint");

// One poll loop per in-flight job id, so navigating away from an older
// tile (e.g. a fresh full-gallery reload) never disrupts a newer job's
// own polling — each loop only ever touches its own tile.
const activePolls = new Set();

function statusLabel(status) {
  if (status === "queued") return "Waiting in the queue…";
  if (status === "running") return "Starting…";
  if (status === "error") return "Failed";
  if (status === "cancelled") return "Cancelled";
  return "";
}

const isActive = (job) => job.status === "queued" || job.status === "running";

/** The server stores naive UTC timestamps — parse them as UTC. */
function parseCreatedAt(iso) {
  return new Date(/(Z|[+-]\d\d:\d\d)$/.test(iso) ? iso : `${iso}Z`);
}

function formatDuration(totalSeconds) {
  const s = Math.max(0, Math.round(totalSeconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** The in-progress tile: stage text, a progress bar (real percentage when the engine reports one, otherwise an
 * indeterminate pulse), elapsed time / rough time left, and a Cancel button. Updated in place by
 * updateActiveTile so the bar animates smoothly between polls instead of being rebuilt. */
function buildActiveTile(tile, job) {
  tile.dataset.active = "true";
  tile.dataset.createdAt = parseCreatedAt(job.created_at).getTime();
  tile.innerHTML = `<div class="h-full w-full flex flex-col items-center justify-center gap-2 p-3 text-center">
      <span data-stage class="text-xs font-medium text-slate-300"></span>
      <div class="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
        <div data-bar class="h-full rounded-full bg-brand-500 transition-[width] duration-[2000ms] ease-linear"></div>
      </div>
      <span data-detail class="text-[11px] text-slate-500 tabular-nums"></span>
      <button type="button" data-cancel
        class="mt-1 rounded-md border border-slate-700 px-2.5 py-1 text-xs text-slate-300 hover:border-red-500 hover:text-red-400 transition-colors">Cancel</button>
    </div>`;
  tile.querySelector("[data-cancel]").addEventListener("click", (event) => cancelJob(job.id, event.currentTarget));
  updateActiveTile(tile, job);
}

function updateActiveTile(tile, job) {
  const bar = tile.querySelector("[data-bar]");
  const known = typeof job.progress === "number";
  tile.dataset.eta = job.eta_seconds ?? "";
  tile.querySelector("[data-stage]").textContent = job.stage || statusLabel(job.status);
  bar.style.width = known ? `${Math.max(job.progress, 3)}%` : "100%";
  bar.classList.toggle("animate-pulse", !known);
  bar.classList.toggle("opacity-40", !known);
  tile.querySelector("[data-detail]").textContent = detailText(tile, known ? Math.round(job.progress) : null);
}

function detailText(tile, percent) {
  const elapsed = formatDuration((Date.now() - Number(tile.dataset.createdAt)) / 1000);
  const parts = [percent === null ? null : `${percent}%`, `${elapsed} elapsed`];
  if (tile.dataset.eta) parts.push(`~${formatDuration(Number(tile.dataset.eta))} left`);
  return parts.filter(Boolean).join(" · ");
}

// Keeps every in-progress tile's elapsed time ticking between the (slower) job polls.
setInterval(() => {
  for (const tile of galleryEl.querySelectorAll('[data-active="true"]')) {
    const detail = tile.querySelector("[data-detail]");
    const percent = detail.textContent.match(/^(\d+)%/);
    detail.textContent = detailText(tile, percent ? Number(percent[1]) : null);
  }
}, 1000);

async function cancelJob(jobId, button) {
  button.disabled = true;
  button.textContent = "Cancelling…";
  try {
    replaceTile(await api(`/api/image-generation/jobs/${jobId}/cancel`, { method: "POST" }));
  } catch (_) {
    button.disabled = false;
    button.textContent = "Cancel";
  }
}

/** Builds one gallery tile for `job` — an image once complete, a
 * spinner-ish placeholder while queued/running, or an error badge. */
function renderTile(job) {
  const tile = document.createElement("div");
  tile.className = "aspect-square rounded-lg bg-slate-900 border border-slate-800 overflow-hidden relative group";
  tile.dataset.jobId = job.id;

  if (job.status === "complete" && job.url) {
    tile.innerHTML = window.completeTileHtml(job); // picture + Save link; click-to-enlarge in image_generation_viewer.js
  } else if (job.status === "error") {
    tile.innerHTML = `<div class="h-full w-full flex flex-col items-center justify-center gap-1 p-2 text-center">
        <span class="text-xs text-red-400">Failed</span>
        <span class="text-[11px] text-slate-500 truncate w-full">${escapeHtml(job.error_message || "")}</span>
      </div>`;
  } else if (isActive(job)) {
    buildActiveTile(tile, job);
  } else {
    tile.innerHTML = `<div class="h-full w-full flex items-center justify-center text-xs text-slate-500">
        ${statusLabel(job.status)}
      </div>`;
  }
  if (!isActive(job)) tile.appendChild(buildRemoveButton(job));
  return tile;
}

/** The × in a finished tile's corner (always visible on a failed/cancelled one, on hover for a real image). */
function buildRemoveButton(job) {
  const button = document.createElement("button");
  button.type = "button";
  button.title = job.status === "complete" ? "Delete this image" : "Remove";
  button.setAttribute("aria-label", button.title);
  button.className =
    "absolute top-1.5 right-1.5 h-6 w-6 rounded-full bg-black/60 text-slate-200 hover:bg-red-600 hover:text-white transition-colors flex items-center justify-center " +
    (job.status === "complete" ? "opacity-0 group-hover:opacity-100 focus:opacity-100" : "");
  button.innerHTML =
    '<svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6 6 18M6 6l12 12"/></svg>';
  button.addEventListener("click", () => removeJob(job, button));
  return button;
}

async function removeJob(job, button) {
  if (job.status === "complete" && !confirm("Delete this image? This can't be undone.")) return;
  button.disabled = true;
  try {
    await api(`/api/image-generation/jobs/${job.id}`, { method: "DELETE" });
    galleryEl.querySelector(`[data-job-id="${job.id}"]`)?.remove();
    galleryEmptyEl.classList.toggle("hidden", galleryEl.children.length > 0);
  } catch (_) {
    button.disabled = false;
  }
}

function replaceTile(job) {
  const existing = galleryEl.querySelector(`[data-job-id="${job.id}"]`);
  if (existing?.dataset.active === "true" && isActive(job)) {
    updateActiveTile(existing, job); // in place, so the bar animates instead of resetting
    return;
  }
  const fresh = renderTile(job);
  if (existing) existing.replaceWith(fresh);
  else galleryEl.prepend(fresh);
}

/** Polls GET .../jobs/{id} every ~2s until status is complete/error,
 * updating that one tile in place — independent of any other job's own
 * loop or of a later full-gallery reload. */
async function pollJob(jobId) {
  if (activePolls.has(jobId)) return;
  activePolls.add(jobId);
  try {
    while (true) {
      const job = await api(`/api/image-generation/jobs/${jobId}`);
      replaceTile(job);
      if (!isActive(job)) return;
      await new Promise((resolve) => setTimeout(resolve, 1500));
    }
  } catch (_) {
    // A transient poll failure isn't worth surfacing per-tile — the next
    // full gallery reload (switching tabs, reopening the page) will show
    // the real state either way.
  } finally {
    activePolls.delete(jobId);
  }
}

async function loadGallery() {
  const jobs = await api("/api/image-generation/jobs");
  galleryEl.innerHTML = "";
  galleryEmptyEl.classList.toggle("hidden", jobs.length > 0);
  for (const job of jobs) {
    galleryEl.appendChild(renderTile(job));
    if (job.status === "queued" || job.status === "running") pollJob(job.id);
  }
}

async function loadCheckpoints() {
  try {
    const { checkpoints } = await api("/api/image-generation/checkpoints");
    checkpointSelect.innerHTML = checkpoints.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("");
    document.getElementById("comfyui-unavailable")?.classList.add("hidden");
  } catch (_) {
    document.getElementById("comfyui-unavailable")?.classList.remove("hidden");
  }
}

generateBtn.addEventListener("click", async () => {
  const prompt = document.getElementById("gen-prompt").value.trim();
  if (!prompt) return;
  if (!checkpointSelect.value) {
    statusEl.textContent = "System is not configured for images yet, ask your admin.";
    return;
  }

  const body = {
    prompt,
    negative_prompt: document.getElementById("gen-negative-prompt").value.trim() || null,
    checkpoint: checkpointSelect.value,
    width: parseInt(document.getElementById("gen-width").value, 10),
    height: parseInt(document.getElementById("gen-height").value, 10),
    steps: parseInt(document.getElementById("gen-steps").value, 10),
    cfg: parseFloat(document.getElementById("gen-cfg").value),
    seed: parseInt(document.getElementById("gen-seed").value, 10),
  };

  const sourceControls = window.generationSource;
  if (sourceControls.isImageMode()) {
    if (!sourceControls.hasImage()) {
      statusEl.textContent = "Choose a source image first.";
      return;
    }
    body.mode = "image_to_image";
    body.init_image = sourceControls.pngBase64(body.width, body.height);
    body.strength = sourceControls.strength();
  }

  generateBtn.disabled = true;
  statusEl.textContent = "Submitting…";
  try {
    const job = await api("/api/image-generation/jobs", { method: "POST", body: JSON.stringify(body) });
    galleryEmptyEl.classList.add("hidden");
    galleryEl.prepend(renderTile(job));
    pollJob(job.id);
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = `Failed to submit: ${err.message}`;
  } finally {
    generateBtn.disabled = false;
  }
});

loadCheckpoints();
loadGallery();
