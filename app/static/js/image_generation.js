/**
 * Image generation page controller: submit a prompt to ComfyUI (see
 * app/routers/image_generation.py), then poll each job's own status
 * until it's complete/errored, rendering a gallery tile per generation.
 * No websocket/live-progress — plain polling only, per this feature's
 * own scope decision (see the plan this was built from).
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
  if (status === "queued") return "Queued…";
  if (status === "running") return "Generating…";
  if (status === "error") return "Failed";
  return "";
}

/** Builds one gallery tile for `job` — an image once complete, a
 * spinner-ish placeholder while queued/running, or an error badge. */
function renderTile(job) {
  const tile = document.createElement("div");
  tile.className = "aspect-square rounded-lg bg-slate-900 border border-slate-800 overflow-hidden relative group";
  tile.dataset.jobId = job.id;

  if (job.status === "complete" && job.url) {
    tile.innerHTML = `<img src="${job.url}" alt="${escapeHtml(job.prompt)}" class="h-full w-full object-cover">
      <div class="absolute inset-x-0 bottom-0 bg-black/60 px-2 py-1 text-xs text-slate-200 truncate opacity-0 group-hover:opacity-100 transition-opacity">
        ${escapeHtml(job.prompt)}
      </div>`;
  } else if (job.status === "error") {
    tile.innerHTML = `<div class="h-full w-full flex flex-col items-center justify-center gap-1 p-2 text-center">
        <span class="text-xs text-red-400">Failed</span>
        <span class="text-[11px] text-slate-500 truncate w-full">${escapeHtml(job.error_message || "")}</span>
      </div>`;
  } else {
    tile.innerHTML = `<div class="h-full w-full flex items-center justify-center text-xs text-slate-500">
        ${statusLabel(job.status)}
      </div>`;
  }
  return tile;
}

function replaceTile(job) {
  const existing = galleryEl.querySelector(`[data-job-id="${job.id}"]`);
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
      if (job.status === "complete" || job.status === "error") return;
      await new Promise((resolve) => setTimeout(resolve, 2000));
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
