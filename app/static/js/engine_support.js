/**
 * Stats page's "Supported architectures" view — every model architecture and quantization (dequant) type the
 * active engine can load, from GET /api/stats/engine-support (see app.services.engine_support_service).
 * Matricxon-only: Ollama has no API listing what it can load, so this view's sidebar button stays hidden unless
 * /health reports Matricxon as the active engine (see initEngineSupportNav below).
 *
 * The raw names come live from Matricxon's own registries — the friendly labels below are display-only, and any
 * name missing from them still renders (just with its raw name), so a newly supported architecture or quant type
 * shows up here the moment Matricxon adds it, with no change needed on this side.
 */

const ARCHITECTURE_LABELS = {
  llama: "Llama (Llama 2/3, Vicuna, LLaVA, SmolVLM, …)",
  mistral3: "Mistral 3 / Ministral",
  gemma4: "Gemma 4",
  phi2: "Phi-2 (moondream2)",
  qwen2: "Qwen 2 / 2.5",
  qwen3: "Qwen 3",
  granite: "IBM Granite",
  granitemoe: "IBM Granite MoE",
  nemotron_h: "NVIDIA Nemotron-H (hybrid Mamba)",
  "command-r": "Cohere Command R",
  bert: "BERT",
  "nomic-bert": "Nomic BERT",
};

// Encoder-only architectures — embeddings for document search (RAG), not chat.
const EMBEDDING_ARCHITECTURES = new Set(["bert", "nomic-bert"]);

// Checked in order — the first matching group wins; anything unmatched lands in "Other".
const QUANT_GROUPS = [
  { title: "Full / half precision", hint: "Unquantized weights — largest files, reference quality.", test: (q) => /^(F32|F16|BF16)$/.test(q) },
  { title: "K-quants", hint: "Q4_K_M/Q5_K_M/Q6_K files use these — the usual best size/quality balance.", test: (q) => /^Q\d_K$/.test(q) },
  { title: "Legacy quants", hint: "Older, simpler block formats (Q4_0, Q8_0, …).", test: (q) => /^Q\d_\d$/.test(q) },
];

function _engineSupportChip(text, extraClass = "") {
  return `<span class="inline-flex items-center rounded-md border border-slate-700 bg-slate-800/60 px-2 py-0.5 font-mono text-xs text-slate-200 ${extraClass}">${escapeHtml(text)}</span>`;
}

function _renderArchitectures(architectures) {
  const section = document.createElement("div");
  const cards = [...architectures]
    .sort((a, b) => Number(EMBEDDING_ARCHITECTURES.has(a)) - Number(EMBEDDING_ARCHITECTURES.has(b)) || a.localeCompare(b))
    .map((name) => {
      const kind = EMBEDDING_ARCHITECTURES.has(name) ? "Embeddings" : "Text generation";
      return (
        `<div class="rounded-lg bg-slate-900 border border-slate-800 px-4 py-3">` +
        `<p class="text-sm font-medium text-slate-100">${escapeHtml(ARCHITECTURE_LABELS[name] || name)}</p>` +
        `<p class="mt-1 flex items-center gap-2">${_engineSupportChip(name)}` +
        `<span class="text-[10px] uppercase tracking-wide text-slate-500">${kind}</span></p></div>`
      );
    });
  section.innerHTML =
    `<h3 class="text-sm font-semibold text-slate-200 mb-3">Architectures (${architectures.length})</h3>` +
    `<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">${cards.join("")}</div>`;
  return section;
}

function _renderQuantizations(quantizations) {
  const section = document.createElement("div");
  const remaining = [...quantizations];
  const groups = QUANT_GROUPS.map((group) => ({ ...group, items: remaining.filter(group.test) }));
  const other = remaining.filter((q) => !QUANT_GROUPS.some((group) => group.test(q)));
  if (other.length) groups.push({ title: "Other", hint: "", items: other });
  const rows = groups
    .filter((group) => group.items.length)
    .map(
      (group) =>
        `<div class="px-4 py-3"><p class="text-xs font-semibold text-slate-300">${escapeHtml(group.title)}` +
        `<span class="font-normal text-slate-500"> — ${escapeHtml(group.hint)}</span></p>` +
        `<div class="mt-2 flex flex-wrap gap-1.5">${group.items.map((q) => _engineSupportChip(q)).join("")}</div></div>`
    );
  section.innerHTML =
    `<h3 class="text-sm font-semibold text-slate-200 mb-1">Quantization / dequant types (${quantizations.length})</h3>` +
    `<p class="text-xs text-slate-500 mb-3">A GGUF file whose quantization isn't listed here (e.g. IQ4_XS) can't be loaded by this engine.</p>` +
    `<div class="rounded-lg bg-slate-900 border border-slate-800 divide-y divide-slate-800">${rows.join("")}</div>`;
  return section;
}

async function loadEngineSupport() {
  const contentEl = document.getElementById("engine-support-content");
  const errorEl = document.getElementById("engine-support-error");
  try {
    const support = await api("/api/stats/engine-support");
    if (!support.available) {
      contentEl.innerHTML =
        '<p class="text-sm text-slate-400">Only available on the Matricxon engine — Ollama has no API listing the architectures it supports.</p>';
    } else if (support.error) {
      throw new Error(support.error);
    } else {
      contentEl.replaceChildren(_renderArchitectures(support.architectures), _renderQuantizations(support.quantizations));
    }
    errorEl.classList.add("hidden");
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.classList.remove("hidden");
  }
}

/** Reveals this view's sidebar button only when Matricxon is the active engine — same /health endpoint
 * telemetry.js reads its own "Currently on" note from. Left hidden on any /health failure. */
async function initEngineSupportNav() {
  try {
    const { active_engine: activeEngine } = await api("/health");
    document.getElementById("stats-nav-architectures").classList.toggle("hidden", activeEngine !== "matricxon");
  } catch (_err) {
    // Best-effort — the rest of the Stats page works regardless.
  }
}

initEngineSupportNav();
