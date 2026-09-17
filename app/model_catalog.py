"""
A curated list of chat models this app knows how to offer in Settings,
beyond whatever the user has already pulled into Ollama manually. Each
entry is a *potential* model — app/routers/settings.py cross-references
this against what's actually installed (`ollama list`) and against this
machine's hardware (app/hardware.py) to decide, per entry, whether to
show "select", "pull", or a disabled row with a minimum-requirements
message.

Sizes/requirements below come from each model's public spec sheet at
the time this catalog was written — see ROADMAP.md for how to keep this
current as new sizes/models are released. An entry pulled from here
(not uninstalled, just dropped from this default list) may still have
its verified Hugging Face repo/file/params kept in
tested-models-params.md, so re-adding it later doesn't mean
re-verifying from scratch.

`min_ram_gb` is deliberately generous (roughly 1.25x the on-disk weight
size) to leave headroom for the context-window KV cache and Ollama's own
runtime overhead — better to under-promise than to gate a model in as
"fits" and have it OOM or swap itself into uselessness.
"""

CATALOG = [
    {
        # Google's own official GGUF repo (not a community re-quant) —
        # confirmed live: google/gemma-4-E2B-it-qat-q4_0-gguf, arch
        # "gemma4", 4.6B total params, 131072 ctx.
        "family": "Gemma 4",
        "vendor": "Google",
        "tag": "hf.co/google/gemma-4-E2B-it-qat-q4_0-gguf:gemma-4-E2B_q4_0-it",
        "parameter_size": "2.3B effective (MoE)",
        "context_length": 128_000,
        "download_gb": 7.2,
        "min_ram_gb": 9,
        "locally_runnable": True,
    },
    {
        # Google's own official GGUF repo — confirmed live:
        # google/gemma-4-E4B-it-qat-q4_0-gguf, arch "gemma4", 7.5B total
        # params, 131072 ctx.
        "family": "Gemma 4",
        "vendor": "Google",
        "tag": "hf.co/google/gemma-4-E4B-it-qat-q4_0-gguf:gemma-4-E4B_q4_0-it",
        "parameter_size": "4.5B effective (MoE)",
        "context_length": 128_000,
        "download_gb": 9.6,
        "min_ram_gb": 12,
        "locally_runnable": True,
    },
    {
        # This is the model this app's "Default vision model" setting used
        # during development/testing, so its own vision capability was
        # verified end-to-end before migrating, not just inferred from the
        # sibling entries below: pulled the real repo (7.0GB main file +
        # its 175MB "mmproj-*.gguf" projector, both auto-paired by
        # Ollama's hf.co/ pull), then confirmed via `ollama show` —
        # Capabilities: tools, completion, vision, audio, 11.9B params
        # (matches "12B" here). Google's own official GGUF repo:
        # google/gemma-4-12B-it-qat-q4_0-gguf.
        "family": "Gemma 4",
        "vendor": "Google",
        "tag": "hf.co/google/gemma-4-12B-it-qat-q4_0-gguf:gemma-4-12b-it-qat-q4_0",
        "parameter_size": "12B",
        "context_length": 256_000,
        "download_gb": 7.6,
        "min_ram_gb": 10,
        "locally_runnable": True,
        "vision": True,
    },
    {
        # Google's own official GGUF repo — confirmed live:
        # google/gemma-4-26B-A4B-it-qat-q4_0-gguf, arch "gemma4", 25.2B
        # total params, 262144 ctx — "A4B" (~4B active) matches this
        # entry's "3.8B active" exactly.
        "family": "Gemma 4",
        "vendor": "Google",
        "tag": "hf.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf:gemma-4-26B_q4_0-it",
        "parameter_size": "25.2B (3.8B active, MoE)",
        "context_length": 256_000,
        "download_gb": 19,
        "min_ram_gb": 24,
        "locally_runnable": True,
    },
    {
        # Google's own official GGUF repo — confirmed live:
        # google/gemma-4-31B-it-qat-q4_0-gguf, arch "gemma4", 30.7B
        # total params, 262144 ctx — matches this entry exactly.
        "family": "Gemma 4",
        "vendor": "Google",
        "tag": "hf.co/google/gemma-4-31B-it-qat-q4_0-gguf:gemma-4-31B_q4_0-it",
        "parameter_size": "30.7B",
        "context_length": 256_000,
        "download_gb": 20,
        "min_ram_gb": 25,
        "locally_runnable": True,
    },
    {
        # Vision-capable (see model_catalog_service.installed_vision_models
        # and Settings > System's "Default vision model" picker) — Vicuna-7B
        # fine-tuned with a CLIP ViT-L vision encoder. The standard, most
        # widely-used size for this family; noticeably lighter than
        # gemma4:12b for image-attachment replies (see
        # app/services/chat_attachment_service.py). Sourced from Hugging
        # Face — confirmed live: second-state/Llava-v1.6-Vicuna-7B-GGUF,
        # arch "llama", 6.74B params, 4096 ctx, with a Q4_K_M main file
        # plus its own "-mmproj-model-f16.gguf" projector in the same
        # repo. Ollama's hf.co/ pull syntax was confirmed (via a real
        # pull + `ollama show`, not just repo inspection) to auto-detect
        # and pull a same-repo "mmproj"-named file alongside the tagged
        # one and keep vision working — see the sibling gemma4:12b
        # entry's own comment for exactly how that was verified.
        "family": "LLaVA",
        "vendor": "LLaVA / Ollama",
        "tag": "hf.co/second-state/Llava-v1.6-Vicuna-7B-GGUF:llava-v1.6-vicuna-7b-Q4_K_M",
        "parameter_size": "7B",
        "context_length": 4096,
        "download_gb": 4.7,
        "min_ram_gb": 6,
        "locally_runnable": True,
        "vision": True,
    },
    {
        # Same LLaVA vision approach on a much smaller Phi-3-mini base —
        # the lightest well-known local vision option, worth offering
        # for CPU-only hardware where gemma4:12b/llava:7b are too slow.
        # Sourced from xtuner's own Hugging Face repo (xtuner authored
        # this LLaVA-Phi-3 model originally, not a third-party re-quant)
        # — confirmed live: xtuner/llava-phi-3-mini-gguf, arch "llama",
        # 3.82B params, 4096 ctx, with its own int4-quantized main file
        # plus a same-repo "-mmproj-f16.gguf" projector (auto-paired by
        # Ollama's hf.co/ pull — see gemma4:12b's own comment).
        "family": "LLaVA",
        "vendor": "LLaVA / Ollama",
        "tag": "hf.co/xtuner/llava-phi-3-mini-gguf:llava-phi-3-mini-int4",
        "parameter_size": "3.8B",
        "context_length": 4096,
        "download_gb": 2.9,
        "min_ram_gb": 4,
        "locally_runnable": True,
        "vision": True,
    },
    {
        # Not LLaVA-family, but the smallest practical local vision model
        # available through Ollama — included alongside the two LLaVA
        # entries above as the lightest possible option for weak/CPU-only
        # hardware. Sourced from moondream's own official Hugging Face
        # repo — confirmed live end-to-end (real pull, then `ollama
        # show`): moondream/moondream2-gguf, arch "phi2", 1.42B params,
        # 2048 ctx, its main text-model file paired automatically by
        # Ollama's hf.co/ pull with the repo's own "-mmproj-f16.gguf"
        # projector — `ollama show` afterward listed Capabilities:
        # completion, vision, confirming the pairing isn't just present
        # in the repo but actually recognized by Ollama.
        "family": "Moondream",
        "vendor": "vikhyat",
        "tag": "hf.co/moondream/moondream2-gguf:moondream2-text-model-f16",
        "parameter_size": "1.8B",
        "context_length": 2048,
        "download_gb": 1.7,
        "min_ram_gb": 3,
        "locally_runnable": True,
        "vision": True,
    },
    {
        # MiniMax M3 (428B total / 23B active, MoE) is a genuine local
        # option, unlike Kimi K2 — Ollama can pull a real GGUF
        # quantization of it straight from Hugging Face via the `hf.co/`
        # tag syntax, so this doesn't need Kimi's special-cased
        # "locally_runnable: False" treatment. It's real weights running
        # locally through Ollama the same way llama3/gemma4 do; it just
        # needs enormous hardware (the smallest practical quant is still
        # ~128GB) — the existing hardware gate handles that honestly on
        # its own, showing it as unavailable on virtually any single
        # machine without any extra code.
        "family": "MiniMax M3",
        "vendor": "MiniMax AI",
        "tag": "hf.co/unsloth/MiniMax-M3-GGUF:UD-IQ1_M",
        "parameter_size": "428B (23B active, MoE)",
        "context_length": 1_048_576,
        "download_gb": 128,
        "min_ram_gb": 133,
        "locally_runnable": True,
    },
]


def find_entry(tag: str) -> dict | None:
    """Looks up a catalog entry by its exact Ollama tag."""
    return next((entry for entry in CATALOG if entry["tag"] == tag), None)


# Embedding models — used only for the RAG knowledge-base feature (app/rag.py), never for chat. Kept in a
# wholly separate list from CATALOG above so an embedding-only model can never leak into the chat-model
# picker/switcher (build_model_catalog, installed_chat_models) — those only ever iterate CATALOG, which this
# list is deliberately not part of.
EMBEDDING_CATALOG = [
    {
        # nomic-ai's own official GGUF repo (not a community re-quant) — confirmed live: arch "nomic-bert",
        # 137M params, 2048 ctx, 768-dim embeddings, Capabilities: embedding. A real /api/embed call returned
        # a correct 768-dim vector. This is the default (see EMBEDDING_MODEL in app/config.py) — higher
        # retrieval quality than the lighter MiniLM entry below, at roughly 6x the parameter count.
        "family": "Nomic Embed Text",
        "vendor": "Nomic AI",
        "tag": "hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0",
        "parameter_size": "137M",
        "context_length": 2048,
        "embedding_dim": 768,
        "download_gb": 0.15,
        "min_ram_gb": 1,
        "locally_runnable": True,
    },
    {
        # Highest-downloaded (402k) community GGUF conversion of sentence-transformers' all-MiniLM-L6-v2 —
        # confirmed live: arch "bert", 22.6M params, 512 ctx, 384-dim embeddings, Capabilities: embedding. A
        # real /api/embed call returned a correct 384-dim vector. Offered as the lighter/faster alternative,
        # not the default — see EMBEDDING_MODEL's own comment in app/config.py for the trade-off.
        "family": "MiniLM L6 v2",
        "vendor": "sentence-transformers (leliuga GGUF)",
        "tag": "hf.co/leliuga/all-MiniLM-L6-v2-GGUF:Q8_0",
        "parameter_size": "22.6M",
        "context_length": 512,
        "embedding_dim": 384,
        "download_gb": 0.03,
        "min_ram_gb": 1,
        "locally_runnable": True,
    },
]


def find_embedding_entry(tag: str) -> dict | None:
    """Looks up an embedding-catalog entry by its exact Ollama tag — mirrors find_entry above, kept separate
    since EMBEDDING_CATALOG is a deliberately distinct list (see its own docstring)."""
    return next((entry for entry in EMBEDDING_CATALOG if entry["tag"] == tag), None)
