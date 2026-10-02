/**
 * The ML stages *inside* the engine for the Stats page's "Visual chat workflow" diagram — the leaves
 * behind the two "ML engine" nodes in workflow_diagram_data.js: the chat model's whole inference
 * pass (a band of rows under the "ML engine" node) and the RAG embedding model's (a small lane in
 * the RAG column). Appended to WORKFLOW_NODES/WORKFLOW_EDGES, so load this after
 * workflow_diagram_data.js. Same node/edge shape as there; an edge may add `dir: "v"` to force a
 * top/bottom connection where the automatic choice would run through a neighbouring node.
 *
 * Laid out in rows inside the diagram's existing width (x 30–1300; six columns for the chat model) as a snake — each
 * row ends where the next begins — so the new stages only add height, never a horizontal scrollbar.
 * Written from Matricxon's real pipeline (app/runtime/chat_engine.py, app/architectures/*,
 * app/runtime/sampler.py, app/runtime/tokenizer.py); Ollama (llama.cpp) runs the same stages in
 * C++ with its own cache and scheduler details, which each body mentions only where they differ.
 * All of it runs in the ML engine's process, not in this app.
 */

const ML_COL = [30, 244, 458, 672, 886, 1100]; // six 176px columns, 38px apart (right edge 1276)
const ML_W = 176;
const ML_H = 64;
const ML_ROW = { r1: 1040, r2: 1140, r2b: 1240, r3: 1340, r4: 1440 };

// Colour group per stage (see WORKFLOW_CATEGORY_COLORS): where in the forward pass it sits / what it does.
const ML_GROUP = {
  ml_load: "ml_pre", ml_template: "ml_pre", ml_tokenize: "ml_pre", ml_prompt_cache: "ml_pre",
  ml_embed: "ml_pre", ml_vision: "ml_pre", ml_emb_tok: "ml_pre", ml_emb_pos: "ml_pre",
  ml_norm_attn: "ml_norm", ml_norm_ffn: "ml_norm",
  ml_attention: "ml_attn", ml_linear_attn: "ml_attn", ml_emb_enc: "ml_attn",
  ml_ffn: "ml_ffn",
  ml_layer_loop: "ml_loop", ml_decode_loop: "ml_loop",
  ml_final_norm: "ml_post", ml_lm_head: "ml_post", ml_sampler: "ml_post", ml_stop_check: "ml_post",
  ml_detok: "ml_post", ml_ndjson: "ml_post", ml_emb_pool: "ml_post", ml_emb_norm: "ml_post",
  ml_kernels: "ml_support", ml_kv: "ml_support",
};

// Stages that run on Matricxon's native C kernels (app/native/src/*.c) when that backend is built —
// the quantized matmuls, embedding-row dequant, the gated delta rule, and the fused RMSNorm / RoPE /
// decode-attention / sampler ops (app/native/src/mx_fused_ops.c, mx_attention.c, mx_sample.c). Marked "K".
const ML_NATIVE = new Set([
  "ml_embed", "ml_attention", "ml_linear_attn", "ml_ffn", "ml_lm_head", "ml_kernels",
  "ml_norm_attn", "ml_norm_ffn", "ml_final_norm", "ml_sampler",
]);

// Stages that can run on the GPU (Matricxon's experimental MATRICXON_DEVICE=cuda mode): the forward
// pass proper. Tokenizing, sampling, the stop check, the vision tower and MoE routing stay on the CPU.
// Marked "G"; a stage that is also in ML_NATIVE shows "K|G" (the native C kernels on the CPU, or the
// GPU when the engine runs on one).
const ML_GPU = new Set([
  "ml_embed", "ml_norm_attn", "ml_attention", "ml_linear_attn", "ml_norm_ffn", "ml_ffn",
  "ml_final_norm", "ml_lm_head", "ml_kernels", "ml_kv", "ml_emb_pos", "ml_emb_enc",
]);

function mlNode(id, label, col, y, title, location, body, size = {}) {
  return {
    id,
    label,
    category: ML_GROUP[id],
    native: ML_NATIVE.has(id),
    gpu: ML_GPU.has(id),
    x: size.x ?? ML_COL[col],
    y,
    w: size.w ?? ML_W,
    h: size.h ?? ML_H,
    detail: { server: "ML engine", title, location, body },
  };
}

const WORKFLOW_ML_NODES = [
  // ---- Row 1: preparing the request ------------------------------------
  mlNode(
    "ml_load", "Model get_or_load\n(GGUF file → memory)", 0, ML_ROW.r1,
    "Getting the model into memory",
    "Matricxon: app/models/manager.py — get_or_load · Ollama: its model scheduler",
    "The first request for a model tag finds its GGUF file, checks it fits in free RAM, picks float32 or bf16 for the parts that stay unquantized, and memory-maps the file. Weights are unpacked lazily on the first forward pass; later requests reuse the loaded model until it is unloaded or evicted. This is the cold-start delay on a model's first message."
  ),
  mlNode(
    "ml_template", "Chat template\n(messages → prompt text)", 1, ML_ROW.r1,
    "Rendering the model's own chat template",
    "app/runtime/chat_template.py — ChatTemplatePromptBuilder",
    "The conversation (system prompt, history, new question) is turned into the single text string this particular model was trained on. The template ships inside the GGUF file as Jinja; it decides the role markers (for Qwen, <|im_start|>user …), where images and tool definitions go, and whether the reply starts with an empty or open thinking block."
  ),
  mlNode(
    "ml_tokenize", "Tokenizer\n(text → token ids)", 2, ML_ROW.r1,
    "Splitting the prompt into tokens",
    "app/runtime/tokenizer.py — GGUFTokenizer",
    "Byte-pair encoding with the vocabulary and merge rules stored in the GGUF: text becomes a list of integer token ids (often 32k–250k possible values). Special tokens such as <|im_start|> map to single ids. A long conversation can run to thousands of tokens, which is what num_ctx limits."
  ),
  mlNode(
    "ml_prompt_cache", "Prompt cache\n(reuse shared prefix)", 3, ML_ROW.r1,
    "Skipping work already done last turn",
    "app/runtime/prompt_cache.py — PromptCache",
    "The app resends the whole conversation every turn, but the model already computed most of it last time. The engine keeps the previous turn's cache and finds the longest run of identical leading token ids; only the tokens after it are processed (prefill). Hybrid models with recurrent layers (Qwen 3.5, Nemotron-H) cannot rewind, so they restore a snapshot of the recurrent state taken during the previous prefill instead."
  ),
  mlNode(
    "ml_embed", "Embedding lookup\n(token ids → vectors)", 4, ML_ROW.r1,
    "From ids to vectors",
    "app/architectures/*.py — token_embd lookup",
    "Each token id indexes a row of a learned table and becomes a vector of a few thousand numbers (2,560 for a 4B model). Everything after this point is arithmetic on these vectors, never on text. If the message carried images, their embeddings overwrite the placeholder positions here."
  ),
  mlNode(
    "ml_vision", "Vision tower + projector\n(images → embeddings)", 5, ML_ROW.r1,
    "Only when the message has an image",
    "app/vision/ — ClipVisionEncoder / Qwen3VLVisionEncoder, app/runtime/*fusion.py",
    "Needs a vision-capable model plus its matching projector file. The image is resized and normalized, cut into 16×16 patches, run through a Vision Transformer, and a projector maps the result into the language model's own vector space — one vector per image token, spliced in at the embedding step. Qwen 3.5 also gives these tokens 3-axis (time/row/column) position ids."
  ),

  // ---- Row 2: one transformer block (repeated for every layer) ---------
  mlNode(
    "ml_norm_attn", "RMSNorm\n(before attention)", 4, ML_ROW.r2,
    "Normalizing each token's vector",
    "app/architectures/layers.py — RMSNorm · app/native/src/mx_fused_ops.c",
    "Rescales every token vector to a steady magnitude (root-mean-square normalization) and applies a learned per-dimension gain. Keeps the numbers in a range the following matrix multiplications handle well. Each layer starts with one; the block below repeats for every layer of the model. With the native backend this is one C call instead of about seven separate tensor operations."
  ),
  mlNode(
    "ml_attention", "Self-attention\nQ·K·V · RoPE · KV cache", 3, ML_ROW.r2,
    "Letting tokens look at earlier tokens",
    "app/architectures/qwen_layers.py / mistral3_layers.py — attention · app/native/src/mx_attention.c",
    "Three projections turn each vector into a query, key and value. Rotary position embedding (RoPE) rotates queries and keys so relative distance is encoded. Each query is scored against every earlier key, softmaxed, and used to blend the values — this is how \"it\" gets tied to the right noun. New keys and values are appended to the KV cache so later tokens never recompute them. When one new token is decoded, the RoPE rotation and the whole score → softmax → blend step run as single native C calls (longer prompts use PyTorch's matrix-multiply attention instead)."
  ),
  mlNode(
    "ml_linear_attn", "Linear attention / SSM\n(Mamba-2, Gated DeltaNet)", 3, ML_ROW.r2b,
    "The alternative to attention in hybrid models",
    "app/architectures/qwen35_deltanet.py, nemotron_h_mamba2.py",
    "Hybrid models replace most attention layers with a recurrent one: a short causal convolution followed by a fixed-size state that is decayed and updated per token (Mamba-2, or Qwen 3.5's Gated DeltaNet). Cost per token stays constant instead of growing with the context, and the state replaces the KV cache for these layers. Dense models (Llama, Mistral, Qwen 3) have only the attention path."
  ),
  mlNode(
    "ml_norm_ffn", "Residual add\n+ RMSNorm", 2, ML_ROW.r2,
    "Adding the result back, normalizing again",
    "app/architectures/qwen_layers.py — decoder layer",
    "The attention output is added back onto the token's original vector (the residual connection that lets very deep stacks train and keeps information flowing), then normalized again before the feed-forward step."
  ),
  mlNode(
    "ml_ffn", "Feed-forward\nSwiGLU (or MoE experts)", 1, ML_ROW.r2,
    "The per-token knowledge step",
    "app/architectures/layers.py — SwiGLUMLP · llama_moe.py / granitemoe.py for MoE",
    "Each token vector goes up to a wider hidden size through two projections (a gate and an up path multiplied together, SwiGLU), then back down. Most of a model's parameters, and most of its stored knowledge, live here. Mixture-of-Experts models route each token to a few of many such blocks and only compute those."
  ),
  mlNode(
    "ml_layer_loop", "Repeat for all N layers\n(residual stream)", 0, ML_ROW.r2,
    "Stacking the block",
    "app/architectures/*.py — _forward_impl layer loop",
    "The same block (norm → attention or recurrent layer → add → norm → feed-forward → add) runs once per layer, 28–80 times depending on the model, each with its own weights. The engine checks for a stop request between layers, so cancelling a reply does not wait for a whole pass."
  ),
  mlNode(
    "ml_decode_loop", "Decode loop\n(1 new token per step)", 5, ML_ROW.r2,
    "Generating one token at a time",
    "app/runtime/chat_engine.py — ChatEngine.stream",
    "The first pass over the prompt (prefill) is parallel and produces the first reply token. After that every token needs its own pass: the new token alone goes back through the embedding lookup and the layer stack, reading the cached keys and values (or recurrent state) of everything before it. This loop is why reply speed is quoted in tokens per second."
  ),

  // ---- Row 3: from hidden state to the next token ----------------------
  mlNode(
    "ml_final_norm", "Final RMSNorm\n(last hidden state)", 0, ML_ROW.r3,
    "Normalizing the last layer's output",
    "app/architectures/*.py — output_norm",
    "One more normalization of the final layer's vectors. Only the last position's vector matters for choosing the next token, so engines skip the rest of the prompt here."
  ),
  mlNode(
    "ml_lm_head", "LM head\n(hidden → vocab logits)", 1, ML_ROW.r3,
    "Scoring every word in the vocabulary",
    "app/architectures/*.py — lm_head",
    "A single large matrix multiplication turns the last vector into one score (logit) per vocabulary entry — 150k–250k numbers. It is one of the biggest matrices in the model, which is why engines compute it for the last position only."
  ),
  mlNode(
    "ml_sampler", "Sampler\ntemp · top-k · top-p", 2, ML_ROW.r3,
    "Choosing the next token",
    "app/runtime/sampler.py — Sampler · app/native/src/mx_sample.c",
    "Logits are adjusted by the repeat penalty, divided by the temperature, trimmed to the top-k and top-p candidates, turned into probabilities, and one token is drawn at random (or the highest one at temperature 0). These are the same options a chat's settings pass down (temperature, top_p, top_k, repeat_penalty, seed). Over a 150k–250k-word vocabulary this used to cost tens of milliseconds per token in PyTorch (a full sort and several softmaxes); the native version does it in one pass in about a millisecond."
  ),
  mlNode(
    "ml_stop_check", "Stop check\n(EOS · num_predict)", 3, ML_ROW.r3,
    "Deciding whether the reply is over",
    "app/runtime/chat_engine.py — ChatEngine.stream",
    "Generation ends when the model emits an end-of-sequence token, when the num_predict limit or the context window is reached, or when a stop is requested from outside (a cancelled message). Otherwise the token continues down to be streamed out and also loops back as the next step's input."
  ),
  mlNode(
    "ml_detok", "Incremental detokenizer\n(ids → text)", 4, ML_ROW.r3,
    "Turning token ids back into text",
    "app/runtime/tokenizer.py — IncrementalTextDecoder",
    "Tokens map back to pieces of text, but a character can span several tokens (emoji, non-Latin scripts), so the decoder holds back incomplete bytes until they form a full character. Special tokens are filtered out so a reply never shows literal <|im_end|> markers."
  ),
  mlNode(
    "ml_ndjson", `NDJSON line out\n(→ back to ${APP_NAME})`, 5, ML_ROW.r3,
    "Streaming the piece out",
    "app/routers/chat_router.py — the /api/chat NDJSON stream",
    "Each piece of text leaves the engine immediately as one JSON line ({\"message\": {\"content\": …}, \"done\": false}), and a final line carries done: true plus token counts and timings. The engine process is finished with the reply at this point; everything after belongs to this app. The decode loop meanwhile has already started the next token."
  ),
  mlNode(
    "ml_kernels", "Weight matmul kernels\n(dequant GEMV: C / Numba)", 0, ML_ROW.r4,
    "Where the arithmetic actually happens",
    "app/native/src/*.c · app/gguf/dequant/quantized_gemv*.py",
    "Every projection above is a matrix multiplication against weights stored in a quantized GGUF format (4–8 bits per weight). Rather than expanding the whole model to 32-bit floats, fused kernels read the packed bytes and multiply directly: a native C library with OpenMP threads, or the slower Numba fallback. On a GPU (experimental) the weights are either dequantized to bf16 up front or kept packed in VRAM and dequantized per call with torch ops - there is no fused GPU kernel yet. During decoding this is memory-bandwidth bound, which is why smaller quantizations run faster."
  ),
  mlNode(
    "ml_kv", "KV cache + recurrent state\n(grows per token)", 1, ML_ROW.r4,
    "What the model remembers about the conversation so far",
    "app/runtime/kv_cache.py · mamba_cache.py",
    "For every attention layer, the keys and values of all tokens so far are kept so each new token only computes its own step. Memory grows linearly with context length (num_ctx), which is the main reason long contexts need so much RAM. Recurrent layers keep a fixed-size state instead."
  ),
];

const WORKFLOW_ML_EDGES = [
  // Chat model: header → prepare → one block → logits → token out
  { from: "ml_engine_server", to: "ml_load", branch: false, dir: "v" },
  { from: "ml_load", to: "ml_template", branch: false },
  { from: "ml_template", to: "ml_tokenize", branch: false },
  { from: "ml_tokenize", to: "ml_prompt_cache", branch: false },
  { from: "ml_prompt_cache", to: "ml_embed", branch: false },
  { from: "ml_vision", to: "ml_embed", branch: true },
  { from: "ml_embed", to: "ml_norm_attn", branch: false },
  { from: "ml_norm_attn", to: "ml_attention", branch: false },
  { from: "ml_norm_attn", to: "ml_linear_attn", branch: true },
  { from: "ml_attention", to: "ml_norm_ffn", branch: false },
  { from: "ml_linear_attn", to: "ml_norm_ffn", branch: true },
  { from: "ml_norm_ffn", to: "ml_ffn", branch: false },
  { from: "ml_ffn", to: "ml_layer_loop", branch: false },
  { from: "ml_layer_loop", to: "ml_final_norm", branch: false },
  { from: "ml_final_norm", to: "ml_lm_head", branch: false },
  { from: "ml_lm_head", to: "ml_sampler", branch: false },
  { from: "ml_sampler", to: "ml_stop_check", branch: false },
  { from: "ml_stop_check", to: "ml_detok", branch: false },
  { from: "ml_detok", to: "ml_ndjson", branch: false },
  { from: "ml_ndjson", to: "broadcast_publish", branch: false, dir: "v" },
  // Side paths: the token loop and the supporting components
  { from: "ml_ndjson", to: "ml_decode_loop", branch: true },
  { from: "ml_decode_loop", to: "ml_norm_attn", branch: true },
  { from: "ml_lm_head", to: "ml_kernels", branch: true },
  { from: "ml_sampler", to: "ml_kv", branch: true },
];

// Small captions above each row of the chat-model band, so "before / inside / after the layers" reads at
// a glance (rendered by workflow_diagram.js; x is the column-3 edge, clear of every edge in the band).
const WORKFLOW_ANNOTATIONS = [
  { text: "① BEFORE THE LAYERS", x: ML_COL[2], y: ML_ROW.r1 - 12, group: "ml_pre" },
  { text: "② INSIDE EVERY LAYER (×N)", x: ML_COL[2], y: ML_ROW.r2 - 12, group: "ml_attn" },
  { text: "③ AFTER THE LAYERS", x: ML_COL[2], y: ML_ROW.r3 - 12, group: "ml_post" },
];

WORKFLOW_NODES.push(...WORKFLOW_ML_NODES);
WORKFLOW_EDGES.push(...WORKFLOW_ML_EDGES);
