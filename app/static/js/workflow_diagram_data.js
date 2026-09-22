/**
 * Content for the Stats page's "Visual chat workflow" diagram — a hand-laid-
 * out (not force-simulated) node graph documenting exactly what this
 * app's own chat pipeline does, end to end, verified against the real
 * source (app/routers/chat.py, app/services/chat_service.py,
 * app/services/chat_history_service.py, app/services/reply_generation_service.py,
 * app/services/inference_client.py, app/services/ollama_client.py, app/services/ollama_pool.py,
 * app/services/matricxon_client.py, app/services/matricxon_pool.py,
 * app/services/model_catalog_service.py, app/services/reply_broadcast_service.py,
 * app/services/document_retrieval.py, app/services/reply_termination_service.py,
 * app/services/reply_cross_instance_service.py, app/services/title_service.py,
 * app/services/chat_attachment_service.py). Kept separate from
 * workflow_diagram.js (rendering only) so this substantial amount of
 * descriptive text can be edited independently of the D3 mechanics.
 *
 * Node shape: {id, label, category, x, y, w, h, detail: {server, title, location, body}}.
 * `detail.server` is which actual process runs this step — "Browser"
 * (chat.js, in the user's own tab), "pAIring" (this app's own backend),
 * or "ML engine" (the separate model-serving process this app talks to —
 * Ollama or Matricxon, whichever Settings > External servers has active
 * right now; see app.services.inference_client/engine_service) —
 * shown as a badge in the click-to-expand detail panel.
 *
 * Edge shape: {from, to, branch}. `branch: true` edges are the side
 * paths (RAG, attachments, error handling, live delivery to other
 * viewers, channel differences, smart titles) forking off the main
 * top-to-bottom chain — rendered dashed so the diagram reads as one
 * clear happy path with real detail branching off it, not one tangle.
 */

// The app's own display name — read from <body data-app-name> (see app/config.py's APP_NAME, the single source
// of truth every server-rendered page passes down; base.html sets this attribute on every page's <body>) rather
// than hardcoded here, so every "pAIring" label below stays correct even if the app is ever renamed/rebranded.
const APP_NAME = document.body.dataset.appName || "pAIring";

// Themed via ThemeColors (see app/static/js/theme.js) rather than fixed hex — each category keeps the same
// --chart-N slot across every theme, so a returning user's mental map of "frontend = the first/coolest hue"
// stays intact even though the actual color shifts with their chosen theme.
const WORKFLOW_CATEGORY_COLORS = {
  frontend: ThemeColors.chart(1),
  router: ThemeColors.chart(2),
  service: ThemeColors.chart(4),
  background_task: ThemeColors.chart(7),
  engine: ThemeColors.chart(8),
  broadcast: ThemeColors.chart(5),
};

const WORKFLOW_CATEGORY_LABELS = {
  frontend: "Frontend (browser)",
  router: "FastAPI router",
  service: `${APP_NAME} layer`,
  background_task: "Detached background task",
  engine: "ML engine server (models)",
  broadcast: `Pub/sub broadcast (${APP_NAME})`,
};

const WORKFLOW_NODES = [
  // ---- Main chain (the linear happy path) --------------------------
  {
    id: "chat_js_send",
    label: "sendMessage()",
    category: "frontend",
    x: 560,
    y: 40,
    w: 200,
    h: 64,
    detail: {
      server: "Browser",
      title: "Optimistic send",
      location: "app/static/js/chat.js — sendMessage",
      body: "Renders the user's own bubble, and (if asking the AI) an empty streaming assistant bubble, immediately client-side — before any network request has even started. The UI feels instant regardless of server/model latency.",
    },
  },
  {
    id: "chat_js_stream",
    label: "readAssistantReplyStream()",
    category: "frontend",
    x: 560,
    y: 150,
    w: 220,
    h: 64,
    detail: {
      server: "Browser",
      title: "The actual HTTP request",
      location: "app/static/js/chat.js — readAssistantReplyStream",
      body: "Sends POST /api/chat/{conversation_id}/stream via XMLHttpRequest — deliberately not fetch(). The reason: an attachment upload needs a real progress bar, and only xhr.upload.onprogress exposes upload progress; fetch() has no such event. The body is FormData (content, ask_ai, one or more attachments files), not JSON — the browser sets its own multipart boundary. The response is still genuinely SSE-shaped (data: {...}\\n\\n frames) — this code just parses those frames by hand out of the growing xhr.responseText instead of using the native EventSource API, since EventSource only supports body-less GET requests.",
    },
  },
  {
    id: "chat_router_stream",
    label: "POST /api/chat/{id}/stream",
    category: "router",
    x: 560,
    y: 260,
    w: 220,
    h: 64,
    detail: {
      server: APP_NAME,
      title: "Router entry point",
      location: "app/routers/chat.py — stream_reply",
      body: "Everything that can fail with a clean 4xx happens here, BEFORE the StreamingResponse starts (once that response's body generator is first read from, the 200 status is already locked in): (1) membership/ownership check, (2) a channel-only 409 if that channel already has a reply in flight, (3) attachment validation and saving to disk. Only once all of that passes does it return StreamingResponse(event_stream(), media_type=\"text/event-stream\").",
    },
  },
  {
    id: "chat_service_build",
    label: "build_reply_stream()",
    category: "service",
    x: 560,
    y: 370,
    w: 220,
    h: 64,
    detail: {
      server: APP_NAME,
      title: "Orchestration begins",
      location: "app/services/chat_service.py — build_reply_stream",
      body: "Creates and commits the user's own Message row immediately (durably saved regardless of whether generation later succeeds). In \"simple\" title mode, sets the conversation's title here too — pure local text processing, no model call. Yields {user_message_id} as the very first SSE event so the browser can tag its optimistic bubble. Also resolves conversation.model against whichever engine is currently active (model_catalog_service.resolve_installed_model) — an admin switching engines leaves every existing conversation's model column pointed at a tag the newly active engine may never have heard of, so this falls back to something actually installed there instead of failing outright. Builds a trimmed conversation history (chat_history_service.trim_history, kept under a token-budget proxy based on num_ctx) — which also folds any run of consecutive same-role messages into one turn (e.g. several channel posts before anyone asked the AI): a strict chat template rejects back-to-back same-role turns outright, so this merges them, each line attributed to its own sender, before the model ever sees them. Then hands off to system-prompt assembly.",
    },
  },
  {
    id: "chat_prompt_service",
    label: "build_system_prompt()",
    category: "service",
    x: 560,
    y: 480,
    w: 220,
    h: 64,
    detail: {
      server: APP_NAME,
      title: "Assembling what the model actually sees",
      location: "app/services/chat_prompt_service.py — build_system_prompt",
      body: "Layers the system prompt in order: (1) any pinned persona/rules/skill notes, (2) RAG retrieval (see the branch below), (3) any ad-hoc text-attachment content folded in last. The result plus the trimmed history becomes ollama_messages. If the message has an image attachment, its vision-model override is also applied here (see the Attachments branch) before handing off to generation.",
    },
  },
  {
    id: "reply_gen_service",
    label: "reply_generation_service.stream_reply()",
    category: "service",
    x: 560,
    y: 590,
    w: 250,
    h: 64,
    detail: {
      server: APP_NAME,
      title: "Handing off to a detached background task",
      location: "app/services/reply_generation_service.py — stream_reply",
      body: "Creates and commits the placeholder assistant Message row (status=\"streaming\"), registers it with the in-process broadcast hub (reply_broadcast_service.start_topic), then SCHEDULES the real generation as a fully detached asyncio.Task rather than awaiting it inline. This function itself then becomes a pure relay/subscriber to that same broadcast topic, re-yielding whatever the detached task publishes back up through this same request's own StreamingResponse.",
    },
  },
  {
    id: "run_generation_task",
    label: "_run_generation() [detached task]",
    category: "background_task",
    x: 560,
    y: 700,
    w: 240,
    h: 64,
    detail: {
      server: APP_NAME,
      title: "Why detached, not awaited inline",
      location: "app/services/reply_generation_service.py — _run_generation",
      body: "Runs as an independent asyncio.Task with its OWN fresh database session (not the original request's, which may already be gone by the time this finishes) — this is deliberate: it means the sender closing their tab, navigating away, or the original HTTP request disconnecting only ever cancels the relay/subscriber side, never silently loses generation that was already underway. A personal chat's task IS still cancelled on sender-disconnect (cancel_on_disconnect=True); a channel's task never is, since other members may still be relying on it. Reads the admin-configured reply-timeout setting fresh on every call.",
    },
  },
  {
    id: "inference_client_stream",
    label: "inference_client.chat_stream()",
    category: "service",
    x: 560,
    y: 810,
    w: 220,
    h: 64,
    detail: {
      server: APP_NAME,
      title: "Resolving the active engine and calling it",
      location: "app/services/inference_client.py — chat_stream",
      body: "This is the step that actually hands the conversation off to the AI language model for inference. Reads engine_service.current_engine() — an admin-configurable choice, Settings > External servers' \"Active engine\" picker — and dispatches to that engine's own client unchanged: ollama_client.chat_stream (host failover via ollama_pool — local mode: the one Ollama process this app manages itself; remote mode: the admin's configured host list) or matricxon_client.chat_stream (same failover shape via matricxon_pool). Either engine's own error type is normalized into one InferenceError here, so nothing downstream needs to know which engine actually served the request. If the model is a \"thinking\" model, raw reasoning tokens are stripped before anything is yielded upward (Ollama's own ollama_client only — Matricxon implements no thinking-capable architecture yet).",
    },
  },
  {
    id: "ml_engine_server",
    label: "ML engine",
    category: "engine",
    x: 560,
    y: 920,
    w: 180,
    h: 64,
    detail: {
      server: "ML engine",
      title: "Where the actual inference happens",
      location: "The active engine's own process/host — Ollama or Matricxon, not this app's own code",
      body: "Runs the real AI/machine learning model (loading its trained weights into RAM/VRAM as needed) and streams its response back as NDJSON — one JSON object per line, each carrying the next piece of generated text, not the SSE format the browser eventually receives (that translation happens back in inference_client.chat_stream). Ollama is the original third-party server this app started against; Matricxon is this project's own from-scratch GGUF inference runtime (no llama.cpp dependency), built deliberately Ollama-API-compatible so this whole pipeline needed no changes to support it. This is also where an image attachment's base64 bytes are processed by a vision-capable model (Ollama only — Matricxon has no vision/mmproj support yet), or an embedding request for RAG is run through a separate embedding model (see the RAG branch's own dedicated ML engine step).",
    },
  },
  {
    id: "broadcast_publish",
    label: "publish_chunk()",
    category: "broadcast",
    x: 560,
    y: 1030,
    w: 200,
    h: 64,
    detail: {
      server: APP_NAME,
      title: "Fanning one chunk out to every interested listener",
      location: "app/services/reply_broadcast_service.py — publish_chunk",
      body: "Every chunk from the active ML engine is immediately published to this conversation's in-process pub/sub topic (so any live viewer sees it with zero added delay) AND separately batched into the database — message.content is only actually committed every ~200 characters or ~0.4 seconds, whichever comes first, deliberately decoupled from the live broadcast so a live viewer never notices the batching.",
    },
  },
  {
    id: "relay_back",
    label: "relay loop re-yields",
    category: "service",
    x: 560,
    y: 1140,
    w: 210,
    h: 64,
    detail: {
      server: APP_NAME,
      title: "Back through the same request that started it",
      location: "app/services/reply_generation_service.py — stream_reply's own relay loop",
      body: "The subscriber loop started two steps ago picks up each broadcast event and re-yields it back up through chat_service.build_reply_stream, then app/routers/chat.py's own SSE encoder — arriving on the SAME original StreamingResponse the browser's XHR request is reading, even though the actual generation is running in a completely separate asyncio.Task by this point.",
    },
  },
  {
    id: "chat_js_render",
    label: "chat.js renders incrementally",
    category: "frontend",
    x: 560,
    y: 1250,
    w: 220,
    h: 64,
    detail: {
      server: "Browser",
      title: "Back in the browser",
      location: "app/static/js/chat.js — readAssistantReplyStream's frame parser",
      body: "Each SSE frame parsed out of xhr.responseText updates the assistant bubble's rendered text immediately — the same incremental Markdown render runs on every chunk, not just once at the end.",
    },
  },
  {
    id: "stream_done",
    label: "status=\"complete\" + publish_done",
    category: "service",
    x: 560,
    y: 1360,
    w: 230,
    h: 64,
    detail: {
      server: APP_NAME,
      title: "Finishing the reply",
      location: "app/services/reply_generation_service.py — _run_generation's success path",
      body: "On the model's stream ending cleanly: message.content is set to the full accumulated reply, status becomes \"complete\", sources (if RAG contributed) are attached, and it's all committed — then publish_done fires the final {done: true, sources, title} event. This is the ONLY path to status=\"complete\" — every other exit (an InferenceError from either engine, a timeout, a cancellation, an unhandled exception) instead leaves status=\"error\", by design, so a reply never just silently vanishes.",
    },
  },

  // ---- Branch: RAG retrieval ----------------------------------------
  {
    id: "rag_retrieve",
    label: "retrieve_context()",
    category: "service",
    x: 860,
    y: 430,
    w: 200,
    h: 60,
    detail: {
      server: APP_NAME,
      title: "RAG — deciding scope",
      location: "app/services/document_retrieval.py — retrieve_context, called from chat_prompt_service.build_system_prompt",
      body: "Runs BEFORE generation starts, as part of assembling the system prompt. Scope is decided right here: a personal chat passes the real user's own id (sees their own private documents PLUS every shared/global document); a channel conversation passes None (global documents only — a channel never exposes one member's private uploads to the rest of the channel).",
    },
  },
  {
    id: "rag_embed",
    label: "inference_client.embed()",
    category: "service",
    x: 1100,
    y: 430,
    w: 200,
    h: 60,
    detail: {
      server: APP_NAME,
      title: "RAG — sending the question to an embedding model",
      location: "app/services/inference_client.py — embed, using model_catalog_service.get_default_embedding_model",
      body: `${APP_NAME}'s own code builds and sends this request — a SEPARATE, smaller machine learning model from the conversation's own chat model, purpose-built for similarity search rather than text generation, and which model that actually is comes from the admin-configured default (Settings > System — falls back to whichever embedding model is actually installed if never configured, or if the configured one was since uninstalled). Same active-engine dispatch as the main chat_stream step above. The actual embedding computation doesn't happen here; it happens on the ML engine, the next step.`,
    },
  },
  {
    id: "rag_embed_ml_engine",
    label: "ML engine",
    category: "engine",
    x: 1100,
    y: 505,
    w: 160,
    h: 40,
    detail: {
      server: "ML engine",
      title: "Where the embedding vector is actually computed",
      location: "The active engine's own process/host — running the admin's configured default embedding model",
      body: "Runs the embedding model's own neural-network inference over the message text, producing a fixed-length numeric vector that captures its meaning. This is the same kind of machine learning model family as the main chat model (both are neural networks served by the active engine) — just trained and shaped for similarity search instead of generating language.",
    },
  },
  {
    id: "rag_rank",
    label: "cosine-similarity ranking",
    category: "service",
    x: 1100,
    y: 575,
    w: 200,
    h: 60,
    detail: {
      server: APP_NAME,
      title: "RAG — finding the best-matching chunks",
      location: "app/services/document_retrieval.py — retrieve_context",
      body: "Plain math, not machine learning — every candidate stored Chunk (already scoped by owner above) is scored by cosine similarity against the embedding vector the AI model just produced, sorted descending, and the top-k (default 4, adjustable per conversation as rag_top_k) are kept.",
    },
  },
  {
    id: "rag_augment",
    label: "build_augmented_system_prompt()",
    category: "service",
    x: 860,
    y: 575,
    w: 200,
    h: 60,
    detail: {
      server: APP_NAME,
      title: "RAG — folding results into the prompt",
      location: "app/services/document_retrieval.py — build_augmented_system_prompt",
      body: "The top-k matched chunks are appended to the system prompt as clearly-labeled reference material, with explicit instructions for the model to use them only if relevant. These same results also become the \"sources\" shown to the user under the finished reply. If RAG itself fails (e.g. the embedding model is unreachable), this degrades silently to no RAG context rather than failing the whole chat request.",
    },
  },

  // ---- Branch: attachments -------------------------------------------
  {
    id: "attachment_save",
    label: "save_attachments()",
    category: "service",
    x: 240,
    y: 260,
    w: 200,
    h: 60,
    detail: {
      server: APP_NAME,
      title: "Attachments — validated and saved up front",
      location: "app/services/chat_attachment_service.py — save_attachments, called from chat.py: stream_reply",
      body: "Extension, count, and size limits are all checked and the files written to disk BEFORE the StreamingResponse begins — so a rejected attachment is a clean 400 error, not a broken stream.",
    },
  },
  {
    id: "attachment_vision",
    label: "apply_image_attachment()",
    category: "service",
    x: 30,
    y: 400,
    w: 220,
    h: 60,
    detail: {
      server: APP_NAME,
      title: "Attachments — an image swaps the model, just for this reply",
      location: "app/services/chat_attachment_service.py — apply_image_attachment",
      body: "If the message carries an image, this reply uses the admin-configured default AI vision model — a machine learning model trained to understand image content, not just text — instead of the conversation's own model, but only for this one reply. conversation.model itself is never touched, so the very next message reverts automatically.",
    },
  },
  {
    id: "attachment_text_fold",
    label: "fold text into system prompt",
    category: "service",
    x: 270,
    y: 400,
    w: 220,
    h: 60,
    detail: {
      server: APP_NAME,
      title: "Attachments — a text file folds straight into the prompt",
      location: "app/services/chat_attachment_service.py",
      body: "A non-image attachment's extracted text content is folded directly into the system prompt (same layering approach as RAG results) — no model override, since there's no vision involved.",
    },
  },

  // ---- Branch: error / timeout / cancellation ------------------------
  {
    id: "termination_service",
    label: "reply_termination_service",
    category: "service",
    x: 860,
    y: 700,
    w: 230,
    h: 64,
    detail: {
      server: APP_NAME,
      title: "When a reply doesn't finish cleanly",
      location: "app/services/reply_termination_service.py",
      body: "An InferenceError (either engine's own failure, normalized — see inference_client.chat_stream above) or a reply-timeout both mark the message status=\"error\", persisting whatever partial content was generated so far — never silently discarded. A timeout ALSO force-unloads the model (inference_client.stop_model, dispatched to whichever engine is active): closing the HTTP connection to the engine alone doesn't reliably stop an in-progress compute phase, especially prompt/image processing, which doesn't appear to check for client disconnection at all. A personal chat's sender-disconnect cancellation can't safely await anything in its own already-cancelled task, so it schedules a brand-new, uncancelled task to do this same persistence instead.",
    },
  },

  // ---- Branch: live delivery to other viewers ------------------------
  {
    id: "subscribe_router",
    label: "GET /{id}/subscribe",
    category: "router",
    x: 860,
    y: 1010,
    w: 220,
    h: 60,
    detail: {
      server: APP_NAME,
      title: "How a DIFFERENT tab or member sees the same reply live",
      location: "app/routers/chat.py — GET /{conversation_id}/subscribe",
      body: "A second, completely independent StreamingResponse per other viewer, subscribed to the exact same in-process pub/sub topic as the sender's own request. Sends an initial {sync: true, ...} catch-up frame built from an in-memory snapshot, so a viewer connecting mid-reply doesn't just see a blank bubble. Explicitly in-process-memory only — no Redis or broker — so this does not work across multiple app instances.",
    },
  },
  {
    id: "cross_instance_fallback",
    label: "reply_cross_instance_service",
    category: "service",
    x: 860,
    y: 1140,
    w: 220,
    h: 60,
    detail: {
      server: APP_NAME,
      title: "The multi-instance safety net",
      location: "app/services/reply_cross_instance_service.py",
      body: "Because the broadcast hub above is in-process-only, the sender's own relay loop (reply_generation_service.stream_reply) has a 3-second timeout on waiting for a broadcast event — if nothing arrives in time (e.g. generation is actually running on a sibling instance), it falls back to polling the database directly instead. This is specifically for the SENDER's own reconnect case, not general delivery to other viewers.",
    },
  },

  // ---- Branch: channel-specific differences --------------------------
  {
    id: "channel_ask_ai_false",
    label: "ask_ai=false: post-only",
    category: "service",
    x: 240,
    y: 530,
    w: 220,
    h: 60,
    detail: {
      server: APP_NAME,
      title: "Channels — posting without asking the AI",
      location: "app/services/chat_service.py — build_reply_stream's early-return branch",
      body: "In a channel, a member can post a plain message with no AI reply requested at all. The user Message row is saved exactly as normal, but essentially everything downstream — history building, RAG, the assistant Message row, reply_generation_service — is skipped entirely. A personal chat always asks the AI regardless of what the client sends, since there's no other reason to be in a 1:1 chat.",
    },
  },
  {
    id: "channel_delivery_mode",
    label: "channel delivery mode",
    category: "service",
    x: 240,
    y: 1030,
    w: 220,
    h: 60,
    detail: {
      server: APP_NAME,
      title: "Channels — how OTHER members find out",
      location: "app/services/chat_settings_service.py — get_channel_delivery_mode (admin setting)",
      body: "Controls how members who AREN'T the sender learn about a reply. \"cheap\" (the default): other tabs periodically poll for new/changed messages — works correctly across multiple app instances, since every instance reads the same database. \"real\": other tabs get live, token-by-token pushes via the same broadcast/subscribe mechanism the sender's own tab uses — explicitly only reliable with a single app instance, since the broadcast hub has no cross-instance broker.",
    },
  },

  // ---- Branch: smart title generation --------------------------------
  {
    id: "smart_title",
    label: "schedule_smart_title_generation()",
    category: "background_task",
    x: 860,
    y: 1360,
    w: 250,
    h: 64,
    detail: {
      server: APP_NAME,
      title: "Generating a real title, after the fact",
      location: "app/services/title_service.py — schedule_smart_title_generation",
      body: "In \"smart\" title mode (personal chats only), a SEPARATE detached task fires only after the reply's own {done} event has already reached the browser — an inline await at that point would just be cancelled, since chat.js disconnects right after \"done\". Makes a short, non-streaming call asking the AI language model itself to generate a title, sanity-checks the result isn't obviously truncated reasoning, then commits it directly. chat.js polls the conversation a few times afterward to pick up the new title once it lands.",
    },
  },
];

const WORKFLOW_EDGES = [
  // Main chain
  { from: "chat_js_send", to: "chat_js_stream", branch: false },
  { from: "chat_js_stream", to: "chat_router_stream", branch: false },
  { from: "chat_router_stream", to: "chat_service_build", branch: false },
  { from: "chat_service_build", to: "chat_prompt_service", branch: false },
  { from: "chat_prompt_service", to: "reply_gen_service", branch: false },
  { from: "reply_gen_service", to: "run_generation_task", branch: false },
  { from: "run_generation_task", to: "inference_client_stream", branch: false },
  { from: "inference_client_stream", to: "ml_engine_server", branch: false },
  { from: "ml_engine_server", to: "broadcast_publish", branch: false },
  { from: "broadcast_publish", to: "relay_back", branch: false },
  { from: "relay_back", to: "chat_js_render", branch: false },
  { from: "chat_js_render", to: "stream_done", branch: false },

  // RAG branch
  { from: "chat_prompt_service", to: "rag_retrieve", branch: true },
  { from: "rag_retrieve", to: "rag_embed", branch: true },
  { from: "rag_embed", to: "rag_embed_ml_engine", branch: true },
  { from: "rag_embed_ml_engine", to: "rag_rank", branch: true },
  { from: "rag_rank", to: "rag_augment", branch: true },
  { from: "rag_augment", to: "chat_prompt_service", branch: true },

  // Attachments branch
  { from: "chat_router_stream", to: "attachment_save", branch: true },
  { from: "attachment_save", to: "attachment_vision", branch: true },
  { from: "attachment_save", to: "attachment_text_fold", branch: true },
  { from: "attachment_vision", to: "chat_prompt_service", branch: true },
  { from: "attachment_text_fold", to: "chat_prompt_service", branch: true },

  // Error/timeout/cancellation branch
  { from: "run_generation_task", to: "termination_service", branch: true },
  { from: "inference_client_stream", to: "termination_service", branch: true },

  // Live delivery branch
  { from: "broadcast_publish", to: "subscribe_router", branch: true },
  { from: "relay_back", to: "cross_instance_fallback", branch: true },

  // Channel differences branch
  { from: "chat_service_build", to: "channel_ask_ai_false", branch: true },
  { from: "broadcast_publish", to: "channel_delivery_mode", branch: true },

  // Smart title branch
  { from: "stream_done", to: "smart_title", branch: true },
];
