/**
 * Educational "How it works" sections for the Visual chat workflow's
 * detail panel, keyed by node id (see workflow_diagram_data.js). Kept
 * separate from the diagram data so the step-by-step ML explanations
 * can grow without bloating the node definitions. Engine-neutral on
 * purpose: this describes how a language model works in general, not
 * any particular inference server.
 */
const WORKFLOW_LEARN = {
  ml_engine_server: {
    heading: "How it works: generating a reply",
    steps: [
      {
        term: "Tokenization",
        text: "The prompt text is split into tokens: whole words, pieces of words or punctuation, each mapped to an integer ID from the model's fixed vocabulary (often 32k–150k entries). A chat template first wraps each message with special role tokens, so the model can tell system, user and assistant turns apart.",
      },
      {
        term: "Embedding",
        text: "Each token ID is looked up in a learned table and becomes a vector: a list of a few thousand numbers. From here on, the model works only with these numbers, never with text.",
      },
      {
        term: "Transformer layers",
        text: "The vectors pass through dozens of stacked layers. In each one, self-attention lets every token weigh how relevant each earlier token is to it (this is how \"it\" gets linked to the right noun), and then a feed-forward network transforms the result. The billions of learned weights in these layers are the model. They are often quantized (for example to 4-bit) so they fit in RAM/VRAM.",
      },
      {
        term: "Prefill and the KV cache",
        text: "The whole prompt is processed in one parallel pass (prefill), and each layer's attention keys and values are stored in a KV cache. Every later token only has to compute its own step against that cache instead of reprocessing the whole conversation. The context window is the maximum number of tokens this cache can hold.",
      },
      {
        term: "Logits and sampling",
        text: "The last layer outputs one score (a logit) for every token in the vocabulary. Softmax turns those scores into probabilities, and one token is picked. Temperature makes the choice sharper (low) or more adventurous (high), and top-k/top-p cut off the unlikely tail. At temperature 0 the most likely token always wins, so the output is deterministic.",
      },
      {
        term: "The autoregressive loop",
        text: "The chosen token is appended to the sequence and the model runs again to predict the next one, one token at a time. Each token is streamed out as soon as it is picked, which is why the reply appears to type itself. The loop stops at an end-of-turn token or at the maximum length.",
      },
      {
        term: "Why speed varies",
        text: "Generating each token means reading essentially all of the model's weights from memory, so tokens per second is limited mostly by memory bandwidth and model size, not raw compute. Prefill is much faster per token because the whole prompt is processed in parallel.",
      },
    ],
  },
  rag_embed_ml_engine: {
    heading: "How it works: computing an embedding",
    steps: [
      {
        term: "Tokenization",
        text: "Just like chat, the text is split into tokens and mapped to vocabulary IDs.",
      },
      {
        term: "One forward pass",
        text: "The tokens run through the model's transformer layers once. There is no generation loop and no sampling: the goal is to understand the text, not to continue it.",
      },
      {
        term: "Pooling",
        text: "The layers produce one vector per token. These are combined into a single vector for the whole text, usually by averaging them or by taking a special summary token's vector.",
      },
      {
        term: "Normalization",
        text: "The vector is scaled to length 1, so comparing two texts reduces to the angle between their vectors (cosine similarity). Texts with similar meanings point in similar directions, even when they share no words.",
      },
    ],
  },
};

/** Renders a node's optional "How it works" section as HTML, or "" if it has none. */
function renderWorkflowLearn(nodeId) {
  const learn = WORKFLOW_LEARN[nodeId];
  if (!learn) return "";
  const items = learn.steps
    .map(
      (step) => `
        <li class="leading-relaxed">
          <span class="font-semibold text-slate-200">${escapeHtml(step.term)}.</span>
          ${escapeHtml(step.text)}
        </li>`
    )
    .join("");
  return `
    <div class="mt-5 border-t border-slate-800 pt-4">
      <p class="text-xs font-semibold uppercase tracking-wide text-slate-400">${escapeHtml(learn.heading)}</p>
      <ol class="mt-2 list-decimal pl-5 space-y-2 text-sm text-slate-300">${items}</ol>
    </div>`;
}
