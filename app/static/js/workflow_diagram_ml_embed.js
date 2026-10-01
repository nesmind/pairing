/**
 * The RAG embedding model's stages for the Stats page's "Visual chat workflow" diagram — the lane behind
 * the second "ML engine" node in workflow_diagram_data.js (an encoder-only model: it reads the whole text
 * at once and returns one vector, no generation). Same helpers/shape as workflow_diagram_ml.js, so load
 * this after it; nodes are narrower so the five fit inside the RAG column without widening the diagram.
 */

const WORKFLOW_ML_EMBED_NODES = [
  mlNode(
    "ml_emb_tok", "Tokenizer\n(WordPiece / BPE)", 0, 575,
    "Embedding model — tokenizing the question",
    "app/runtime/tokenizer.py (embedding models use their own vocabulary)",
    "The question is split into tokens with the embedding model's own vocabulary — separate from the chat model's — and wrapped in its start/end markers.",
    { x: 1176, w: 124, h: 52 }
  ),
  mlNode(
    "ml_emb_pos", "Embeddings\n+ position", 0, 575,
    "Embedding model — token and position vectors",
    "app/architectures/bert.py · nomic_bert.py",
    "Token ids become vectors and position information is added (learned position vectors for BERT, rotary embeddings for Nomic BERT), so word order is visible to the layers.",
    { x: 1022, w: 124, h: 52 }
  ),
  mlNode(
    "ml_emb_enc", "Encoder layers ×N\n(bidirectional)", 0, 575,
    "Embedding model — reading the whole text at once",
    "app/architectures/bert_layers.py",
    "Unlike the chat model, an embedding model is an encoder: every token attends to every other token in both directions, with no causal mask and no generation, then a feed-forward step — repeated over its layers (typically 6–24). One pass over the text, no cache, no sampling.",
    { x: 868, w: 124, h: 52 }
  ),
  mlNode(
    "ml_emb_pool", "Pooling\n(mean or [CLS])", 0, 655,
    "Embedding model — one vector for the whole text",
    "app/runtime/embedding_engine.py",
    "The per-token vectors are collapsed into a single vector for the whole question, by averaging them (mean pooling) or taking the special [CLS] token's vector, depending on how the model was trained.",
    { x: 868, w: 124, h: 52 }
  ),
  mlNode(
    "ml_emb_norm", "L2 normalize\n(unit vector)", 0, 655,
    "Embedding model — making vectors comparable",
    "app/runtime/embedding_engine.py",
    "The vector is scaled to length 1, so comparing two texts reduces to the angle between their vectors (cosine similarity). This is the vector the app receives and ranks stored document chunks against in the next step.",
    { x: 1022, w: 124, h: 52 }
  ),
];

const WORKFLOW_ML_EMBED_EDGES = [
  // Embedding model (RAG): entry → encoder → one unit vector → ranking
  { from: "rag_embed_ml_engine", to: "ml_emb_tok", branch: true },
  { from: "ml_emb_tok", to: "ml_emb_pos", branch: true },
  { from: "ml_emb_pos", to: "ml_emb_enc", branch: true },
  { from: "ml_emb_enc", to: "ml_emb_pool", branch: true },
  { from: "ml_emb_pool", to: "ml_emb_norm", branch: true },
  { from: "ml_emb_norm", to: "rag_rank", branch: true, dir: "v" },
];

WORKFLOW_NODES.push(...WORKFLOW_ML_EMBED_NODES);
WORKFLOW_EDGES.push(...WORKFLOW_ML_EMBED_EDGES);
