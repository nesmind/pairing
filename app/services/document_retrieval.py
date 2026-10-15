"""The read side of RAG: turns a question into the most relevant stored chunks (written by document_ingest.py)."""

from pathlib import Path

import numpy as np
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import RAG_RERANK_CANDIDATES, RAG_TOP_K
from app.models import Chunk, Document
from app.services import model_catalog_service
from app.services.document_rerank import ChunkReranker
from app.services.inference_client import embed


def _bytes_to_vector(data: bytes) -> np.ndarray:
    """Reverses document_ingest._vector_to_bytes."""
    return np.frombuffer(data, dtype=np.float32)


async def retrieve_context(
    db: AsyncSession,
    query: str,
    user_id: str | None,
    top_k: int = RAG_TOP_K,
) -> list[dict]:
    """Embeds `query` and returns the top_k most similar chunks `user_id` may see, best first. `user_id` set =
    personal chat: own private documents plus shared "global" ones. `user_id=None` = channel chat
    (Conversation.channel_id): global only, so one member's private uploads are never exposed to the channel
    (chat_service.py call site). Each result has the chunk text and its document's display name (not the
    owner-scoped path), to build the prompt (build_augmented_system_prompt) and show sources. Empty list if
    nothing exists in scope.
    """
    scope = (
        Document.owner_id.is_(None)
        if user_id is None
        else or_(Document.owner_id.is_(None), Document.owner_id == user_id)
    )
    chunks = (await db.execute(select(Chunk).join(Document).where(scope))).scalars().all()
    if not chunks:
        return []

    embedding_model = await model_catalog_service.get_default_embedding_model(db)
    if embedding_model is None:
        # No embedding model installed (or the chunks' model was uninstalled): degrade as document_sync does, no hard
        # failure.
        return []

    query_vector = np.asarray(await embed(query, embedding_model), dtype=np.float32)
    query_norm = np.linalg.norm(query_vector) or 1.0

    scored = []
    for chunk in chunks:
        # Only vectors from the current model compare: another model's (recorded, or of another size when
        # unknown) waits for the next sync to re-embed it instead of failing or matching badly.
        recorded = chunk.document.embedding_model
        if recorded is not None and recorded != embedding_model:
            continue
        vector = _bytes_to_vector(chunk.embedding)
        if vector.shape != query_vector.shape:
            continue
        norm = np.linalg.norm(vector) or 1.0
        # Cosine similarity: direction alignment, ignoring magnitude; the standard embedding comparison.
        similarity = float(np.dot(query_vector, vector) / (query_norm * norm))
        # chunk.document is selectin-loaded (models/document.py), so access is safe under the async session.
        scored.append((similarity, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    # A reranker, if set, picks the best top_k from a wider similarity shortlist.
    candidates = [chunk for _similarity, chunk in scored[: max(top_k, RAG_RERANK_CANDIDATES)]]
    best = await ChunkReranker(db).order(query, candidates, top_k)
    return [
        {"text": chunk.text, "filename": Path(chunk.document.filename).name, "document_id": chunk.document_id}
        for chunk in best
    ]


def build_augmented_system_prompt(base_prompt: str, retrieved_chunks: list[dict]) -> str:
    """Appends retrieved excerpts to the system prompt, labeled so the model tells background material from
    instructions."""
    if not retrieved_chunks:
        return base_prompt
    context_block = "\n\n---\n\n".join(chunk["text"] for chunk in retrieved_chunks)
    return (
        f"{base_prompt}\n\n"
        "Use the following reference material to answer the user's next "
        "question if it's relevant. If it isn't relevant, ignore it and "
        "answer normally. Do not mention that you were given reference "
        "material.\n\n"
        f"### Reference material\n{context_block}"
    )


def build_attachment_system_prompt(base_prompt: str, filename: str, text: str) -> str:
    """Like build_augmented_system_prompt, for one ad-hoc file attached to this message
    (chat_attachment_service, chat_service.build_reply_stream); labeled by filename since there's only one.
    """
    return (
        f"{base_prompt}\n\n"
        f'The user has attached a file named "{filename}" with this message. '
        "Use its content to answer if relevant.\n\n"
        f"### Attached file: {filename}\n{text}"
    )
