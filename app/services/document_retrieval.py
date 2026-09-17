"""
The read side of RAG: turning a live question into the most relevant
stored chunks. See app/services/document_ingest.py for how those chunks
got there in the first place.
"""

from pathlib import Path

import numpy as np
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import RAG_TOP_K
from app.models import Chunk, Document
from app.services.ollama_client import embed


def _bytes_to_vector(data: bytes) -> np.ndarray:
    """Reverses document_ingest._vector_to_bytes."""
    return np.frombuffer(data, dtype=np.float32)


async def retrieve_context(
    db: AsyncSession,
    query: str,
    user_id: str | None,
    top_k: int = RAG_TOP_K,
) -> list[dict]:
    """Embeds `query` and returns the top_k most similar chunks the
    caller is allowed to see, best match first. `user_id` set means a
    personal chat: `user_id`'s own private documents plus every shared
    "global" one. `user_id=None` means a channel chat (see
    app/models/conversation.py's channel_id) — global documents only,
    since a channel has no single private RAG scope of its own and must
    never expose one member's private uploads to the rest of the channel
    (see app/services/chat_service.py's call site). Each result carries
    the chunk's text plus which document it came from (as a plain
    display name, not its full owner-scoped path), so callers can both
    build the model's prompt (build_augmented_system_prompt below) and
    show the user which sources were actually used. Returns an empty
    list if nothing in that scope exists yet.
    """
    scope = (
        Document.owner_id.is_(None)
        if user_id is None
        else or_(Document.owner_id.is_(None), Document.owner_id == user_id)
    )
    chunks = (await db.execute(select(Chunk).join(Document).where(scope))).scalars().all()
    if not chunks:
        return []

    query_vector = np.asarray(await embed(query), dtype=np.float32)
    query_norm = np.linalg.norm(query_vector) or 1.0

    scored = []
    for chunk in chunks:
        vector = _bytes_to_vector(chunk.embedding)
        norm = np.linalg.norm(vector) or 1.0
        # Cosine similarity: how aligned two vectors are in direction,
        # ignoring magnitude — the standard way to compare embeddings.
        similarity = float(np.dot(query_vector, vector) / (query_norm * norm))
        # chunk.document is loaded via the "selectin" relationship
        # strategy (see app/models/document.py) so this attribute access
        # is safe under the app's async session without an explicit
        # eager-load option on the query above.
        scored.append((similarity, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {"text": chunk.text, "filename": Path(chunk.document.filename).name, "document_id": chunk.document_id}
        for _similarity, chunk in scored[:top_k]
    ]


def build_augmented_system_prompt(base_prompt: str, retrieved_chunks: list[dict]) -> str:
    """Appends retrieved document excerpts to the conversation's system
    prompt, clearly labeled so the model can distinguish "background
    material it was given" from "instructions about how to behave"."""
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
    """Same layering approach as build_augmented_system_prompt above, for
    a single ad-hoc file the user attached to this one message (see
    app.services.chat_attachment_service and
    app.services.chat_service.build_reply_stream's `attachment` handling)
    rather than a RAG-retrieved chunk — labeled by its own filename since
    there's only ever one, not a list of sources to disambiguate."""
    return (
        f"{base_prompt}\n\n"
        f'The user has attached a file named "{filename}" with this message. '
        "Use its content to answer if relevant.\n\n"
        f"### Attached file: {filename}\n{text}"
    )
