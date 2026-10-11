"""Chunks one document's text, embeds every chunk and writes the Document/Chunk rows. Text step:
document_extract.py; folder orchestration (once per new/changed file): document_sync.py.
"""

from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import KNOWLEDGE_DIR
from app.models import Chunk, Document
from app.services.document_extract import chunk_text, vector_to_bytes
from app.services.inference_client import InferenceError, embed

# Privacy: private uploads live in KNOWLEDGE_DIR/users/<id>/ and match only that user's chats
# (document_retrieval.py). KNOWLEDGE_DIR/global/ (admin) is visible to everyone (Document.owner_id NULL = "global").
GLOBAL_DIR = KNOWLEDGE_DIR / "global"
USERS_DIR = KNOWLEDGE_DIR / "users"
GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
USERS_DIR.mkdir(parents=True, exist_ok=True)

# Max files per upload request (document_upload.py), so one upload can't queue a huge, slow embedding batch.
MAX_UPLOAD_FILES = 5


def user_folder(user_id: str) -> Path:
    """One user's private documents folder, created on first use (most users never upload)."""
    path = USERS_DIR / user_id
    path.mkdir(parents=True, exist_ok=True)
    return path


async def ingest_document_stream(
    db: AsyncSession,
    filename: str,
    content_hash: str,
    text: str,
    existing: Document | None,
    owner_id: str | None,
    size_bytes: int,
    embedding_model: str,
):
    """Chunks `text`, embeds every chunk, and only *then* writes to the DB (replacing `existing`'s chunks or
    creating a Document). Embedding is slow and failure-prone, so a mid-way engine failure leaves the DB as it
    was and the file is retried at the next sync.

    An async generator: it yields `{"stage": "chunk_progress", "completed", "total"}` after each chunk's
    embedding. On CPU a chunk takes a noticeable moment and a document may have dozens, so a caller streaming
    progress (document_sync.sync_folder_stream / the upload endpoint) would otherwise look hung. The last item
    is `{"stage": "ingested", "skipped_chunks": int}`.

    A chunk whose embedding fails is skipped, not the whole document. RAG_CHUNK_SIZE (app/config.py) leaves
    margin under the model's context limit, but tokens per character depend on script (Hebrew and other
    non-Latin text can be 3-4x English), so a chunk can still overflow. Otherwise one such chunk, even near the
    end, would discard all the other embeddings.
    """
    pieces = chunk_text(text)
    kept: list[tuple[str, list[float]]] = []
    skipped_chunks = 0
    for completed, piece in enumerate(pieces, start=1):
        try:
            vector = await embed(piece, embedding_model)
        except InferenceError:
            skipped_chunks += 1
        else:
            kept.append((piece, vector))
        yield {"stage": "chunk_progress", "completed": completed, "total": len(pieces)}

    if not kept:
        raise InferenceError(f"All {len(pieces)} chunk(s) of this file failed to embed.")

    if existing is not None:
        await db.execute(delete(Chunk).where(Chunk.document_id == existing.id))
        document = existing
        document.content_hash = content_hash
    else:
        document = Document(filename=filename, content_hash=content_hash, owner_id=owner_id)
        db.add(document)
        await db.flush()  # assigns document.id without committing

    for piece, vector in kept:
        db.add(Chunk(document_id=document.id, text=piece, embedding=vector_to_bytes(vector)))
    document.chunk_count = len(kept)
    document.size_bytes = size_bytes
    await db.commit()
    yield {"stage": "ingested", "skipped_chunks": skipped_chunks}
