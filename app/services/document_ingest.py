"""
Chunks one document's text and embeds every chunk, writing the resulting
Document/Chunk rows. See app/services/document_extract.py for the text-
to-chunks step this builds on, and app/services/document_sync.py for the
folder-level orchestration that calls this once per new/changed file.
"""

from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import KNOWLEDGE_DIR
from app.models import Chunk, Document
from app.services.document_extract import chunk_text, vector_to_bytes
from app.services.ollama_client import OllamaError, embed

# How each user's documents stay private: everyone's private uploads
# live in their own subfolder under KNOWLEDGE_DIR/users/, and only ever
# get matched against that same user's chats (see
# app/services/document_retrieval.py). Anything an admin puts in
# KNOWLEDGE_DIR/global/ instead is visible to every user — the one
# shared part of an otherwise per-user knowledge base (see
# Document.owner_id: NULL there means "global").
GLOBAL_DIR = KNOWLEDGE_DIR / "global"
USERS_DIR = KNOWLEDGE_DIR / "users"
GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
USERS_DIR.mkdir(parents=True, exist_ok=True)

# How many files a single upload request may contain (see
# app/services/document_upload.py) — a small, deliberate cap so one
# upload can't accidentally queue up a huge, slow embedding batch.
MAX_UPLOAD_FILES = 5


def user_folder(user_id: str) -> Path:
    """The folder holding one user's own private documents, created on
    first use — most users never upload anything, so there's no reason
    to pre-create every registered user's folder up front."""
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
):
    """Chunks `text` and embeds every chunk, and only *then* writes
    anything to the database — either replacing `existing`'s chunks or
    creating a brand-new Document. Doing all the (slow, failure-prone)
    embedding calls before any write means a mid-way Ollama failure
    leaves the database exactly as it was, so the file is naturally
    retried on the next sync rather than getting stuck in a half-
    ingested state.

    An async generator, not a plain function: it yields
    `{"stage": "chunk_progress", "completed", "total"}` after every
    single chunk's embedding call. Embedding one chunk can itself take
    a noticeable moment on CPU-only hardware, and a real document can
    easily be dozens of chunks — without this, a caller streaming
    progress to a client (see document_sync.sync_folder_stream / the
    upload endpoint) would have nothing to report for the file's
    *entire* processing time, which looks exactly like the request
    hanging. The final item yielded is always
    `{"stage": "ingested", "skipped_chunks": int}`.

    A chunk whose own embedding call fails is skipped rather than
    failing the whole document — RAG_CHUNK_SIZE (app/config.py) leaves
    real margin under the embedding model's context limit, but how many
    tokens a chunk of a given length produces depends on the *script*
    (Hebrew and other non-Latin text can run 3-4x more tokens per
    character than English against a Latin-oriented tokenizer), so an
    occasional chunk can still overflow it despite that margin. Without
    this, one such chunk anywhere in a long document — even near the
    very end — would discard every other chunk's already-successful
    embedding work along with it.
    """
    pieces = chunk_text(text)
    kept: list[tuple[str, list[float]]] = []
    skipped_chunks = 0
    for completed, piece in enumerate(pieces, start=1):
        try:
            vector = await embed(piece)
        except OllamaError:
            skipped_chunks += 1
        else:
            kept.append((piece, vector))
        yield {"stage": "chunk_progress", "completed": completed, "total": len(pieces)}

    if not kept:
        raise OllamaError(f"All {len(pieces)} chunk(s) of this file failed to embed.")

    if existing is not None:
        await db.execute(delete(Chunk).where(Chunk.document_id == existing.id))
        document = existing
        document.content_hash = content_hash
    else:
        document = Document(filename=filename, content_hash=content_hash, owner_id=owner_id)
        db.add(document)
        await db.flush()  # assigns document.id without committing yet

    for piece, vector in kept:
        db.add(Chunk(document_id=document.id, text=piece, embedding=vector_to_bytes(vector)))
    document.chunk_count = len(kept)
    document.size_bytes = size_bytes
    await db.commit()
    yield {"stage": "ingested", "skipped_chunks": skipped_chunks}
