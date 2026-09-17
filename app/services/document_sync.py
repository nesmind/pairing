"""
Folder-level orchestration: makes the Document/Chunk rows for one owner
match whatever's actually sitting on disk, calling
app/services/document_ingest.py once per new/changed file. Used at app
startup (app/main.py) and right after a browser upload (see
app/services/document_upload.py).
"""

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import EMBEDDING_MODEL, KNOWLEDGE_DIR
from app.models import Document
from app.services.document_extract import SUPPORTED_EXTENSIONS, extract_text, hash_bytes
from app.services.document_ingest import GLOBAL_DIR, USERS_DIR, ingest_document_stream, user_folder
from app.services.ollama_client import OllamaError, has_embedding_model


async def sync_folder_stream(db: AsyncSession, folder: Path, owner_id: str | None):
    """Makes the Document/Chunk rows for one owner (a specific user, or
    the shared `None`/global scope) match whatever's currently sitting
    in `folder`: new files get chunked+embedded, edited files get
    re-ingested, and files removed from the folder have their database
    rows deleted. Safe to call repeatedly — on every app startup, and
    right after an upload — since unchanged files are skipped via their
    stored content hash instead of being re-embedded every time.

    An async generator rather than a plain function so a caller that
    cares (the upload endpoint, for a progress bar) can watch each file
    resolve one at a time; `_drain_sync_stats` below is for callers that
    just want the final tally. The very last item yielded is always
    `{"stage": "done", "stats": {...}}`.

    Each file's own text-extraction and embedding is wrapped in its own
    try/except: a single corrupt or unusually-structured file (a
    password-protected or malformed PDF, say) reports itself as one
    per-file error and the rest of the batch still goes through — it
    used to be that an uncaught exception from *any* file's parsing
    would blow up the whole sync, silently abandoning every file after
    it (including ones that had nothing wrong with them).
    """
    stats = {"added": 0, "updated": 0, "removed": 0, "skipped": 0, "errors": []}

    existing_docs = (await db.execute(select(Document).where(Document.owner_id == owner_id))).scalars().all()
    existing_by_relpath = {doc.filename: doc for doc in existing_docs}
    on_disk = (
        {path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS}
        if folder.is_dir()
        else set()
    )
    on_disk_relpaths = {str(path.relative_to(KNOWLEDGE_DIR)) for path in on_disk}

    removed_any = False
    for relpath, document in existing_by_relpath.items():
        if relpath not in on_disk_relpaths:
            await db.delete(document)
            stats["removed"] += 1
            removed_any = True
    if removed_any:
        await db.commit()

    # Removing stale rows above only needed the database, but adding or
    # re-embedding files needs the embedding model — check once up front
    # rather than failing once per file with the same underlying cause.
    embeddings_available = (not on_disk) or await has_embedding_model()
    if not embeddings_available:
        stats["errors"].append(
            "No embedding model available — new/changed files were not (re-)ingested. "
            f"Run 'ollama pull {EMBEDDING_MODEL}'.",
        )

    files = sorted(on_disk)
    for index, path in enumerate(files, start=1):
        relpath = str(path.relative_to(KNOWLEDGE_DIR))
        raw = path.read_bytes()
        content_hash = hash_bytes(raw)
        existing = existing_by_relpath.get(relpath)

        if existing is not None and existing.content_hash == content_hash:
            stats["skipped"] += 1
            yield {
                "stage": "file_done",
                "filename": path.name,
                "index": index,
                "total": len(files),
                "status": "skipped",
            }
            continue
        if not embeddings_available:
            yield {
                "stage": "file_done",
                "filename": path.name,
                "index": index,
                "total": len(files),
                "status": "error",
                "error": "No embedding model available.",
            }
            continue

        yield {"stage": "file_start", "filename": path.name, "index": index, "total": len(files)}

        try:
            text = extract_text(raw, path.suffix.lower())
        except Exception as exc:
            error = f"couldn't read this file ({exc})."
            stats["errors"].append(f"{path.name}: {error}")
            yield {
                "stage": "file_done",
                "filename": path.name,
                "index": index,
                "total": len(files),
                "status": "error",
                "error": error,
            }
            continue

        if not text.strip():
            error = "no extractable text found (empty file, or a scanned/image-only PDF?)"
            stats["errors"].append(f"{path.name}: {error}")
            yield {
                "stage": "file_done",
                "filename": path.name,
                "index": index,
                "total": len(files),
                "status": "error",
                "error": error,
            }
            continue

        try:
            skipped_chunks = 0
            chunks_total = 0
            async for progress in ingest_document_stream(db, relpath, content_hash, text, existing, owner_id, len(raw)):
                if progress["stage"] == "chunk_progress":
                    chunks_total = progress["total"]
                    yield {
                        "stage": "chunk_progress",
                        "filename": path.name,
                        "index": index,
                        "total": len(files),
                        "chunk": progress["completed"],
                        "chunks_total": chunks_total,
                    }
                elif progress["stage"] == "ingested":
                    skipped_chunks = progress["skipped_chunks"]
            status = "updated" if existing is not None else "added"
            stats[status] += 1
            if skipped_chunks:
                # Still a success overall — just missing a bit of this
                # file's content from the knowledge base, not the whole
                # file — see ingest_document_stream's docstring for why
                # an individual chunk can fail on its own.
                stats["errors"].append(
                    f"{path.name}: {skipped_chunks} of {chunks_total} chunk(s) couldn't be "
                    "embedded and were skipped (the rest of the file was ingested normally).",
                )
            yield {"stage": "file_done", "filename": path.name, "index": index, "total": len(files), "status": status}
        except OllamaError as exc:
            await db.rollback()
            stats["errors"].append(f"{path.name}: {exc}")
            yield {
                "stage": "file_done",
                "filename": path.name,
                "index": index,
                "total": len(files),
                "status": "error",
                "error": str(exc),
            }
        except Exception as exc:
            # Anything else (a bug in a specific parser, an odd file
            # structure pypdf/python-docx/etc. doesn't expect, ...) —
            # reported the same way rather than taking the whole request
            # down with it.
            await db.rollback()
            error = f"unexpected error while processing this file ({exc})."
            stats["errors"].append(f"{path.name}: {error}")
            yield {
                "stage": "file_done",
                "filename": path.name,
                "index": index,
                "total": len(files),
                "status": "error",
                "error": error,
            }

    yield {"stage": "done", "stats": stats}


async def _drain_sync_stats(folder: Path, owner_id: str | None, db: AsyncSession) -> dict:
    """Runs sync_folder_stream to completion and returns just the final
    stats — for callers (startup, and sync_global/sync_user below) that
    don't care about per-file progress, only the end result."""
    async for event in sync_folder_stream(db, folder, owner_id):
        if event["stage"] == "done":
            return event["stats"]
    raise AssertionError("sync_folder_stream ended without a 'done' event")


def merge_sync_stats(a: dict, b: dict) -> dict:
    """Combines two sync-stats dicts (see sync_folder_stream) into one, for
    callers that sync more than one folder in a single request/startup
    step (e.g. global + one user's own folder) and want a single
    combined result."""
    return {
        "added": a["added"] + b["added"],
        "updated": a["updated"] + b["updated"],
        "removed": a["removed"] + b["removed"],
        "skipped": a["skipped"] + b["skipped"],
        "errors": a["errors"] + b["errors"],
    }


async def sync_global(db: AsyncSession) -> dict:
    """Syncs the shared knowledge base every user's chats can draw on."""
    return await _drain_sync_stats(GLOBAL_DIR, None, db)


async def sync_user(db: AsyncSession, user_id: str) -> dict:
    """Syncs one user's own private knowledge base."""
    return await _drain_sync_stats(user_folder(user_id), user_id, db)


async def sync_all_on_startup(db: AsyncSession) -> dict:
    """Syncs the global folder plus every user subfolder that currently
    exists on disk, for app startup — covers files dropped directly
    onto the server filesystem while the app wasn't running. Doesn't
    iterate the full user table: a user who's never uploaded anything
    has no folder yet, and there's nothing to sync for them."""
    stats = await sync_global(db)
    if USERS_DIR.is_dir():
        for entry in sorted(USERS_DIR.iterdir()):
            if entry.is_dir():
                stats = merge_sync_stats(stats, await _drain_sync_stats(entry, entry.name, db))
    return stats
