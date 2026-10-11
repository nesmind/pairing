"""Folder-level orchestration: makes one owner's Document/Chunk rows match the disk, calling
document_ingest.py per new/changed file. Used at startup (app/main.py) and after browser uploads (document_upload.py).
"""

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import EMBEDDING_MODEL, KNOWLEDGE_DIR
from app.models import Document
from app.services import model_catalog_service
from app.services.document_extract import SUPPORTED_EXTENSIONS, extract_text, hash_bytes
from app.services.document_ingest import GLOBAL_DIR, USERS_DIR, ingest_document_stream, user_folder
from app.services.inference_client import InferenceError


async def sync_folder_stream(db: AsyncSession, folder: Path, owner_id: str | None):
    """Makes one owner's (a user or the shared `None`/global scope) rows match `folder`: new files are
    chunked+embedded, edited ones re-ingested, removed ones deleted. Safe to repeat (startup, after upload):
    unchanged files are skipped by content hash.

    An async generator so the upload endpoint can show per-file progress; `_drain_sync_stats` is for callers
    wanting only the final tally. The last item is `{"stage": "done", "stats": {...}}`.

    Each file's extraction and embedding has its own try/except: a corrupt file (e.g. a password-protected or
    malformed PDF) reports one per-file error and the rest still go through (an uncaught exception once
    abandoned every later file).
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

    # Removing stale rows needs only the DB, but adding/re-embedding needs the embedding model: resolve it once up
    # front, not per file.
    embedding_model = await model_catalog_service.get_default_embedding_model(db)
    embeddings_available = (not on_disk) or embedding_model is not None
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
            async for progress in ingest_document_stream(
                db, relpath, content_hash, text, existing, owner_id, len(raw), embedding_model
            ):
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
                # Still a success: this file just lacks some content (see ingest_document_stream on per-chunk failures).
                stats["errors"].append(
                    f"{path.name}: {skipped_chunks} of {chunks_total} chunk(s) couldn't be "
                    "embedded and were skipped (the rest of the file was ingested normally).",
                )
            yield {"stage": "file_done", "filename": path.name, "index": index, "total": len(files), "status": status}
        except InferenceError as exc:
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
            # Anything else (parser bug, odd file structure): reported the same way, not taking the request down.
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
    """Runs sync_folder_stream to completion and returns the final stats, for callers that don't need per-file
    progress."""
    async for event in sync_folder_stream(db, folder, owner_id):
        if event["stage"] == "done":
            return event["stats"]
    raise AssertionError("sync_folder_stream ended without a 'done' event")


def merge_sync_stats(a: dict, b: dict) -> dict:
    """Combines two sync-stats dicts (sync_folder_stream) for callers syncing several folders in one step (e.g. global +
    a user's)."""
    return {
        "added": a["added"] + b["added"],
        "updated": a["updated"] + b["updated"],
        "removed": a["removed"] + b["removed"],
        "skipped": a["skipped"] + b["skipped"],
        "errors": a["errors"] + b["errors"],
    }


async def sync_global(db: AsyncSession) -> dict:
    """Syncs the shared knowledge base."""
    return await _drain_sync_stats(GLOBAL_DIR, None, db)


async def sync_user(db: AsyncSession, user_id: str) -> dict:
    """Syncs one user's private knowledge base."""
    return await _drain_sync_stats(user_folder(user_id), user_id, db)


async def sync_all_on_startup(db: AsyncSession) -> dict:
    """Syncs the global folder plus every existing user subfolder at startup (covers files dropped while the app
    was down). Skips the user table: no folder means nothing to sync.
    """
    stats = await sync_global(db)
    if USERS_DIR.is_dir():
        for entry in sorted(USERS_DIR.iterdir()):
            if entry.is_dir():
                stats = merge_sync_stats(stats, await _drain_sync_stats(entry, entry.name, db))
    return stats
