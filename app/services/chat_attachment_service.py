"""
Validation and on-disk storage for a chat/channel message's own
attachments — up to app.config.MAX_ATTACHMENT_DOCUMENTS text/PDF/docx/etc.
files and MAX_ATTACHMENT_IMAGES image, combined in one message. Split out
from app/services/chat_service.py to keep that file focused and under
CLAUDE.md's file-size rule.

Deliberately not the same machinery as app/services/document_upload.py:
that's the shared/personal RAG knowledge base (a whole folder synced as a
standing collection, chunked and embedded for later retrieval across
*every* future message). An attachment here is tied to one specific
message, saved and read once, with no chunking/embedding pipeline at all
— see app/services/chat_service.py's build_reply_stream for how each type
is actually used.
"""

import base64
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import ATTACHMENTS_DIR, MAX_ATTACHMENT_DOCUMENTS, MAX_ATTACHMENT_IMAGES, MAX_ATTACHMENT_MB
from app.models import Message
from app.services import model_catalog_service
from app.services.document_extract import SUPPORTED_EXTENSIONS, extract_text
from app.services.document_retrieval import build_attachment_system_prompt
from app.services.document_upload import safe_filename

# Classified as `type="image"` — sent to the default vision model as
# part of the request (see chat_service.py). Anything in
# document_extract.SUPPORTED_EXTENSIONS instead becomes `"text"` — its
# content is extracted and folded into the system prompt, using whatever
# model the conversation already had (see build_attachment_system_prompt
# in app/services/document_retrieval.py).
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


@dataclass
class AttachmentInfo:
    path: Path
    filename: str
    type: str  # "image" | "text"


def _classify(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if suffix in SUPPORTED_EXTENSIONS:
        return "text"
    raise HTTPException(status_code=400, detail=f"Unsupported attachment type: {suffix or 'unknown'}")


async def save_attachments(uploads: list[UploadFile], conversation_id: str) -> list[AttachmentInfo]:
    """Validates every file in `uploads` (extension, combined per-type
    counts, and each one's size) and writes them all to
    ATTACHMENTS_DIR/<conversation_id>/, returning where each ended up.
    Raises HTTPException(400) — naming the actual limit — for anything
    unsupported, oversized, or over MAX_ATTACHMENT_DOCUMENTS/
    MAX_ATTACHMENT_IMAGES, rejecting the whole batch before writing
    anything: meant to be called directly from app/routers/chat.py, before
    build_reply_stream/StreamingResponse ever starts, same as that
    router's other pre-stream validation (see its own comment on why)."""
    classified = [
        (upload, safe_filename(upload.filename), _classify(safe_filename(upload.filename))) for upload in uploads
    ]

    image_count = sum(1 for _, _, t in classified if t == "image")
    document_count = sum(1 for _, _, t in classified if t == "text")
    if image_count > MAX_ATTACHMENT_IMAGES:
        raise HTTPException(status_code=400, detail=f"Only {MAX_ATTACHMENT_IMAGES} image is allowed per message.")
    if document_count > MAX_ATTACHMENT_DOCUMENTS:
        raise HTTPException(
            status_code=400, detail=f"Only {MAX_ATTACHMENT_DOCUMENTS} document files are allowed per message."
        )

    target_dir = ATTACHMENTS_DIR / conversation_id
    target_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = MAX_ATTACHMENT_MB * 1024 * 1024

    infos = []
    for upload, name, attachment_type in classified:
        raw = await upload.read()
        if len(raw) > max_bytes:
            raise HTTPException(status_code=400, detail=f"Attachment exceeds the {MAX_ATTACHMENT_MB}MB limit.")
        # Prefixed with a short random token (not the message id — these
        # files are written before the placeholder Message row exists) so
        # two attachments with the same original filename in one
        # conversation never collide.
        stored_name = f"{uuid.uuid4().hex[:12]}_{name}"
        path = target_dir / stored_name
        path.write_bytes(raw)
        infos.append(AttachmentInfo(path=path, filename=name, type=attachment_type))

    return infos


async def delete_attachment_files(db: AsyncSession, message: Message) -> None:
    """Unlinks every one of a single message's attachment files from disk
    and deletes its MessageAttachment rows — used by
    conversation_service._cancel_and_clear when a message carrying one or
    more is soft-deleted. No-op if it has none. File-first, same ordering
    as app/routers/documents.py's delete_document (a stray file with no
    row self-heals; a stray row with no file wouldn't).

    Explicitly refreshes `attachments` first rather than trusting it's
    already loaded: `message` often reaches here freshly constructed and
    committed within the same request/test (see the module-split import-
    binding-adjacent gotcha this codebase already documents elsewhere for
    a relationship added via db.add() directly rather than through its
    parent's collection) — a persistent object's never-yet-accessed
    relationship attribute triggers a real lazy-load on first touch,
    which fails outside an awaited context. One cheap extra query
    guarantees this always works regardless of how the caller got here."""
    await db.refresh(message, attribute_names=["attachments"])
    for attachment in list(message.attachments):
        (ATTACHMENTS_DIR / attachment.path).unlink(missing_ok=True)
        await db.delete(attachment)


def delete_conversation_attachments(conversation_id: str) -> None:
    """Removes every attachment file for a whole conversation at once —
    used when the conversation itself (or its owning channel) is being
    deleted outright. Every attachment lives under one folder per
    conversation (see save_attachments above), so this is a single
    rmtree rather than tracking each message's file individually. The
    MessageAttachment rows themselves are cascade-deleted by the ORM the
    same way Message rows already are (see Message.attachments)."""
    shutil.rmtree(ATTACHMENTS_DIR / conversation_id, ignore_errors=True)


def extract_attachment_text(path: Path) -> str:
    """Reads a `type="text"` attachment's content for folding into the
    system prompt — reuses document_extract.extract_text (the same
    per-format parsing the RAG knowledge base already relies on) rather
    than duplicating it. Called fresh at send time only; the file itself
    stays on disk purely so the message can still show "📎 filename" and
    be re-downloaded later."""
    return extract_text(path.read_bytes(), path.suffix.lower())


def fold_text_attachments_into_prompt(system_prompt: str, attachments: list[AttachmentInfo]) -> str:
    """Folds every `type="text"` attachment's extracted content in, each
    as its own section (a `type="image"` one is a no-op here — see
    apply_image_attachment below for its own handling). Called by
    chat_service.build_reply_stream *before* ollama_messages is built
    from system_prompt + history, since a text attachment's content
    needs to already be part of the system message by then."""
    for attachment in attachments:
        if attachment.type == "text":
            system_prompt = build_attachment_system_prompt(
                system_prompt, attachment.filename, extract_attachment_text(attachment.path)
            )
    return system_prompt


async def apply_image_attachment(
    db: AsyncSession, attachments: list[AttachmentInfo], ollama_messages: list[dict]
) -> str | None:
    """Finds the at-most-one image among `attachments` (a `type="text"`
    one is a no-op here — already folded into the system prompt by
    fold_text_attachments_into_prompt above) and attaches its raw bytes,
    base64-encoded, to the *last* entry of `ollama_messages` (the user's
    own turn chat_service just added) as the ML engine's own per-message
    "images" field — ollama_client.chat_stream passes `messages` straight
    through, so nothing else needs to know this key exists. Returns the
    admin-configured default vision model to use for *this* reply only
    (None if no image, or none installed) — see
    model_catalog_service.get_default_vision_model's own docstring for
    why this never touches the conversation's own model."""
    image = next((a for a in attachments if a.type == "image"), None)
    if image is None:
        return None
    vision_model = await model_catalog_service.get_default_vision_model(db)
    if vision_model:
        image_b64 = base64.b64encode(image.path.read_bytes()).decode("ascii")
        ollama_messages[-1] = {**ollama_messages[-1], "images": [image_b64]}
    return vision_model
