"""Validation and storage for a message's attachments: up to MAX_ATTACHMENT_DOCUMENTS text/PDF/docx files
and MAX_ATTACHMENT_IMAGES images per message. Split from chat_service.py for the file-size rule.

Unlike document_upload.py (the RAG knowledge base: a synced folder, chunked and embedded for every later
message), an attachment belongs to one message, is saved and read once, and is never embedded
(see chat_service.build_reply_stream).
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

# `type="image"` goes to the default vision model with the request (chat_service.py). Extensions in
# document_extract.SUPPORTED_EXTENSIONS become `"text"`: extracted into the system prompt, using the
# conversation's model (build_attachment_system_prompt in document_retrieval.py).
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


@dataclass
class AttachmentInfo:
    path: Path
    filename: str
    type: str


def _classify(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if suffix in SUPPORTED_EXTENSIONS:
        return "text"
    raise HTTPException(status_code=400, detail=f"Unsupported attachment type: {suffix or 'unknown'}")


async def save_attachments(uploads: list[UploadFile], conversation_id: str) -> list[AttachmentInfo]:
    """Validates every file (extension, per-type counts, size) and writes them to ATTACHMENTS_DIR/<conversation_id>/,
    returning where each ended up. Raises HTTPException(400) naming the limit, rejecting the whole batch before
    writing anything; call from routers/chat.py before streaming starts.
    """
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
        # Prefixed with a random token (the placeholder Message doesn't exist yet) so same-named files in one
        # conversation don't collide.
        stored_name = f"{uuid.uuid4().hex[:12]}_{name}"
        path = target_dir / stored_name
        path.write_bytes(raw)
        infos.append(AttachmentInfo(path=path, filename=name, type=attachment_type))

    return infos


async def delete_attachment_files(db: AsyncSession, message: Message) -> None:
    """Unlinks a message's attachment files and deletes its MessageAttachment rows (used by
    conversation_service._cancel_and_clear on soft-delete). No-op without attachments. File first, as in
    routers/documents.py delete_document (a stray file self-heals, a stray row wouldn't).

    Refreshes `attachments` explicitly: a message added via db.add may have a never-loaded relationship, and
    touching it outside an await fails. One cheap query makes it always safe.
    """
    await db.refresh(message, attribute_names=["attachments"])
    for attachment in list(message.attachments):
        (ATTACHMENTS_DIR / attachment.path).unlink(missing_ok=True)
        await db.delete(attachment)


def delete_conversation_attachments(conversation_id: str) -> None:
    """Removes all attachment files of a conversation (or its channel) at once: one folder per conversation
    (save_attachments), so a single rmtree. Rows are cascade-deleted (Message.attachments).
    """
    shutil.rmtree(ATTACHMENTS_DIR / conversation_id, ignore_errors=True)


def extract_attachment_text(path: Path) -> str:
    """Reads a `type="text"` attachment for the system prompt, via document_extract.extract_text (shared with RAG).
    Called at send time only; the file stays for the "📎 filename" display and re-download.
    """
    return extract_text(path.read_bytes(), path.suffix.lower())


def fold_text_attachments_into_prompt(system_prompt: str, attachments: list[AttachmentInfo]) -> str:
    """Folds each `type="text"` attachment's content into the system prompt as its own section (images are
    handled by apply_image_attachment). Called by chat_service.build_reply_stream before ollama_messages is built.
    """
    for attachment in attachments:
        if attachment.type == "text":
            system_prompt = build_attachment_system_prompt(
                system_prompt, attachment.filename, extract_attachment_text(attachment.path)
            )
    return system_prompt


async def apply_image_attachment(
    db: AsyncSession, attachments: list[AttachmentInfo], ollama_messages: list[dict]
) -> str | None:
    """Finds the at-most-one image among `attachments` and attaches its base64 bytes to the last entry of
    `ollama_messages` (the user's new turn) as the engine's per-message "images" field (ollama_client.chat_stream
    passes messages through). Returns the default vision model for this reply only, or None if no image or none
    installed (model_catalog_service.get_default_vision_model; the conversation's own model is untouched).
    """
    image = next((a for a in attachments if a.type == "image"), None)
    if image is None:
        return None
    vision_model = await model_catalog_service.get_default_vision_model(db)
    if vision_model:
        image_b64 = base64.b64encode(image.path.read_bytes()).decode("ascii")
        ollama_messages[-1] = {**ollama_messages[-1], "images": [image_b64]}
    return vision_model
