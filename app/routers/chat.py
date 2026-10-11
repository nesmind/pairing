"""The core chat endpoint: takes a user message, streams the reply over SSE, persists both sides.
Logic: app/services/chat_service.py; this file wraps its events in SSE.

SSE over WebSockets: simpler for one-way streams, survives proxies, and chat.js needs only
`fetch` + `ReadableStream`.
"""

import json
import mimetypes

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import ATTACHMENTS_DIR
from app.database import get_db
from app.models import Message, MessageAttachment, User
from app.services import chat_attachment_service, conversation_service, reply_broadcast_service
from app.services.auth_service import get_current_user
from app.services.chat_service import build_reply_stream
from app.services.model_catalog_service import HiddenModelTags

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/{conversation_id}/stream")
async def stream_reply(
    conversation_id: str,
    content: str = Form(..., min_length=1),
    ask_ai: bool = Form(True),
    attachments: list[UploadFile] = File(default=[]),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    conversation = await conversation_service.get_accessible_conversation_or_404(db, conversation_id, user)
    if ask_ai and await HiddenModelTags(db).blocks(user, conversation.model):
        raise HTTPException(status_code=409, detail="This model is disabled by an admin. Pick another model.")

    # Raise before StreamingResponse is built: Starlette locks in 200 once it starts pulling the body, so a
    # 409 from inside build_reply_stream would be too late. Best-effort, not race-free
    # (conversation_service.has_active_reply).
    if ask_ai and conversation.channel_id is not None and await conversation_service.has_active_reply(db, conversation):
        raise HTTPException(
            status_code=409,
            detail=(
                "The AI is still replying to another message in this channel — "
                "wait for it to finish, or send this as a normal message instead."
            ),
        )

    # Validated and written before streaming starts, so a bad batch gives a clean 400.
    attachments = [a for a in attachments if a.filename]
    attachment_infos = (
        await chat_attachment_service.save_attachments(attachments, conversation_id) if attachments else []
    )

    async def event_stream():
        # SSE format "data: <payload>\n\n"; JSON payloads tell text chunks from the final "done".
        async for event in build_reply_stream(
            db, conversation, user, content, ask_ai=ask_ai, attachments=attachment_infos
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # Stops nginx-style proxies buffering chunks.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/attachments/{attachment_id}")
async def get_attachment(
    attachment_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """The only way attachments are served (not under /static), so the same access check applies
    (conversation_service.get_accessible_conversation_or_404).
    """
    attachment = await db.get(MessageAttachment, attachment_id)
    if attachment is None:
        raise HTTPException(status_code=404, detail="Attachment not found")
    # Look up by attachment.message_id: `attachment.message` lacks lazy="selectin" and would crash
    # outside an awaited context (MissingGreenlet).
    message = await db.get(Message, attachment.message_id)
    await conversation_service.get_accessible_conversation_or_404(db, message.conversation_id, user)
    # "text" attachments may be.html/.htm; guessing Content-Type could return text/html, which a
    # browser would run. Force a generic non-executable type (Content-Disposition: attachment is only a second layer).
    # Images keep their real type for chat.js's <img>.
    media_type = mimetypes.guess_type(attachment.filename or "")[0] or "application/octet-stream"
    if attachment.type != "image":
        media_type = "application/octet-stream"
    return FileResponse(ATTACHMENTS_DIR / attachment.path, filename=attachment.filename, media_type=media_type)


@router.get("/{conversation_id}/subscribe")
async def subscribe_to_reply(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Read-only SSE stream to watch an in-flight or just-finished reply without sending it (chat.js): a channel
    member's tab in "real" delivery mode (chat_settings_service.get_channel_delivery_mode), or an owner
    reconnecting after a dropped connection. Taps reply_broadcast_service's hub; independent of whether
    generation is still running. Gated by the access check only; "cheap" vs "real" delivery is a frontend choice.
    """
    # Only the access check matters; the conversation itself is not needed.
    await conversation_service.get_accessible_conversation_or_404(db, conversation_id, user)

    async def event_stream():
        state, queue = reply_broadcast_service.subscribe(conversation_id)
        try:
            if state is not None:
                # Shows the reply's partial content at once for someone joining mid-generation.
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "sync": True,
                            "message_id": state.message_id,
                            "content": state.content,
                            "status": state.status,
                            "sources": state.sources,
                            "tool_events": state.tool_events,
                        }
                    )
                    + "\n\n"
                )
                if state.status != "streaming":
                    return
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("done") or event.get("error") or event.get("deleted"):
                    return
        finally:
            reply_broadcast_service.unsubscribe(conversation_id, queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
