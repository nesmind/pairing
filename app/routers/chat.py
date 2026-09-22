"""
The core chat endpoint: takes a new user message, streams back the
model's reply token-by-token over Server-Sent Events (SSE), and persists
both sides of the exchange once it's done. See
app/services/chat_service.py for the actual business logic — this file
only wraps its structured events into the SSE wire format.

SSE (rather than WebSockets) was chosen because it's simpler for a
one-way "server keeps talking, client just listens" stream, works over
plain HTTP (so it survives proxies/load balancers more easily), and the
browser's built-in EventSource-style `fetch` + `ReadableStream` handling
is all the frontend needs (see app/static/js/chat.js).
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

    # Must be checked and raised here, before StreamingResponse is ever
    # constructed below — Starlette locks in this response's 200 status
    # the moment it starts pulling from event_stream()'s body, which only
    # happens *after* this function has already returned that
    # StreamingResponse object to it. Raising from inside
    # build_reply_stream's own generator body instead would be too late
    # to ever produce a clean 409 here. See
    # conversation_service.has_active_reply's own docstring for why this
    # check is best-effort, not race-free.
    if ask_ai and conversation.channel_id is not None and await conversation_service.has_active_reply(db, conversation):
        raise HTTPException(
            status_code=409,
            detail=(
                "The AI is still replying to another message in this channel — "
                "wait for it to finish, or send this as a normal message instead."
            ),
        )

    # Same reasoning as the 409 above: validated (and, on success,
    # written to disk) before StreamingResponse ever starts, so a bad
    # batch cleanly 400s instead of surfacing as a broken stream.
    attachments = [a for a in attachments if a.filename]
    attachment_infos = (
        await chat_attachment_service.save_attachments(attachments, conversation_id) if attachments else []
    )

    async def event_stream():
        # SSE wire format: each event is "data: <payload>\n\n". The
        # payload is JSON so the frontend can tell a text chunk apart
        # from the final "done" marker unambiguously.
        async for event in build_reply_stream(
            db, conversation, user, content, ask_ai=ask_ai, attachments=attachment_infos
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # Disables buffering in nginx-style reverse proxies so
            # chunks reach the browser as they're generated instead of
            # being held until the response closes.
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
    """The only way a chat/channel message's attachment is ever served —
    not mounted under /static, so access stays gated by the same
    membership/ownership check every other endpoint here already uses
    (see conversation_service.get_accessible_conversation_or_404),
    rather than being reachable by anyone who guesses a path."""
    attachment = await db.get(MessageAttachment, attachment_id)
    if attachment is None:
        raise HTTPException(status_code=404, detail="Attachment not found")
    # Looked up by attachment.message_id directly rather than via
    # attachment.message (that relationship has no lazy="selectin" —
    # unlike every other relationship read straight off of an object this
    # session's own code touches — so reading it here would trigger a
    # real lazy load outside of an awaited context and crash with
    # sqlalchemy.exc.MissingGreenlet, confirmed live).
    message = await db.get(Message, attachment.message_id)
    await conversation_service.get_accessible_conversation_or_404(db, message.conversation_id, user)
    # A "text" attachment (see chat_attachment_service.SUPPORTED_IMAGE_EXTENSIONS
    # for what isn't one) can be an .html/.htm file — served with no
    # explicit media_type, FileResponse would guess Content-Type from the
    # filename and could hand back "text/html", which a browser renders
    # (executing any embedded script) if it's ever opened directly rather
    # than downloaded. Forcing a generic, never-executable type here is
    # the actual fix; the automatic Content-Disposition: attachment
    # (from passing `filename` below) forcing a download today is only a
    # second, coincidental layer, not something to rely on alone. An
    # image needs its real type instead, since chat.js renders it inline
    # via <img src=...>.
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
    """A read-only SSE stream for watching an in-flight (or just-
    finished) reply live without being the one who sent it — used two
    ways (see chat.js): a channel member's own tab, for "real" channel
    delivery mode (see app.services.chat_settings_service.get_channel_delivery_mode);
    and a personal chat's own owner, reconnecting (a reload, a different
    tab/device) mid-reply after whatever connection originally sent it
    dropped, so they see the reply as it finishes rather than a blank
    spot where it should be. Taps directly into
    app.services.reply_broadcast_service's hub, the same one
    app.services.reply_generation_service's own relay for the sender
    publishes to — so this endpoint has nothing to do with whether
    generation is still running by the time a caller connects; it only
    affects who else finds out about it, and how. Gated only by the
    existing access check (channel membership, or personal ownership) —
    a channel's "cheap" vs "real" delivery-mode setting is purely a
    frontend decision about whether to bother opening this connection at
    all; a personal chat has no such setting since only its owner can
    ever open it.
    """
    # Only the access check matters here (channel membership, or
    # personal ownership) — the conversation itself isn't otherwise
    # needed, unlike the /stream endpoint above.
    await conversation_service.get_accessible_conversation_or_404(db, conversation_id, user)

    async def event_stream():
        state, queue = reply_broadcast_service.subscribe(conversation_id)
        try:
            if state is not None:
                # Lets a member who opens the chat mid-generation see the
                # reply's current partial content immediately, instead of
                # a blank bubble until the next chunk happens to arrive.
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "sync": True,
                            "message_id": state.message_id,
                            "content": state.content,
                            "status": state.status,
                            "sources": state.sources,
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
