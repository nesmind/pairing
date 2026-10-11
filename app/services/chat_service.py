"""Core chat logic: trims history to the context window, builds the system prompt from notes + RAG, calls
the engine and persists both sides. routers/chat.py only wraps build_reply_stream's events into SSE.
Title generation lives in title_service.py and history trimming/merging in chat_history_service.py
(file-size rule).
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import ATTACHMENTS_DIR
from app.models import Conversation, Message, MessageAttachment, User
from app.services import (
    channel_history_setting,
    chat_attachment_service,
    chat_history_service,
    chat_prompt_service,
    chat_settings_service,
    context_window_setting,
    conversation_service,
    model_catalog_service,
    reply_generation_service,
    title_service,
)
from app.services.chat_attachment_service import AttachmentInfo


async def build_reply_stream(
    db: AsyncSession,
    conversation: Conversation,
    user: User,
    content: str,
    *,
    ask_ai: bool = True,
    attachments: list[AttachmentInfo] | None = None,
):
    """Async generator of events for routers/chat.py to SSE-encode:
      {"user_message_id": str}                      first event, always
      {"chunk": str}                                while streaming
      {"error": str}                                the engine failed mid-stream
      {"done": True, "title": str, "sources": [...]} once persisted
    ("title" only for personal chats. With `ask_ai` False, "done" follows "user_message_id" directly.)

    "user_message_id" lets the sender's tab tag its optimistic bubble, so a later channel poll/live-watch
    (reply_generation_service.stream_reply's cancel_on_disconnect, chat.js ownSendInFlight) doesn't duplicate it.

    `ask_ai=False` (chat.js, channels only; personal chats always ask) saves the message but generates no
    reply: no RAG/history, no stream_reply, no assistant Message. It still counts as history for later asks
    (chat_history_service.trim_history reads every message).

    The user's message is saved first, so it survives a failed reply. Title depends on the admin mode
    (chat_settings_service.get_title_mode): "simple" (title_service.maybe_set_title) sets it with the user's
    message, needing no reply; "smart" (title_service.maybe_generate_title_with_model) asks the model after
    "done" is yielded, so it never delays the reply. Both skipped for channels (title is the channel name,
    conversation_service.update_conversation refuses edits).

    The reply runs in reply_generation_service as a detached background task. On disconnect
    (stream_reply's cancel_on_disconnect): channels always finish (others may watch); personal chats are
    cancelled. Either way a clean status="error" record keeps what was generated. This function adds the
    title handling and the title in personal "done" events (channels have none).
    """
    attachments = attachments or []
    # num_ctx is one system-wide value (Settings > System > Context window).
    params = {**conversation.params, "num_ctx": await context_window_setting.get_num_ctx(db)}
    is_personal = conversation.channel_id is None
    title_mode = await chat_settings_service.get_title_mode(db) if is_personal else None
    # Personal chats always ask the AI, whatever the caller passes (no other human to message). Enforced here
    # as well as in the frontend, in case a client sends ask_ai=False.
    ask_ai = ask_ai or is_personal

    user_message = Message(conversation_id=conversation.id, role="user", content=content, sender_id=user.id)
    # Appended through the relationship: user_message.id isn't assigned until flush, and SQLAlchemy wires the FK.
    user_message.attachments = [
        MessageAttachment(
            # Stored relative to ATTACHMENTS_DIR so moving that folder doesn't strand urls.
            path=str(attachment.path.relative_to(ATTACHMENTS_DIR)),
            filename=attachment.filename,
            type=attachment.type,
        )
        for attachment in attachments
    ]
    db.add(user_message)

    if title_mode == "simple":
        title_service.maybe_set_title(conversation, content)

    await db.commit()

    yield {"user_message_id": user_message.id}

    # "Simple" mode already knows the title, so the sidebar needn't wait for the full reply. ("Smart" resends
    # the unchanged "New chat" placeholder: harmless.)
    if is_personal:
        yield {"title": conversation.title}

    if not ask_ai:
        done_event = {"done": True, "sources": []}
        if is_personal:
            done_event["title"] = conversation.title
        yield done_event
        return

    # History = conversation_service.visible_messages (excludes admin-deleted) plus user_message explicitly:
    # with expire_on_commit=False (app/database.py) the loaded `conversation.messages` isn't refreshed, and
    # db.add alone doesn't update it, so otherwise history would miss the current question.
    past = conversation_service.visible_messages(conversation)
    if not is_personal and not await channel_history_setting.get_include_plain_messages(db):
        past = chat_history_service.only_addressed_to_ai(past)
    history = chat_history_service.trim_history(
        [*chat_history_service.without_failed_replies(past), user_message],
        params["num_ctx"],
        # Room for a full reply (num_predict); else a long chat cuts the reply short.
        reserved_tokens=max(512, params.get("num_predict", 0)),
    )

    # Notes/RAG/attachment layering: see chat_prompt_service.build_system_prompt.
    system_prompt, turn_context, source_filenames = await chat_prompt_service.build_system_prompt(
        db, conversation, user, content, params, attachments
    )
    chat_prompt_service.prepend_turn_context(history, turn_context)
    # Always sent, even when "": an explicit empty system message tells the engine there is no preamble,
    # whereas omitting it makes some chat-template layers (e.g. Matricxon's VicunaPromptBuilder) substitute their own
    # default.
    ollama_messages = [{"role": "system", "content": system_prompt}] + history

    # None (stream_reply falls back to conversation.model) unless `attachments` has an image, then the
    # default vision model for this reply only (apply_image_attachment). Also the has_image signal below.
    reply_model = (
        await chat_attachment_service.apply_image_attachment(db, attachments, ollama_messages) if attachments else None
    )
    if reply_model is None:
        # get_default_vision_model only returns installed models, so only conversation.model needs checking
        # (resolve_installed_model: an engine switch can leave the stored tag unknown to the new engine).
        resolved_model = await model_catalog_service.resolve_installed_model(db, user, conversation.model)
        if resolved_model != conversation.model:
            conversation.model = resolved_model
            await db.commit()

    # Channels always finish generating (other members may rely on it); a personal chat has one viewer, so
    # leaving stops generation (cancel_on_disconnect=True). Both leave a clean status="error" record
    # (reply_generation_service.stream_reply).
    async for event in reply_generation_service.stream_reply(
        db,
        conversation,
        ollama_messages,
        source_filenames,
        reply_to_message_id=user_message.id,
        cancel_on_disconnect=is_personal,
        model=reply_model,
        has_image=reply_model is not None,
        params=params,
    ):
        if is_personal and event.get("done"):
            # "Simple" mode: the title was set before streaming, so it is final. "Smart": still the old value
            # (usually "New chat"); chat.js polls for the scheduled result. reply_generation_service knows nothing of
            # titles (channels have none), so it is layered on here.
            event = {**event, "title": conversation.title}
        yield event

        if is_personal and event.get("done") and title_mode == "smart":
            # Scheduled as a detached task (title_service.schedule_smart_title_generation), not awaited: chat.js stops
            # reading after "done" and ASGI then cancels this generator, so an await would silently drop the title.
            title_service.schedule_smart_title_generation(conversation.id, content)
