"""Admin "Reset channel": wipes a channel's chat history but keeps the channel, its members and pinned notes."""

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Channel, Message, MessageAttachment
from app.services import (
    chat_attachment_service,
    conversation_service,
    reply_broadcast_service,
    reply_generation_service,
)


class ChannelResetService:
    @staticmethod
    async def reset(db: AsyncSession, channel: Channel) -> int:
        """Deletes every message (and attachment file) of the channel's shared conversation, cancelling any reply
        still generating. Returns how many messages were removed. No undo."""
        conversation = channel.conversation
        streaming = [m.id for m in conversation.messages if m.status == "streaming"]
        for message_id in streaming:  # no `await` between the two: see conversation_service._cancel_and_clear
            reply_generation_service.mark_message_deleted(message_id)
            reply_generation_service.cancel_generation(message_id)
        if streaming:
            reply_broadcast_service.publish_deleted(conversation.id)
        count = len(conversation.messages)
        chat_attachment_service.delete_conversation_attachments(conversation.id)
        ids = [m.id for m in conversation.messages]
        await db.execute(delete(MessageAttachment).where(MessageAttachment.message_id.in_(ids)))
        await db.execute(delete(Message).where(Message.conversation_id == conversation.id))
        conversation.title = "New chat"
        await db.commit()
        db.expire(conversation, ["messages"])
        await conversation_service._drop_engine_cache(conversation.id)
        return count
