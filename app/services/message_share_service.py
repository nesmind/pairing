"""Shares one message from a private chat into a channel: a text copy posted as the sharer's own message,
never triggering an AI reply. HTTP side: routers/conversations.py share endpoint; UI: chat.js
(chat_share.js) button and "Shared from a private chat" remark.
"""

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation, Message, User
from app.models._base import utcnow
from app.services import channel_service

SHAREABLE_ROLES = ("user", "assistant")


class MessageShareService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def share_to_channel(
        self, user: User, conversation: Conversation, message_id: str, channel_id: str
    ) -> Message:
        """Copies `message_id` (from `conversation`, which the caller confirmed `user` may open) into `channel_id`'s
        shared conversation as a "user" message from `user`, tagged with the original role (`Message.shared_from`).
        Only text travels: no attachments, sources or link back. Raises 400 for a non-private conversation or an
        unshareable message (streaming, failed, empty, system), 404 for an unknown message/channel, 403 if `user`
        isn't a channel member.
        """
        if conversation.channel_id is not None:
            raise HTTPException(status_code=400, detail="Only a private chat's messages can be shared to a channel.")
        source = next((m for m in conversation.messages if m.id == message_id), None)
        if source is None or source.status == "deleted":
            raise HTTPException(status_code=404, detail="Message not found")
        if source.role not in SHAREABLE_ROLES or source.status != "complete" or not source.content.strip():
            raise HTTPException(status_code=400, detail="This message can't be shared.")

        channel = await channel_service.get_channel_or_404(self._db, channel_id)
        if not any(member.user_id == user.id for member in channel.members):
            raise HTTPException(status_code=403, detail="You're not a member of that channel.")

        shared = Message(
            conversation_id=channel.conversation.id,
            role="user",
            content=source.content,
            sender_id=user.id,
            shared_from=source.role,
        )
        self._db.add(shared)
        channel.conversation.updated_at = utcnow()
        await self._db.commit()
        # `sender`/`attachments` are selectin relationships a new row hasn't loaded; MessageOut reads both and a lazy
        # load under AsyncSession would fail, so load them explicitly.
        await self._db.refresh(shared, attribute_names=["sender", "attachments"])
        return shared
