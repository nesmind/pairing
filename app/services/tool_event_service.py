"""Keeps the tool calls a reply made: stored on the message and sent live to everyone watching."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Message
from app.services import reply_broadcast_service

# Shown (red) when a reply used tools but the model then wrote nothing.
NO_ANSWER = "The model used a tool but gave no answer. Try again or rephrase."


async def record(db: AsyncSession, message: Message, event: dict, conversation_id: str) -> None:
    message.tool_events = [*(message.tool_events or []), event]  # new list so the JSON change is tracked
    await db.commit()
    reply_broadcast_service.publish_tool(conversation_id, event)
