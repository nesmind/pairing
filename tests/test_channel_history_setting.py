"""Channel messages posted without asking the AI: the admin setting and the history filter."""

import pytest

from app.models import Message
from app.routers import channel_history_admin
from app.schemas.settings import ChannelPlainMessages
from app.services.chat_history_service import only_addressed_to_ai


@pytest.mark.asyncio
async def test_setting_defaults_to_include_and_can_be_switched_off(db):
    assert (await channel_history_admin.get_setting(db=db, _admin=None)).include is True
    await channel_history_admin.set_setting(ChannelPlainMessages(include=False), db=db, _admin=None)
    assert (await channel_history_admin.get_setting(db=db, _admin=None)).include is False


def test_only_messages_the_ai_answered_are_kept():
    asked = Message(id="q1", role="user", content="asked the AI")
    chatter = Message(id="q2", role="user", content="just chatting")
    failed_ask = Message(id="q3", role="user", content="asked, reply failed")
    messages = [
        asked,
        Message(id="a1", role="assistant", content="answer", reply_to_message_id="q1"),
        chatter,
        failed_ask,
        Message(id="a3", role="assistant", content="", status="error", reply_to_message_id="q3"),
    ]

    kept = only_addressed_to_ai(messages)

    assert [m.id for m in kept] == ["q1", "a1", "q3", "a3"]
