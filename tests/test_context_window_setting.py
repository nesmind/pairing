"""The system-wide context window (Settings > System > Context window): setting, endpoint, and its use by chats."""

import pytest
from pydantic import ValidationError

from app.routers import context_window_admin
from app.schemas import ConversationCreate
from app.schemas.settings import ContextWindow
from app.services import chat_service, context_window_setting, conversation_service, reply_generation_service


@pytest.mark.asyncio
async def test_defaults_to_the_built_in_value_and_can_be_changed(db):
    assert await context_window_setting.get_num_ctx(db) == 4096
    await context_window_admin.set_setting(ContextWindow(num_ctx=8192), db=db, _admin=None)
    assert (await context_window_admin.get_setting(db=db, _admin=None)).num_ctx == 8192
    await context_window_setting.set_num_ctx(db, 2048)  # a second save updates the row
    assert await context_window_setting.get_num_ctx(db) == 2048


@pytest.mark.parametrize("value", [255, 32769])
def test_out_of_range_values_are_rejected(value):
    with pytest.raises(ValidationError):
        ContextWindow(num_ctx=value)


@pytest.mark.asyncio
async def test_a_chat_uses_the_system_wide_value_not_its_own_saved_one(db, user, monkeypatch):
    seen: list[int] = []

    async def fake_chat_stream(_model, _messages, params):
        seen.append(params["num_ctx"])
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)
    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    conversation.params = {**conversation.params, "num_ctx": 1024}  # an old per-chat value
    await db.commit()

    await context_window_setting.set_num_ctx(db, 6144)
    async for _event in chat_service.build_reply_stream(db, conversation, user, "hi"):
        pass
    await context_window_setting.set_num_ctx(db, 3072)  # changed live - the next message picks it up
    async for _event in chat_service.build_reply_stream(db, conversation, user, "again"):
        pass

    assert seen == [6144, 3072]
    assert conversation.params["num_ctx"] == 1024  # the saved per-chat value is left alone
