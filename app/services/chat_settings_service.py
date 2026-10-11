"""Admin settings governing how a chat reply behaves (title mode, channel delivery mode, reply timeouts),
split from settings_service.py for the file-size rule; read by chat_service / reply_generation_service.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SYSTEM_OWNER_ID, AppSetting

TITLE_MODE_KEY = "title_mode"
CHANNEL_DELIVERY_MODE_KEY = "channel_delivery_mode"
REPLY_TIMEOUT_SECONDS_KEY = "reply_timeout_seconds"
VISION_REPLY_TIMEOUT_SECONDS_KEY = "vision_reply_timeout_seconds"

# "simple" (default): plain local text processing for titles; instant and immune to small/"thinking"
# model quirks (chat_service._summarize_as_title). "smart": the model invents the title; reads better but
# costs a second, slower call (off the critical path, see chat_service.build_reply_stream) that sometimes yields nothing
# useful.
_VALID_TITLE_MODES = {"simple", "smart"}
DEFAULT_TITLE_MODE = "simple"

# How other members see an in-progress/finished reply (reply_generation_service.py, reply_broadcast_service.py):
# "cheap" (default) polls; "real" pushes tokens live over the in-process hub, which reaches every member
# only with a single instance (see reply_broadcast_service).
_VALID_CHANNEL_DELIVERY_MODES = {"cheap", "real"}
DEFAULT_CHANNEL_DELIVERY_MODE = "cheap"

# Wall-clock limit for a text-only reply before it is marked failed: a backstop against hung or very slow
# models (ollama_client's httpx client has no competing timeout for chat). 0 = no timeout, for slow hardware
# that needs minutes per reply. Default 300s: generous for CPU-only, short enough to catch a stuck request.
DEFAULT_REPLY_TIMEOUT_SECONDS = 300
_MAX_REPLY_TIMEOUT_SECONDS = 3600

# Same backstop for replies with an image (apply_image_attachment). Separate because encoding the image
# (vision projector pass, before any token streams) is far slower on CPU (several minutes), and one shared
# timeout forces a bad choice. Default 300s; admins can raise it independently.
DEFAULT_VISION_REPLY_TIMEOUT_SECONDS = 300


async def get_title_mode(db: AsyncSession) -> str:
    """ "simple" local text processing or "smart" (ask the model); read by chat_service.build_reply_stream.
    System-wide, admin-configured.
    """
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, TITLE_MODE_KEY))
    if row and row.value.get("mode") in _VALID_TITLE_MODES:
        return row.value["mode"]
    return DEFAULT_TITLE_MODE


async def set_title_mode(db: AsyncSession, mode: str) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, TITLE_MODE_KEY))
    value = {"mode": mode}
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=TITLE_MODE_KEY, value=value))
    else:
        row.value = value
    await db.commit()


async def get_channel_delivery_mode(db: AsyncSession) -> str:
    """See _VALID_CHANNEL_DELIVERY_MODES. Read fresh on every call (unlike settings_service.get_proxy_mode):
    routers/chat.py `/subscribe` and chat.js need changes at once.
    """
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, CHANNEL_DELIVERY_MODE_KEY))
    if row and row.value.get("mode") in _VALID_CHANNEL_DELIVERY_MODES:
        return row.value["mode"]
    return DEFAULT_CHANNEL_DELIVERY_MODE


async def set_channel_delivery_mode(db: AsyncSession, mode: str) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, CHANNEL_DELIVERY_MODE_KEY))
    value = {"mode": mode}
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=CHANNEL_DELIVERY_MODE_KEY, value=value))
    else:
        row.value = value
    await db.commit()


async def get_reply_timeout_seconds(db: AsyncSession) -> int:
    """See DEFAULT_REPLY_TIMEOUT_SECONDS. Read fresh by reply_generation_service at each generation, so changes
    apply to the next message. A stored value outside [0, _MAX_REPLY_TIMEOUT_SECONDS] (only via a hand-edited
    row) falls back to the default.
    """
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, REPLY_TIMEOUT_SECONDS_KEY))
    if row:
        value = row.value.get("timeout_seconds")
        if isinstance(value, int) and 0 <= value <= _MAX_REPLY_TIMEOUT_SECONDS:
            return value
    return DEFAULT_REPLY_TIMEOUT_SECONDS


async def set_reply_timeout_seconds(db: AsyncSession, timeout_seconds: int) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, REPLY_TIMEOUT_SECONDS_KEY))
    value = {"timeout_seconds": timeout_seconds}
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=REPLY_TIMEOUT_SECONDS_KEY, value=value))
    else:
        row.value = value
    await db.commit()


async def get_vision_reply_timeout_seconds(db: AsyncSession) -> int:
    """See DEFAULT_VISION_REPLY_TIMEOUT_SECONDS. Used instead of get_reply_timeout_seconds only for replies with
    an image (reply_generation_service.stream_reply's has_image); read fresh.
    """
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, VISION_REPLY_TIMEOUT_SECONDS_KEY))
    if row:
        value = row.value.get("timeout_seconds")
        if isinstance(value, int) and 0 <= value <= _MAX_REPLY_TIMEOUT_SECONDS:
            return value
    return DEFAULT_VISION_REPLY_TIMEOUT_SECONDS


async def set_vision_reply_timeout_seconds(db: AsyncSession, timeout_seconds: int) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, VISION_REPLY_TIMEOUT_SECONDS_KEY))
    value = {"timeout_seconds": timeout_seconds}
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=VISION_REPLY_TIMEOUT_SECONDS_KEY, value=value))
    else:
        row.value = value
    await db.commit()
