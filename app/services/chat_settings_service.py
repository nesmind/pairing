"""Admin-configured settings that govern how a chat reply itself
behaves, as opposed to app/services/settings_service.py's account/
upload-limit/instance settings — split out purely to keep that file
under CLAUDE.md's file-size rule as this list has grown (title mode,
channel delivery mode, and now the reply timeout below), not because
these settings share any runtime behavior with each other beyond being
read by the reply-generation path (app.services.chat_service /
app.services.reply_generation_service).
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SYSTEM_OWNER_ID, AppSetting

TITLE_MODE_KEY = "title_mode"
CHANNEL_DELIVERY_MODE_KEY = "channel_delivery_mode"
REPLY_TIMEOUT_SECONDS_KEY = "reply_timeout_seconds"
VISION_REPLY_TIMEOUT_SECONDS_KEY = "vision_reply_timeout_seconds"

# "simple" (the out-of-the-box default) generates a new chat's title
# with plain local text processing — instant, and immune to however a
# small/"thinking" model might otherwise behave (see
# app.services.chat_service._summarize_as_title). "smart" asks the
# model itself to invent a title instead — can read more naturally, at
# the cost of a second, slower generation call (kept off the critical
# path — see chat_service.build_reply_stream) that occasionally still
# produces nothing useful.
_VALID_TITLE_MODES = {"simple", "smart"}
DEFAULT_TITLE_MODE = "simple"

# How other channel members find out about an in-progress/finished reply
# while sitting in that chat (see app/services/reply_generation_service.py
# and app/services/reply_broadcast_service.py) — "cheap" (the
# out-of-the-box default) has them poll periodically, no extra infra
# needed; "real" pushes it to them live, token-by-token, over the
# in-process broadcast hub, which only reliably reaches every member when
# running as a single app instance (see reply_broadcast_service's own
# docstring for that limitation).
_VALID_CHANNEL_DELIVERY_MODES = {"cheap", "real"}
DEFAULT_CHANNEL_DELIVERY_MODE = "cheap"

# How long (wall-clock, not per-chunk) app.services.reply_generation_service
# waits for a model to finish generating a text-only reply before giving
# up and marking it failed — a backstop against a hung or pathologically
# slow model tying up resources indefinitely (app.services.ollama_client's
# own httpx client has no competing timeout of its own for a chat call —
# this is the one real deadline). 0 means no timeout — some local hardware
# genuinely needs minutes for one reply (see this app's own README on
# running without a GPU), so admins running on slow hardware need a real
# way to turn this off rather than just setting it very high. 300s (5
# minutes) out of the box: generous enough for slow CPU-only inference,
# short enough to still catch a genuinely stuck request well before
# someone gives up waiting and assumes the app itself is broken.
DEFAULT_REPLY_TIMEOUT_SECONDS = 300
_MAX_REPLY_TIMEOUT_SECONDS = 3600

# Same backstop, but for a reply that has an image attached (see
# app.services.chat_attachment_service.apply_image_attachment) — kept as
# its own separate setting rather than reusing DEFAULT_REPLY_TIMEOUT_SECONDS
# because the two aren't comparable: encoding an image (the vision
# projector pass, before any token even starts streaming) is dramatically
# slower than plain text generation on CPU-only hardware, confirmed live —
# a modest attached image alone can take several minutes to just finish
# encoding. A single shared timeout forced a choice between cutting image
# replies short or leaving a stuck text reply tying up resources for far
# longer than it should. Same 300s starting point as the text one out of
# the box — the point of splitting these is letting an admin raise this
# one independently once they see how slow vision actually is on their
# own hardware, not assuming that need up front.
DEFAULT_VISION_REPLY_TIMEOUT_SECONDS = 300


async def get_title_mode(db: AsyncSession) -> str:
    """Whether a new chat's title comes from "simple" local text
    processing or by asking the model itself ("smart") — see
    app.services.chat_service.build_reply_stream, the only reader of
    this. System-wide, admin-configured from Settings > System — not a
    per-user preference."""
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
    """See _VALID_CHANNEL_DELIVERY_MODES above. System-wide, admin-
    configured from Settings > System — read fresh on every call (unlike
    app.services.settings_service.get_proxy_mode), since app/routers/chat.py's
    `/subscribe` endpoint and chat.js both need this to reflect a change
    immediately, not just after a restart."""
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
    """See DEFAULT_REPLY_TIMEOUT_SECONDS above — the only reader of this
    is app.services.reply_generation_service, which reads it fresh at
    the start of every single generation (not cached), so a change here
    takes effect on the very next message, not just after a restart.
    A stored value outside [0, _MAX_REPLY_TIMEOUT_SECONDS] (only
    possible via a hand-edited row, since set_reply_timeout_seconds
    below and the API layer's own schema validation both enforce this
    range) falls back to the default rather than applying a nonsense
    timeout."""
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
    """See DEFAULT_VISION_REPLY_TIMEOUT_SECONDS above — used instead of get_reply_timeout_seconds only for a reply
    with an image attached (app.services.reply_generation_service.stream_reply's own has_image), read fresh the
    same way."""
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
