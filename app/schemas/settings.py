"""Request/response shapes for the "global"/admin half of
app/routers/settings.py — system info and the various admin-configured
system-wide knobs (title mode, channel delivery mode, reply timeout,
retention). See app/schemas/model_catalog.py for the model-catalog
shapes (split out once that grew past this file's own line cap) and
app/schemas/user.py for account management shapes instead."""

from typing import Literal

from pydantic import BaseModel, Field


class HardwareSummary(BaseModel):
    """This machine's capacity, shown in Settings next to whichever
    models are hardware-blocked so the reason is concrete, not just
    "unavailable" (see app/hardware.py)."""

    ram_gb: float
    vram_gb: float
    total_gb: float


class RagAvailability(BaseModel):
    """Whether the embedding model the knowledge base needs is actually
    installed (see GET /api/settings/rag-availability)."""

    available: bool


class DefaultModel(BaseModel):
    """The model a brand-new conversation of the current user's starts
    with, if they didn't specify one (see
    app.services.conversation_service.create_conversation and
    app.services.settings_service's default-model helpers). Stored the
    same per-user/system-fallback way as GenerationParams defaults."""

    model: str


class UiTheme(BaseModel):
    """Which app-wide theme (see app.theme_config.UI_THEMES) the current user's whole UI — backgrounds, text,
    accent colors, fonts — renders with. One choice per account, same as DefaultModel above."""

    theme: str


class SystemInfo(BaseModel):
    """Admin-only snapshot shown in Settings' System tab (see
    app/routers/settings.py:system_info). The user list itself lives
    behind its own GET /api/settings/users instead of here, since it
    needs to be refetched after every add/edit without also re-fetching
    hardware info. local_install_supported (see app.hardware's own
    function of the same name) also gates the External servers tab's
    "Install from GitHub" button — fetched from here rather than a
    dedicated endpoint since it's cheap and this snapshot already exists."""

    hardware: HardwareSummary
    app_version: str
    local_install_supported: bool


class TitleMode(BaseModel):
    """Whether a new chat's title comes from quick local text processing
    ("simple") or by asking the model itself ("smart") — see
    app.services.chat_settings_service.get_title_mode/set_title_mode.
    System-wide, admin-configured from Settings > System — not a
    per-user preference."""

    mode: Literal["simple", "smart"]


class ChannelDeliveryMode(BaseModel):
    """How other members of a channel see an in-progress/finished reply
    while they're sitting in that chat — "cheap" (they poll
    periodically) or "real" (pushed to them live, token-by-token). See
    app.services.chat_settings_service.get_channel_delivery_mode/
    set_channel_delivery_mode and app.services.reply_broadcast_service
    for the mechanism "real" mode relies on. System-wide, admin-
    configured from Settings > System — not a per-user preference. Every
    logged-in user can read this value (chat.js needs it to decide how
    to behave in a channel chat), but only an admin can change it."""

    mode: Literal["cheap", "real"]


class ReplyTimeout(BaseModel):
    """How long (in seconds) the app waits for a model to finish
    generating a reply — personal chat or channel alike — before giving
    up and marking it failed (see
    app.services.chat_settings_service.get_reply_timeout_seconds/
    set_reply_timeout_seconds, and app.services.reply_generation_service,
    the only reader of this). System-wide, admin-configured from
    Settings > System — not a per-user preference, and admin-only to
    both read and change: unlike ChannelDeliveryMode above, nothing in
    the frontend needs to branch on this value, it's a pure backend
    operational cap. 0 means no timeout — see
    chat_settings_service.DEFAULT_REPLY_TIMEOUT_SECONDS for why that's a
    real, intentional option rather than just an edge case."""

    timeout_seconds: int = Field(ge=0, le=3600)


class RetentionSettings(BaseModel):
    """How long (in days) the Stats page's Telemetry/System views keep their poller-written history before
    app.services.retention_poller sweeps it — see app.services.retention_settings_service for storage/defaults.
    System-wide, admin-configured from Settings > System, admin-only to both read and change, same reasoning as
    ReplyTimeout above: a pure backend operational cap, nothing in the frontend branches on it."""

    telemetry_days: int = Field(ge=1, le=365)
    system_metrics_days: int = Field(ge=1, le=365)
