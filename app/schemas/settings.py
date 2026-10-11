"""Shapes for the global/admin half of routers/settings.py: system info and system-wide knobs (title mode,
channel delivery, reply timeout, retention). Model catalog shapes: schemas/model_catalog.py; accounts: schemas/user.py.
"""

from typing import Literal

from pydantic import BaseModel, Field


class HardwareSummary(BaseModel):
    """This machine's capacity, shown beside hardware-blocked models (app/hardware.py)."""

    ram_gb: float
    vram_gb: float
    total_gb: float


class RagAvailability(BaseModel):
    """Whether the embedding model is installed AND the active engine can serve embeddings
    (GET /api/settings/rag-availability). A connector like RunPod can't, so `reason` names which is the problem.
    """

    available: bool
    reason: str | None = None


class DefaultModel(BaseModel):
    """The model a new conversation starts with if none is specified (conversation_service.create_conversation,
    settings_service default-model helpers); stored per-user with a system fallback, like GenerationParams defaults.
    """

    model: str


class UiTheme(BaseModel):
    """The app-wide theme (app.theme_config.UI_THEMES) for the user's whole UI; one per account."""

    theme: str


class SystemInfo(BaseModel):
    """Admin snapshot for Settings' System tab (routers/settings.py:system_info). The user list has its own
    GET /api/settings/users so it can refresh without refetching hardware. local_install_supported
    (app.hardware) also gates the "Install from GitHub" button.
    """

    hardware: HardwareSummary
    app_version: str
    local_install_supported: bool


class TitleMode(BaseModel):
    """Whether a new chat's title comes from quick text processing ("simple") or the model ("smart")
    (chat_settings_service.get_title_mode/set_title_mode). System-wide, admin-configured.
    """

    mode: Literal["simple", "smart"]


class ContextWindow(BaseModel):
    """The system-wide context window in tokens (context_window_setting)."""

    num_ctx: int = Field(ge=256, le=32768)


class ChannelPlainMessages(BaseModel):
    """Whether plain channel messages (posted without asking the AI) are included in what the model reads later
    (channel_history_setting). Admin-only, system-wide.
    """

    include: bool


class ChannelDeliveryMode(BaseModel):
    """How other channel members see a reply in progress: "cheap" (polling) or "real" (pushed live, token by
    token) (chat_settings_service.get_channel_delivery_mode/set_..., reply_broadcast_service). System-wide,
    admin-set; every user can read it (chat.js needs it), only admins change it.
    """

    mode: Literal["cheap", "real"]


class ReplyTimeout(BaseModel):
    """Seconds to wait for a reply (personal or channel) before marking it failed
    (chat_settings_service.get_reply_timeout_seconds/set_..., reply_generation_service). System-wide,
    admin-only to read and change (a backend cap). 0 = no timeout, intentionally
    (chat_settings_service.DEFAULT_REPLY_TIMEOUT_SECONDS).
    """

    timeout_seconds: int = Field(ge=0, le=3600)


class RetentionSettings(BaseModel):
    """Days of Telemetry/System history kept before retention_poller sweeps it (retention_settings_service).
    System-wide, admin-only, like ReplyTimeout.
    """

    telemetry_days: int = Field(ge=1, le=365)
    system_metrics_days: int = Field(ge=1, le=365)
