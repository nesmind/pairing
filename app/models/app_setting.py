"""The generic per-user/system key-value settings row — see
app/services/settings_service.py for every key actually stored this way."""

from sqlalchemy import JSON, Column, String

from app.database import Base
from app.models._base import ID_LEN

# Sentinel owner_id for an app-wide (not-yet-personalized) AppSetting
# row. Deliberately a real string rather than NULL: standard SQL never
# treats two NULLs as equal for uniqueness purposes (true of SQLite,
# MySQL, and Postgres alike), so NULL would silently let more than one
# "system" row exist for the same key — a plain sentinel value
# sidesteps that entirely, on any of them.
SYSTEM_OWNER_ID = "__system__"


class AppSetting(Base):
    """Per-user key/value settings — currently each user's own default
    model/generation params/code theme for new chats, edited from the
    Settings page (see app/services/settings_service.py). The
    SYSTEM_OWNER_ID row for a given key is the app-wide fallback used
    before a user has saved their own."""

    __tablename__ = "app_settings"

    owner_id = Column(String(ID_LEN), primary_key=True, default=SYSTEM_OWNER_ID)
    key = Column(String(100), primary_key=True)
    value = Column(JSON, nullable=False)
