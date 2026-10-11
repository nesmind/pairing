"""Generic per-user/system key-value settings row (keys: app/services/settings_service.py)."""

from sqlalchemy import JSON, Column, String

from app.database import Base
from app.models._base import ID_LEN

# Owner id for app-wide settings. A real string, not NULL: SQL treats NULLs as distinct, so NULL
# would allow duplicate "system" rows per key.
SYSTEM_OWNER_ID = "__system__"


class AppSetting(Base):
    """Per-user settings (default model, generation params, code theme), edited on the Settings page.
    The SYSTEM_OWNER_ID row is the app-wide fallback.
    """

    __tablename__ = "app_settings"

    owner_id = Column(String(ID_LEN), primary_key=True, default=SYSTEM_OWNER_ID)
    key = Column(String(100), primary_key=True)
    value = Column(JSON, nullable=False)
