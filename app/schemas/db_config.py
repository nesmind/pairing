"""
Schemas for Settings > System > Database (see
app/services/db_config_service.py) — the admin-configurable SQLite/MySQL
connection an installation runs on.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SqliteParams(BaseModel):
    directory: str = Field(min_length=1)
    filename: str = Field(min_length=1, default="pAIring.db")


class MysqlParams(BaseModel):
    host: str = Field(min_length=1)
    port: int = Field(default=3306, ge=1, le=65535)
    user: str = Field(min_length=1)
    # Empty means "keep whatever password is already saved" — see
    # db_config_service._build_url. Never populated on a read (see
    # DbConfig.has_password instead).
    password: str = ""
    database: str = Field(min_length=1)


class DbConfig(BaseModel):
    """What Settings > System > Database currently shows. Reflects the
    connection saved on disk (.env), not necessarily what the running
    process is actually using — see restart_required. Saving through
    this page always keeps the two in sync (see
    db_config_service.save_and_apply); restart_required only ever turns
    true if .env was edited by hand outside the UI."""

    db_type: Literal["sqlite", "mysql"]
    sqlite: SqliteParams | None = None
    mysql: MysqlParams | None = None
    has_password: bool = False
    restart_required: bool = False


class DbConfigUpdate(BaseModel):
    db_type: Literal["sqlite", "mysql"]
    sqlite: SqliteParams | None = None
    mysql: MysqlParams | None = None

    @model_validator(mode="after")
    def _require_matching_params(self) -> "DbConfigUpdate":
        if self.db_type == "sqlite" and self.sqlite is None:
            raise ValueError("sqlite params are required when db_type is 'sqlite'")
        if self.db_type == "mysql" and self.mysql is None:
            raise ValueError("mysql params are required when db_type is 'mysql'")
        return self


class DbActionResult(BaseModel):
    ok: bool
    message: str


class InternalSwitchDatabaseRequest(BaseModel):
    """Body of POST /api/settings/database/internal-switch — sent only
    by another local instance's own app.services.instance_db_broadcast,
    never by a browser (see that router endpoint's own docstring for the
    loopback-only guard)."""

    database_url: str
