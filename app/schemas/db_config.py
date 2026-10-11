"""Schemas for Settings > System > Database (services/db_config_service.py): the SQLite/MySQL connection."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SqliteParams(BaseModel):
    directory: str = Field(min_length=1)
    filename: str = Field(min_length=1, default="pAIring.db")


class MysqlParams(BaseModel):
    host: str = Field(min_length=1)
    port: int = Field(default=3306, ge=1, le=65535)
    user: str = Field(min_length=1)
    # Empty keeps the saved password (db_config_service._build_url); never filled on read (see DbConfig.has_password).
    password: str = ""
    database: str = Field(min_length=1)


class DbConfig(BaseModel):
    """What the Database settings show: the connection saved in .env, not necessarily the running one
    (restart_required). Saving here keeps both in sync (save_and_apply); restart_required is true only
    after a manual.env edit.
    """

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
    """Body of POST /api/settings/database/internal-switch, sent only by another instance's
    instance_db_broadcast (loopback-only, see the endpoint).
    """

    database_url: str
