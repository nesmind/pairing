"""Admin settings > Update: what each updatable component reports."""

from typing import Literal

from pydantic import BaseModel, Field

UpdateMethod = Literal["git", "download"]
RestartMode = Literal["script", "systemd", "docker", "manual"]


class SourceUpdateStatus(BaseModel):
    name: str
    path: str | None = None
    # False when this install can't be updated from here; `note` says why.
    available: bool = False
    method: UpdateMethod | None = None
    current: str | None = None
    latest: str | None = None
    # None when unknown (e.g. a download install with no recorded version).
    update_available: bool | None = None
    behind: int | None = None
    changes: list[str] = Field(default_factory=list)
    note: str | None = None


class PairingRestartInfo(BaseModel):
    mode: RestartMode
    can_restart: bool
    hint: str


class UpdateOverview(BaseModel):
    pairing: SourceUpdateStatus
    # None when Matricxon isn't installed locally.
    matricxon: SourceUpdateStatus | None = None
    restart: PairingRestartInfo
