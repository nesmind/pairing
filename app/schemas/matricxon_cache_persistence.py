from pydantic import BaseModel, Field


class MatricxonCachePersistence(BaseModel):
    """State of Matricxon's encrypted on-disk prompt cache, summed over every configured host.
    `enabled` is true only when every reachable host has it on; the limits are the first host's."""

    enabled: bool
    budget_mb: int
    ttl_hours: int
    files: int
    used_bytes: int
    hosts: int
    unreachable: list[str] = Field(default_factory=list)


class MatricxonCachePersistenceUpdate(BaseModel):
    """Fields left out stay unchanged. Turning `enabled` off also deletes every stored cache."""

    enabled: bool | None = None
    budget_mb: int | None = Field(default=None, ge=0, le=1048576)
    ttl_hours: int | None = Field(default=None, ge=1, le=8760)
