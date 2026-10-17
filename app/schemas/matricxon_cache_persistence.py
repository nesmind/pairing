from pydantic import BaseModel, Field


class MatricxonCachePersistence(BaseModel):
    """State of Matricxon's encrypted on-disk prompt cache over all hosts. `enabled` only if every reachable
    host has it on; limits are the first host's.
    """

    enabled: bool
    budget_mb: int
    ttl_hours: int
    files: int
    # Distinct chats with a saved copy; `files` can be more (branches of one chat).
    chats: int = 0
    used_bytes: int
    hosts: int
    unreachable: list[str] = Field(default_factory=list)


class MatricxonCachePersistenceUpdate(BaseModel):
    """Omitted fields stay unchanged. Turning `enabled` off deletes every stored cache."""

    enabled: bool | None = None
    budget_mb: int | None = Field(default=None, ge=0, le=1048576)
    ttl_hours: int | None = Field(default=None, ge=1, le=8760)
