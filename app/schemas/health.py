"""Schema for GET /health (see app/routers/health.py)."""

from typing import Literal

from pydantic import BaseModel

from app.schemas.common import EngineName


class HealthStatus(BaseModel):
    status: Literal["ok", "unhealthy"]
    database: bool
    # Whichever engine (Ollama or Matricxon) is actually live right now (see
    # app.services.engine_service) — `engine_hosts` reports THAT engine's own host reachability,
    # not always Ollama's, so this stays meaningful after an admin switches the active engine.
    active_engine: EngineName
    engine_hosts: dict[str, bool]
