"""Schema for GET /health (app/routers/health.py)."""

from typing import Literal

from pydantic import BaseModel

from app.schemas.common import EngineName


class HealthStatus(BaseModel):
    status: Literal["ok", "unhealthy"]
    database: bool
    # The live engine (engine_service); `engine_hosts` reports that engine's host reachability.
    active_engine: EngineName
    engine_hosts: dict[str, bool]
