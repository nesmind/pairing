"""Schema for GET /health (see app/routers/health.py)."""

from typing import Literal

from pydantic import BaseModel


class HealthStatus(BaseModel):
    status: Literal["ok", "unhealthy"]
    database: bool
    ollama_hosts: dict[str, bool]
