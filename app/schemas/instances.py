"""Schemas for the "local instances" admin (Settings > System): supervisor services/instance_service.py;
built-in balancer instance_pool.py + instance_proxy.py.
"""

from typing import Literal

from pydantic import BaseModel, Field


class InstanceStatus(BaseModel):
    """One process: index 0 is the primary (human-started), 1+ are spawned siblings. `pid`/`healthy` are
    None for an untracked index (only briefly while reconciling).
    """

    index: int
    port: int
    pid: int | None
    alive: bool
    healthy: bool | None


class InstancesConfig(BaseModel):
    """GET/PUT /api/settings/instances: `count` is the saved target, `instances` what runs now (always
    includes index 0; may lag `count` while siblings start or stop).
    """

    count: int
    instances: list[InstanceStatus]


class InstancesUpdate(BaseModel):
    count: int = Field(ge=1, le=8)


class ProxyMode(BaseModel):
    """GET/PUT /api/settings/proxy-mode. "proxy": an external reverse proxy (if any) balances. "local": the
    primary balances across local instances (only matters with count > 1, instance_pool.pick_instance).
    """

    mode: Literal["proxy", "local"]
