"""Schemas for the "local instances" admin feature (Settings > System) —
see app/services/instance_service.py for the supervisor that spawns/
tracks the sibling processes these describe, and
app/services/instance_pool.py + instance_proxy.py for the "proxy mode"
built-in load balancer that routes across them."""

from typing import Literal

from pydantic import BaseModel, Field


class InstanceStatus(BaseModel):
    """One process — index 0 is always the primary (the one a human
    actually started); 1+ are siblings the primary spawned itself.
    `pid`/`healthy` are None/unknown for an index that isn't currently
    tracked as running at all (shouldn't normally happen outside a brief
    window while reconciling, but the shape allows for it)."""

    index: int
    port: int
    pid: int | None
    alive: bool
    healthy: bool | None


class InstancesConfig(BaseModel):
    """GET/PUT /api/settings/instances response — `count` is the admin's
    saved target, `instances` is what's actually observed running right
    now (always includes index 0; may briefly lag `count` immediately
    after a change while siblings are still starting/stopping)."""

    count: int
    instances: list[InstanceStatus]


class InstancesUpdate(BaseModel):
    count: int = Field(ge=1, le=8)


class ProxyMode(BaseModel):
    """GET/PUT /api/settings/proxy-mode. "proxy": unchanged behavior —
    an external reverse proxy (if any) balances across instances
    directly; this app does nothing extra. "local": the primary itself
    load-balances requests across every local instance — only does
    anything once instance count > 1 (see instance_pool.pick_instance)."""

    mode: Literal["proxy", "local"]
