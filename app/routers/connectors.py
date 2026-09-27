"""
GET/PUT for the Connectors page — admin-configured third-party engines (see
app.services.connectors for what a Connector is, app.services.engines.runpod_engine
for the one real engine a connector currently backs). Admin-only: this is where API
keys get entered. Deliberately its own router rather than folded into
engine_admin.py: a Connector is a distinct concept from engine selection itself (see
app.services.connectors.base's own docstring).
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import ConnectorConfigUpdate, ConnectorEnabledUpdate, ConnectorOut, ConnectorsResponse, OkResponse
from app.services import connector_config_cache, connector_config_service, server_pool_broadcast
from app.services.auth_service import require_admin
from app.services.connectors import CONNECTORS, get_connector

logger = logging.getLogger("llama_chat")

router = APIRouter(prefix="/api/connectors", tags=["connectors"])


async def _to_out(db: AsyncSession, connector) -> ConnectorOut:
    # Imported here, not at module level: app.services.engines.registry (which this needs for is_ready()) sits
    # above app.services.connectors in the dependency direction — importing it back from a module-level import
    # here would be circular the same way app.services.engine_support_checker.EngineSupportSet.load already
    # documents for its own identical case.
    from app.services.engines.registry import registry

    values = await connector_config_service.get_config_for_display(db, connector.id)
    configured = await connector_config_service.is_configured(db, connector.id)
    enabled = await connector_config_service.get_enabled(db, connector.id)
    last_test_passed, last_test_message = await connector_config_service.get_test_result(db, connector.id)
    ready = await registry.get(connector.engine_name).is_ready()
    return ConnectorOut(
        id=connector.id,
        engine_name=connector.engine_name,
        display_name=connector.display_name,
        description=connector.description,
        config_fields=[
            {
                "name": f.name,
                "label": f.label,
                "field_type": f.field_type,
                "required": f.required,
                "secret": f.secret,
                "help_text": f.help_text,
            }
            for f in connector.config_fields
        ],
        values=values,
        configured=configured,
        enabled=enabled,
        ready=ready,
        last_test_passed=last_test_passed,
        last_test_message=last_test_message,
    )


@router.get("", response_model=ConnectorsResponse)
async def list_connectors(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return ConnectorsResponse(connectors=[await _to_out(db, connector) for connector in CONNECTORS])


@router.put("/{connector_id}/config", response_model=ConnectorOut)
async def update_connector_config(
    connector_id: str,
    body: ConnectorConfigUpdate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    try:
        connector = get_connector(connector_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    await connector_config_service.set_config(db, connector_id, body.values)
    await connector_config_cache.load_cache_from_db(db)
    failures = await server_pool_broadcast.broadcast_refresh("/api/connectors/internal-refresh")
    if failures:
        logger.warning("Some instances did not pick up the updated %s config: %s", connector_id, "; ".join(failures))
    return await _to_out(db, connector)


@router.put("/{connector_id}/enabled", response_model=ConnectorOut)
async def update_connector_enabled(
    connector_id: str,
    body: ConnectorEnabledUpdate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    try:
        connector = get_connector(connector_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    await connector_config_service.set_enabled(db, connector_id, body.enabled)
    await connector_config_cache.load_cache_from_db(db)
    failures = await server_pool_broadcast.broadcast_refresh("/api/connectors/internal-refresh")
    if failures:
        logger.warning("Some instances did not pick up %s's enabled state: %s", connector_id, "; ".join(failures))
    return await _to_out(db, connector)


@router.post("/{connector_id}/test", response_model=ConnectorOut)
async def test_connector_config(
    connector_id: str, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
):
    """Actually attempts to reach the real service with the connector's *currently saved* config (see
    app.services.engines.base.InferenceEngine.test_connection) — a connector can't become ready (selectable as
    the active engine) until this has passed, even if every required field is filled in and it's enabled (see
    app.services.connector_config_cache.is_ready)."""
    # Imported here, not at module level — same circular-import reasoning as _to_out's own import.
    from app.services.engines.registry import registry

    try:
        connector = get_connector(connector_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    config = await connector_config_service.get_config(db, connector_id)
    passed, message = await registry.get(connector.engine_name).test_connection(config)
    await connector_config_service.set_test_result(db, connector_id, passed, message)
    await connector_config_cache.load_cache_from_db(db)
    failures = await server_pool_broadcast.broadcast_refresh("/api/connectors/internal-refresh")
    if failures:
        logger.warning("Some instances did not pick up %s's test result: %s", connector_id, "; ".join(failures))
    return await _to_out(db, connector)


@router.post("/internal-refresh", response_model=OkResponse)
async def internal_refresh(request: Request, db: AsyncSession = Depends(get_db)):
    """Called only by another local instance's own server_pool_broadcast.broadcast_refresh right after an
    admin's connector save — see app/routers/engine_admin.py's internal_refresh for the identical loopback-only
    guard and reasoning."""
    if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail="Internal endpoint — not reachable externally.")
    await connector_config_cache.load_cache_from_db(db)
    return OkResponse(ok=True)
