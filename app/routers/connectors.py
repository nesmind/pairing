"""GET/PUT for the Connectors page: admin-configured third-party engines (app.services.connectors;
runpod_engine is the one real backend). Admin-only since API keys are entered here. Separate from
engine_admin.py: a Connector is distinct from engine selection (connectors.base).
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import (
    ConnectorConfigUpdate,
    ConnectorEnabledUpdate,
    ConnectorOut,
    ConnectorsResponse,
    ConnectorSummary,
    OkResponse,
)
from app.services import connector_config_cache, connector_config_service, server_pool_broadcast
from app.services.auth_service import require_admin
from app.services.connectors import CONNECTORS, get_connector

logger = logging.getLogger("llama_chat")

router = APIRouter(prefix="/api/connectors", tags=["connectors"])


async def _to_out(db: AsyncSession, connector) -> ConnectorOut:
    # Imported here to avoid a cycle: engines.registry (needs is_ready) sits above connectors
    # (as in engine_support_checker.EngineSupportSet.load).
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
    """Tries the real service with the saved config (InferenceEngine.test_connection). A connector is not
    ready (selectable as engine) until this passes, even if enabled and filled in (connector_config_cache.is_ready).
    """
    # Imported here for the same circular-import reason as _to_out.
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


@router.get("/{connector_id}/summary", response_model=ConnectorSummary)
async def connector_summary(
    connector_id: str, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
):
    """State, settings, model counts and usage, for the External servers tab."""
    # Imported here for the same circular-import reason as _to_out.
    from app.services.connector_summary import ConnectorSummaryBuilder

    try:
        builder = ConnectorSummaryBuilder(db, connector_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return await builder.build()


@router.post("/internal-refresh", response_model=OkResponse)
async def internal_refresh(request: Request, db: AsyncSession = Depends(get_db)):
    """Called only by another local instance's server_pool_broadcast.broadcast_refresh after a connector save;
    see engine_admin.internal_refresh for the loopback-only guard.
    """
    if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail="Internal endpoint — not reachable externally.")
    await connector_config_cache.load_cache_from_db(db)
    return OkResponse(ok=True)
