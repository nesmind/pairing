"""Settings > System > Database admin (admin-only): read the backend, test a candidate connection, build
the schema and starter data on an empty target, and save, which switches the live app without a restart
(app/services/db_config_service.py). A wrong choice here has a large blast radius.
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request

from app import database
from app.models import User
from app.schemas import DbActionResult, DbConfig, DbConfigUpdate, InternalSwitchDatabaseRequest, OkResponse
from app.services import db_config_service, db_config_url
from app.services.auth_service import require_admin

router = APIRouter(prefix="/api/settings/database", tags=["database"])


@router.get("", response_model=DbConfig)
async def get_database_config(_admin: User = Depends(require_admin)) -> DbConfig:
    return db_config_url.get_current_config()


@router.put("", response_model=DbActionResult)
async def save_database_config(update: DbConfigUpdate, _admin: User = Depends(require_admin)) -> DbActionResult:
    return await db_config_service.save_and_apply(update)


@router.post("/test-connection", response_model=DbActionResult)
async def test_database_connection(update: DbConfigUpdate, _admin: User = Depends(require_admin)) -> DbActionResult:
    return await asyncio.to_thread(db_config_service.test_connection, update)


@router.post("/build-schema", response_model=DbActionResult)
async def build_database_schema(update: DbConfigUpdate, _admin: User = Depends(require_admin)) -> DbActionResult:
    return await db_config_service.build_schema(update)


@router.post("/internal-switch", response_model=OkResponse)
async def internal_switch_database(body: InternalSwitchDatabaseRequest, request: Request) -> OkResponse:
    """Called only by another local instance's instance_db_broadcast.broadcast_database_switch after a live
    switch, never by a browser. It sits under "/api/settings/database", covered by
    instance_proxy_http.EXEMPT_PREFIXES, so it is never load-balanced to another instance: a proxied
    request looks loopback-sourced to a sibling and would defeat the check below.
    """
    if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail="Internal endpoint — not reachable externally.")
    await database.switch_database(body.database_url)
    return OkResponse(ok=True)
