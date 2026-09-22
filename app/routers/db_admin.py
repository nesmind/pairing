"""
Settings > System > Database admin endpoints: read the currently
configured backend, test a candidate connection before committing to
it, build the schema (plus starter data) on an empty target, and save —
which switches the live app over immediately, no restart required (see
app/services/db_config_service.py for how). All admin-only; a wrong DB
choice here has a much bigger blast radius than a wrong RAG limit.
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
    """Called only by another local instance's own
    app.services.instance_db_broadcast.broadcast_database_switch right
    after an admin's live database switch — never by a browser, so no
    require_admin session here. Deliberately placed under this router's
    own "/api/settings/database" prefix, which
    app.services.instance_proxy_http.EXEMPT_PREFIXES already covers (a
    startswith() prefix match) — so this can never get load-balanced to
    a random *different* instance the way an ordinary request would,
    which matters here specifically: a request proxied from the primary
    to a sibling always looks loopback-sourced to the sibling (the
    primary's own httpx call originates from 127.0.0.1), which would
    silently defeat the check below if this path could ever be proxied.
    """
    if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail="Internal endpoint — not reachable externally.")
    await database.switch_database(body.database_url)
    return OkResponse(ok=True)
