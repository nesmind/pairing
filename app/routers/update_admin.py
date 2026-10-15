"""Admin settings > Update: check for and apply updates to pAIring and a local Matricxon (SSE progress)."""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import OkResponse
from app.schemas.update import UpdateOverview
from app.services import http_proxy_service, settings_service
from app.services.auth_service import require_admin
from app.services.extended_model_catalog_service import ExtendedModelCatalog
from app.services.update_targets import MatricxonUpdate, PairingUpdate

router = APIRouter(prefix="/api/settings/update", tags=["update"])


async def _proxy_url(db: AsyncSession) -> str | None:
    return (await http_proxy_service.get_http_proxy_config(db)).proxy_url()


def _sse(events: AsyncIterator[dict]) -> StreamingResponse:
    async def body() -> AsyncIterator[str]:
        async for event in events:
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@router.get("/status", response_model=UpdateOverview)
async def status(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)) -> UpdateOverview:
    proxy_url = await _proxy_url(db)
    pairing = PairingUpdate(proxy_url)
    matricxon = MatricxonUpdate(await settings_service.get_matricxon_server_config(db), proxy_url)
    return UpdateOverview(
        pairing=await pairing.status(), matricxon=await matricxon.status(), restart=pairing.restart_info()
    )


@router.post("/pairing")
async def update_pairing(
    db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> StreamingResponse:
    pairing = PairingUpdate(await _proxy_url(db))
    if pairing.in_container():
        raise HTTPException(status_code=400, detail="Running in a container: update by pulling a new image.")
    return _sse(pairing.updater.update_stream())


@router.post("/matricxon")
async def update_matricxon(
    db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> StreamingResponse:
    matricxon = MatricxonUpdate(await settings_service.get_matricxon_server_config(db), await _proxy_url(db))
    if matricxon.updater is None:
        raise HTTPException(status_code=400, detail="Matricxon is not installed locally.")
    # Read before streaming: the session closes once the response starts.
    await ExtendedModelCatalog(db).clear()
    return _sse(matricxon.update_stream())


@router.post("/pairing/restart", response_model=OkResponse)
async def restart_pairing(_admin: User = Depends(require_admin)) -> OkResponse:
    try:
        PairingUpdate().restart()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return OkResponse()
