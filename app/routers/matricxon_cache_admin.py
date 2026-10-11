"""Settings > Matricxon > Saved chat caches: admin view and controls for Matricxon's encrypted on-disk
prompt cache (logic: app.services.matricxon_cache_persistence).
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import MatricxonCachePersistence, MatricxonCachePersistenceUpdate
from app.services import engine_service, settings_service
from app.services.auth_service import require_admin
from app.services.matricxon_cache_persistence import MatricxonCachePersistenceService
from app.services.matricxon_client import MatricxonError

router = APIRouter(prefix="/api/settings/matricxon/cache-persistence", tags=["matricxon"])


async def _require_matricxon_engine(db: AsyncSession) -> None:
    """The disk cache is Matricxon-only; Ollama has its own."""
    if await engine_service.get_active_engine(db) != "matricxon":
        raise HTTPException(status_code=409, detail="Saved chat caches are only available with the Matricxon engine.")


async def _shared_folder(db: AsyncSession) -> bool:
    """In cluster mode every node reads the same saved-caches folder."""
    return (await settings_service.get_matricxon_server_config(db)).cluster_role != "off"


def _bad_gateway(exc: MatricxonError) -> HTTPException:
    return HTTPException(status_code=502, detail=str(exc))


@router.get("", response_model=MatricxonCachePersistence)
async def get_cache_persistence(
    db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> MatricxonCachePersistence:
    await _require_matricxon_engine(db)
    try:
        return await MatricxonCachePersistenceService(shared_folder=await _shared_folder(db)).status()
    except MatricxonError as exc:
        raise _bad_gateway(exc) from exc


@router.put("", response_model=MatricxonCachePersistence)
async def update_cache_persistence(
    body: MatricxonCachePersistenceUpdate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> MatricxonCachePersistence:
    await _require_matricxon_engine(db)
    try:
        return await MatricxonCachePersistenceService(shared_folder=await _shared_folder(db)).update(body)
    except MatricxonError as exc:
        raise _bad_gateway(exc) from exc


@router.delete("", response_model=MatricxonCachePersistence)
async def clear_cache_persistence(
    db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> MatricxonCachePersistence:
    await _require_matricxon_engine(db)
    try:
        return await MatricxonCachePersistenceService(shared_folder=await _shared_folder(db)).clear()
    except MatricxonError as exc:
        raise _bad_gateway(exc) from exc
