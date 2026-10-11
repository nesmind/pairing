"""Test for routers/telemetry.py, called directly, not via TestClient/ASGI (see tests/test_instance_proxy_http.py).
The aggregation is in tests/test_telemetry_service.py; this proves only that the router is admin-only, as
routers/stats.py
(see its test on why a direct require_admin call, not a router call with a plain `_user`, proves it).
"""

import pytest
from fastapi import HTTPException

from app.models import TelemetryEvent
from app.models._base import utcnow
from app.routers import telemetry
from app.services.auth_service import require_admin


@pytest.mark.asyncio
async def test_get_ollama_summary_is_reachable_by_an_admin(db, admin_user):
    db.add(TelemetryEvent(kind="chat", host="http://x:11434", success=True, started_at=utcnow(), duration_ms=1.0))
    await db.commit()

    summary = await telemetry.get_ollama_summary(db=db, _user=admin_user)

    assert summary.total_requests == 1


@pytest.mark.asyncio
async def test_get_matricxon_summary_is_reachable_by_an_admin_and_stays_isolated_from_ollamas(db, admin_user):
    db.add(
        TelemetryEvent(
            engine="ollama", kind="chat", host="http://x:11434", success=True, started_at=utcnow(), duration_ms=1.0
        )
    )
    db.add(
        TelemetryEvent(
            engine="matricxon", kind="chat", host="http://x:8420", success=True, started_at=utcnow(), duration_ms=1.0
        )
    )
    await db.commit()

    summary = await telemetry.get_matricxon_summary(db=db, _user=admin_user)

    assert summary.total_requests == 1


@pytest.mark.asyncio
async def test_a_plain_user_is_rejected_by_require_admin(user):
    with pytest.raises(HTTPException) as exc_info:
        require_admin(user)
    assert exc_info.value.status_code == 403
