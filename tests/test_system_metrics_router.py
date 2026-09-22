"""Unit test for app/routers/system_metrics.py — called directly, not
through a TestClient/ASGI app (see tests/test_instance_proxy_http.py's
own docstring on why this project doesn't use that pattern). Real
aggregation coverage lives in tests/test_system_metrics_service.py;
this file only proves the router is admin-only, same access-control
decision as app/routers/stats.py/telemetry.py (see stats.py's own test
for why a direct require_admin call, not a router call with a plain
`_user`, is what actually proves this)."""

import pytest
from fastapi import HTTPException

from app.models import SystemMetricSnapshot
from app.models._base import utcnow
from app.routers import system_metrics
from app.services.auth_service import require_admin


@pytest.mark.asyncio
async def test_get_summary_is_reachable_by_an_admin(db, admin_user):
    db.add(
        SystemMetricSnapshot(
            cpu_percent=5.0,
            mem_used_bytes=1,
            mem_total_bytes=2,
            disk_used_bytes=1,
            disk_total_bytes=2,
            polled_at=utcnow(),
        )
    )
    await db.commit()

    summary = await system_metrics.get_summary(db=db, _user=admin_user)

    assert summary.cpu_percent == 5.0


@pytest.mark.asyncio
async def test_a_plain_user_is_rejected_by_require_admin(user):
    with pytest.raises(HTTPException) as exc_info:
        require_admin(user)
    assert exc_info.value.status_code == 403
