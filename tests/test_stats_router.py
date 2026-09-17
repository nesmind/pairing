"""Unit test for app/routers/stats.py — called directly, not through a
TestClient/ASGI app (see tests/test_instance_proxy_http.py's own
docstring on why this project doesn't use that pattern). Real
aggregation coverage lives in tests/test_stats_service.py; this file
only proves the router is admin-only (see its own docstring for why) —
calling require_admin directly with a plain `user`, since a direct
router call bypasses FastAPI's own dependency resolution and would
never actually run that check otherwise."""

import pytest
from fastapi import HTTPException

from app.models import Conversation
from app.routers import stats
from app.services.auth_service import require_admin


@pytest.mark.asyncio
async def test_get_summary_is_reachable_by_an_admin(db, admin_user):
    db.add(Conversation(owner_id=admin_user.id, model="llama3:latest"))
    await db.commit()

    summary = await stats.get_summary(db=db, _user=admin_user)

    assert summary.conversations.total == 1
    assert summary.conversations.personal == 1


@pytest.mark.asyncio
async def test_a_plain_user_is_rejected_by_require_admin(user):
    with pytest.raises(HTTPException) as exc_info:
        require_admin(user)
    assert exc_info.value.status_code == 403
