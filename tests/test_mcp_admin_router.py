"""app/routers/mcp_admin.py, called directly like the other router tests."""

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.routers import mcp_admin
from app.schemas.mcp import McpLimits, McpServerIn
from tests.mcp_fake_server import FakeMcpServer


@pytest.fixture(scope="module")
def server():
    fake = FakeMcpServer().start()
    yield fake
    fake.stop()


async def test_crud_round_trip_never_returns_header_values(db, server) -> None:
    body = McpServerIn(name="fake", url=server.url, headers={"Authorization": "Bearer k"})
    created = await mcp_admin.create_server(body, db=db, _admin=None)

    listed = await mcp_admin.list_servers(db=db, _admin=None)
    assert [s.name for s in listed] == ["fake"]
    assert listed[0].header_names == ["Authorization"]
    assert "Bearer k" not in listed[0].model_dump_json()

    updated = await mcp_admin.update_server(
        created.id, McpServerIn(name="fake", url=server.url, enabled=False), db=db, _admin=None
    )
    assert updated.enabled is False and updated.header_names == ["Authorization"]

    await mcp_admin.delete_server(created.id, db=db, _admin=None)
    assert await mcp_admin.list_servers(db=db, _admin=None) == []


async def test_duplicate_is_409_and_unknown_is_404(db, server) -> None:
    body = McpServerIn(name="fake", url=server.url)
    await mcp_admin.create_server(body, db=db, _admin=None)

    with pytest.raises(HTTPException) as dup:
        await mcp_admin.create_server(body, db=db, _admin=None)
    assert dup.value.status_code == 409
    for call in (
        mcp_admin.delete_server("nope", db=db, _admin=None),
        mcp_admin.test_server("nope", db=db, _admin=None),
        mcp_admin.update_server("nope", body, db=db, _admin=None),
    ):
        with pytest.raises(HTTPException) as missing:
            await call
        assert missing.value.status_code == 404


async def test_test_button_lists_tools_or_reports_the_error(db, server) -> None:
    good = await mcp_admin.create_server(McpServerIn(name="good", url=server.url), db=db, _admin=None)
    bad = await mcp_admin.create_server(McpServerIn(name="bad", url="http://127.0.0.1:1/mcp"), db=db, _admin=None)

    ok = await mcp_admin.test_server(good.id, db=db, _admin=None)
    failed = await mcp_admin.test_server(bad.id, db=db, _admin=None)

    assert ok.ok and "good__add" in [t.name for t in ok.tools]
    assert not failed.ok and failed.error


async def test_connection_can_be_tested_before_saving(db, server) -> None:
    ok = await mcp_admin.test_connection(McpServerIn(name="new", url=server.url), db=db, _admin=None)
    failed = await mcp_admin.test_connection(McpServerIn(name="new", url="http://127.0.0.1:1/mcp"), db=db, _admin=None)

    assert ok.ok and "new__add" in [t.name for t in ok.tools]
    assert not failed.ok and failed.error
    assert await mcp_admin.list_servers(db=db, _admin=None) == []  # nothing was saved


async def test_testing_an_edit_without_headers_uses_the_saved_ones(db, server, monkeypatch) -> None:
    saved = await mcp_admin.create_server(
        McpServerIn(name="s", url=server.url, headers={"X-Key": "abc"}), db=db, _admin=None
    )
    seen: list[dict] = []

    async def fake_probe(name, url, headers):
        seen.append(headers)
        return mcp_admin.McpTestResult(ok=True)

    monkeypatch.setattr(mcp_admin.McpToolCatalog, "probe", staticmethod(fake_probe))

    await mcp_admin.test_connection(McpServerIn(name="s", url=server.url), server_id=saved.id, db=db, _admin=None)
    await mcp_admin.test_connection(
        McpServerIn(name="s", url=server.url, headers={"X-Key": "new"}), server_id=saved.id, db=db, _admin=None
    )

    assert seen == [{"X-Key": "abc"}, {"X-Key": "new"}]


async def test_global_switch_defaults_on_and_can_be_turned_off(db) -> None:
    assert (await mcp_admin.get_mcp_enabled(db=db, _admin=None)).enabled is True

    await mcp_admin.set_mcp_enabled(mcp_admin.McpEnabled(enabled=False), db=db, _admin=None)

    assert (await mcp_admin.get_mcp_enabled(db=db, _admin=None)).enabled is False


async def test_limits_default_set_and_validated(db) -> None:
    default = await mcp_admin.get_limits(db=db, _admin=None)
    assert (default.rounds, default.result_chars) == (5, 4000)
    await mcp_admin.set_limits(McpLimits(rounds=9, result_chars=1500), db=db, _admin=None)
    saved = await mcp_admin.get_limits(db=db, _admin=None)
    assert (saved.rounds, saved.result_chars) == (9, 1500)
    for bad in (
        {"rounds": 0, "result_chars": 4000},
        {"rounds": 5, "result_chars": 100},
        {"rounds": 21, "result_chars": 4000},
    ):
        with pytest.raises(ValidationError):
            McpLimits(**bad)
