"""Tests for the Inference engines connector summary: services/connector_summary.py (GET /api/connectors/{id}/summary)
and services/cloudflare_usage.py (token status and Workers AI usage from Cloudflare's analytics). No network."""

import json
from datetime import UTC, datetime

import httpx
import pytest
from fastapi import HTTPException

from app.routers import connectors
from app.services import cloudflare_client, connector_config_cache, connector_config_service
from app.services.cloudflare_usage import CloudflareAccountReport
from app.services.engines.cloudflare_engine import CloudflareEngine

_CONFIG = {"account_id": "acc-1", "api_token": "tok"}
_NOW = datetime(2026, 10, 14, 9, 30, tzinfo=UTC)


@pytest.fixture(autouse=True)
def reset_cache():
    connector_config_cache._cache = {}
    yield
    connector_config_cache._cache = {}


def _fake_api(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))


def _week_row(model, count, neurons):
    return {
        "count": count,
        "sum": {"totalNeurons": neurons, "totalInputTokens": 100, "totalOutputTokens": 50},
        "dimensions": {"modelId": model},
    }


class TestCloudflareUsage:
    @pytest.mark.asyncio
    async def test_usage_is_summed_with_the_top_models(self, monkeypatch):
        seen = {}

        def handler(request):
            seen["variables"] = json.loads(request.content)["variables"]
            week = [_week_row("@cf/a", 3, 12.34), _week_row("@cf/b", 1, 2.0)]
            account = {"week": week, "today": [{"count": 2, "sum": {"totalNeurons": 5.55}}]}
            return httpx.Response(200, json={"data": {"viewer": {"accounts": [account]}}})

        _fake_api(monkeypatch, handler)
        usage, note = await CloudflareAccountReport(_CONFIG, now=_NOW).usage_summary()

        assert note is None
        expected = {"account": "acc-1", "week": "2026-10-07T09:30:00Z", "today": "2026-10-14T00:00:00Z"}
        assert seen["variables"] == expected
        assert usage["requests"] == 4 and usage["units"] == 14.3 and usage["units_label"] == "neurons"
        assert usage["input_tokens"] == 200 and usage["output_tokens"] == 100
        assert usage["today_requests"] == 2 and usage["today_units"] == 5.5
        assert [m["model"] for m in usage["top_models"]] == ["@cf/a", "@cf/b"]

    @pytest.mark.asyncio
    async def test_a_token_without_analytics_gets_a_note(self, monkeypatch):
        errors = [{"message": "not authorized for that account", "extensions": {"code": "authz"}}]
        _fake_api(monkeypatch, lambda _r: httpx.Response(200, json={"data": None, "errors": errors}))
        usage, note = await CloudflareAccountReport(_CONFIG, now=_NOW).usage_summary()
        assert usage is None and "Account Analytics: Read" in note

    @pytest.mark.asyncio
    async def test_token_status_is_reported(self, monkeypatch):
        _fake_api(monkeypatch, lambda _r: httpx.Response(200, json={"result": {"status": "active"}}))
        details = await CloudflareAccountReport(_CONFIG).account_details()
        assert ("Token status", "Active") in details

    @pytest.mark.asyncio
    async def test_nothing_is_asked_without_a_saved_token(self, monkeypatch):
        _fake_api(monkeypatch, lambda _r: pytest.fail("no call expected"))
        report = CloudflareAccountReport({})
        assert await report.account_details() == []
        assert await report.usage_summary() == (None, None)


class TestSummaryRoute:
    @pytest.mark.asyncio
    async def test_a_not_ready_connector_shows_only_its_state(self, db):
        summary = await connectors.connector_summary("cloudflare", db=db, _admin=None)
        assert summary.display_name == "Cloudflare Workers AI"
        assert summary.ready is False and summary.chat_models is None and summary.usage is None

    @pytest.mark.asyncio
    async def test_a_ready_connector_shows_settings_models_and_usage(self, db, monkeypatch):
        await connector_config_service.set_config(db, "cloudflare", _CONFIG)
        await connector_config_service.set_enabled(db, "cloudflare", True)
        await connector_config_service.set_test_result(db, "cloudflare", True, "Connected successfully.")
        await connector_config_cache.load_cache_from_db(db)

        async def models():
            return [{"name": "@cf/a", "capabilities": ["completion"]}, {"name": "@cf/e", "capabilities": ["embedding"]}]

        async def details(_self):
            return [("Token status", "Active")]

        async def usage(_self):
            return None, "no analytics permission"

        monkeypatch.setattr(cloudflare_client, "list_models", models)
        monkeypatch.setattr(CloudflareEngine, "account_details", details)
        monkeypatch.setattr(CloudflareEngine, "usage_summary", usage)

        summary = await connectors.connector_summary("cloudflare", db=db, _admin=None)

        rows = {(r.label, r.value) for r in summary.details}
        assert ("Account ID", "acc-1") in rows and ("API token", "Saved") in rows
        assert ("Token status", "Active") in rows
        assert all(r.value != "tok" for r in summary.details)  # the token itself never leaves
        assert summary.chat_models == 1 and summary.embedding_models == 1
        assert summary.usage is None and summary.usage_note == "no analytics permission"

    @pytest.mark.asyncio
    async def test_an_unknown_connector_is_404(self, db):
        with pytest.raises(HTTPException) as exc:
            await connectors.connector_summary("nope", db=db, _admin=None)
        assert exc.value.status_code == 404
