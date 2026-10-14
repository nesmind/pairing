"""RunPod is unlisted until it can be tested on a real endpoint (connectors/registry.py): its code stays, but it can't
be configured or become ready, so it can't be picked as the engine."""

import pytest

from app.services import connector_config_cache, connector_config_service
from app.services.connectors import CONNECTORS, get_connector
from app.services.engines.runpod_engine import RunPodEngine


def test_runpod_is_not_offered():
    assert [c.id for c in CONNECTORS] == ["cloudflare", "digitalocean"]
    with pytest.raises(KeyError):
        get_connector("runpod")


@pytest.mark.asyncio
async def test_runpod_never_becomes_ready(db):
    # Even a full saved config from before can't make it ready: the cache only loads listed connectors.
    await connector_config_service.set_enabled(db, "runpod", True)
    await connector_config_service.set_test_result(db, "runpod", True, "ok")
    await connector_config_cache.load_cache_from_db(db)
    assert await RunPodEngine().is_ready() is False
