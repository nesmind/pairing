import pytest

from app.services.cluster_nodes import ClusterNodes, NodeNotAllowedError


class _Cfg:
    models_path = None
    prompt_cache_path = None
    cluster_role = "seed"


@pytest.mark.asyncio
async def test_adopt_paths_reads_the_cluster_folders(monkeypatch):
    nodes = ClusterNodes.__new__(ClusterNodes)

    async def seed(_config):
        return "http://a", {"config": {"cluster/paths": {"models_dir": "/m", "prompt_cache_dir": "/c"}}}

    monkeypatch.setattr(nodes, "_seed_consensus", seed)
    cfg = _Cfg()
    await nodes._adopt_paths(cfg)
    assert (cfg.models_path, cfg.prompt_cache_path) == ("/m", "/c")


@pytest.mark.asyncio
async def test_adopt_paths_keeps_saved_values_and_tolerates_no_cluster(monkeypatch):
    nodes = ClusterNodes.__new__(ClusterNodes)

    async def none(_config):
        return None

    monkeypatch.setattr(nodes, "_seed_consensus", none)
    cfg = _Cfg()
    await nodes._adopt_paths(cfg)
    assert cfg.models_path is None
    cfg.models_path, cfg.prompt_cache_path = "/x", "/y"
    await nodes._adopt_paths(cfg)
    assert (cfg.models_path, cfg.prompt_cache_path) == ("/x", "/y")
    assert NodeNotAllowedError
