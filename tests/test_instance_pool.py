"""Tests for services/instance_pool.py, the "fewest in-flight, health-aware" selector instance_proxy.py's
balancer uses across local instances: tests/test_ollama_pool.py's pattern for instance indices. Every test
fakes instance_process.read_tracking/is_alive, no real processes.
"""

import pytest

from app.services import instance_pool, instance_process


@pytest.fixture
def three_alive(monkeypatch):
    """Indices 1 and 2 tracked and alive, plus index 0 (the primary, always a candidate), isolated per test with a fresh
    _pool as test_ollama_pool.py's `hosts` fixture."""
    monkeypatch.setattr(instance_process, "read_tracking", lambda: {1: {"pid": 101}, 2: {"pid": 102}})
    monkeypatch.setattr(instance_process, "is_alive", lambda pid: True)
    monkeypatch.setattr(instance_pool, "_pool", {})
    return [0, 1, 2]


def test_pick_instance_with_only_the_primary_always_returns_zero(monkeypatch):
    monkeypatch.setattr(instance_process, "read_tracking", lambda: {})
    monkeypatch.setattr(instance_pool, "_pool", {})
    assert instance_pool.pick_instance() == 0


def test_pick_instance_prefers_fewest_in_flight(three_alive):
    instance_pool._state(0).in_flight = 3
    instance_pool._state(1).in_flight = 0
    instance_pool._state(2).in_flight = 1
    assert instance_pool.pick_instance() == 1


def test_pick_instance_excludes_given_indices(three_alive):
    instance_pool._state(0).in_flight = 0
    assert instance_pool.pick_instance(exclude=frozenset({0})) in (1, 2)


def test_mark_failure_cools_an_index_down_and_pick_instance_avoids_it(three_alive):
    instance_pool.mark_failure(1)
    for _ in range(10):
        assert instance_pool.pick_instance() != 1


def test_mark_recovered_clears_cooldown_immediately(three_alive):
    # 1 and 2 are kept busier than 0 so the outcome stays deterministic despite pick_instance's random tie-breaking;
    # otherwise "no longer cooling" would make 0 one of three equally likely picks.
    instance_pool._state(1).in_flight = 1
    instance_pool._state(2).in_flight = 1
    instance_pool.mark_failure(0)
    assert instance_pool.pick_instance() != 0
    instance_pool.mark_recovered(0)
    assert instance_pool.pick_instance() == 0


def test_pick_instance_falls_back_when_every_candidate_is_cooling(three_alive):
    for i in (0, 1, 2):
        instance_pool.mark_failure(i)
    assert instance_pool.pick_instance() in (0, 1, 2)


def test_pick_instance_spreads_load_across_tied_candidates(three_alive):
    """Guards a bug before the fix: plain min over a tie always picked index 0, so light
    sequential traffic (every candidate at 0 in flight, the common case) never reached a sibling. Random
    tie-breaking (see pick_instance) fixes it; this proves it statistically, not just "can return any of them".
    """
    picks = {instance_pool.pick_instance() for _ in range(50)}
    assert len(picks) > 1


def test_candidate_set_changes_between_calls_as_tracking_changes(monkeypatch):
    """Unlike Ollama's remote host list, the candidate set changes live with the instance count (Settings)."""
    monkeypatch.setattr(instance_pool, "_pool", {})
    tracked = {}
    monkeypatch.setattr(instance_process, "read_tracking", lambda: tracked)
    monkeypatch.setattr(instance_process, "is_alive", lambda pid: True)

    assert instance_pool._candidate_indices() == {0}
    tracked[1] = {"pid": 201}
    assert instance_pool._candidate_indices() == {0, 1}


@pytest.mark.asyncio
async def test_track_request_increments_and_decrements_in_flight(three_alive):
    assert instance_pool._state(1).in_flight == 0
    async with instance_pool.track_request(1):
        assert instance_pool._state(1).in_flight == 1
    assert instance_pool._state(1).in_flight == 0


@pytest.mark.asyncio
async def test_track_request_decrements_even_on_exception(three_alive):
    with pytest.raises(ValueError):
        async with instance_pool.track_request(1):
            raise ValueError("boom")
    assert instance_pool._state(1).in_flight == 0


def test_cached_proxy_mode_round_trips(monkeypatch):
    monkeypatch.setattr(instance_pool, "_proxy_mode", "proxy")
    assert instance_pool.get_cached_proxy_mode() == "proxy"
    instance_pool.set_cached_proxy_mode("local")
    assert instance_pool.get_cached_proxy_mode() == "local"
