"""Tests for services/instance_service.py (async orchestration and locking) and the pure decision logic in
instance_process.py. Every test patches the subprocess/PID/network primitives: no real process, no real
filesystem (as db_config_service's tests stub the sync-driver connect).
"""

import asyncio

import pytest

from app.config import APP_PORT
from app.services import instance_process, instance_service, settings_service


@pytest.fixture(autouse=True)
def isolated_tracking_file(tmp_path, monkeypatch):
    """Each test gets its own tracking file in tmp_path; otherwise tests would read/write the project's
    data/instances.json."""
    monkeypatch.setattr(instance_process, "_TRACKING_FILE", tmp_path / "instances.json")


# Pure logic (instance_process.plan_changes / port_for_index)


def test_plan_changes_scales_up_from_empty():
    to_spawn, to_terminate = instance_process.plan_changes(4, set())
    assert to_spawn == {1, 2, 3}
    assert to_terminate == set()


def test_plan_changes_scales_down_terminates_only_the_excess():
    # desired_count=2 wants only index 1 (range(1, 2) == [1]); 2 and 3 are excess and terminated.
    to_spawn, to_terminate = instance_process.plan_changes(2, {1, 2, 3})
    assert to_spawn == set()
    assert to_terminate == {2, 3}


def test_plan_changes_is_a_noop_when_already_matching():
    to_spawn, to_terminate = instance_process.plan_changes(3, {1, 2})
    assert to_spawn == set()
    assert to_terminate == set()


def test_plan_changes_never_wants_to_spawn_index_zero():
    # Index 0 is the primary, never a sibling plan_changes could spawn: "wanted" is always range(1, desired_count).
    to_spawn, _to_terminate = instance_process.plan_changes(8, set())
    assert 0 not in to_spawn


def test_port_for_index_is_app_port_plus_index():
    assert instance_process.port_for_index(3) == APP_PORT + 3


# Orchestration (instance_service)


@pytest.mark.asyncio
async def test_reconcile_on_startup_spawns_nothing_at_default_count(db, monkeypatch):
    """The default saved count is 1 (just the primary): reconcile must spawn nothing on a fresh install."""
    spawned = []
    monkeypatch.setattr(instance_process, "spawn_sibling", lambda index: spawned.append(index) or 1000 + index)

    await instance_service.reconcile_on_startup(db)

    assert spawned == []


@pytest.mark.asyncio
async def test_reconcile_on_startup_spawns_missing_indices(db, monkeypatch):
    await settings_service.set_instance_count(db, 3)
    spawned = []
    monkeypatch.setattr(instance_process, "spawn_sibling", lambda index: spawned.append(index) or 1000 + index)

    await instance_service.reconcile_on_startup(db)

    assert spawned == [1, 2]
    tracked = instance_process.read_tracking()
    assert set(tracked) == {1, 2}


@pytest.mark.asyncio
async def test_reconcile_on_startup_terminates_extras_on_count_decrease(db, monkeypatch):
    instance_process.write_tracking({1: {"pid": 1001}, 2: {"pid": 1002}, 3: {"pid": 1003}})
    monkeypatch.setattr(instance_process, "is_alive", lambda pid: True)
    terminated = []
    monkeypatch.setattr(instance_process, "terminate_sibling", lambda pid: terminated.append(pid))
    # count=2 wants only index 1: 2 and 3 are excess.
    await settings_service.set_instance_count(db, 2)

    await instance_service.reconcile_on_startup(db)

    assert sorted(terminated) == [1002, 1003]
    assert set(instance_process.read_tracking()) == {1}


@pytest.mark.asyncio
async def test_reconcile_on_startup_is_idempotent_when_state_already_matches(db, monkeypatch):
    instance_process.write_tracking({1: {"pid": 1001}, 2: {"pid": 1002}})
    monkeypatch.setattr(instance_process, "is_alive", lambda pid: True)
    monkeypatch.setattr(instance_process, "spawn_sibling", lambda index: pytest.fail("should not spawn"))
    monkeypatch.setattr(instance_process, "terminate_sibling", lambda pid: pytest.fail("should not terminate"))
    await settings_service.set_instance_count(db, 3)

    await instance_service.reconcile_on_startup(db)


@pytest.mark.asyncio
async def test_reconcile_on_startup_respawns_a_tracked_pid_that_is_no_longer_alive(db, monkeypatch):
    """Covers PID reuse or a sibling dying unexpectedly: a tracked index whose is_alive now fails is dead and respawned,
    not assumed to be the same process."""
    instance_process.write_tracking({1: {"pid": 1001}})
    monkeypatch.setattr(instance_process, "is_alive", lambda pid: False)
    spawned = []
    monkeypatch.setattr(instance_process, "spawn_sibling", lambda index: spawned.append(index) or 2001)
    await settings_service.set_instance_count(db, 2)

    await instance_service.reconcile_on_startup(db)

    assert spawned == [1]
    assert instance_process.read_tracking()[1]["pid"] == 2001


@pytest.mark.asyncio
async def test_set_instance_count_rejects_out_of_range(db):
    with pytest.raises(ValueError):
        await instance_service.set_instance_count(db, 0)
    with pytest.raises(ValueError):
        await instance_service.set_instance_count(db, 9)


@pytest.mark.asyncio
async def test_set_instance_count_persists_and_reconciles_in_one_call(db, monkeypatch):
    monkeypatch.setattr(instance_process, "spawn_sibling", lambda index: 3000 + index)
    monkeypatch.setattr(instance_process, "is_alive", lambda pid: True)
    monkeypatch.setattr(instance_service, "_ping_health", _fake_ping_health(healthy=True))

    config = await instance_service.set_instance_count(db, 3)

    assert await settings_service.get_instance_count(db) == 3
    assert config.count == 3
    assert {inst.index for inst in config.instances} == {0, 1, 2}


def _fake_ping_health(healthy: bool):
    async def _fake(port: int) -> bool:
        return healthy

    return _fake


@pytest.mark.asyncio
async def test_get_status_reports_alive_and_healthy(db, monkeypatch):
    instance_process.write_tracking({1: {"pid": 1001}})
    monkeypatch.setattr(instance_process, "is_alive", lambda pid: True)
    monkeypatch.setattr(instance_service, "_ping_health", _fake_ping_health(healthy=True))
    await settings_service.set_instance_count(db, 2)

    config = await instance_service.get_status(db)

    assert config.count == 2
    by_index = {inst.index: inst for inst in config.instances}
    assert by_index[0].alive is True
    assert by_index[0].healthy is True
    assert by_index[1].pid == 1001
    assert by_index[1].healthy is True


@pytest.mark.asyncio
async def test_reconcile_on_startup_serializes_concurrent_calls(monkeypatch):
    """Proves _lock gives mutual exclusion: two concurrent reconcile_on_startup calls must never run their critical
    sections (_reconcile_locked) together. Tracks concurrent entries with a counter rather than forcing a
    spawn-count race through timing: asyncio yields only at real suspension points, so fakes that never yield
    can't reproduce interleaving even without the lock; a counter proves it deterministically.
    """
    in_progress = 0
    max_concurrent = 0

    async def fake_reconcile_locked(_db):
        nonlocal in_progress, max_concurrent
        in_progress += 1
        max_concurrent = max(max_concurrent, in_progress)
        await asyncio.sleep(0.01)
        in_progress -= 1
        return {}

    monkeypatch.setattr(instance_service, "_reconcile_locked", fake_reconcile_locked)

    await asyncio.gather(
        instance_service.reconcile_on_startup(None),
        instance_service.reconcile_on_startup(None),
    )

    assert max_concurrent == 1


@pytest.mark.asyncio
async def test_terminate_all_siblings_clears_tracking_and_terminates_each(db, monkeypatch):
    instance_process.write_tracking({1: {"pid": 1001}, 2: {"pid": 1002}})
    terminated = []
    monkeypatch.setattr(instance_process, "terminate_sibling", lambda pid: terminated.append(pid))

    await instance_service.terminate_all_siblings()

    assert sorted(terminated) == [1001, 1002]
    assert instance_process.read_tracking() == {}
