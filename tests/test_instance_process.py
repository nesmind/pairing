"""Unit tests for app/services/instance_process.py's port_for_index — a real bug lived here: called from a
non-primary sibling instance (see app.services.server_pool_broadcast._other_live_ports, the only caller that can
run from anywhere but the primary), it used to compute this process's own port as the "base" and add `index` to
it — correct only for the primary itself, since a sibling's own APP_PORT env var (see spawn_sibling below) is
already offset by its own index. Patches this module's own imported APP_PORT/INSTANCE_INDEX bindings directly
(not app.config's), same "module-split import-binding gotcha" this codebase already tracks elsewhere."""

from app.services import instance_process


def test_port_for_index_from_the_primarys_own_perspective(monkeypatch):
    monkeypatch.setattr(instance_process, "APP_PORT", 8000)
    monkeypatch.setattr(instance_process, "INSTANCE_INDEX", 0)

    assert instance_process.port_for_index(0) == 8000
    assert instance_process.port_for_index(1) == 8001
    assert instance_process.port_for_index(2) == 8002


def test_port_for_index_from_a_siblings_own_perspective_still_recovers_the_true_base_port(monkeypatch):
    """The exact scenario that was broken: called from within sibling 1's own process, whose own APP_PORT (set
    by spawn_sibling) is already 8001, not the primary's 8000."""
    monkeypatch.setattr(instance_process, "APP_PORT", 8001)
    monkeypatch.setattr(instance_process, "INSTANCE_INDEX", 1)

    assert instance_process.port_for_index(0) == 8000  # the primary's real port, not this process's own 8001
    assert instance_process.port_for_index(1) == 8001  # its own port, correctly recovered too
    assert instance_process.port_for_index(2) == 8002


def test_port_for_index_from_a_second_siblings_own_perspective(monkeypatch):
    monkeypatch.setattr(instance_process, "APP_PORT", 8002)
    monkeypatch.setattr(instance_process, "INSTANCE_INDEX", 2)

    assert instance_process.port_for_index(0) == 8000
    assert instance_process.port_for_index(2) == 8002
