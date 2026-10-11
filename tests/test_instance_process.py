"""Tests for instance_process.py's port_for_index. A real bug lived here: called from a sibling
(server_pool_broadcast._other_live_ports,
the only caller that can run off the primary), it used this process's own port as the "base" and added `index`,
correct only on the primary since a sibling's APP_PORT (spawn_sibling) is already offset by its index. Patches
this module's APP_PORT/INSTANCE_INDEX bindings, not app.config's (module-split gotcha).
"""

from app.services import instance_process


def test_port_for_index_from_the_primarys_own_perspective(monkeypatch):
    monkeypatch.setattr(instance_process, "APP_PORT", 8000)
    monkeypatch.setattr(instance_process, "INSTANCE_INDEX", 0)

    assert instance_process.port_for_index(0) == 8000
    assert instance_process.port_for_index(1) == 8001
    assert instance_process.port_for_index(2) == 8002


def test_port_for_index_from_a_siblings_own_perspective_still_recovers_the_true_base_port(monkeypatch):
    """The broken scenario: called inside sibling 1, whose APP_PORT (set by spawn_sibling) is already 8001, not the
    primary's 8000."""
    monkeypatch.setattr(instance_process, "APP_PORT", 8001)
    monkeypatch.setattr(instance_process, "INSTANCE_INDEX", 1)

    assert instance_process.port_for_index(0) == 8000
    assert instance_process.port_for_index(1) == 8001
    assert instance_process.port_for_index(2) == 8002


def test_port_for_index_from_a_second_siblings_own_perspective(monkeypatch):
    monkeypatch.setattr(instance_process, "APP_PORT", 8002)
    monkeypatch.setattr(instance_process, "INSTANCE_INDEX", 2)

    assert instance_process.port_for_index(0) == 8000
    assert instance_process.port_for_index(2) == 8002
