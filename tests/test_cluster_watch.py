"""What pAIring logs when the cluster changes between two readings."""

import logging

from app.schemas.cluster import ClusterHost
from app.services.cluster_watch import ClusterWatch


def _host(name: str, up: bool = True, ready: bool = True, active: bool = True, role: str | None = None):
    consensus = {"role": role} if role else None
    return ClusterHost(
        host=name, reachable=up, ready=ready, active=active, error=None if up else "refused", consensus=consensus
    )


def _lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "llama_chat"]


def test_the_first_reading_is_only_a_baseline(caplog):
    caplog.set_level(logging.INFO, logger="llama_chat")
    ClusterWatch().observe([_host("http://a"), _host("http://b", up=False)], "lost")
    assert _lines(caplog) == []


def test_a_node_going_down_and_coming_back_is_logged(caplog):
    caplog.set_level(logging.INFO, logger="llama_chat")
    watch = ClusterWatch()
    watch.observe([_host("http://a"), _host("http://b")], "ok")
    watch.observe([_host("http://a"), _host("http://b", up=False)], "ok")
    watch.observe([_host("http://a"), _host("http://b")], "ok")
    text = "\n".join(_lines(caplog))
    assert "node http://b is not responding (refused)" in text
    assert "node http://b is responding again" in text
    assert "http://a" not in text


def test_nodes_joining_leaving_becoming_active_and_the_leader_moving_are_logged(caplog):
    caplog.set_level(logging.INFO, logger="llama_chat")
    watch = ClusterWatch()
    watch.observe([_host("http://a", role="leader"), _host("http://b", active=False)], "ok")
    watch.observe([_host("http://a", role="follower"), _host("http://b", role="leader"), _host("http://c")], "ok")
    watch.observe([_host("http://a", role="follower"), _host("http://b", role="leader")], "ok")
    text = "\n".join(_lines(caplog))
    assert "node http://b is now active in the cluster" in text
    assert "node http://c joined the host list" in text and "node http://c left the host list" in text
    assert "the leader is now http://b" in text


def test_losing_and_regaining_the_quorum_is_logged_but_off_to_on_is_not(caplog):
    caplog.set_level(logging.INFO, logger="llama_chat")
    watch = ClusterWatch()
    watch.observe([_host("http://a")], "off")
    watch.observe([_host("http://a")], "ok")
    watch.observe([_host("http://a")], "lost")
    watch.observe([_host("http://a")], "ok")
    text = "\n".join(_lines(caplog))
    assert text.count("NO QUORUM") == 1 and text.count("quorum is back") == 1
