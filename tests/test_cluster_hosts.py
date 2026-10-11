"""Nodes are one list, whichever field stores them."""

from app.schemas import MatricxonServerConfig
from app.services.cluster_hosts import ClusterHosts
from app.services.matricxon_pool import LOCAL_MATRICXON_HOST as LOCAL

A, B, C = "http://a:8420", "http://b:8420", "http://c:8420"


def _config(**fields) -> MatricxonServerConfig:
    return MatricxonServerConfig(**fields)


def test_the_nodes_are_one_list_in_both_modes():
    assert ClusterHosts.nodes(_config(mode="local", cluster_hosts=[A])) == [LOCAL, A]
    assert ClusterHosts.nodes(_config(mode="remote", remote_hosts=[B], cluster_hosts=[A])) == [B, A]
    assert ClusterHosts.nodes(_config(mode="remote", remote_hosts=[A], cluster_hosts=[A])) == [A]


def test_taking_a_node_out_removes_it_from_wherever_it_is_stored():
    config = _config(mode="remote", remote_hosts=[A, B], cluster_hosts=[B, C])
    ClusterHosts.without(config, B)
    assert ClusterHosts.nodes(config) == [A, C]
    ClusterHosts.without(config, A)
    assert (config.mode, ClusterHosts.nodes(config)) == ("remote", [C])


def test_taking_out_the_node_pairing_manages_switches_to_remote_mode_with_the_others_as_hosts():
    config = _config(mode="local", cluster_hosts=[A, B])
    ClusterHosts.without(config, LOCAL)
    assert (config.mode, config.remote_hosts, config.cluster_hosts) == ("remote", [A, B], [])


def _save(previous: MatricxonServerConfig, **form) -> MatricxonServerConfig:
    """What the form sends: no cluster_hosts, cluster role as loaded."""
    saved = _config(cluster_role=previous.cluster_role, **form)
    ClusterHosts.keep_through_a_save(previous, saved)
    return saved


def test_without_a_cluster_the_forms_own_host_list_is_the_list():
    previous = _config(mode="remote", remote_hosts=[A], cluster_role="off")
    saved = _save(previous, mode="remote", remote_hosts=[A, B])
    assert ClusterHosts.nodes(saved) == [A, B]


def test_in_cluster_mode_the_form_cannot_change_the_nodes_nor_lose_one_by_switching_mode():
    previous = _config(mode="local", cluster_hosts=[A], cluster_role="seed")
    typed = _save(previous, mode="local", remote_hosts=[C])
    assert ClusterHosts.nodes(typed) == [LOCAL, A]

    to_remote = _save(previous, mode="remote", remote_hosts=[C])
    assert ClusterHosts.nodes(to_remote) == [LOCAL, A] and to_remote.mode == "remote"

    back = _save(to_remote, mode="local")
    assert ClusterHosts.nodes(back) == [LOCAL, A] and back.mode == "local"


def test_the_form_never_wipes_what_the_cluster_page_owns():
    """Bug: every form Save emptied the advertise address; the restarted seed lost its quorum."""
    previous = _config(
        mode="local",
        cluster_role="seed",
        advertise_url="http://host.containers.internal:8420",
        models_path="/mnt/shared/models",
        prompt_cache_path="/mnt/shared/cache",
        cluster_hosts=[A],
    )
    saved = _config(mode="local", cluster_role="off", models_path="/elsewhere", log_level=2)
    ClusterHosts.keep_through_a_save(previous, saved)

    assert saved.advertise_url == "http://host.containers.internal:8420"
    assert saved.cluster_role == "seed"  # stale "off" does not turn the cluster off
    assert (saved.models_path, saved.prompt_cache_path) == ("/mnt/shared/models", "/mnt/shared/cache")
    assert saved.cluster_hosts == [A]
    assert saved.log_level == 2

    alone = _config(cluster_role="off", advertise_url="http://x:8420", models_path="/mine")
    own = _config(cluster_role="off", models_path="/changed")
    ClusterHosts.keep_through_a_save(alone, own)
    assert own.models_path == "/changed" and own.advertise_url == "http://x:8420"


def test_a_form_save_keeps_the_chosen_first_node():
    previous = _config(cluster_role="seed", first_node="http://n1:8420")
    saved = _config(cluster_role="off", first_node=None)
    ClusterHosts.keep_through_a_save(previous, saved)
    assert saved.first_node == "http://n1:8420"
