"""Unit tests for app/services/sdcpp_pool.py — thin wrapper over HostPool (see tests/test_server_pool.py)."""

from app.config import SDCPP_HOST
from app.schemas import SdCppConfig
from app.services import sdcpp_pool


def test_remote_config_swaps_in_the_remote_hosts_and_local_restores_the_default():
    sdcpp_pool.refresh_from_config(SdCppConfig(mode="remote", remote_hosts=["http://a:1", "http://b:2"]))
    assert sdcpp_pool.get_effective_hosts() == ["http://a:1", "http://b:2"]
    sdcpp_pool.refresh_from_config(SdCppConfig())
    assert sdcpp_pool.get_effective_hosts() == [SDCPP_HOST]
