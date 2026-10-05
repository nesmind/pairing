"""sdcpp_process.spawn against tiny stand-in 'sd-server' scripts — startup success and failure reporting."""

import os
import signal

import pytest

from app.services import sdcpp_process


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setattr(sdcpp_process, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sdcpp_process, "_POLL_S", 0.02)


def _script(tmp_path, body: str) -> str:
    path = tmp_path / "sd-server"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return str(path)


def test_unsupported_model_raises_a_clear_error(tmp_path):
    binary = _script(
        tmp_path,
        "sleep 0.2; echo \"[ERROR] VAE tensor 'x' not in model metadata\"; "
        "echo '[ERROR] model metadata validation failed'; exit 1",
    )
    with pytest.raises(sdcpp_process.StartupError, match="not supported"):
        sdcpp_process.spawn(binary, "m.gguf", None)


def test_other_exit_points_to_the_log(tmp_path):
    binary = _script(tmp_path, "echo 'address already in use'; exit 1")
    with pytest.raises(sdcpp_process.StartupError, match="sdcpp.log"):
        sdcpp_process.spawn(binary, "m.gguf", None)


def test_old_log_lines_do_not_leak_into_the_next_start(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "sdcpp.log").write_text("not in model metadata\n")
    binary = _script(tmp_path, "exit 1")
    with pytest.raises(sdcpp_process.StartupError, match="sdcpp.log"):
        sdcpp_process.spawn(binary, "m.gguf", None)


def test_returns_pid_once_listening(tmp_path):
    binary = _script(tmp_path, "echo '[INFO] listening on: http://127.0.0.1:8189'; sleep 5")
    pid = sdcpp_process.spawn(binary, "m.gguf", None)
    try:
        assert pid and sdcpp_process.is_alive(pid, binary)
    finally:
        if pid:
            os.killpg(pid, signal.SIGTERM)


def test_missing_binary_returns_none(tmp_path):
    assert sdcpp_process.spawn(str(tmp_path / "nope"), "m.gguf", None) is None
