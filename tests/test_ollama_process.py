"""Unit tests for app/services/ollama_process.py — the local Ollama
process supervisor behind Settings > External servers. The actual real
stop/start of a real Ollama process isn't exercised here — see this
project's live end-to-end verification instead (real process, real
curl), same as every other feature this session that manages an OS
process (app/services/instance_process.py/comfyui_process.py's own
tests take the same approach). subprocess/pgrep/pkill and httpx are all
monkeypatched."""

import subprocess

import httpx
import pytest

from app.schemas import OllamaServerConfig
from app.services import ollama_process


def _config(**overrides) -> OllamaServerConfig:
    return OllamaServerConfig(**overrides)


def test_find_binary_prefers_path(monkeypatch):
    monkeypatch.setattr(ollama_process.shutil, "which", lambda name: "/usr/bin/ollama" if name == "ollama" else None)
    assert ollama_process._find_binary() == "/usr/bin/ollama"


def test_find_binary_falls_back_when_not_on_path(monkeypatch, tmp_path):
    fallback = tmp_path / "ollama"
    fallback.write_text("#!/bin/sh\n")
    fallback.chmod(0o755)
    monkeypatch.setattr(ollama_process.shutil, "which", lambda name: None)
    monkeypatch.setattr(ollama_process, "_FALLBACK_BIN_PATHS", (str(fallback),))
    assert ollama_process._find_binary() == str(fallback)


def test_find_binary_returns_none_when_nowhere_found(monkeypatch):
    monkeypatch.setattr(ollama_process.shutil, "which", lambda name: None)
    monkeypatch.setattr(ollama_process, "_FALLBACK_BIN_PATHS", ("/nonexistent/ollama",))
    assert ollama_process._find_binary() is None


def test_is_installed_reflects_find_binary(monkeypatch):
    monkeypatch.setattr(ollama_process, "_find_binary", lambda _binary_path=None: "/usr/bin/ollama")
    assert ollama_process.is_installed() is True
    monkeypatch.setattr(ollama_process, "_find_binary", lambda _binary_path=None: None)
    assert ollama_process.is_installed() is False


def test_auto_detect_binary_delegates_to_find_binary_with_no_override(monkeypatch):
    monkeypatch.setattr(ollama_process, "_find_binary", lambda binary_path=None: ("used", binary_path))
    assert ollama_process.auto_detect_binary() == ("used", None)


def test_find_binary_prefers_a_given_override_path(tmp_path, monkeypatch):
    custom = tmp_path / "my-ollama"
    custom.write_text("#!/bin/sh\n")
    custom.chmod(0o755)
    monkeypatch.setattr(ollama_process.shutil, "which", lambda _name: "/usr/bin/ollama")
    assert ollama_process._find_binary(str(custom)) == str(custom)


def test_find_binary_falls_back_to_auto_detect_when_override_does_not_exist(monkeypatch):
    monkeypatch.setattr(ollama_process.shutil, "which", lambda _name: "/usr/bin/ollama")
    assert ollama_process._find_binary("/no/such/binary") == "/usr/bin/ollama"


def test_find_binary_falls_back_to_auto_detect_when_no_override_given(monkeypatch):
    monkeypatch.setattr(ollama_process.shutil, "which", lambda _name: "/usr/bin/ollama")
    assert ollama_process._find_binary() == "/usr/bin/ollama"


def test_build_env_omits_unset_fields(monkeypatch):
    # The real ambient environment (this shell's own) may already carry
    # some of these — cleared so the assertion reflects _build_env's own
    # behavior, not whatever happens to be set outside this test.
    for key in ("OLLAMA_NUM_PARALLEL", "OLLAMA_KEEP_ALIVE", "OLLAMA_MAX_LOADED_MODELS", "OLLAMA_CONTEXT_LENGTH"):
        monkeypatch.delenv(key, raising=False)
    env = ollama_process._build_env(_config())
    assert "OLLAMA_NUM_PARALLEL" not in env
    assert "OLLAMA_KEEP_ALIVE" not in env
    assert "OLLAMA_MAX_LOADED_MODELS" not in env
    assert "OLLAMA_CONTEXT_LENGTH" not in env
    assert "OLLAMA_MODELS" in env  # always set


def test_build_env_includes_every_configured_field():
    env = ollama_process._build_env(_config(num_parallel=4, keep_alive="24h", max_loaded_models=2, context_length=8192))
    assert env["OLLAMA_NUM_PARALLEL"] == "4"
    assert env["OLLAMA_KEEP_ALIVE"] == "24h"
    assert env["OLLAMA_MAX_LOADED_MODELS"] == "2"
    assert env["OLLAMA_CONTEXT_LENGTH"] == "8192"


def test_build_env_sets_proxy_vars_when_given():
    env = ollama_process._build_env(_config(), proxy_url="http://10.0.0.5:8080")
    assert env["HTTP_PROXY"] == "http://10.0.0.5:8080"
    assert env["HTTPS_PROXY"] == "http://10.0.0.5:8080"
    assert env["http_proxy"] == "http://10.0.0.5:8080"
    assert env["https_proxy"] == "http://10.0.0.5:8080"


def test_build_env_omits_proxy_vars_when_not_given():
    env = ollama_process._build_env(_config())
    assert "HTTP_PROXY" not in env
    assert "http_proxy" not in env


@pytest.mark.asyncio
async def test_get_status_reports_not_running(monkeypatch):
    monkeypatch.setattr(ollama_process, "_is_running", lambda _binary_path=None: False)
    status = await ollama_process.get_status()
    assert status.running is False
    assert status.pid is None


@pytest.mark.asyncio
async def test_get_status_reports_running_and_healthy(monkeypatch):
    monkeypatch.setattr(ollama_process, "_is_running", lambda _binary_path=None: True)

    async def fake_ping():
        return True

    monkeypatch.setattr(ollama_process, "_ping_health", fake_ping)
    status = await ollama_process.get_status()
    assert status.running is True
    assert status.healthy is True


@pytest.mark.asyncio
async def test_start_rejects_on_a_non_primary_instance(monkeypatch):
    monkeypatch.setattr(ollama_process, "IS_PRIMARY", False)
    with pytest.raises(ValueError):
        await ollama_process.start(_config())


@pytest.mark.asyncio
async def test_start_is_a_noop_when_already_running(monkeypatch):
    monkeypatch.setattr(ollama_process, "_is_running", lambda _binary_path=None: True)

    async def fail_if_called():
        pytest.fail("should not spawn again")

    monkeypatch.setattr(ollama_process, "_start", lambda _config, _proxy_url=None: fail_if_called())

    async def fake_ping():
        return True

    monkeypatch.setattr(ollama_process, "_ping_health", fake_ping)
    status = await ollama_process.start(_config())
    assert status.running is True


@pytest.mark.asyncio
async def test_start_raises_runtime_error_when_it_never_becomes_healthy(monkeypatch):
    monkeypatch.setattr(ollama_process, "_is_running", lambda _binary_path=None: False)
    monkeypatch.setattr(ollama_process, "_start", lambda _config, _proxy_url=None: None)
    monkeypatch.setattr(ollama_process, "_START_TIMEOUT_SECONDS", 0)

    async def fake_ping():
        return False

    monkeypatch.setattr(ollama_process, "_ping_health", fake_ping)
    with pytest.raises(RuntimeError):
        await ollama_process.start(_config())


@pytest.mark.asyncio
async def test_stop_rejects_on_a_non_primary_instance(monkeypatch):
    monkeypatch.setattr(ollama_process, "IS_PRIMARY", False)
    with pytest.raises(ValueError):
        await ollama_process.stop()


@pytest.mark.asyncio
async def test_stop_calls_the_stop_primitive(monkeypatch):
    called = []
    monkeypatch.setattr(ollama_process, "_stop", lambda _binary_path=None: called.append(True))
    monkeypatch.setattr(ollama_process, "_is_running", lambda _binary_path=None: False)
    await ollama_process.stop()
    assert called == [True]


@pytest.mark.asyncio
async def test_apply_local_config_rejects_on_a_non_primary_instance(monkeypatch):
    monkeypatch.setattr(ollama_process, "IS_PRIMARY", False)
    with pytest.raises(ValueError):
        await ollama_process.apply_local_config(_config())


@pytest.mark.asyncio
async def test_apply_local_config_stops_then_starts(monkeypatch):
    calls = []
    monkeypatch.setattr(ollama_process, "_stop", lambda _binary_path=None: calls.append("stop"))
    monkeypatch.setattr(ollama_process, "_start", lambda _config, _proxy_url=None: calls.append("start"))
    monkeypatch.setattr(ollama_process, "_is_running", lambda _binary_path=None: False)

    async def fake_ping():
        return True

    monkeypatch.setattr(ollama_process, "_ping_health", fake_ping)
    await ollama_process.apply_local_config(_config(num_parallel=2))
    assert calls == ["stop", "start"]


@pytest.mark.asyncio
async def test_start_passes_proxy_url_through_to_start_primitive(monkeypatch):
    captured = {}
    monkeypatch.setattr(ollama_process, "_is_running", lambda _binary_path=None: False)
    monkeypatch.setattr(
        ollama_process, "_start", lambda config, proxy_url=None: captured.update(config=config, proxy_url=proxy_url)
    )

    async def fake_ping():
        return True

    monkeypatch.setattr(ollama_process, "_ping_health", fake_ping)
    await ollama_process.start(_config(), proxy_url="http://10.0.0.5:8080")

    assert captured["proxy_url"] == "http://10.0.0.5:8080"


@pytest.mark.asyncio
async def test_apply_local_config_passes_proxy_url_through_to_start_primitive(monkeypatch):
    captured = {}
    monkeypatch.setattr(ollama_process, "_stop", lambda _binary_path=None: None)
    monkeypatch.setattr(
        ollama_process, "_start", lambda config, proxy_url=None: captured.update(config=config, proxy_url=proxy_url)
    )
    monkeypatch.setattr(ollama_process, "_is_running", lambda _binary_path=None: False)

    async def fake_ping():
        return True

    monkeypatch.setattr(ollama_process, "_ping_health", fake_ping)
    await ollama_process.apply_local_config(_config(), proxy_url="http://1.2.3.4:3128")

    assert captured["proxy_url"] == "http://1.2.3.4:3128"


def test_is_running_reflects_pgrep(monkeypatch):
    class _Result:
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: _Result())
    assert ollama_process._is_running() is True


def test_process_name_defaults_to_the_literal_ollama():
    assert ollama_process._process_name() == "ollama"
    assert ollama_process._process_name(None) == "ollama"


def test_process_name_derives_from_a_given_binary_paths_own_filename():
    assert ollama_process._process_name("/opt/custom/my-ollama") == "my-ollama"


def test_process_name_truncates_to_15_bytes_to_match_the_kernels_comm_field():
    # A real, confirmed-live gap: a custom binary_path not literally named
    # "ollama" made Start genuinely work while every is_running/stop check
    # kept reporting it as not running — pgrep/pkill -x need an exact
    # match against the kernel's own comm, which is always truncated to
    # exactly 15 bytes (TASK_COMM_LEN), same as this codebase's own
    # sibling-instance naming already accounts for (see
    # instance_process.py).
    name = ollama_process._process_name("/opt/custom/ollama-a-genuinely-long-custom-filename")
    assert name == "ollama-a-genuin"
    assert len(name) == 15


def test_is_running_matches_a_derived_process_name(monkeypatch):
    captured = []

    class _Result:
        returncode = 0

    def fake_run(argv, **_kw):
        captured.append(argv)
        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert ollama_process._is_running("/opt/custom/my-ollama") is True
    assert captured == [["pgrep", "-x", "my-ollama"]]


def test_stop_targets_the_derived_process_name(monkeypatch):
    calls = []

    def fake_run(argv, **_kw):
        calls.append(argv)

        class _Result:
            # Running for the first pgrep check (so _stop actually
            # proceeds to pkill), not running for every check after.
            returncode = 0 if argv[0] == "pgrep" and len(calls) == 1 else 1

        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    ollama_process._stop("/opt/custom/my-ollama")
    assert calls[0] == ["pgrep", "-x", "my-ollama"]  # the initial _is_running check
    assert calls[1] == ["pkill", "-x", "my-ollama"]


@pytest.mark.asyncio
async def test_ping_health_false_on_http_error(monkeypatch):
    class _FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, *_a, **_kw):
            raise httpx.ConnectError("refused", request=httpx.Request("GET", "http://x"))

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FailingClient())
    assert await ollama_process._ping_health() is False
