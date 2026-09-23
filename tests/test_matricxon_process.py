"""Unit tests for app/services/matricxon_process.py — the local Matricxon process supervisor behind Settings >
External servers. The actual real start/stop of a real Matricxon process isn't exercised here — see
tests/test_ollama_process.py's own docstring for why (subprocess/os.kill/httpx are all monkeypatched)."""

import httpx
import pytest
from pydantic import ValidationError

from app.schemas import MatricxonServerConfig
from app.services import matricxon_process


def _config(**overrides) -> MatricxonServerConfig:
    return MatricxonServerConfig(**overrides)


def _make_checkout(tmp_path):
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts" / "start.sh").write_text("#!/usr/bin/env bash\n")
    return tmp_path


def test_find_project_dir_prefers_a_given_override(tmp_path):
    checkout = _make_checkout(tmp_path / "override")
    assert matricxon_process._find_project_dir(str(checkout)) == checkout


def test_find_project_dir_falls_back_to_default_when_override_missing(tmp_path, monkeypatch):
    default_checkout = _make_checkout(tmp_path / "default")
    monkeypatch.setattr(matricxon_process, "_DEFAULT_PROJECT_DIR", default_checkout)
    assert matricxon_process._find_project_dir("/no/such/checkout") == default_checkout


def test_find_project_dir_returns_none_when_nowhere_found(tmp_path, monkeypatch):
    monkeypatch.setattr(matricxon_process, "_DEFAULT_PROJECT_DIR", tmp_path / "nowhere")
    assert matricxon_process._find_project_dir() is None


def test_is_installed_reflects_find_project_dir(monkeypatch):
    monkeypatch.setattr(matricxon_process, "_find_project_dir", lambda _project_dir=None: "/some/dir")
    assert matricxon_process.is_installed() is True
    monkeypatch.setattr(matricxon_process, "_find_project_dir", lambda _project_dir=None: None)
    assert matricxon_process.is_installed() is False


def test_auto_detect_project_dir_stringifies_the_found_path(tmp_path, monkeypatch):
    checkout = _make_checkout(tmp_path)
    monkeypatch.setattr(matricxon_process, "_find_project_dir", lambda: checkout)
    assert matricxon_process.auto_detect_project_dir() == str(checkout)


def test_auto_detect_project_dir_is_none_when_not_found(monkeypatch):
    monkeypatch.setattr(matricxon_process, "_find_project_dir", lambda: None)
    assert matricxon_process.auto_detect_project_dir() is None


def test_read_pid_returns_none_when_no_pid_file(tmp_path):
    assert matricxon_process._read_pid(tmp_path) is None


def test_read_pid_returns_none_on_corrupt_pid_file(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "matricxon.pid").write_text("not-a-number")
    assert matricxon_process._read_pid(tmp_path) is None


def test_read_pid_parses_a_real_pid_file(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "matricxon.pid").write_text("4242\n")
    assert matricxon_process._read_pid(tmp_path) == 4242


def test_is_running_false_with_no_pid_file(tmp_path):
    assert matricxon_process._is_running(tmp_path) is False


def test_is_running_true_when_kill_succeeds(tmp_path, monkeypatch):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "matricxon.pid").write_text("4242")
    monkeypatch.setattr(matricxon_process.os, "kill", lambda _pid, _sig: None)
    assert matricxon_process._is_running(tmp_path) is True


def test_is_running_false_when_process_lookup_fails(tmp_path, monkeypatch):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "matricxon.pid").write_text("4242")

    def fake_kill(_pid, _sig):
        raise ProcessLookupError

    monkeypatch.setattr(matricxon_process.os, "kill", fake_kill)
    assert matricxon_process._is_running(tmp_path) is False


def test_build_env_omits_unset_fields():
    env = matricxon_process._build_env(_config())
    assert "MATRICXON_MAX_LOADED_MODELS" not in env
    assert "MATRICXON_MEMORY_SAFETY_MARGIN" not in env
    assert "MATRICXON_MODELS_DIR" not in env
    assert "MATRICXON_TORCH_THREADS" not in env


def test_build_env_includes_every_configured_field():
    env = matricxon_process._build_env(_config(max_loaded_models=3, memory_safety_margin=1.2, torch_threads=2))
    assert env["MATRICXON_MAX_LOADED_MODELS"] == "3"
    assert env["MATRICXON_MEMORY_SAFETY_MARGIN"] == "1.2"
    assert env["MATRICXON_TORCH_THREADS"] == "2"


@pytest.mark.parametrize("value", [0, 257])
def test_torch_threads_rejects_out_of_range_values(value):
    with pytest.raises(ValidationError):
        MatricxonServerConfig(torch_threads=value)


def test_build_env_honors_a_models_path_override():
    env = matricxon_process._build_env(_config(models_path="/mnt/big-disk/matricxon-models"))
    assert env["MATRICXON_MODELS_DIR"] == "/mnt/big-disk/matricxon-models"


def test_resolve_project_dir_prefers_a_given_override(tmp_path):
    checkout = _make_checkout(tmp_path / "override")
    assert matricxon_process.resolve_project_dir(str(checkout)) == checkout


def test_resolve_project_dir_falls_back_to_default_when_no_override_given(tmp_path, monkeypatch):
    default_checkout = _make_checkout(tmp_path / "default")
    monkeypatch.setattr(matricxon_process, "_DEFAULT_PROJECT_DIR", default_checkout)
    assert matricxon_process.resolve_project_dir() == default_checkout


def test_auto_detect_models_dir_appends_data_models_to_the_found_project_dir(tmp_path, monkeypatch):
    checkout = _make_checkout(tmp_path)
    monkeypatch.setattr(matricxon_process, "_find_project_dir", lambda: checkout)
    assert matricxon_process.auto_detect_models_dir() == str(checkout / "data" / "models")


def test_auto_detect_models_dir_is_none_when_no_checkout_found(monkeypatch):
    monkeypatch.setattr(matricxon_process, "_find_project_dir", lambda: None)
    assert matricxon_process.auto_detect_models_dir() is None


@pytest.mark.parametrize("value", [1.0, 1.09, 1.81, 2.0])
def test_memory_safety_margin_rejects_values_outside_the_matricxon_supported_range(value):
    """1.1-1.8 mirrors the same bound matricxon's own Settings.memory_safety_margin enforces
    server-side (see ../matricxon/app/config.py) — rejected here too so a bad value is caught at
    Save time with a clear message instead of only failing on matricxon's next restart."""
    with pytest.raises(ValidationError):
        MatricxonServerConfig(memory_safety_margin=value)


def test_build_env_defaults_quantized_native_compute_and_log_level_off():
    """Unlike max_loaded_models/memory_safety_margin, these two are never omitted — they're always concrete
    (bool/int, not Optional), and their defaults already match matricxon's own (see
    ../matricxon/app/config.py's Settings.enable_quantized_native_compute/log_level)."""
    env = matricxon_process._build_env(_config())
    assert env["MATRICXON_ENABLE_QUANTIZED_NATIVE_COMPUTE"] == "false"
    assert env["MATRICXON_GEMV_BACKEND"] == "numba"
    assert env["MATRICXON_LOG_LEVEL"] == "0"


def test_build_env_passes_through_quantized_native_compute_and_log_level():
    env = matricxon_process._build_env(
        _config(enable_quantized_native_compute=True, gemv_backend="native", log_level=2)
    )
    assert env["MATRICXON_ENABLE_QUANTIZED_NATIVE_COMPUTE"] == "true"
    assert env["MATRICXON_GEMV_BACKEND"] == "native"
    assert env["MATRICXON_LOG_LEVEL"] == "2"


def test_gemv_backend_rejects_unknown_values():
    """Only matricxon's own two Settings.gemv_backend values - rejected at Save time, same reasoning as
    memory_safety_margin above."""
    with pytest.raises(ValidationError):
        MatricxonServerConfig(gemv_backend="cuda")


@pytest.mark.parametrize("value", [-1, 3])
def test_log_level_rejects_values_outside_the_matricxon_supported_range(value):
    """0-2 mirrors matricxon's own Settings.log_level bound (../matricxon/app/config.py) — same
    "reject at Save time" reasoning as memory_safety_margin above."""
    with pytest.raises(ValidationError):
        MatricxonServerConfig(log_level=value)


def test_build_env_sets_proxy_vars_when_given():
    env = matricxon_process._build_env(_config(), proxy_url="http://10.0.0.5:8080")
    assert env["HTTP_PROXY"] == "http://10.0.0.5:8080"
    assert env["HTTPS_PROXY"] == "http://10.0.0.5:8080"


def test_build_env_omits_proxy_vars_when_not_given():
    env = matricxon_process._build_env(_config())
    assert "HTTP_PROXY" not in env


@pytest.mark.asyncio
async def test_get_status_reports_not_installed(monkeypatch):
    monkeypatch.setattr(matricxon_process, "_find_project_dir", lambda _project_dir=None: None)
    status = await matricxon_process.get_status()
    assert status.running is False
    assert status.installed is False


@pytest.mark.asyncio
async def test_get_status_reports_installed_but_not_running(tmp_path, monkeypatch):
    checkout = _make_checkout(tmp_path)
    monkeypatch.setattr(matricxon_process, "_find_project_dir", lambda _project_dir=None: checkout)
    status = await matricxon_process.get_status()
    assert status.running is False
    assert status.installed is True


@pytest.mark.asyncio
async def test_get_status_reports_running_and_healthy(tmp_path, monkeypatch):
    checkout = _make_checkout(tmp_path)
    monkeypatch.setattr(matricxon_process, "_find_project_dir", lambda _project_dir=None: checkout)
    monkeypatch.setattr(matricxon_process, "_is_running", lambda _project_dir: True)
    monkeypatch.setattr(matricxon_process, "_read_pid", lambda _project_dir: 4242)

    async def fake_ping():
        return True

    monkeypatch.setattr(matricxon_process, "_ping_health", fake_ping)
    status = await matricxon_process.get_status()
    assert status.running is True
    assert status.pid == 4242
    assert status.healthy is True


@pytest.mark.asyncio
async def test_start_rejects_on_a_non_primary_instance(monkeypatch):
    monkeypatch.setattr(matricxon_process, "IS_PRIMARY", False)
    with pytest.raises(ValueError):
        await matricxon_process.start(_config())


@pytest.mark.asyncio
async def test_start_raises_when_no_checkout_found(monkeypatch):
    monkeypatch.setattr(matricxon_process, "_find_project_dir", lambda _project_dir=None: None)
    with pytest.raises(RuntimeError):
        await matricxon_process.start(_config())


@pytest.mark.asyncio
async def test_start_is_a_noop_when_already_running(tmp_path, monkeypatch):
    checkout = _make_checkout(tmp_path)
    monkeypatch.setattr(matricxon_process, "_find_project_dir", lambda _project_dir=None: checkout)
    monkeypatch.setattr(matricxon_process, "_is_running", lambda _project_dir: True)
    monkeypatch.setattr(matricxon_process, "_read_pid", lambda _project_dir: 4242)

    async def fail_if_called(*_a, **_kw):
        pytest.fail("should not spawn again")

    monkeypatch.setattr(matricxon_process, "_start", fail_if_called)

    async def fake_ping():
        return True

    monkeypatch.setattr(matricxon_process, "_ping_health", fake_ping)
    status = await matricxon_process.start(_config())
    assert status.running is True


@pytest.mark.asyncio
async def test_start_raises_runtime_error_when_it_never_becomes_healthy(tmp_path, monkeypatch):
    checkout = _make_checkout(tmp_path)
    monkeypatch.setattr(matricxon_process, "_find_project_dir", lambda _project_dir=None: checkout)
    monkeypatch.setattr(matricxon_process, "_is_running", lambda _project_dir: False)
    monkeypatch.setattr(matricxon_process, "_start", lambda _dir, _config, _proxy_url=None: None)
    monkeypatch.setattr(matricxon_process, "_START_TIMEOUT_SECONDS", 0)

    async def fake_ping():
        return False

    monkeypatch.setattr(matricxon_process, "_ping_health", fake_ping)
    with pytest.raises(RuntimeError):
        await matricxon_process.start(_config())


@pytest.mark.asyncio
async def test_stop_rejects_on_a_non_primary_instance(monkeypatch):
    monkeypatch.setattr(matricxon_process, "IS_PRIMARY", False)
    with pytest.raises(ValueError):
        await matricxon_process.stop()


@pytest.mark.asyncio
async def test_stop_calls_the_stop_primitive(tmp_path, monkeypatch):
    checkout = _make_checkout(tmp_path)
    monkeypatch.setattr(matricxon_process, "_find_project_dir", lambda _project_dir=None: checkout)
    called = []
    monkeypatch.setattr(matricxon_process, "_stop", lambda _dir: called.append(True))
    monkeypatch.setattr(matricxon_process, "_is_running", lambda _project_dir: False)
    await matricxon_process.stop()
    assert called == [True]


@pytest.mark.asyncio
async def test_stop_is_a_noop_when_no_checkout_found(monkeypatch):
    monkeypatch.setattr(matricxon_process, "_find_project_dir", lambda _project_dir=None: None)
    status = await matricxon_process.stop()
    assert status.running is False


@pytest.mark.asyncio
async def test_apply_local_config_stops_then_starts(tmp_path, monkeypatch):
    checkout = _make_checkout(tmp_path)
    monkeypatch.setattr(matricxon_process, "_find_project_dir", lambda _project_dir=None: checkout)
    calls = []
    monkeypatch.setattr(matricxon_process, "_stop", lambda _dir: calls.append("stop"))
    monkeypatch.setattr(matricxon_process, "_start", lambda _dir, _config, _proxy_url=None: calls.append("start"))
    monkeypatch.setattr(matricxon_process, "_is_running", lambda _project_dir: False)

    async def fake_ping():
        return True

    monkeypatch.setattr(matricxon_process, "_ping_health", fake_ping)
    await matricxon_process.apply_local_config(_config())
    assert calls == ["stop", "start"]


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
    assert await matricxon_process._ping_health() is False


def test_run_script_invokes_bash_with_the_scripts_own_path(tmp_path, monkeypatch):
    checkout = _make_checkout(tmp_path)
    captured = {}

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["cwd"] = kwargs.get("cwd")
        return _Result()

    monkeypatch.setattr(matricxon_process.subprocess, "run", fake_run)
    matricxon_process._run_script(checkout, "start.sh", {"FOO": "bar"})
    assert captured["argv"] == ["bash", str(checkout / "scripts" / "start.sh")]
    assert captured["cwd"] == checkout


def test_start_primitive_raises_on_nonzero_exit(tmp_path, monkeypatch):
    checkout = _make_checkout(tmp_path)

    class _Result:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(matricxon_process, "_run_script", lambda *_a, **_kw: _Result())
    with pytest.raises(RuntimeError):
        matricxon_process._start(checkout, _config())


def test_stop_primitive_is_a_noop_when_not_running(tmp_path, monkeypatch):
    checkout = _make_checkout(tmp_path)
    monkeypatch.setattr(matricxon_process, "_is_running", lambda _dir: False)

    def fail_if_called(*_a, **_kw):
        pytest.fail("should not run stop.sh when nothing is running")

    monkeypatch.setattr(matricxon_process, "_run_script", fail_if_called)
    matricxon_process._stop(checkout)


def test_write_env_file_writes_every_local_mode_field(tmp_path):
    checkout = _make_checkout(tmp_path)
    config = _config(
        project_dir=str(checkout),
        models_path="/mnt/big-disk/matricxon-models",
        max_loaded_models=3,
        memory_safety_margin=1.2,
        enable_quantized_native_compute=True,
        gemv_backend="native",
        torch_threads=2,
        log_level=2,
    )

    matricxon_process.write_env_file(config)

    env_text = (checkout / ".env").read_text()
    assert "MATRICXON_MODELS_DIR=/mnt/big-disk/matricxon-models" in env_text
    assert "MATRICXON_MAX_LOADED_MODELS=3" in env_text
    assert "MATRICXON_MEMORY_SAFETY_MARGIN=1.2" in env_text
    assert "MATRICXON_ENABLE_QUANTIZED_NATIVE_COMPUTE=true" in env_text
    assert "MATRICXON_GEMV_BACKEND=native" in env_text
    assert "MATRICXON_TORCH_THREADS=2" in env_text
    assert "MATRICXON_LOG_LEVEL=2" in env_text


def test_write_env_file_omits_unset_optional_fields(tmp_path):
    checkout = _make_checkout(tmp_path)

    matricxon_process.write_env_file(_config(project_dir=str(checkout)))

    env_text = (checkout / ".env").read_text()
    assert "MATRICXON_MODELS_DIR" not in env_text
    assert "MATRICXON_MAX_LOADED_MODELS" not in env_text
    assert "MATRICXON_MEMORY_SAFETY_MARGIN" not in env_text
    assert "MATRICXON_TORCH_THREADS" not in env_text
    # These two are never Optional (see MatricxonServerConfig) — always written, real defaults.
    assert "MATRICXON_ENABLE_QUANTIZED_NATIVE_COMPUTE=false" in env_text
    assert "MATRICXON_LOG_LEVEL=0" in env_text


def test_write_env_file_removes_a_previously_set_field_once_cleared(tmp_path):
    """Confirmed important: clearing models_path back to "use Matricxon's own default" must remove the
    line entirely, not write MATRICXON_MODELS_DIR= (empty) — an empty override is still an override,
    and would keep pointing Matricxon at an empty path instead of falling back to its own default."""
    checkout = _make_checkout(tmp_path)
    matricxon_process.write_env_file(_config(project_dir=str(checkout), models_path="/custom/path"))
    assert "MATRICXON_MODELS_DIR=/custom/path" in (checkout / ".env").read_text()

    matricxon_process.write_env_file(_config(project_dir=str(checkout), models_path=None))

    assert "MATRICXON_MODELS_DIR" not in (checkout / ".env").read_text()


def test_write_env_file_preserves_unrelated_lines(tmp_path):
    """An admin-set env var this app's own config form doesn't expose (e.g. MATRICXON_DEVICE — see
    MatricxonServerConfig's own docstring on why keep_alive_seconds/device were dropped from the form
    but still work if set directly) must survive a save untouched."""
    checkout = _make_checkout(tmp_path)
    (checkout / ".env").write_text("MATRICXON_DEVICE=cpu\n")

    matricxon_process.write_env_file(_config(project_dir=str(checkout), max_loaded_models=2))

    env_text = (checkout / ".env").read_text()
    assert "MATRICXON_DEVICE=cpu" in env_text
    assert "MATRICXON_MAX_LOADED_MODELS=2" in env_text


def test_write_env_file_is_a_noop_when_the_project_dir_cannot_be_resolved(tmp_path, monkeypatch):
    monkeypatch.setattr(matricxon_process, "_DEFAULT_PROJECT_DIR", tmp_path / "nowhere")
    matricxon_process.write_env_file(_config())  # must not raise
