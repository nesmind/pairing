"""Tests for services/comfyui_installer.py. Every subprocess call (git/venv/pip) goes through a fake
asyncio.create_subprocess_exec; no network, clone or pip install happens.
"""

import shutil

import pytest

from app.services import comfyui_installer


@pytest.fixture(autouse=True)
def _assume_avx2_by_default(monkeypatch):
    """install_stream's AVX2 gate (test_install_stream_blocks_...) would make every other test depend on the real CPU;
    pinned to "supported" so the clone/venv/pip flow is deterministic.
    """
    monkeypatch.setattr(comfyui_installer, "has_avx2", lambda: True)


@pytest.fixture(autouse=True)
def _assume_linux_by_default(monkeypatch):
    """Same as _assume_avx2_by_default, for the OS gate (test_install_stream_blocks_on_an_unsupported_os): pinned to
    "supported" so tests don't depend on the real OS.
    """
    monkeypatch.setattr(comfyui_installer, "local_install_supported", lambda: True)


def test_is_installed_false_when_nothing_there(tmp_path, monkeypatch):
    monkeypatch.setattr(comfyui_installer, "EXTERNAL_DIR", tmp_path)
    assert comfyui_installer.is_installed() is False


def test_is_installed_false_while_pip_install_still_running(tmp_path, monkeypatch):
    """A confirmed-live gap: git clone and venv creation finish long before pip installs requirements.txt, so
    main.py/venv/bin/python existing must NOT report "installed" mid-install (see is_installed).
    """
    monkeypatch.setattr(comfyui_installer, "EXTERNAL_DIR", tmp_path)
    comfyui_installer.main_py_path().parent.mkdir(parents=True)
    comfyui_installer.main_py_path().write_text("")
    comfyui_installer.python_path().parent.mkdir(parents=True)
    comfyui_installer.python_path().write_text("")
    assert comfyui_installer.is_installed() is False


def test_is_installed_true_once_the_completion_marker_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(comfyui_installer, "EXTERNAL_DIR", tmp_path)
    comfyui_installer.main_py_path().parent.mkdir(parents=True)
    comfyui_installer.main_py_path().write_text("")
    comfyui_installer.python_path().parent.mkdir(parents=True)
    comfyui_installer.python_path().write_text("")
    comfyui_installer._marker_path().write_text("")
    assert comfyui_installer.is_installed() is True


@pytest.mark.asyncio
async def test_install_stream_reports_error_when_git_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(comfyui_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    events = [event async for event in comfyui_installer.install_stream()]

    assert len(events) == 1
    assert "git" in events[0]["error"]


@pytest.mark.asyncio
async def test_install_stream_blocks_on_an_unsupported_os(monkeypatch, tmp_path):
    monkeypatch.setattr(comfyui_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(comfyui_installer, "local_install_supported", lambda: False)
    monkeypatch.setattr(comfyui_installer.platform, "system", lambda: "Darwin")

    called = []
    monkeypatch.setattr(shutil, "which", lambda _name: called.append(True) or "/usr/bin/git")

    events = [event async for event in comfyui_installer.install_stream()]

    assert len(events) == 1
    assert "Darwin" in events[0]["error"]
    assert called == []


@pytest.mark.asyncio
async def test_install_stream_blocks_before_downloading_anything_when_cpu_lacks_avx2(monkeypatch, tmp_path):
    """Catches up front a confirmed-live crash: without AVX2, ComfyUI's prebuilt PyTorch/NumPy dependencies hard-crash
    ("Illegal instruction") when run. This app can't catch that, so it refuses before spending several GB of
    download/pip time.
    """
    monkeypatch.setattr(comfyui_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(comfyui_installer, "has_avx2", lambda: False)

    called = []
    monkeypatch.setattr(shutil, "which", lambda _name: called.append(True) or "/usr/bin/git")

    events = [event async for event in comfyui_installer.install_stream()]

    assert len(events) == 1
    assert "AVX2" in events[0]["error"]
    assert called == []


class _AsyncLineIterator:
    def __init__(self, lines: list[bytes]):
        self._lines = lines

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for line in self._lines:
            yield line


class _FakeSubprocess:
    def __init__(self, lines: list[bytes], returncode: int):
        self.stdout = _AsyncLineIterator(lines)
        self._returncode_to_report = returncode
        # None until wait reaps it, like asyncio.subprocess.Process; _run_streamed's finally checks it to decide if a
        # kill is needed.
        self.returncode = None

    async def wait(self):
        self.returncode = self._returncode_to_report
        return self.returncode


@pytest.mark.asyncio
async def test_install_stream_runs_clone_venv_and_pip_in_order(monkeypatch, tmp_path):
    monkeypatch.setattr(comfyui_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/git")
    monkeypatch.setattr(comfyui_installer, "get_gpu_vram_gb", lambda: 8.0)

    calls = []

    async def fake_create_subprocess_exec(*argv, **_kwargs):
        calls.append(argv)
        if argv[0] == "git":
            # Simulate git creating the target directory + requirements.txt, as a real clone would.
            target_dir = tmp_path / "ComfyUI"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "requirements.txt").write_text("some-package\n")
            (target_dir / "main.py").write_text("")
            return _FakeSubprocess([b"Cloning...\n"], 0)
        if "venv" in argv:
            # Simulate venv creation producing the interpreter is_installed checks for.
            comfyui_installer.python_path().parent.mkdir(parents=True, exist_ok=True)
            comfyui_installer.python_path().write_text("")
        return _FakeSubprocess([b"ok\n"], 0)

    monkeypatch.setattr(comfyui_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    events = [event async for event in comfyui_installer.install_stream()]

    assert events[-1]["done"] is True
    assert events[-1]["step"] == 3
    assert events[-1]["total_steps"] == 3
    assert events[-1]["python_path"] == str(comfyui_installer.python_path())
    assert events[-1]["main_py_path"] == str(comfyui_installer.main_py_path())
    assert events[-1]["extra_args"] is None
    assert calls[0][0] == "git"
    assert calls[1][1:3] == ("-m", "venv")
    assert "pip" in calls[2]
    assert comfyui_installer._marker_path().exists()
    assert comfyui_installer.is_installed() is True

    clone_events = [e for e in events if e.get("step") == 1]
    venv_events = [e for e in events if e.get("step") == 2]
    pip_events = [e for e in events if e.get("step") == 3 and not e.get("done")]
    assert all(e["step_label"] == "Cloning" for e in clone_events)
    assert all(e["step_label"] == "Creating a Python environment" for e in venv_events)
    assert all(e["step_label"].startswith("Installing dependencies") for e in pip_events)


def test_preserve_and_restore_models_dir_round_trips_real_files(tmp_path):
    target_dir = tmp_path / "ComfyUI"
    checkpoints_dir = target_dir / "models" / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    (checkpoints_dir / "sd_xl.safetensors").write_text("fake checkpoint")

    preserved = comfyui_installer._preserve_models_dir(target_dir)

    assert preserved is not None
    assert not (target_dir / "models").exists()
    assert (preserved / "checkpoints" / "sd_xl.safetensors").read_text() == "fake checkpoint"

    comfyui_installer._restore_models_dir(preserved, target_dir)

    assert (target_dir / "models" / "checkpoints" / "sd_xl.safetensors").read_text() == "fake checkpoint"
    assert not preserved.parent.exists()


def test_preserve_models_dir_is_a_noop_when_theres_nothing_there(tmp_path):
    target_dir = tmp_path / "ComfyUI"
    target_dir.mkdir()
    assert comfyui_installer._preserve_models_dir(target_dir) is None


@pytest.mark.asyncio
async def test_install_stream_preserves_previously_pulled_checkpoints_across_a_reinstall(monkeypatch, tmp_path):
    """Regression: reinstalling used to `shutil.rmtree` the whole target_dir, destroying every pulled checkpoint
    (often several GB) in ComfyUI's default models/, pAIring's managed library rather than a disposable install
    artifact.
    """
    monkeypatch.setattr(comfyui_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/git")
    monkeypatch.setattr(comfyui_installer, "get_gpu_vram_gb", lambda: 8.0)

    existing_target = tmp_path / "ComfyUI"
    existing_checkpoints = existing_target / "models" / "checkpoints"
    existing_checkpoints.mkdir(parents=True)
    (existing_checkpoints / "already-pulled.safetensors").write_text("real weights, several GB in spirit")

    async def fake_create_subprocess_exec(*argv, **_kwargs):
        if argv[0] == "git":
            existing_target.mkdir(parents=True, exist_ok=True)
            (existing_target / "requirements.txt").write_text("some-package\n")
            (existing_target / "main.py").write_text("")
            return _FakeSubprocess([b"Cloning...\n"], 0)
        if "venv" in argv:
            comfyui_installer.python_path().parent.mkdir(parents=True, exist_ok=True)
            comfyui_installer.python_path().write_text("")
        return _FakeSubprocess([b"ok\n"], 0)

    monkeypatch.setattr(comfyui_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    events = [event async for event in comfyui_installer.install_stream()]

    assert events[-1]["done"] is True
    restored = existing_target / "models" / "checkpoints" / "already-pulled.safetensors"
    assert restored.exists()
    assert restored.read_text() == "real weights, several GB in spirit"


@pytest.mark.asyncio
async def test_install_stream_restores_preserved_models_even_when_the_clone_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(comfyui_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/git")

    existing_target = tmp_path / "ComfyUI"
    existing_checkpoints = existing_target / "models" / "checkpoints"
    existing_checkpoints.mkdir(parents=True)
    (existing_checkpoints / "already-pulled.safetensors").write_text("must survive a failed reinstall too")

    async def fake_create_subprocess_exec(*_argv, **_kwargs):
        return _FakeSubprocess([b"fatal: could not resolve host\n"], 1)

    monkeypatch.setattr(comfyui_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    events = [event async for event in comfyui_installer.install_stream()]

    assert "error" in events[-1]
    restored = existing_target / "models" / "checkpoints" / "already-pulled.safetensors"
    assert restored.exists()
    assert restored.read_text() == "must survive a failed reinstall too"


@pytest.mark.asyncio
async def test_install_stream_sets_proxy_env_for_clone_and_pip_but_not_venv(monkeypatch, tmp_path):
    """git clone and pip install make network requests and must honor the proxy; venv creation doesn't, so it is
    left alone (env=None, inheriting this process's env).
    """
    monkeypatch.setattr(comfyui_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/git")
    monkeypatch.setattr(comfyui_installer, "get_gpu_vram_gb", lambda: 8.0)

    calls = []

    async def fake_create_subprocess_exec(*argv, **kwargs):
        calls.append((argv, kwargs.get("env")))
        if argv[0] == "git":
            target_dir = tmp_path / "ComfyUI"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "requirements.txt").write_text("some-package\n")
            (target_dir / "main.py").write_text("")
            return _FakeSubprocess([b"Cloning...\n"], 0)
        if "venv" in argv:
            comfyui_installer.python_path().parent.mkdir(parents=True, exist_ok=True)
            comfyui_installer.python_path().write_text("")
        return _FakeSubprocess([b"ok\n"], 0)

    monkeypatch.setattr(comfyui_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    [event async for event in comfyui_installer.install_stream(proxy_url="http://10.0.0.5:8080")]

    git_env, venv_env, pip_env = calls[0][1], calls[1][1], calls[2][1]
    for env in (git_env, pip_env):
        assert env["HTTP_PROXY"] == "http://10.0.0.5:8080"
        assert env["HTTPS_PROXY"] == "http://10.0.0.5:8080"
        assert env["http_proxy"] == "http://10.0.0.5:8080"
        assert env["https_proxy"] == "http://10.0.0.5:8080"
    assert venv_env is None


@pytest.mark.asyncio
async def test_install_stream_passes_no_env_override_when_no_proxy_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(comfyui_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/git")
    monkeypatch.setattr(comfyui_installer, "get_gpu_vram_gb", lambda: 8.0)

    calls = []

    async def fake_create_subprocess_exec(*argv, **kwargs):
        calls.append(kwargs.get("env"))
        if argv[0] == "git":
            target_dir = tmp_path / "ComfyUI"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "requirements.txt").write_text("some-package\n")
            (target_dir / "main.py").write_text("")
            return _FakeSubprocess([b"Cloning...\n"], 0)
        if "venv" in argv:
            comfyui_installer.python_path().parent.mkdir(parents=True, exist_ok=True)
            comfyui_installer.python_path().write_text("")
        return _FakeSubprocess([b"ok\n"], 0)

    monkeypatch.setattr(comfyui_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    [event async for event in comfyui_installer.install_stream()]

    assert calls == [None, None, None]


def test_proxied_env_returns_none_when_no_proxy_given():
    assert comfyui_installer._proxied_env(None) is None


def test_proxied_env_sets_both_cases_when_a_proxy_is_given():
    env = comfyui_installer._proxied_env("http://1.2.3.4:3128")
    assert env["HTTP_PROXY"] == "http://1.2.3.4:3128"
    assert env["http_proxy"] == "http://1.2.3.4:3128"


@pytest.mark.asyncio
async def test_run_streamed_passes_env_through_to_the_subprocess(monkeypatch):
    captured = {}

    async def fake_create_subprocess_exec(*_argv, **kwargs):
        captured.update(kwargs)
        return _FakeSubprocess([b"ok\n"], 0)

    monkeypatch.setattr(comfyui_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    [line async for line in comfyui_installer._run_streamed(["echo", "hi"], env={"FOO": "bar"})]

    assert captured["env"] == {"FOO": "bar"}


@pytest.mark.asyncio
async def test_install_stream_suggests_cpu_flag_when_no_gpu_present(monkeypatch, tmp_path):
    """A confirmed-live crash: ComfyUI's requirements.txt pulls a CUDA torch build, which raises "RuntimeError:
    Found no NVIDIA driver on your system" on a GPU-less machine unless launched with --cpu.
    """
    monkeypatch.setattr(comfyui_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/git")
    monkeypatch.setattr(comfyui_installer, "get_gpu_vram_gb", lambda: 0.0)

    async def fake_create_subprocess_exec(*argv, **_kwargs):
        if argv[0] == "git":
            target_dir = tmp_path / "ComfyUI"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "requirements.txt").write_text("some-package\n")
            (target_dir / "main.py").write_text("")
        if "venv" in argv:
            comfyui_installer.python_path().parent.mkdir(parents=True, exist_ok=True)
            comfyui_installer.python_path().write_text("")
        return _FakeSubprocess([b"ok\n"], 0)

    monkeypatch.setattr(comfyui_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    events = [event async for event in comfyui_installer.install_stream()]

    assert events[-1]["extra_args"] == "--cpu"


@pytest.mark.asyncio
async def test_install_stream_uses_a_given_repo_and_version_override(monkeypatch, tmp_path):
    monkeypatch.setattr(comfyui_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/git")
    monkeypatch.setattr(comfyui_installer, "get_gpu_vram_gb", lambda: 0.0)

    calls = []

    async def fake_create_subprocess_exec(*argv, **_kwargs):
        calls.append(argv)
        if argv[0] == "git":
            target_dir = tmp_path / "ComfyUI"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "requirements.txt").write_text("some-package\n")
            (target_dir / "main.py").write_text("")
        return _FakeSubprocess([b"ok\n"], 0)

    monkeypatch.setattr(comfyui_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    events = [event async for event in comfyui_installer.install_stream(repo="me/comfy-fork", version="v9.9.9")]

    assert calls[0] == (
        "git",
        "clone",
        "--branch",
        "v9.9.9",
        "--depth",
        "1",
        "https://github.com/me/comfy-fork.git",
        str(tmp_path / "ComfyUI"),
    )
    assert "v9.9.9" in events[0]["status"]


@pytest.mark.asyncio
async def test_install_stream_removes_a_partial_clone_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(comfyui_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/git")

    async def fake_create_subprocess_exec(*argv, **_kwargs):
        target_dir = tmp_path / "ComfyUI"
        target_dir.mkdir(parents=True, exist_ok=True)
        return _FakeSubprocess([b"error\n"], 1)

    monkeypatch.setattr(comfyui_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    events = [event async for event in comfyui_installer.install_stream()]

    assert "error" in events[-1]
    assert not (tmp_path / "ComfyUI").exists()


class _FakeInfiniteSubprocess:
    """A still-running child (git/pip) whose stdout produces lines forever: the live defect where a client
    disconnecting mid-install left this process running orphaned.
    """

    def __init__(self):
        self.killed = False
        self.returncode = None
        self.stdout = self

    def __aiter__(self):
        return self

    async def __anext__(self):
        return b"still running...\n"

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


@pytest.mark.asyncio
async def test_run_streamed_kills_the_subprocess_if_torn_down_early(monkeypatch):
    fake_process = _FakeInfiniteSubprocess()

    async def fake_create_subprocess_exec(*_argv, **_kwargs):
        return fake_process

    monkeypatch.setattr(comfyui_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    gen = comfyui_installer._run_streamed(["some-long-running-cmd"])
    line = await gen.__anext__()
    assert line == "still running..."
    assert fake_process.killed is False

    await gen.aclose()

    assert fake_process.killed is True


@pytest.mark.asyncio
async def test_run_streamed_does_not_kill_a_process_that_already_exited_cleanly(monkeypatch):
    fake_process = _FakeSubprocess([b"done\n"], 0)
    fake_process.killed = False
    fake_process.kill = lambda: setattr(fake_process, "killed", True)

    async def fake_create_subprocess_exec(*_argv, **_kwargs):
        return fake_process

    monkeypatch.setattr(comfyui_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    lines = [line async for line in comfyui_installer._run_streamed(["ok-cmd"])]

    assert lines == ["done"]
    assert fake_process.killed is False


@pytest.mark.asyncio
async def test_install_stream_removes_a_pre_existing_directory_before_cloning(monkeypatch, tmp_path):
    monkeypatch.setattr(comfyui_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/git")
    stale_dir = comfyui_installer.install_dir()
    stale_dir.mkdir(parents=True)
    (stale_dir / "stale-file").write_text("leftover from a failed attempt")

    async def fake_create_subprocess_exec(*argv, **_kwargs):
        return _FakeSubprocess([], 1)

    monkeypatch.setattr(comfyui_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    [event async for event in comfyui_installer.install_stream()]

    assert not (stale_dir / "stale-file").exists()
