"""Unit tests for app/services/comfyui_installer.py — every subprocess
call (git/venv/pip) is monkeypatched via a fake asyncio.create_subprocess_exec;
no real network access, clone, or pip install happens in this test
suite."""

import shutil

import pytest

from app.services import comfyui_installer


@pytest.fixture(autouse=True)
def _assume_avx2_by_default(monkeypatch):
    """install_stream's own AVX2 gate (see test_install_stream_blocks_...
    below for that behavior itself) would otherwise make every other
    test here depend on whichever real CPU happens to run the test suite
    — pinned to "supported" so the rest of this file's tests exercise
    the clone/venv/pip flow deterministically regardless of that."""
    monkeypatch.setattr(comfyui_installer, "has_avx2", lambda: True)


@pytest.fixture(autouse=True)
def _assume_linux_by_default(monkeypatch):
    """Same reasoning as _assume_avx2_by_default above, for install_stream's OS gate (see
    test_install_stream_blocks_on_an_unsupported_os below) — pinned to "supported" so the rest of this file's
    tests aren't accidentally sensitive to whichever real OS runs the test suite."""
    monkeypatch.setattr(comfyui_installer, "local_install_supported", lambda: True)


def test_is_installed_false_when_nothing_there(tmp_path, monkeypatch):
    monkeypatch.setattr(comfyui_installer, "EXTERNAL_DIR", tmp_path)
    assert comfyui_installer.is_installed() is False


def test_is_installed_false_while_pip_install_still_running(tmp_path, monkeypatch):
    """A real, confirmed-live gap: git clone and venv creation both
    finish long before pip finishes installing requirements.txt, so
    main.py/venv/bin/python existing alone must NOT report "installed"
    while the install is still actively running — see is_installed's own
    docstring."""
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
    assert called == []  # refused before even checking for git


@pytest.mark.asyncio
async def test_install_stream_blocks_before_downloading_anything_when_cpu_lacks_avx2(monkeypatch, tmp_path):
    """A real, confirmed-live crash this now catches up front instead of
    letting it happen mid-run: without AVX2, ComfyUI's own prebuilt
    PyTorch/NumPy dependencies hard-crash with "Illegal instruction" the
    moment they actually execute — not something this app can catch any
    other way, so the honest thing to do is refuse before spending
    several GB of download/pip-install time getting there."""
    monkeypatch.setattr(comfyui_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(comfyui_installer, "has_avx2", lambda: False)

    called = []
    monkeypatch.setattr(shutil, "which", lambda _name: called.append(True) or "/usr/bin/git")

    events = [event async for event in comfyui_installer.install_stream()]

    assert len(events) == 1
    assert "AVX2" in events[0]["error"]
    assert called == []  # never even checked for git — refused before any real work started


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
        # None until wait() actually reaps it — matches real
        # asyncio.subprocess.Process, and is what _run_streamed's own
        # finally block checks to decide whether a kill() is still needed.
        self.returncode = None

    async def wait(self):
        self.returncode = self._returncode_to_report
        return self.returncode


@pytest.mark.asyncio
async def test_install_stream_runs_clone_venv_and_pip_in_order(monkeypatch, tmp_path):
    monkeypatch.setattr(comfyui_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/git")
    monkeypatch.setattr(comfyui_installer, "get_gpu_vram_gb", lambda: 8.0)  # has a GPU — no --cpu suggested

    calls = []

    async def fake_create_subprocess_exec(*argv, **_kwargs):
        calls.append(argv)
        if argv[0] == "git":
            # Simulate git actually creating the target directory + requirements.txt,
            # same as a real clone would.
            target_dir = tmp_path / "ComfyUI"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "requirements.txt").write_text("some-package\n")
            (target_dir / "main.py").write_text("")
            return _FakeSubprocess([b"Cloning...\n"], 0)
        if "venv" in argv:
            # Simulate venv creation actually producing the interpreter
            # is_installed() checks for.
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
    assert events[-1]["extra_args"] is None  # a GPU is present — no --cpu suggested
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
    assert all(e["step_label"] == "Installing dependencies" for e in pip_events)


@pytest.mark.asyncio
async def test_install_stream_sets_proxy_env_for_clone_and_pip_but_not_venv(monkeypatch, tmp_path):
    """git clone and pip install both make real network requests and
    must honor the configured proxy; venv creation doesn't touch the
    network at all and must be left alone (env=None, inheriting this
    app's own process env unchanged)."""
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
    """A real, confirmed-live crash: ComfyUI's own requirements.txt pulls
    a CUDA-enabled torch build unconditionally, which raises
    "RuntimeError: Found no NVIDIA driver on your system" the instant it
    starts on a GPU-less machine, unless launched with ComfyUI's own
    --cpu flag."""
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
    """Simulates a still-running child process (git/pip) whose stdout
    keeps producing lines forever — stands in for the real defect found
    live: a client disconnecting mid-install must not leave this process
    running orphaned forever."""

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
    assert fake_process.killed is False  # still genuinely running at this point

    await gen.aclose()

    assert fake_process.killed is True


@pytest.mark.asyncio
async def test_run_streamed_does_not_kill_a_process_that_already_exited_cleanly(monkeypatch):
    fake_process = _FakeSubprocess([b"done\n"], 0)
    fake_process.killed = False
    fake_process.kill = lambda: setattr(fake_process, "killed", True)  # should never be called below

    async def fake_create_subprocess_exec(*_argv, **_kwargs):
        return fake_process

    monkeypatch.setattr(comfyui_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    lines = [line async for line in comfyui_installer._run_streamed(["ok-cmd"])]

    assert lines == ["done"]
    assert fake_process.killed is False  # normal completion — nothing left to kill


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
